import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d_plugin._compat import BaseModule, auto_fp16
from mmdet3d_plugin.ops import SmoothSampler, grid_sample_3d


class LaplaceDensity(nn.Module):
    """Laplace density used to transform SDF values into density values."""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter(
            "beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False)
        )
        self.register_parameter(
            "beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()
        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        return self.beta.abs() + self.beta_min


class SingleVarianceNetwork(nn.Module):
    """Variance network used by NeuS alpha computation."""

    def __init__(self, init_val):
        super().__init__()
        self.register_parameter(
            "variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(
            self.variance * 10.0
        )

    def get_variance(self):
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class SDFDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=256, n_blocks=5):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for layer_idx in range(self.num_layers - 1):
            setattr(self, f"lin{layer_idx}", nn.Linear(dims[layer_idx], dims[layer_idx + 1]))

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for _ in range(self.num_layers - 1)]
        )
        self.fc_p = nn.Linear(3, hidden_size)
        self.activation = nn.Softplus(beta=100)

    def forward(self, points, point_feats):
        x = self.fc_p(points)
        for layer_idx in range(self.num_layers - 1):
            x = x + self.fc_c[layer_idx](point_feats)
            x = getattr(self, f"lin{layer_idx}")(x)
            if layer_idx < self.num_layers - 2:
                x = self.activation(x)
        return x


class LiDARDecoder(nn.Module):
    """Predict intensity, raydrop, and depth correction from per-ray features."""

    def __init__(self, in_dim=32, hidden_dim=128, out_dim=3, num_layers=5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.output_head = nn.Linear(hidden_dim, out_dim)

        import math

        nn.init.constant_(self.output_head.bias[1], -math.log(9.0))
        nn.init.constant_(self.output_head.bias[2], 0.0)

    def forward(self, ray_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(ray_features))
        for block in self.blocks:
            x = x + block(x)
        return self.output_head(x)


class SDFFieldLidarOnly(BaseModule):
    """SDF field for LiDAR rendering with no RGB decoder or RGB outputs."""

    def __init__(
        self,
        voxel_size,
        pc_range,
        voxel_shape,
        scale_factor,
        sdf_decoder_cfg,
        interpolate_cfg,
        beta_init,
        **kwargs,
    ):
        super().__init__()
        self.fp16_enabled = kwargs.get("fp16_enabled", False)
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.voxel_shape = voxel_shape
        self.beta_init = beta_init
        self.interpolate_cfg = interpolate_cfg
        self.scale_factor = scale_factor
        self.debug_numerics = kwargs.get("debug_numerics", False)
        self.sdf_decoder = SDFDecoder(**sdf_decoder_cfg)
        self.laplace_density = LaplaceDensity(init_val=self.beta_init)
        self.deviation_network = SingleVarianceNetwork(init_val=self.beta_init)
        self._cos_anneal_ratio = 1.0

    def set_cos_anneal_ratio(self, anneal):
        self._cos_anneal_ratio = anneal

    def get_alpha(self, ray_samples, sdf, gradients):
        inv_s = self.deviation_network.get_variance()
        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self._cos_anneal_ratio)
            + F.relu(-true_cos) * self._cos_anneal_ratio
        )

        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        return ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

    def _assert_finite(self, name, tensor):
        if not self.debug_numerics:
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        finite_count = int(finite.sum().item())
        total_count = tensor.numel()
        finite_values = tensor[finite]
        min_value = finite_values.min().item() if finite_count else "none"
        max_value = finite_values.max().item() if finite_count else "none"
        raise FloatingPointError(
            f"{name} has non-finite values: "
            f"shape={tuple(tensor.shape)}, finite={finite_count}/{total_count}, "
            f"min={min_value}, max={max_value}"
        )

    def interpolate_feats(self, pts, feats_volume):
        pc_range = pts.new_tensor(self.pc_range)
        norm_coords = (pts / self.scale_factor - pc_range[:3]) / (
            pc_range[3:] - pc_range[:3]
        )
        assert (
            self.voxel_shape[0] == feats_volume.shape[3]
            and self.voxel_shape[1] == feats_volume.shape[2]
            and self.voxel_shape[2] == feats_volume.shape[1]
        )
        norm_coords = norm_coords * 2 - 1
        self._assert_finite("sdf_field.points", pts)
        self._assert_finite("sdf_field.norm_coords", norm_coords)
        self._assert_finite("sdf_field.feature_volume", feats_volume)
        if (
            self.interpolate_cfg["type"] == "SmoothSampler"
            and SmoothSampler is not None
        ):
            feats = (
                SmoothSampler.apply(
                    feats_volume.unsqueeze(0),
                    norm_coords[None, None, ...],
                    self.interpolate_cfg["padding_mode"],
                    True,
                    False,
                )
                .squeeze(0)
                .squeeze(1)
                .permute(1, 2, 0)
            )
        else:
            feats = (
                grid_sample_3d(feats_volume.unsqueeze(0), norm_coords[None, None, ...])
                .squeeze(0)
                .squeeze(1)
                .permute(1, 2, 0)
            )
        self._assert_finite("sdf_field.point_features", feats)
        return feats

    @auto_fp16(apply_to=("points", "feature_volume"))
    def get_sdf(self, points, feature_volume):
        point_features = self.interpolate_feats(points, feature_volume)
        decoded = self.sdf_decoder(points, point_features)
        sdf, geo_features = decoded[..., :1], decoded[..., 1:]
        return sdf, geo_features, point_features

    def get_density(self, ray_samples, feature_volume):
        points = ray_samples.frustums.get_start_positions()
        sdf, _, _ = self.get_sdf(points, feature_volume)
        return self.laplace_density(sdf)

    def get_occupancy(self, sdf):
        return torch.sigmoid(-10.0 * sdf)

    @auto_fp16(out_fp32=True)
    def forward(self, ray_samples, feature_volume, return_alphas=False):
        outputs = {}
        points = ray_samples.frustums.get_start_positions()
        points.requires_grad_(True)
        with torch.enable_grad():
            sdf, _, point_features = self.get_sdf(points, feature_volume)

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=d_output,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )[0]
        density = self.laplace_density(sdf)

        outputs.update(
            {
                "density": density,
                "sdf": sdf,
                "gradients": gradients,
                "point_features": point_features,
            }
        )
        if return_alphas:
            outputs["alphas"] = self.get_alpha(ray_samples, sdf, gradients)
        return outputs
