import torch
import torch.nn as nn
import numpy as np

from mmdet3d_plugin.models.dense_heads.render_utils.models import NeuSLidarOnly
from mmdet3d_plugin.models.dense_heads.render_utils.rays import RayBundle


class LidarOnlyRenderHead(nn.Module):
    """Render head for lidar-only training (no RGB rays).

    Training: uses GT point directions as rays (including non-return rays for
    raydrop supervision). Each GT point becomes exactly one ray.

    Inference: uses a predefined 32x1080 beam grid. The raydrop head filters
    rays that don't hit surfaces.
    """

    def __init__(self, unified_voxel_size, unified_voxel_shape, pc_range, render_cfg):
        super().__init__()
        render_cfg = dict(render_cfg)
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.unified_voxel_size = np.array(unified_voxel_size, dtype=np.float32)
        self.unified_voxel_shape = np.array(unified_voxel_shape, dtype=np.int32)

        self.close_radius = render_cfg.pop("close_radius", 1.0)
        self.far_radius = render_cfg.pop("far_radius", 80.0)
        self.part = render_cfg.pop("part", 8192)  # chunk size for inference
        self.max_train_rays = render_cfg.pop(
            "max_train_rays", 0
        )  # 0 = all rays, >0 = subsample

        # Pop out legacy keys that should not go to NeuSLidarOnly
        render_cfg.pop("use_predefined_rays", None)
        render_cfg.pop("use_predefine_rays", None)
        render_cfg.pop("ray_sampling_mode", None)

        predefined_rays_cfg = render_cfg.pop("predefined_rays_cfg", None)
        if predefined_rays_cfg is None:
            predefined_rays_cfg = render_cfg.pop("predefine_rays_cfg", None)
        self.predefined_rays_cfg = dict(predefined_rays_cfg or {})
        render_cfg.pop("predefined_elevation_tolerance_deg", None)

        self.render_model = NeuSLidarOnly(
            pc_range=self.pc_range,
            voxel_size=self.unified_voxel_size,
            voxel_shape=self.unified_voxel_shape,
            **render_cfg,
        )

    @property
    def scale_factor(self):
        return self.render_model.scale_factor

    @staticmethod
    def _cfg_get(cfg: dict, keys: tuple[str, ...], default=None):
        for key in keys:
            if key in cfg:
                return cfg[key]
        return default

    def _predefined_spec(self, device, dtype):
        cfg = self.predefined_rays_cfg
        azimuth_range = self._cfg_get(cfg, ("azimuth_range",), [0.0, 360.0])
        azimuth_min = float(
            self._cfg_get(cfg, ("azimuth_min_deg", "azimuth_min"), azimuth_range[0])
        )
        azimuth_max = float(
            self._cfg_get(cfg, ("azimuth_max_deg", "azimuth_max"), azimuth_range[1])
        )
        azimuth_res = self._cfg_get(
            cfg, ("azimuth_res", "azimuth_resolution_deg"), None
        )
        num_azimuth = self._cfg_get(
            cfg, ("num_azimuth", "azimuth_beams", "azimuth_bins"), None
        )
        if num_azimuth is None:
            if azimuth_res is not None:
                num_azimuth = round((azimuth_max - azimuth_min) / float(azimuth_res))
            else:
                num_azimuth = 1080
        num_azimuth = int(num_azimuth)

        elevation_angles = self._cfg_get(cfg, ("elevation_angles_deg",), None)
        if elevation_angles is None:
            elevation_range = self._cfg_get(cfg, ("elevation_range",), [-30.67, 10.67])
            elevation_min = float(
                self._cfg_get(
                    cfg, ("elevation_min_deg", "elevation_min"), elevation_range[0]
                )
            )
            elevation_max = float(
                self._cfg_get(
                    cfg, ("elevation_max_deg", "elevation_max"), elevation_range[1]
                )
            )
            num_beams = int(self._cfg_get(cfg, ("num_beams", "elevation_beams"), 32))
            elevations_deg = torch.linspace(
                elevation_min, elevation_max, num_beams, device=device, dtype=dtype
            )
        else:
            elevations_deg = torch.as_tensor(
                elevation_angles, device=device, dtype=dtype
            )
            num_beams = int(elevations_deg.numel())

        azimuths_deg = torch.linspace(
            azimuth_min,
            azimuth_max,
            num_azimuth + 1,
            device=device,
            dtype=dtype,
        )[:-1]

        return (
            elevations_deg,
            azimuths_deg,
            (num_beams, num_azimuth),
            (azimuth_min, azimuth_max),
        )

    def _predefined_ray_directions(self, device, dtype):
        elevations_deg, azimuths_deg, shape, azimuth_range = self._predefined_spec(
            device, dtype
        )
        elevations = torch.deg2rad(elevations_deg)
        azimuths = torch.deg2rad(azimuths_deg)
        elev_grid, azim_grid = torch.meshgrid(elevations, azimuths, indexing="ij")
        elev_flat = elev_grid.reshape(-1)
        azim_flat = azim_grid.reshape(-1)
        ray_d = torch.stack(
            [
                torch.cos(elev_flat) * torch.cos(azim_flat),
                torch.cos(elev_flat) * torch.sin(azim_flat),
                torch.sin(elev_flat),
            ],
            dim=-1,
        )
        return ray_d, shape

    def forward(
        self,
        uni_feats: torch.Tensor,
        rays: list[dict],
        occupancy_mask: torch.Tensor | None = None,
    ) -> list[dict]:
        """Forward pass: render lidar rays from voxel features."""
        batch_ret = []
        batch_size = uni_feats.shape[0]

        for bs_idx in range(batch_size):
            i_ray_o = rays[bs_idx]["ray_o"]
            i_ray_d = rays[bs_idx]["ray_d"]
            i_ray_depth = rays[bs_idx].get("depth", None)
            i_occ_mask = occupancy_mask[bs_idx] if occupancy_mask is not None else None

            if self.training:
                if "scaled_points" not in rays[bs_idx]:
                    raise KeyError("Training lidar rays must include 'scaled_points'.")
                scaled_points = rays[bs_idx]["scaled_points"]
                ray_bundle = RayBundle(
                    origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth
                )
                preds_dict = self.render_model(
                    ray_bundle, uni_feats[bs_idx], points=scaled_points,
                    occupancy_mask=i_occ_mask,
                )
            else:
                # Chunk to avoid OOM during inference
                num_rays = i_ray_o.shape[0]
                num_parts = (num_rays - 1) // self.part + 1
                part_ret = []
                for p in range(num_parts):
                    s = p * self.part
                    e = min((p + 1) * self.part, num_rays)
                    ray_bundle = RayBundle(
                        origins=i_ray_o[s:e], directions=i_ray_d[s:e]
                    )
                    part_preds = self.render_model(
                        ray_bundle, uni_feats[bs_idx],
                        occupancy_mask=i_occ_mask,
                    )
                    part_ret.append(
                        {
                            k: v.detach() if v is not None else None
                            for k, v in part_preds.items()
                        }
                    )
                # Merge chunks
                preds_dict = {}
                for p_ret in part_ret:
                    for k, v in p_ret.items():
                        if k not in preds_dict:
                            preds_dict[k] = []
                        preds_dict[k].append(v)
                for k, v in preds_dict.items():
                    if v[0] is None:
                        preds_dict[k] = None
                    else:
                        preds_dict[k] = torch.cat(v, dim=0)

            batch_ret.append(preds_dict)

        return batch_ret

    def loss(self, preds_dict: list[dict], targets: list[dict]) -> dict:
        """Compute losses over the batch."""
        loss_dict = {}
        for bs_idx in range(len(targets)):
            i_loss_dict = self.render_model.loss(preds_dict[bs_idx], targets[bs_idx])
            for k, v in i_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = []
                loss_dict[k].append(v)
        for k, v in loss_dict.items():
            loss_dict[k] = torch.stack(v, dim=0).mean()
        return loss_dict

    # ------------------------------------------------------------------
    # Training: point-derived rays with did_return support
    # ------------------------------------------------------------------

    def sample_lidar_rays(
        self,
        points: list[torch.Tensor],
        did_return: list[torch.Tensor] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Sample lidar rays for training from GT point directions.

        Each GT point (including non-return dummy points) becomes one ray.
        Non-return rays train the raydrop head via BCE loss.

        Args:
            points: List of (N_i, >=4) point clouds [x, y, z, intensity, ...].
            did_return: List of (N_i,) bool tensors. If None, all points are returns.
        Returns:
            Tuple of (rays_list, targets_list).
        """
        scale_factor = self.scale_factor
        rays_list = []
        targets_list = []

        for bs_idx in range(len(points)):
            i_pts = points[bs_idx]
            device = i_pts.device

            positions = i_pts[:, :3]
            dis = positions.norm(dim=-1)

            # Get did_return mask
            if did_return is not None:
                i_did_return = did_return[bs_idx].bool()
            else:
                i_did_return = torch.ones(
                    positions.shape[0], dtype=torch.bool, device=device
                )

            # Filter by distance: keep returned points within range,
            # keep ALL non-returned points (same as UniScene)
            valid = ((dis > self.close_radius) & (dis < self.far_radius)) | (
                ~i_did_return
            )
            positions = positions[valid]
            i_did_return = i_did_return[valid]

            # Intensity
            if i_pts.shape[1] > 3:
                intensity = i_pts[valid, 3:4]
            else:
                intensity = torch.zeros(positions.shape[0], 1, device=device)

            # Subsample rays during training to avoid OOM
            num_rays = positions.shape[0]
            if (
                self.training
                and self.max_train_rays > 0
                and num_rays > self.max_train_rays
            ):
                perm = torch.randperm(num_rays, device=device)[: self.max_train_rays]
                positions = positions[perm]
                i_did_return = i_did_return[perm]
                intensity = intensity[perm]

            # Directions from origin to each point
            ranges = positions.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            directions = positions / ranges

            # Origins at (0,0,0)
            origins = torch.zeros_like(positions)

            # Scale
            ray_o = origins * scale_factor
            ray_d = directions
            depth = ranges * scale_factor
            # For non-return rays, set depth to 0 (convention: no valid depth)
            depth = depth * i_did_return.unsqueeze(-1).float()

            # scaled_points for optional SDF supervision (only from returned points)
            returned_positions = positions[i_did_return]
            if returned_positions.shape[0] == 0:
                scaled_points = torch.zeros(1, 3, device=device)
            else:
                scaled_points = returned_positions * scale_factor

            rays_list.append(
                {
                    "ray_o": ray_o,
                    "ray_d": ray_d,
                    "depth": depth,
                    "scaled_points": scaled_points,
                }
            )
            targets_list.append(
                {
                    "depth": depth,
                    "intensity": intensity,
                    "did_return": i_did_return.unsqueeze(-1).float(),
                }
            )

        return rays_list, targets_list

    # ------------------------------------------------------------------
    # Inference: predefined beam grid
    # ------------------------------------------------------------------

    def sample_lidar_inference_rays(
        self,
        points: list[torch.Tensor] | None = None,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        num_beams: int | None = None,
        num_azimuth: int | None = None,
    ) -> list[dict]:
        """Sample a dense predefined grid of lidar rays for inference.

        32 elevation beams x 1080 azimuth bins = 34,560 rays by default.
        The raydrop head will filter rays that don't hit surfaces.
        """
        if points is not None:
            if batch_size is None:
                batch_size = len(points)
            if len(points) > 0:
                if device is None:
                    device = points[0].device
                if dtype is None:
                    dtype = points[0].dtype

        if batch_size is None:
            raise ValueError(
                "sample_lidar_inference_rays requires batch_size or points."
            )
        if device is None:
            raise ValueError("sample_lidar_inference_rays requires device or points.")
        if dtype is None:
            dtype = torch.float32

        device = torch.device(device)
        cfg = dict(self.predefined_rays_cfg)
        if num_beams is not None:
            cfg["num_beams"] = num_beams
        if num_azimuth is not None:
            cfg["num_azimuth"] = num_azimuth

        old_cfg = self.predefined_rays_cfg
        self.predefined_rays_cfg = cfg
        try:
            ray_d, shape = self._predefined_ray_directions(device, dtype)
        finally:
            self.predefined_rays_cfg = old_cfg

        total_rays = ray_d.shape[0]
        ray_o = torch.zeros(total_rays, 3, device=device, dtype=dtype)

        batch_ret = []
        for _ in range(batch_size):
            batch_ret.append(
                {
                    "ray_o": ray_o,
                    "ray_d": ray_d,
                    "lidar_render_shape": shape,
                }
            )

        return batch_ret
