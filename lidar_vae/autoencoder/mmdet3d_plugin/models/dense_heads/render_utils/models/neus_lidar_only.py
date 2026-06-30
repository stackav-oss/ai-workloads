from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from mmdet3d_plugin._compat import BaseModule, HEADS, auto_fp16
from mmdet3d_plugin.models.dense_heads.render_utils import (
    fields,
    ray_samplers,
    scene_colliders,
)
from mmdet3d_plugin.models.dense_heads.render_utils.fields.sdf_field import (
    LiDARDecoder,
)
from mmdet3d_plugin.models.dense_heads.render_utils.renderers import DepthRenderer


class _AttrDict(dict):
    """Dict subclass that allows attribute-style access for loss config."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, dict) and not isinstance(val, _AttrDict):
            val = _AttrDict(val)
            self[key] = val
        return val


@HEADS.register_module()
class NeuSLidarOnly(BaseModule):
    """NeuS-style LiDAR renderer with no RGB field, RGB renderer, or camera loss."""

    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        field_cfg,
        collider_cfg,
        sampler_cfg,
        loss_cfg,
        norm_scene,
        **kwargs,
    ):
        super().__init__()
        self.fp16_enabled = kwargs.get("fp16_enabled", False)
        self.scale_factor = 1.0 / np.max(np.abs(pc_range)) if norm_scene else 1.0

        field_cfg = dict(field_cfg)
        field_cfg.setdefault("type", "SDFFieldLidarOnly")
        field_feature_dim = field_cfg.get("sdf_decoder_cfg", {}).get("in_dim", 32)
        field_type = field_cfg.pop("type")
        self.field = getattr(fields, field_type)(
            voxel_size=voxel_size,
            pc_range=pc_range,
            voxel_shape=voxel_shape,
            scale_factor=self.scale_factor,
            **field_cfg,
        )

        collider_cfg = dict(collider_cfg)
        collider_type = collider_cfg.pop("type")
        self.collider = getattr(scene_colliders, collider_type)(
            scene_box=pc_range,
            scale_factor=self.scale_factor,
            **collider_cfg,
        )

        sampler_cfg = dict(sampler_cfg)
        sampler_type = sampler_cfg.pop("type")
        self.sampler = getattr(ray_samplers, sampler_type)(**sampler_cfg)
        self.depth_renderer = DepthRenderer()

        self.pred_intensity = kwargs.get("pred_intensity", False)
        self.pred_raydrop = kwargs.get("pred_raydrop", False)
        if self.pred_intensity or self.pred_raydrop:
            self.lidar_decoder = LiDARDecoder(
                in_dim=field_feature_dim,
                hidden_dim=128,
                out_dim=3,
            )

        self.loss_cfg = _AttrDict(loss_cfg) if isinstance(loss_cfg, dict) else loss_cfg
        self.anneal_end = 50000

    def sample_and_forward_field(self, ray_bundle, feature_volume, **kwargs):
        occupancy_mask = kwargs.pop("occupancy_mask", None)
        sampler_out_dict = self.sampler(
            ray_bundle,
            occupancy_fn=self.field.get_occupancy,
            sdf_fn=partial(self.field.get_sdf, feature_volume=feature_volume),
            sdf_field=self.field,
            feature_volume=feature_volume,
            occupancy_mask=occupancy_mask,
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, feature_volume, return_alphas=True)
        weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs["alphas"]
        )
        return {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "sampled_points": ray_samples.frustums.get_start_positions(),
            **sampler_out_dict,
        }

    def get_outputs(self, lidar_ray_bundle, feature_volume, **kwargs):
        samples_and_field_outputs = self.sample_and_forward_field(
            lidar_ray_bundle,
            feature_volume,
            **kwargs,
        )
        ray_samples = samples_and_field_outputs["ray_samples"]
        field_outputs = samples_and_field_outputs["field_outputs"]
        weights = samples_and_field_outputs["weights"]

        depth = self.depth_renderer(ray_samples=ray_samples, weights=weights)
        lidar_output = None
        depth_correction = None
        if hasattr(self, "lidar_decoder") and (
            self.pred_intensity or self.pred_raydrop
        ):
            point_feats = field_outputs["point_features"]
            ray_features = (point_feats * weights).sum(dim=1)
            lidar_output = self.lidar_decoder(ray_features)
            depth_correction = lidar_output[:, 2:3] * 0.5

        outputs = {
            "depth": depth,
            "depth_correction": depth_correction,
            "sdf": field_outputs["sdf"],
            "gradients": field_outputs["gradients"],
            "z_vals": ray_samples.frustums.starts,
            "lidar_output": lidar_output,
        }

        if self.training and self.loss_cfg.get("sparse_points_sdf_supervised", False):
            sparse_points_sdf, _, _ = self.field.get_sdf(
                kwargs["points"].unsqueeze(0), feature_volume
            )
            outputs["sparse_points_sdf"] = sparse_points_sdf.squeeze(0)

        return outputs

    @auto_fp16(apply_to=("feature_volume",))
    def forward(self, lidar_ray_bundle, feature_volume, **kwargs):
        lidar_ray_bundle = self.collider(lidar_ray_bundle)
        return self.get_outputs(lidar_ray_bundle, feature_volume, **kwargs)

    def g_loss(self, preds_dict, lidar_targets):
        depth_pred = preds_dict["depth"]
        depth_gt = lidar_targets["depth"]

        loss_dict = {}
        loss_weights = self.loss_cfg.weights

        depth_correction = preds_dict.get("depth_correction", None)
        if depth_correction is not None:
            depth_pred = depth_pred + depth_correction

        valid_gt_mask = depth_gt > 0.0
        did_return = lidar_targets["did_return"].view(valid_gt_mask.shape).bool()
        valid_gt_mask = valid_gt_mask & did_return

        if loss_weights.get("depth_loss", 0.0) > 0:
            depth_loss = torch.sum(
                valid_gt_mask * torch.abs(depth_gt - depth_pred)
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["depth_loss"] = depth_loss * loss_weights.depth_loss

        lidar_output = preds_dict.get("lidar_output", None)
        if lidar_output is not None and self.pred_intensity:
            intensity_pred = lidar_output[:, 0:1].sigmoid()
            intensity_gt = lidar_targets["intensity"]
            intensity_loss = torch.sum(
                valid_gt_mask * (intensity_gt - intensity_pred) ** 2
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["intensity_loss"] = intensity_loss * loss_weights.intensity_loss

        if lidar_output is not None and self.pred_raydrop:
            raydrop_pred = lidar_output[:, 1:2]
            did_return_flat = lidar_targets["did_return"].view(-1)
            raydrop_gt = (~did_return_flat.bool()).float()
            n_rays = did_return_flat.shape[0]
            raydrop_loss = F.binary_cross_entropy_with_logits(
                raydrop_pred.view(-1),
                raydrop_gt,
                reduction="sum",
            ) / max(n_rays, 1)
            loss_dict["raydrop_loss"] = raydrop_loss * loss_weights.raydrop_loss

        pred_sdf = preds_dict["sdf"][..., 0]
        z_vals = preds_dict["z_vals"][..., 0]
        truncation = self.loss_cfg.sensor_depth_truncation * self.scale_factor

        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        if loss_weights.get("free_space_loss", 0.0) > 0:
            free_space_loss = (
                F.relu(truncation - pred_sdf) * front_mask
            ).sum() / torch.clamp(front_mask.sum(), min=1.0)
            loss_dict["free_space_loss"] = (
                free_space_loss * loss_weights.free_space_loss
            )

        if loss_weights.get("sdf_loss", 0.0) > 0:
            sdf_loss = (
                torch.abs(z_vals + pred_sdf - depth_gt) * sdf_mask
            ).sum() / torch.clamp(sdf_mask.sum(), min=1.0)
            loss_dict["sdf_loss"] = sdf_loss * loss_weights.sdf_loss

        if loss_weights.get("eikonal_loss", 0.0) > 0:
            gradients = preds_dict["gradients"]
            eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
            loss_dict["eikonal_loss"] = eikonal_loss * loss_weights.eikonal_loss

        if self.loss_cfg.get("sparse_points_sdf_supervised", False):
            sparse_points_sdf_loss = torch.mean(
                torch.abs(preds_dict["sparse_points_sdf"])
            )
            loss_dict["sparse_points_sdf_loss"] = (
                sparse_points_sdf_loss * loss_weights.sparse_points_sdf_loss
            )

        return loss_dict

    def loss(self, preds_dict, lidar_targets):
        return self.g_loss(preds_dict, lidar_targets)
