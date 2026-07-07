"""Train LiDAR-only rendering from scratch (no pre-trained checkpoint).

Self-contained training script: model is defined inline (no model.py dependency).
Architecture matches UniScene's occ2lidar_render approach:
  - LiDAR encoder → shared 3D feature volume
  - NeuSLidarOnly field renders ALL GT points as rays (same for train/inference)
  - Predicts depth, intensity, raydrop

Usage (inside the lidar_vae Docker container):
    # Single GPU:
    python train_lidar_only.py --max-steps 10000 --overfit-samples 20

    # Multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 train_lidar_only.py --max-steps 10000 --overfit-samples 20
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from jormungand.datasets.nuscenes.nuscenes_lance import NuScenesLanceDataset
from jormungand.datasets.nuscenes.nuscenes_dataframe_utils import (
    NuScenesDataConfig,
    NuScenesDataFrame,
    aggregate_lidar_sweeps,
)
from lidar_encoder import LiDAREncoder
from lidar_only_render_head import LidarOnlyRenderHead

from bev_pillar_pooling import BEVPillarPooling
from bev_2d_fpn import BEV2DFPN
from metrics.chamfer_distance import ChamferDistanceMetrics
from metrics.jsd import JensenShannonDivergence
from metrics.mmd import MaximumMeanDiscrepancy

# Generate occupancy GT for loss computation if we are outputting occupancy from 2D BEV FPN.
def compute_occupancy_gt(
    points_batch: list[torch.Tensor],
    pc_range: list[float],
    voxel_size: list[float],
    grid_shape: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute binary occupancy ground truth from point clouds.

    Voxelizes all input points (from aggregated sweeps) into a binary grid.
    Uses ONLY returned points (not the synthetic missing-ray points).

    Args:
        points_batch: List of (N_i, >=3) point tensors. Should be the
                      `aggregated_points` (real returns only, no missing rays).
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        grid_shape: [nx, ny, nz] = [1024, 1024, 64]
        device: Target device.

    Returns:
        occ_gt: (B, nz, ny, nx) = (B, 64, 1024, 1024) float tensor
                with 1.0 for occupied voxels, 0.0 for empty.
                Shape matches the feature volume layout (B, D, H, W).
    """
    B = len(points_batch)
    nx, ny, nz = grid_shape  # [1024, 1024, 64]
    occ_gt = torch.zeros(B, nz, ny, nx, device=device)

    x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]

    for b in range(B):
        pts = points_batch[b][:, :3].to(device)

        # Compute voxel indices
        vx = ((pts[:, 0] - x_min) / voxel_size[0]).long()
        vy = ((pts[:, 1] - y_min) / voxel_size[1]).long()
        vz = ((pts[:, 2] - z_min) / voxel_size[2]).long()

        # Clamp to valid range
        valid = (vx >= 0) & (vx < nx) & (vy >= 0) & (vy < ny) & (vz >= 0) & (vz < nz)
        vx, vy, vz = vx[valid], vy[valid], vz[valid]

        # Mark occupied voxels
        # Layout: occ_gt[b, z, y, x] = 1.0
        occ_gt[b, vz, vy, vx] = 1.0

    return occ_gt


# Occupancy loss
def occupancy_loss(
    occ_logits: torch.Tensor,
    occ_gt: torch.Tensor,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """Binary cross-entropy loss for occupancy prediction.

    Args:
        occ_logits: (B, 64, 1024, 1024) raw logits from occ_head
        occ_gt:     (B, 64, 1024, 1024) binary GT (1=occupied, 0=empty)
        pos_weight: Weight for positive (occupied) class to handle extreme
                    class imbalance (~0.05% occupied, ~99.95% empty).

    Returns:
        Scalar loss value.
    """
    pw = torch.tensor([pos_weight], device=occ_logits.device, dtype=occ_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        occ_logits, occ_gt, pos_weight=pw, reduction="mean"
    )
    return loss


# ---------------------------------------------------------------------------
# Model (self-contained, no model.py dependency)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_occupancy_counts(occ_logits, occ_gt, threshold=0.0):
    pred = (occ_logits > threshold).bool()
    gt = occ_gt.bool()
    tp = (pred & gt).sum().double()
    fp = (pred & ~gt).sum().double()
    fn = (~pred & gt).sum().double()
    return torch.stack([tp, fp, fn], dim=0)


@torch.no_grad()
def compute_occupancy_metrics(occ_logits, occ_gt, threshold=0.0):
    tp, fp, fn = compute_occupancy_counts(occ_logits, occ_gt, threshold=threshold)
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {
        "occ_iou": iou.item(),
        "occ_precision": precision.item(),
        "occ_recall": recall.item(),
    }


class LidarOnlyVAE(nn.Module):
    """Self-contained lidar-only rendering model.

    Architecture (matches UniScene occ2lidar_render):
      1. LiDAR encoder → pts_feats (B, C, D, H, W)
      2. BEV pooling + BEV FPN/VAE → neural feature grid
      3. Sample rays from ALL GT points (UniScene-style)
      4. NeuSLidarOnly renders rays → depth, intensity, raydrop
    """

    def __init__(
        self,
        lidar_encoder,
        bev_pillar_pooling,
        bev_2d_fpn,
        render_head,
        render_cfg,
        pc_range,
        train_vae: bool = True,
    ):
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.bev_pooling = bev_pillar_pooling
        self.bev_2d_fpn = bev_2d_fpn
        self.render_head = render_head
        self.render_cfg = render_cfg
        self.pc_range = tuple(float(value) for value in pc_range)
        self.train_vae = train_vae
        self.kl_weight_scale = 1.0
        self.debug_numerics = False

    def set_kl_weight_scale(
        self, scale: float, stochastic_sampling: bool = True
    ) -> None:
        """Set the scheduled multiplier for the configured peak KL weight."""
        if not self.train_vae:
            self.kl_weight_scale = 0.0
            if self.bev_2d_fpn is not None:
                self.bev_2d_fpn.set_stochastic_sampling(False)
            return
        if self.bev_2d_fpn is not None:
            self.bev_2d_fpn.set_stochastic_sampling(stochastic_sampling)
        self.kl_weight_scale = max(0.0, float(scale))

    def set_vae_warmup(self, active: bool) -> None:
        """Disable stochastic VAE sampling and KL while the AE path stabilizes."""
        self.set_kl_weight_scale(
            0.0 if active else 1.0,
            stochastic_sampling=not active,
        )

    def set_debug_numerics(self, enabled: bool) -> None:
        self.debug_numerics = enabled
        if self.bev_2d_fpn is not None:
            self.bev_2d_fpn.debug_numerics = enabled
        render_model = getattr(self.render_head, "render_model", None)
        field = getattr(render_model, "field", None)
        sampler = getattr(render_model, "sampler", None)
        pdf_sampler = getattr(sampler, "pdf_sampler", None)
        for module in (field, sampler, pdf_sampler):
            if hasattr(module, "debug_numerics"):
                module.debug_numerics = enabled

    def _assert_finite(self, name: str, tensor: torch.Tensor | None) -> None:
        if not self.debug_numerics or tensor is None:
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

    def _extract_features(
        self,
        aggregated_points: list[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Shared feature extraction for BOTH training AND inference.

        Returns:
            nfg:        (B, 16, 64, 1024, 1024) neural feature grid for render head
            occ_logits: (B, 64, 1024, 1024) occupancy logits (or None)
            z_sampled:  (B, 32, 128, 128) sampled VAE latent or AE bottleneck
            mu:         (B, 32, 128, 128) or None when train_vae=False
            log_var:    (B, 32, 128, 128) or None when train_vae=False
        """
        # ─── LiDAR Encode ───
        pts_feats = None
        if self.lidar_encoder is not None:
            pts_feats = self.lidar_encoder(aggregated_points)
        if pts_feats is None:
            raise RuntimeError("LiDAR encoder did not produce features.")
        uni_feats = pts_feats

        # ─── BEV Pooling ───
        if self.bev_pooling is not None:
            uni_feats = self.bev_pooling(uni_feats)

        # ─── BEV 2D FPN → AE/VAE bottleneck ───
        occ_logits = None
        z_sampled = None
        mu = None
        log_var = None
        nfg = uni_feats  # fallback

        if self.bev_2d_fpn is not None:
            _, z_sampled, occ_logits, nfg, mu, log_var = self.bev_2d_fpn(uni_feats)
            return nfg, occ_logits, z_sampled, mu, log_var

        return nfg, occ_logits, z_sampled, mu, log_var

    def forward(
        self,
        aggregated_points: list[torch.Tensor],
        points: list[torch.Tensor],
        did_return: list[torch.Tensor] | None = None,
        occ_gt: torch.Tensor | None = None,
        use_predefined_rays: bool = False,
    ) -> dict:
        """Forward pass: encode → fuse → sample rays → render → loss."""
        device = points[0].device
        nfg, occ_logits, z_sampled, mu, log_var = self._extract_features(
            aggregated_points
        )
        losses = {}

        # Occupancy Loss
        if occ_logits is not None and occ_gt is not None:
            losses["occupancy_loss"] = (
                occupancy_loss(occ_logits, occ_gt, pos_weight=5.0)
                * self.render_cfg["loss_cfg"]["weights"]["occupancy_loss"]
            )

        # KL Divergence Loss
        if mu is not None and log_var is not None:
            log_var_safe = log_var.clamp(min=-10.0, max=10.0)
            kl_loss = -0.5 * torch.mean(
                1.0 + log_var_safe - mu.pow(2) - torch.exp(log_var_safe)
            )
            losses["kl_loss_raw"] = kl_loss.detach()
            peak_kl_weight = self.render_cfg["loss_cfg"]["weights"]["kl_loss"]
            effective_kl_weight = peak_kl_weight * self.kl_weight_scale
            losses["kl_loss"] = kl_loss * effective_kl_weight
            losses["kl_weight"] = float(effective_kl_weight)
            losses["kl_weight_scale"] = float(self.kl_weight_scale)
            with torch.no_grad():
                true_kl = kl_loss.detach()
                losses["latent_mu_mean"] = mu.detach().mean().item()
                losses["latent_mu_std"] = mu.detach().std().item()
                latent_std = torch.exp(0.5 * log_var_safe)
                losses["latent_std_mean"] = latent_std.mean().item()
                losses["latent_std_std"] = latent_std.std().item()
                losses["latent_z_mean"] = z_sampled.detach().mean().item()
                losses["latent_z_std"] = z_sampled.detach().std().item()
                losses["latent_true_kl"] = true_kl.item()
                z_stats = z_sampled.detach()
                if z_stats.ndim == 4:
                    channel_mean = z_stats.mean(dim=(0, 2, 3))
                    channel_std = z_stats.std(dim=(0, 2, 3))
                    mean_abs_p95 = torch.quantile(channel_mean.abs(), 0.95)
                    std_error_p95 = torch.quantile((channel_std - 1.0).abs(), 0.95)
                    losses["latent_channel_mean_abs_p95"] = mean_abs_p95.item()
                    losses["latent_channel_std_error_p95"] = std_error_p95.item()
                    losses["latent_gaussian_score"] = (
                        mean_abs_p95 + std_error_p95 + true_kl
                    ).item()

        # Sample rays
        if use_predefined_rays:
            B = len(aggregated_points)
            lidar_rays = self.render_head.sample_lidar_inference_rays(
                batch_size=B, device=device, dtype=nfg.dtype
            )
            # Build targets from GT points for loss computation
            _, lidar_targets = self.render_head.sample_lidar_rays(
                points=points, did_return=did_return
            )
        else:
            lidar_rays, lidar_targets = self.render_head.sample_lidar_rays(
                points=points, did_return=did_return
            )

        self._assert_finite("render.nfg", nfg)
        self._assert_finite("render.occ_logits", occ_logits)
        for ray_idx, lidar_ray in enumerate(lidar_rays):
            self._assert_finite(f"lidar_rays[{ray_idx}].ray_o", lidar_ray["ray_o"])
            self._assert_finite(f"lidar_rays[{ray_idx}].ray_d", lidar_ray["ray_d"])
            self._assert_finite(f"lidar_rays[{ray_idx}].depth", lidar_ray.get("depth"))
            self._assert_finite(
                f"lidar_rays[{ray_idx}].scaled_points",
                lidar_ray.get("scaled_points"),
            )

        # Render
        preds = self.render_head(uni_feats=nfg, rays=lidar_rays)

        # Loss
        losses.update(self.render_head.loss(preds, lidar_targets))
        return losses

    @torch.no_grad()
    def render_lidar(
        self,
        aggregated_points: list[torch.Tensor],
        points: list[torch.Tensor],
    ) -> dict:
        """Inference: render a generated LiDAR sweep from predefined sensor rays.

        Validation still receives GT points so the saved file can include GT for
        comparison, but GT points do not define the generated inference rays.
        """
        self.eval()
        B = len(aggregated_points)

        nfg, occ_logits, _, _, _ = self._extract_features(aggregated_points)

        lidar_rays = self.render_head.sample_lidar_inference_rays(
            batch_size=B,
            device=nfg.device,
            dtype=nfg.dtype,
        )

        with torch.cuda.amp.autocast(enabled=True):
            preds = self.render_head(uni_feats=nfg, rays=lidar_rays)

        # Reconstruct predicted point clouds for every batch element. Keep the
        # historical single-sample keys pointed at batch element 0 for callers
        # that save one render at a time.
        scale_factor = self.render_head.scale_factor
        batch_pred_points = []
        batch_pred_ranges = []
        batch_pred_intensity = []
        batch_pred_raydrop = []

        for lid_pred, lidar_ray in zip(preds, lidar_rays, strict=True):
            ray_d = lidar_ray["ray_d"]
            lid_depth = lid_pred["depth"]
            if lid_pred.get("depth_correction") is not None:
                lid_depth = lid_depth + lid_pred["depth_correction"]
            lid_ranges = lid_depth / scale_factor
            lid_points = ray_d * lid_ranges
            batch_pred_points.append(lid_points)
            batch_pred_ranges.append(lid_ranges)

            if lid_pred.get("lidar_output") is not None:
                lidar_out = lid_pred["lidar_output"]
                batch_pred_intensity.append(lidar_out[:, 0:1].sigmoid())
                batch_pred_raydrop.append(lidar_out[:, 1:2].sigmoid())

        result = {
            "pred_points": batch_pred_points[0],
            "pred_ranges": batch_pred_ranges[0],
            "pred_points_batch": torch.stack(batch_pred_points, dim=0),
            "pred_ranges_batch": torch.stack(batch_pred_ranges, dim=0),
            "lidar_render_shape": lidar_rays[0].get("lidar_render_shape"),
            "lidar_render_shapes": [
                lidar_ray.get("lidar_render_shape") for lidar_ray in lidar_rays
            ],
        }
        if batch_pred_intensity:
            result["pred_intensity"] = batch_pred_intensity[0]
            result["pred_raydrop"] = batch_pred_raydrop[0]
            result["pred_intensity_batch"] = torch.stack(batch_pred_intensity, dim=0)
            result["pred_raydrop_batch"] = torch.stack(batch_pred_raydrop, dim=0)

        if occ_logits is not None:
            result["occ_logits"] = occ_logits

        return result

def build_model(
    render_cfg: dict | None = None,
    max_train_rays: int = 0,
    kl_weight: float = 5e-2,
    train_vae: bool = True,
    debug_numerics: bool = False,
) -> LidarOnlyVAE:
    """Build the self-contained lidar-only model."""

    pc_range = [-80.0, -80.0, -4.5, 80.0, 80.0, 4.5]
    pts_voxel_size = [0.15625, 0.15625, 0.140625]
    unified_voxel_size = [0.15625, 0.15625, 0.140625]
    unified_voxel_shape = [
        int((pc_range[3] - pc_range[0]) / unified_voxel_size[0]),
        int((pc_range[4] - pc_range[1]) / unified_voxel_size[1]),
        int((pc_range[5] - pc_range[2]) / unified_voxel_size[2]),
    ]
    encoder_channels = 256

    if render_cfg is None:
        render_cfg = {
            "norm_scene": True,
            "field_cfg": {
                "type": "SDFFieldLidarOnly",
                "debug_numerics": debug_numerics,
                "sdf_decoder_cfg": {
                    "in_dim": 16,
                    "out_dim": 17,
                    "hidden_size": 16,
                    "n_blocks": 3,
                },
                "interpolate_cfg": {"type": "SmoothSampler", "padding_mode": "zeros"},
                "beta_init": 0.3,
            },
            "collider_cfg": {"type": "AABBBoxCollider", "near_plane": 1.0},
            "sampler_cfg": {
                "type": "NeuSSampler",
                "initial_sampler": "UniformSampler",
                "num_samples": 144,
                "num_samples_importance": 36,
                # "num_samples_importance_prior": 36,
                "num_upsample_steps": 1,
                "train_stratified": True,
                "single_jitter": True,
                "debug_numerics": debug_numerics,
            },
            "loss_cfg": {
                "sensor_depth_truncation": 0.1,
                "sparse_points_sdf_supervised": False,
                "weights": {
                    "depth_loss": 10.0,
                    "intensity_loss": 10.0,
                    "raydrop_loss": 0.2,
                    "occupancy_loss": 1.0,
                    "eikonal_loss": 0.1,
                    "sdf_loss": 0.1,
                    "free_space_loss": 0.1,
                    "kl_loss": kl_weight if train_vae else 0.0,
                },
            },
            "pred_intensity": True,
            "pred_raydrop": True,
            "close_radius": 3.0,
            "far_radius": 50.0,
            "max_train_rays": max_train_rays,
            "predefined_rays_cfg": {
                "num_beams": 32,
                "num_azimuth": 1080,
                "elevation_range": [-30.67, 10.67],
                "azimuth_range": [0.0, 360.0],
            },
        }

    sparse_shape = [
        int((pc_range[5] - pc_range[2]) / pts_voxel_size[2]) + 1,
        int((pc_range[4] - pc_range[1]) / pts_voxel_size[1]),
        int((pc_range[3] - pc_range[0]) / pts_voxel_size[0]),
    ]
    lidar_encoder = LiDAREncoder(
        in_channels=5,
        output_channels=encoder_channels,
        pts_voxel_size=pts_voxel_size,
        pc_range=pc_range,
        sparse_shape=sparse_shape,
        unified_voxel_shape=unified_voxel_shape,
        mae_mask_ratio=0.0,
        mae_downsample_scale=8,
    )

    bev_pillar_pooling = BEVPillarPooling(
        in_channels=encoder_channels,
        depth_bins=unified_voxel_shape[2],
        out_channels=unified_voxel_shape[2],
    )

    bev_2d_fpn = BEV2DFPN(
        in_channels=unified_voxel_shape[2],
        bottleneck_channels=32,
        shared_channels=64,
        channels_per_voxel=16,  # per-voxel feature dim for NFG
        height_dim=64,  # height bins
        occupancy_bias=-5.0,  # occupancy head bias initialization
        encoding_channels=[64, 128, 256],
        decoding_channels=[32, 32],
        train_vae=train_vae,
        debug_numerics=debug_numerics,
    )
    # Render head
    render_head = LidarOnlyRenderHead(
        unified_voxel_size=unified_voxel_size,
        unified_voxel_shape=unified_voxel_shape,
        pc_range=pc_range,
        render_cfg=render_cfg,
    )

    return LidarOnlyVAE(
        lidar_encoder=lidar_encoder,
        bev_pillar_pooling=bev_pillar_pooling,
        bev_2d_fpn=bev_2d_fpn,
        render_head=render_head,
        render_cfg=render_cfg,
        pc_range=pc_range,
        train_vae=train_vae,
    )


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, bool]:
    """Initialize DDP if launched via torchrun. Returns (rank, world_size, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        visible_cuda_devices = torch.cuda.device_count()
        if local_rank >= visible_cuda_devices:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {visible_cuda_devices} CUDA "
                "device(s) are visible."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def run_rank_serialized(rank, world_size, is_distributed, rank0_message, fn):
    """Run a startup action one rank at a time (avoids spconv JIT race)."""
    if not is_distributed:
        return fn()
    result = None
    for active_rank in range(world_size):
        if rank == active_rank:
            if is_main_process(rank):
                print(rank0_message, flush=True)
            result = fn()
        dist.barrier()
    return result


def load_compatible_model_state(
    model: nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    """Load checkpoint tensors whose names and shapes match the current model."""
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for name, value in checkpoint_state.items():
        if name not in model_state:
            skipped.append(f"{name}: unexpected in checkpoint")
            continue
        if model_state[name].shape != value.shape:
            skipped.append(
                f"{name}: checkpoint{tuple(value.shape)} != "
                f"model{tuple(model_state[name].shape)}"
            )
            continue
        compatible_state[name] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    return (
        list(incompatible.missing_keys),
        list(incompatible.unexpected_keys),
        skipped,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class RetryBadSamplesDataset(torch.utils.data.Dataset):
    """Retry nearby indices when a Lance row has a corrupt/missing payload."""

    def __init__(self, dataset, *, max_retries: int, rank: int):
        self.dataset = dataset
        self.max_retries = max_retries
        self.rank = rank

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        first_error = None
        dataset_len = len(self.dataset)
        for attempt in range(self.max_retries + 1):
            candidate_index = (index + attempt) % dataset_len
            try:
                frame = self.dataset[candidate_index]
                self._validate_frame(frame, candidate_index)
                if attempt > 0:
                    self._log_retry(index, candidate_index)
                return frame
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                if attempt == 0:
                    self._log_bad_sample(index, exc)
        raise RuntimeError(
            f"Failed to load a valid sample after {self.max_retries + 1} "
            f"attempts starting from dataset index {index}."
        ) from first_error

    def _validate_frame(self, frame, index):
        lidar_rig = frame.lidar_rig
        if lidar_rig is None:
            raise ValueError(f"Sample {index} has no lidar_rig.")
        if lidar_rig.LIDAR_TOP is None:
            raise ValueError(f"Sample {index} has no LIDAR_TOP.")

    def _worker_label(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = "main" if worker is None else str(worker.id)
        return f"rank={self.rank} worker={worker_id}"

    def _log_bad_sample(self, index, exc):
        print(
            f"[DataLoader] Skipping bad sample at index {index} "
            f"({self._worker_label()}): {type(exc).__name__}: {exc}",
            flush=True,
        )

    def _log_retry(self, index, replacement):
        print(
            f"[DataLoader] Replaced bad sample {index} -> {replacement} "
            f"({self._worker_label()})",
            flush=True,
        )


# Predefined sensor spec for missing-ray generation (matches NuScenes HDL-32E)
_PREDEFINED_NUM_BEAMS = 32
_PREDEFINED_NUM_AZIMUTH = 1080
_PREDEFINED_ELEVATION_MIN_DEG = -30.67
_PREDEFINED_ELEVATION_MAX_DEG = 10.67
_PREDEFINED_AZIMUTH_MIN_DEG = 0.0
_PREDEFINED_AZIMUTH_MAX_DEG = 360.0
_MAX_MISSING_RAYS = 4000
_MISSING_RAY_CLOSE_RADIUS = 3.0
_MISSING_RAY_FAR_RADIUS = 50.0


def _compute_missing_rays(
    points_xyz: torch.Tensor,
    max_missing: int = _MAX_MISSING_RAYS,
) -> torch.Tensor:
    """Compute non-return ray directions from predefined grid bins with no GT points.

    For each predefined beam direction that has no nearby GT return, creates a
    synthetic non-return point at a large dummy distance along that direction.
    These will be marked did_return=False in the collated output.

    Args:
        points_xyz: (N, 3) GT point positions.
        max_missing: Maximum number of missing rays to return.
    Returns:
        (M, 5) tensor of [x, y, z, intensity=0, timestamp=0] for missing rays,
        placed at dummy distance 200m along the missing direction.
    """
    # Build predefined ray directions
    elevations_deg = torch.linspace(
        _PREDEFINED_ELEVATION_MIN_DEG,
        _PREDEFINED_ELEVATION_MAX_DEG,
        _PREDEFINED_NUM_BEAMS,
    )
    azimuths_deg = torch.linspace(
        _PREDEFINED_AZIMUTH_MIN_DEG,
        _PREDEFINED_AZIMUTH_MAX_DEG,
        _PREDEFINED_NUM_AZIMUTH + 1,
    )[:-1]

    elevations_rad = torch.deg2rad(elevations_deg)
    azimuths_rad = torch.deg2rad(azimuths_deg)
    elev_grid, azim_grid = torch.meshgrid(elevations_rad, azimuths_rad, indexing="ij")
    elev_flat = elev_grid.reshape(-1)
    azim_flat = azim_grid.reshape(-1)
    # All predefined ray directions: (num_beams * num_azimuth, 3)
    all_ray_d = torch.stack(
        [
            torch.cos(elev_flat) * torch.cos(azim_flat),
            torch.cos(elev_flat) * torch.sin(azim_flat),
            torch.sin(elev_flat),
        ],
        dim=-1,
    )

    num_rays = all_ray_d.shape[0]  # 32 * 1080 = 34560

    # Bin GT points into predefined rays
    dis = points_xyz.norm(dim=-1)
    valid = (dis > _MISSING_RAY_CLOSE_RADIUS) & (dis < _MISSING_RAY_FAR_RADIUS)
    valid_pts = points_xyz[valid]

    if valid_pts.shape[0] == 0:
        # All rays are missing - subsample
        perm = torch.randperm(num_rays)[:max_missing]
        dummy_dist = 200.0
        missing_xyz = all_ray_d[perm] * dummy_dist
        return torch.cat([missing_xyz, torch.zeros(missing_xyz.shape[0], 2)], dim=-1)

    ranges = valid_pts.norm(dim=-1).clamp(min=1e-6)
    azimuth_width = _PREDEFINED_AZIMUTH_MAX_DEG - _PREDEFINED_AZIMUTH_MIN_DEG
    azimuth_step = azimuth_width / _PREDEFINED_NUM_AZIMUTH

    azimuth_deg = torch.rad2deg(torch.atan2(valid_pts[:, 1], valid_pts[:, 0]))
    azimuth_norm = torch.remainder(
        azimuth_deg - _PREDEFINED_AZIMUTH_MIN_DEG, azimuth_width
    )
    azimuth_idx = (
        torch.round(azimuth_norm / azimuth_step).long() % _PREDEFINED_NUM_AZIMUTH
    )

    z_ratio = (valid_pts[:, 2] / ranges).clamp(-1.0, 1.0)
    elevation_deg = torch.rad2deg(torch.asin(z_ratio))
    elevation_delta = torch.abs(elevation_deg[:, None] - elevations_deg[None, :])
    beam_idx = elevation_delta.argmin(dim=1)

    # Compute occupied ray indices
    ray_idx = beam_idx * _PREDEFINED_NUM_AZIMUTH + azimuth_idx
    occupied = torch.zeros(num_rays, dtype=torch.bool)
    occupied.scatter_(0, ray_idx, True)

    # Missing = predefined rays that have no GT point
    missing_mask = ~occupied
    missing_indices = missing_mask.nonzero(as_tuple=False).squeeze(-1)

    if missing_indices.numel() == 0:
        return torch.zeros(0, 5)

    # Subsample if too many
    if missing_indices.numel() > max_missing:
        perm = torch.randperm(missing_indices.numel())[:max_missing]
        missing_indices = missing_indices[perm]

    # Place dummy points at 200m along missing ray directions
    dummy_dist = 200.0
    missing_directions = all_ray_d[missing_indices]
    missing_xyz = missing_directions * dummy_dist
    # intensity=0, timestamp=0
    missing_points = torch.cat(
        [
            missing_xyz,
            torch.zeros(missing_xyz.shape[0], 2),
        ],
        dim=-1,
    )
    return missing_points


def collate_nuscenes(
    batch: list[NuScenesDataFrame],
    num_sweeps: int = 0,
) -> dict:
    """Collate NuScenes frames with LiDAR data only.

    Returns points with associated did_return masks. Missing rays from the
    predefined 32x1080 beam grid are added as non-return points (did_return=False)
    to train the raydrop head.
    """
    aggregated_points_list = []
    points_list = []
    did_return_list = []

    for frame in batch:
        pc = frame.lidar_rig.LIDAR_TOP
        pts_xyz = pc.points.float()
        n = pts_xyz.shape[0]

        intensity = pc.intensity
        if intensity is not None:
            intensity_f = intensity.float().unsqueeze(1) / 255.0
        else:
            intensity_f = torch.zeros(n, 1)
        timestamp_col = torch.zeros(n, 1)
        points = torch.cat([pts_xyz, intensity_f, timestamp_col], dim=1)

        # Compute missing rays (non-returns) from predefined beam grid
        missing_points = _compute_missing_rays(pts_xyz)
        n_missing = missing_points.shape[0]

        # did_return: True for all GT points, False for missing rays
        did_return_gt = torch.ones(n, dtype=torch.bool)
        did_return_missing = torch.zeros(n_missing, dtype=torch.bool)

        # Concatenate GT + missing
        if n_missing > 0:
            all_points = torch.cat([points, missing_points], dim=0)
            did_return = torch.cat([did_return_gt, did_return_missing], dim=0)
        else:
            all_points = points
            did_return = did_return_gt

        points_list.append(all_points)
        did_return_list.append(did_return)

        aggregated_points = points  # aggregated does NOT include missing rays
        if num_sweeps > 0:
            aggregated_frame = aggregate_lidar_sweeps(frame, num_sweeps=num_sweeps)
            aggregated_pc = aggregated_frame.lidar_rig.LIDAR_TOP
            agg_pts_xyz = aggregated_pc.points.float()
            n_agg = agg_pts_xyz.shape[0]
            agg_intensity = aggregated_pc.intensity
            if agg_intensity is not None:
                agg_intensity_f = agg_intensity.float().unsqueeze(1) / 255.0
            else:
                agg_intensity_f = torch.zeros(n_agg, 1)
            aggregated_points = torch.cat(
                [agg_pts_xyz, agg_intensity_f, torch.zeros(n_agg, 1)], dim=1
            )
        aggregated_points_list.append(aggregated_points)

    # Compute occupancy GT on CPU (will be moved to device in training loop)
    pc_range = [-80.0, -80.0, -4.5, 80.0, 80.0, 4.5]
    voxel_size = [0.15625, 0.15625, 0.140625]
    grid_shape = [1024, 1024, 64]  # [nx, ny, nz]
    occ_gt = compute_occupancy_gt(
        aggregated_points_list,
        pc_range=pc_range,
        voxel_size=voxel_size,
        grid_shape=grid_shape,
        device=torch.device("cpu"),
    )

    return {
        "aggregated_points": aggregated_points_list,
        "points": points_list,
        "did_return": did_return_list,
        "occ_gt": occ_gt,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate_lidar(
    model,
    dataset,
    device: torch.device,
    step: int,
    output_dir: Path,
    num_samples: int = 10,
    num_sweeps: int = 9,
) -> None:
    """Run validation: render LiDAR point clouds and save .npz files."""
    model.eval()
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    collate_fn = partial(collate_nuscenes, num_sweeps=num_sweeps)

    occ_metrics_accum = {"occ_iou": 0.0, "occ_precision": 0.0, "occ_recall": 0.0}
    occ_count = 0

    for i, idx in enumerate(indices):
        torch.cuda.empty_cache()
        frame = dataset[idx]
        batch = collate_fn([frame])

        aggregated_points = [p.to(device) for p in batch["aggregated_points"]]
        points = [p.to(device) for p in batch["points"]]
        occ_gt = batch["occ_gt"].to(device)

        preds = model.render_lidar(
            aggregated_points=aggregated_points,
            points=points,
        )

        # Log raydrop statistics
        if "pred_raydrop" in preds:
            total_rays = preds["pred_raydrop"].numel()
            dropped = (preds["pred_raydrop"] >= 0.5).sum().item()
            kept = total_rays - dropped
            print(
                f"    Sample {i}: {total_rays} total rays, "
                f"{kept} kept ({100 * kept / total_rays:.1f}%), "
                f"{dropped} dropped by raydrop head ({100 * dropped / total_rays:.1f}%)"
            )

        # Compute occupancy metrics if model produces occ_logits
        if "occ_logits" in preds and occ_gt is not None:
            metrics = compute_occupancy_metrics(preds["occ_logits"], occ_gt)
            for k, v in metrics.items():
                occ_metrics_accum[k] += v
            occ_count += 1
            print(
                f"    Sample {i}: occ_iou={metrics['occ_iou']:.4f} "
                f"precision={metrics['occ_precision']:.4f} "
                f"recall={metrics['occ_recall']:.4f}"
            )

        _save_point_clouds(
            preds, batch["points"][0], points_dir / f"step{step:06d}_sample{i:02d}.npz"
        )

        del preds, aggregated_points, points, occ_gt
        torch.cuda.empty_cache()

    if occ_count > 0:
        avg_metrics = {k: v / occ_count for k, v in occ_metrics_accum.items()}
        print(
            f"  Avg occ metrics: iou={avg_metrics['occ_iou']:.4f} "
            f"precision={avg_metrics['occ_precision']:.4f} "
            f"recall={avg_metrics['occ_recall']:.4f}"
        )

    model.train()


def _save_point_clouds(preds: dict, gt_points: torch.Tensor, save_path: Path) -> None:
    """Save GT and predicted point clouds as .npz."""
    gt_xyz = gt_points[:, :3].numpy()
    gt_intensity = gt_points[:, 3].numpy() if gt_points.shape[1] > 3 else None

    save_dict = {"gt_xyz": gt_xyz}
    if gt_intensity is not None:
        save_dict["gt_intensity"] = gt_intensity
    if "lidar_render_shape" in preds and preds["lidar_render_shape"] is not None:
        save_dict["lidar_render_shape"] = np.array(preds["lidar_render_shape"])

    if "pred_points" in preds:
        pred_points = preds["pred_points"].cpu().numpy()
        pred_ranges = preds["pred_ranges"].cpu().numpy().squeeze(-1)
        save_dict["pred_xyz"] = pred_points
        save_dict["pred_ranges"] = pred_ranges

        pred_intensity = None
        if "pred_intensity" in preds:
            pred_intensity = preds["pred_intensity"].cpu().numpy().squeeze(-1)
            save_dict["pred_intensity"] = pred_intensity
        if "pred_raydrop" in preds:
            pred_raydrop = preds["pred_raydrop"].cpu().numpy().squeeze(-1)
            save_dict["pred_raydrop"] = pred_raydrop
            returned_mask = pred_raydrop < 0.5
            save_dict["pred_xyz_filtered"] = pred_points[returned_mask]
            if pred_intensity is not None:
                save_dict["pred_intensity_filtered"] = pred_intensity[returned_mask]

    np.savez_compressed(str(save_path), **save_dict)


def _select_point_cloud_sample(preds: dict, sample_idx: int) -> dict:
    """Select one sample from batched render predictions for saving."""
    sample_preds = {}
    batch_keys = {
        "pred_points": "pred_points_batch",
        "pred_ranges": "pred_ranges_batch",
        "pred_intensity": "pred_intensity_batch",
        "pred_raydrop": "pred_raydrop_batch",
    }
    for single_key, batch_key in batch_keys.items():
        if batch_key in preds:
            sample_preds[single_key] = preds[batch_key][sample_idx]
        elif sample_idx == 0 and single_key in preds:
            sample_preds[single_key] = preds[single_key]

    if "lidar_render_shapes" in preds:
        sample_preds["lidar_render_shape"] = preds["lidar_render_shapes"][sample_idx]
    elif "lidar_render_shape" in preds:
        sample_preds["lidar_render_shape"] = preds["lidar_render_shape"]

    return sample_preds


# ---------------------------------------------------------------------------
# Test-set validation
# ---------------------------------------------------------------------------


@dataclass
class PointCloudMetricSuite:
    """LiDAR point-cloud metric accumulators for one validation pass."""

    chamfer: ChamferDistanceMetrics
    jsd: JensenShannonDivergence
    mmd: MaximumMeanDiscrepancy

    def add_sample(
        self,
        *,
        pred_xyz: torch.Tensor,
        pred_raydrop: torch.Tensor | None,
        gt_xyz: torch.Tensor,
        gt_did_return: torch.Tensor,
        sample_id: str,
    ) -> None:
        self.chamfer.add_sample(
            pred_xyz=pred_xyz,
            pred_raydrop=pred_raydrop,
            gt_xyz=gt_xyz,
            gt_did_return=gt_did_return,
            sample_id=sample_id,
        )
        self.jsd.add_sample(
            pred_xyz=pred_xyz,
            pred_raydrop=pred_raydrop,
            gt_xyz=gt_xyz,
            gt_did_return=gt_did_return,
            sample_id=sample_id,
        )
        self.mmd.add_sample(
            pred_xyz=pred_xyz,
            pred_raydrop=pred_raydrop,
            gt_xyz=gt_xyz,
            gt_did_return=gt_did_return,
            sample_id=sample_id,
        )

    def summary_metrics(self) -> dict[str, float | int]:
        chamfer = self.chamfer.aggregate()
        return {
            "point_metric_samples": chamfer["num_samples"],
            "point_metric_valid_samples": chamfer["valid_samples"],
            "chamfer_pred_to_gt_mean": chamfer["chamfer_pred_to_gt_mean"],
            "chamfer_gt_to_pred_mean": chamfer["chamfer_gt_to_pred_mean"],
            "chamfer_symmetric_mean": chamfer["chamfer_symmetric_mean"],
            "chamfer_pred_to_gt_sqrt_m_mean": chamfer[
                "chamfer_pred_to_gt_sqrt_m_mean"
            ],
            "chamfer_gt_to_pred_sqrt_m_mean": chamfer[
                "chamfer_gt_to_pred_sqrt_m_mean"
            ],
            "chamfer_symmetric_sqrt_m_mean": chamfer["chamfer_symmetric_sqrt_m_mean"],
        }


def _build_point_cloud_metric_suite(
    pc_range: tuple[float, float, float, float, float, float],
) -> PointCloudMetricSuite:
    return PointCloudMetricSuite(
        chamfer=ChamferDistanceMetrics(
            max_points=65536,
            raydrop_threshold=0.5,
            pc_range=pc_range,
            use_cuda_extension=True,
            fallback_chunk_size=1024,
        ),
        jsd=JensenShannonDivergence(
            spatial_shape=(1, 100, 100),
            pc_range=pc_range,
            raydrop_threshold=0.5,
        ),
        mmd=MaximumMeanDiscrepancy(
            metric_shape=(1, 100, 100),
            pc_range=pc_range,
            raydrop_threshold=0.5,
            sigma=0.5,
            kernel_chunk_size=32,
        ),
    )


def _merge_point_cloud_metric_suite(
    destination: PointCloudMetricSuite,
    source: PointCloudMetricSuite,
) -> PointCloudMetricSuite:
    destination.chamfer.samples.extend(source.chamfer.samples)
    destination.jsd.p += source.jsd.p
    destination.jsd.q += source.jsd.q
    destination.jsd.samples.extend(source.jsd.samples)
    destination.mmd.gt_set.extend(source.mmd.gt_set)
    destination.mmd.gen_set.extend(source.mmd.gen_set)
    destination.mmd.samples.extend(source.mmd.samples)
    return destination


def _gather_point_cloud_metric_suite(
    metrics: PointCloudMetricSuite,
    rank: int,
) -> PointCloudMetricSuite | None:
    if not (dist.is_available() and dist.is_initialized()):
        return metrics

    gathered = (
        [None for _ in range(dist.get_world_size())] if is_main_process(rank) else None
    )
    dist.gather_object(metrics, object_gather_list=gathered, dst=0)

    if not is_main_process(rank):
        return None

    merged = _build_point_cloud_metric_suite(
        pc_range=tuple(float(value) for value in metrics.chamfer.pc_range)
    )
    for rank_metrics in gathered:
        if rank_metrics is not None:
            _merge_point_cloud_metric_suite(merged, rank_metrics)
    return merged


def _model_pc_range(
    model,
) -> tuple[float, float, float, float, float, float]:
    pc_range = getattr(model, "pc_range", None)
    if pc_range is None and getattr(model, "render_head", None) is not None:
        pc_range = getattr(model.render_head, "pc_range", None)
    if pc_range is None:
        raise AttributeError("Model does not expose pc_range for point-cloud metrics.")
    values = tuple(float(value) for value in pc_range)
    if len(values) != 6:
        raise ValueError(f"Expected model pc_range with 6 values, got {values}.")
    return values


def _format_point_metric_summary(metrics: PointCloudMetricSuite) -> str:
    summary = metrics.summary_metrics()
    return " | ".join(
        [
            f"point_metric_samples: {int(summary['point_metric_samples'])}",
            (
                "point_metric_valid_samples: "
                f"{int(summary['point_metric_valid_samples'])}"
            ),
            f"chamfer_pred_to_gt_mean: {summary['chamfer_pred_to_gt_mean']:.6f}",
            f"chamfer_gt_to_pred_mean: {summary['chamfer_gt_to_pred_mean']:.6f}",
            f"chamfer_symmetric_mean: {summary['chamfer_symmetric_mean']:.6f}",
            (
                "chamfer_pred_to_gt_sqrt_m_mean: "
                f"{summary['chamfer_pred_to_gt_sqrt_m_mean']:.6f}"
            ),
            (
                "chamfer_gt_to_pred_sqrt_m_mean: "
                f"{summary['chamfer_gt_to_pred_sqrt_m_mean']:.6f}"
            ),
            (
                "chamfer_symmetric_sqrt_m_mean: "
                f"{summary['chamfer_symmetric_sqrt_m_mean']:.6f}"
            ),
        ]
    )


@torch.no_grad()
def validate_test_set(
    model,
    test_dataloader,
    device: torch.device,
    step: int,
    output_dir: Path,
    num_renders: int = 20,
    compute_point_metrics: bool = True,
    rank: int = 0,
) -> None:
    """Compute distributed test loss, point-cloud metrics, and rank-0 renders."""
    model.eval()
    points_dir = output_dir / "points"
    if is_main_process(rank):
        points_dir.mkdir(parents=True, exist_ok=True)

    base_metric_names = (
        "depth_loss",
        "intensity_loss",
        "occupancy_loss",
        "raydrop_loss",
    )
    vae_metric_names = (
        "kl_weight",
        "kl_weight_scale",
        "kl_loss",
        "latent_true_kl",
        "latent_gaussian_score",
        "latent_z_mean",
        "latent_z_std",
        "latent_channel_mean_abs_p95",
        "latent_channel_std_error_p95",
    )
    if getattr(model, "train_vae", True):
        metric_names = base_metric_names + vae_metric_names
    else:
        metric_names = base_metric_names
    local_sums = {name: 0.0 for name in metric_names}
    local_batches = 0
    renders_saved = 0
    point_metrics = (
        _build_point_cloud_metric_suite(pc_range=_model_pc_range(model))
        if compute_point_metrics
        else None
    )

    for batch in test_dataloader:
        aggregated_points = [p.to(device) for p in batch["aggregated_points"]]
        points = [p.to(device) for p in batch["points"]]
        did_return = [d.to(device) for d in batch["did_return"]]
        occ_gt = batch["occ_gt"].to(device)

        # Compute test loss using GT rays (same as training, for comparable metrics)
        losses = model(
            aggregated_points=aggregated_points,
            points=points,
            did_return=did_return,
            occ_gt=occ_gt,
        )
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            if k in local_sums:
                local_sums[k] += val

        local_batches += 1

        # Dense rendering is needed for point-cloud metrics. When metrics are
        # disabled, render only the first N rank-0 samples for previews.
        should_render_batch = compute_point_metrics or (
            is_main_process(rank) and renders_saved < num_renders
        )
        if should_render_batch:
            preds = model.render_lidar(
                aggregated_points=aggregated_points,
                points=points,
            )

            if point_metrics is not None:
                for sample_idx, gt_points in enumerate(points):
                    sample_preds = _select_point_cloud_sample(preds, sample_idx)
                    point_metrics.add_sample(
                        pred_xyz=sample_preds["pred_points"],
                        pred_raydrop=sample_preds.get("pred_raydrop"),
                        gt_xyz=gt_points,
                        gt_did_return=did_return[sample_idx],
                        sample_id=(
                            f"rank{rank:02d}_batch{local_batches:06d}_"
                            f"sample{sample_idx:02d}"
                        ),
                    )

            for sample_idx in range(len(points)):
                if renders_saved >= num_renders:
                    break
                if not is_main_process(rank):
                    break
                sample_preds = _select_point_cloud_sample(preds, sample_idx)
                _save_point_clouds(
                    sample_preds,
                    batch["points"][sample_idx],
                    points_dir / f"test_step{step:06d}_sample{renders_saved:02d}.npz",
                )
                renders_saved += 1
            del preds

        del losses, aggregated_points, points, did_return, occ_gt

    if device.type == "cuda":
        torch.cuda.empty_cache()

    metric_tensor = torch.tensor(
        [local_sums[name] for name in metric_names] + [local_batches],
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    gathered_point_metrics = None
    if point_metrics is not None:
        gathered_point_metrics = _gather_point_cloud_metric_suite(point_metrics, rank)

    global_batches = metric_tensor[-1].item()
    if is_main_process(rank) and global_batches > 0:
        avg_losses = {
            name: metric_tensor[i].item() / global_batches
            for i, name in enumerate(metric_names)
        }
        loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(avg_losses.items()))
        print(f"\n  === Test Results (step {step}, {int(global_batches)} batches) ===")
        if gathered_point_metrics is not None:
            loss_str = (
                f"{loss_str} | {_format_point_metric_summary(gathered_point_metrics)}"
            )
        print(f"  {loss_str}")

    model.train()


def kl_weight_scale_for_step(
    step: int,
    max_steps: int,
    schedule: str,
    warmup_steps: int,
    cycle_count: int,
    cycle_ramp_fraction: float,
) -> float:
    """Return the 0..1 multiplier applied to the configured peak KL weight."""
    if warmup_steps > 0 and step < warmup_steps:
        return 0.0

    if schedule == "constant":
        return 1.0

    active_steps = max(1, max_steps - warmup_steps)
    active_step = max(0, step - warmup_steps)
    ramp_fraction = max(cycle_ramp_fraction, 1e-8)

    if schedule == "linear":
        ramp_steps = max(1.0, active_steps * ramp_fraction)
        return min(1.0, active_step / ramp_steps)

    if schedule != "cyclic":
        raise ValueError(f"Unknown KL weight schedule: {schedule}")

    cycle_steps = max(1.0, active_steps / cycle_count)
    cycle_position = (active_step % cycle_steps) / cycle_steps
    if cycle_position >= ramp_fraction:
        return 1.0
    return min(1.0, cycle_position / ramp_fraction)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--data-root", type=str, default="/data/nuscenes")
@click.option("--version", type=str, default="v1.0-trainval")
@click.option("--batch-size", default=1, help="Per-GPU batch size")
@click.option("--max-steps", default=50000)
@click.option("--lr", default=2e-4)
@click.option("--grad-accum-steps", default=1)
@click.option("--log-interval", default=10)
@click.option("--val-interval", default=5000)
@click.option("--val-samples", default=10)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=Path("checkpoints/lidar_only"),
)
@click.option("--device", default="cuda")
@click.option("--overfit-samples", default=0)
@click.option("--num-workers", default=4, help="DataLoader workers per GPU/rank.")
@click.option("--skip-bad-samples/--no-skip-bad-samples", default=True)
@click.option("--bad-sample-retries", default=64)
@click.option("--resume", type=click.Path(path_type=Path, exists=True), default=None)
@click.option(
    "--num-sweeps",
    default=9,
    help="Number of lidar sweeps to aggregate for encoder input.",
)
@click.option(
    "--max-train-rays",
    default=0,
    help="Max rays per sample during training. 0=ALL rays (needs large GPU, e.g. A100 80GB).",
)
@click.option(
    "--test-val/--no-test-val",
    default=True,
    help="Run full test-set validation at each val_interval.",
)
@click.option(
    "--test-metrics/--no-test-metrics",
    default=True,
    help=(
        "During test validation, densely render the test dataloader and report "
        "Chamfer point-cloud metrics."
    ),
)
@click.option(
    "--kl-weight",
    default=5e-2,
    help="Peak KL divergence loss weight. The schedule scales this value.",
)
@click.option(
    "--kl-weight-schedule",
    type=click.Choice(["constant", "linear", "cyclic"], case_sensitive=False),
    default="cyclic",
    help="Schedule for the KL weight after the deterministic VAE warmup.",
)
@click.option(
    "--kl-cycle-count",
    default=4,
    help="Number of KL cycles across the post-warmup training steps.",
)
@click.option(
    "--kl-cycle-ramp-fraction",
    default=0.5,
    help="Fraction of each KL cycle used to ramp from 0 to --kl-weight.",
)
@click.option(
    "--train-vae/--no-train-vae",
    default=True,
    help="Use VAE bottleneck sampling and KL loss. Disable for deterministic AE.",
)
@click.option(
    "--vae-warmup-steps",
    default=10000,
    help="Run deterministic AE bottleneck with KL disabled for this many steps.",
)
@click.option(
    "--debug-numerics/--no-debug-numerics",
    default=False,
    help="Enable finite checks around the VAE, rays, and field sampler.",
)
@click.option("--test-version", type=str, default="v1.0-test")
def main(
    data_root: str,
    version: str,
    batch_size: int,
    max_steps: int,
    lr: float,
    grad_accum_steps: int,
    log_interval: int,
    val_interval: int,
    val_samples: int,
    checkpoint_dir: Path,
    device: str,
    overfit_samples: int,
    num_workers: int,
    skip_bad_samples: bool,
    bad_sample_retries: int,
    resume: Path | None,
    num_sweeps: int,
    max_train_rays: int,
    test_val: bool,
    test_metrics: bool,
    test_version: str,
    kl_weight: float,
    kl_weight_schedule: str,
    kl_cycle_count: int,
    kl_cycle_ramp_fraction: float,
    train_vae: bool,
    vae_warmup_steps: int,
    debug_numerics: bool,
) -> None:
    """Train LiDAR-only rendering from scratch (depth + intensity + raydrop)."""
    rank, world_size, is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    kl_weight_schedule = kl_weight_schedule.lower()
    if kl_weight < 0.0:
        raise ValueError("--kl-weight must be non-negative.")
    if vae_warmup_steps < 0:
        raise ValueError("--vae-warmup-steps must be non-negative.")
    if kl_cycle_count < 1:
        raise ValueError("--kl-cycle-count must be at least 1.")
    if not (0.0 < kl_cycle_ramp_fraction <= 1.0):
        raise ValueError("--kl-cycle-ramp-fraction must be in the interval (0, 1].")

    if is_main_process(rank):
        print("=== LiDAR-Only Training ===")
        print(f"Device: {device} | World size: {world_size}")

    # --- Build model (serialized to avoid spconv JIT races) ---
    def _build():
        return build_model(
            max_train_rays=max_train_rays,
            kl_weight=kl_weight,
            train_vae=train_vae,
            debug_numerics=debug_numerics,
        )

    model = run_rank_serialized(
        rank,
        world_size,
        is_distributed,
        "Rank 0 building model (spconv JIT)...",
        _build,
    )
    model = model.to(device)

    # Resume
    start_step = 0
    start_epoch = 0
    load_optimizer_state = True
    if resume is not None:
        if is_main_process(rank):
            print(f"Resuming from: {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        ckpt_train_vae = ckpt.get("train_vae")
        converting_ae_to_vae = ckpt_train_vae is False and train_vae
        if ckpt_train_vae is not None and ckpt_train_vae != train_vae:
            if not converting_ae_to_vae:
                raise ValueError(
                    "Checkpoint train_vae mode does not match this run: "
                    f"checkpoint train_vae={ckpt_train_vae}, "
                    f"current train_vae={train_vae}."
                )
            missing, unexpected, skipped = load_compatible_model_state(
                model, ckpt["model_state_dict"]
            )
            allowed_missing_prefixes = (
                "bev_2d_fpn.mu_head.",
                "bev_2d_fpn.log_var_head.",
            )
            disallowed_missing = [
                key for key in missing if not key.startswith(allowed_missing_prefixes)
            ]
            if unexpected or skipped or disallowed_missing:
                raise RuntimeError(
                    "Failed to convert deterministic AE checkpoint to VAE. "
                    f"unexpected={unexpected[:10]}, skipped={skipped[:10]}, "
                    f"disallowed_missing={disallowed_missing[:10]}"
                )
            load_optimizer_state = False
            start_step = 0
            start_epoch = 0
            if is_main_process(rank):
                print(
                    "Converted deterministic AE checkpoint to VAE adapter: "
                    "loaded shared weights, kept initialized VAE heads, "
                    "reset optimizer and schedule step to 0."
                )
        else:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to load checkpoint. If this checkpoint was saved with "
                    "an older VAE bottleneck shape, convert from a matching "
                    "--no-train-vae checkpoint or rerun with matching mode."
                ) from exc
            start_step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if is_distributed else model
    raw_model.set_debug_numerics(debug_numerics)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(rank):
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # --- Dataset ---
    data_config = NuScenesDataConfig(
        CAM_FRONT=False,
        CAM_FRONT_LEFT=False,
        CAM_FRONT_RIGHT=False,
        CAM_BACK=False,
        CAM_BACK_LEFT=False,
        CAM_BACK_RIGHT=False,
        lidar=True,
        bounding_boxes_3d=False,
    )

    nuscenes_dataset = NuScenesLanceDataset(
        data_config=data_config,
        version=version,
        data_root=data_root,
        scene_names=None,
        key_frames_only=False,
    )
    if is_main_process(rank):
        print(f"Dataset: {len(nuscenes_dataset)} frames")

    train_dataset = nuscenes_dataset
    if overfit_samples > 0:
        indices = list(range(min(overfit_samples, len(nuscenes_dataset))))
        train_dataset = torch.utils.data.Subset(nuscenes_dataset, indices)
        if is_main_process(rank):
            print(f"  Overfitting on {len(train_dataset)} samples")

    if skip_bad_samples:
        train_dataset = RetryBadSamplesDataset(
            train_dataset, max_retries=bad_sample_retries, rank=rank
        )

    sampler = None
    shuffle = True
    if is_distributed:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False

    train_dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": partial(collate_nuscenes, num_sweeps=num_sweeps),
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_dataloader_kwargs["prefetch_factor"] = 2
        train_dataloader_kwargs["multiprocessing_context"] = "spawn"

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **train_dataloader_kwargs
    )

    # --- Test DataLoader (created once, reused every val_interval) ---
    test_dataloader = None
    if test_val:
        test_sampler = None
        test_dataset = NuScenesLanceDataset(
            data_config=data_config,
            version=test_version,
            data_root=data_root,
            scene_names=None,
            key_frames_only=True,
        )
        if skip_bad_samples:
            test_dataset = RetryBadSamplesDataset(
                test_dataset, max_retries=bad_sample_retries, rank=rank
            )
        if is_distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )

        test_dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "sampler": test_sampler,
            "num_workers": num_workers,
            "collate_fn": partial(collate_nuscenes, num_sweeps=num_sweeps),
            "pin_memory": device.type == "cuda",
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            test_dataloader_kwargs["prefetch_factor"] = 2
            test_dataloader_kwargs["multiprocessing_context"] = "spawn"

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **test_dataloader_kwargs
        )
        if is_main_process(rank):
            print(
                f"Test dataset: {len(test_dataset)} frames ({test_version}, key_frames_only)"
            )

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_iters = min(500, max_steps // 4)

    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return (1.0 / 3.0) + (2.0 / 3.0) * (current_step / warmup_iters)
        progress = (current_step - warmup_iters) / max(1, max_steps - warmup_iters)
        return max(1e-3, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if load_optimizer_state and resume is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # --- Training Loop ---
    if is_main_process(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = start_step
    epoch = start_epoch
    running_losses: dict[str, float] = {}

    if is_main_process(rank):
        ray_mode = (
            "ALL rays"
            if max_train_rays == 0
            else f"subsample {max_train_rays} rays/sample"
        )
        print(f"\nStarting training for {max_steps} steps...")
        print(f"  batch_size={batch_size} x world_size={world_size}")
        print(f"  grad_accum={grad_accum_steps}")
        print(f"  rays: {ray_mode} (points + non-return for raydrop)")
        print(f"  num_sweeps={num_sweeps}")
        print(f"  test_val={test_val}, test_metrics={test_metrics}")
        print(f"  train_vae={train_vae}")
        if train_vae:
            print(f"  kl_weight_peak={kl_weight:g}")
            print(f"  kl_weight_schedule={kl_weight_schedule}")
            if kl_weight_schedule == "cyclic":
                print(
                    f"  kl_cycle_count={kl_cycle_count}, "
                    f"kl_cycle_ramp_fraction={kl_cycle_ramp_fraction:g}"
                )
            if vae_warmup_steps > 0:
                print(
                    f"  vae_warmup_steps={vae_warmup_steps} "
                    "(sampling off, KL weight 0 before schedule starts)"
                )
        elif vae_warmup_steps > 0:
            print("  vae_warmup_steps ignored because train_vae=False")
        if debug_numerics:
            print("  debug_numerics=True")
        print()

    last_vae_warmup_active = None
    while step < max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_dataloader:
            if step >= max_steps:
                break

            if train_vae:
                vae_warmup_active = vae_warmup_steps > 0 and step < vae_warmup_steps
                kl_weight_scale = kl_weight_scale_for_step(
                    step=step,
                    max_steps=max_steps,
                    schedule=kl_weight_schedule,
                    warmup_steps=vae_warmup_steps,
                    cycle_count=kl_cycle_count,
                    cycle_ramp_fraction=kl_cycle_ramp_fraction,
                )
                raw_model.set_kl_weight_scale(
                    kl_weight_scale,
                    stochastic_sampling=not vae_warmup_active,
                )
                if vae_warmup_steps > 0 and vae_warmup_active != last_vae_warmup_active:
                    if is_main_process(rank):
                        if vae_warmup_active:
                            print(
                                f"  VAE warmup active until step {vae_warmup_steps}: "
                                "using z=mu and KL weight 0."
                            )
                        else:
                            print(
                                "  VAE warmup complete: stochastic sampling and "
                                f"{kl_weight_schedule} KL schedule enabled."
                            )
                    last_vae_warmup_active = vae_warmup_active

            aggregated_points = [p.to(device) for p in batch["aggregated_points"]]
            points = [p.to(device) for p in batch["points"]]
            did_return = [d.to(device) for d in batch["did_return"]]
            occ_gt = batch["occ_gt"].to(device)

            losses = model(
                aggregated_points=aggregated_points,
                points=points,
                did_return=did_return,
                occ_gt=occ_gt,
            )

            loss_terms = [
                v
                for k, v in losses.items()
                if k.endswith("_loss") and isinstance(v, torch.Tensor)
            ]
            total_loss = sum(loss_terms)
            if not torch.isfinite(total_loss):
                loss_debug = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in losses.items()
                }
                raise FloatingPointError(f"total_loss is non-finite: {loss_debug}")
            total_loss = total_loss / grad_accum_steps
            total_loss.backward()

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                running_losses[k] = running_losses.get(k, 0.0) + val

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            if step % log_interval == 0 and is_main_process(rank):
                avg_losses = {k: v / log_interval for k, v in running_losses.items()}
                loss_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in sorted(avg_losses.items())
                )
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[Step {step}/{max_steps}] [Epoch {epoch}] lr={current_lr:.2e} | {loss_str}"
                )
                running_losses = {}

            if step % val_interval == 0 and is_main_process(rank):
                ckpt_path = checkpoint_dir / f"step_{step:06d}.pth"
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_vae": train_vae,
                        "kl_weight": kl_weight,
                        "kl_weight_schedule": kl_weight_schedule,
                        "kl_cycle_count": kl_cycle_count,
                        "kl_cycle_ramp_fraction": kl_cycle_ramp_fraction,
                        "vae_warmup_steps": vae_warmup_steps,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"  Saved: {ckpt_path}")

            if step % val_interval == 0:
                val_dir = checkpoint_dir / "val_renders"
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()
                torch.cuda.empty_cache()
                if is_main_process(rank):
                    print(f"\n  Validating ({val_samples} samples)...")
                    # validate_lidar(
                    #     model=raw_model,
                    #     dataset=train_dataset,
                    #     device=device,
                    #     step=step,
                    #     output_dir=val_dir,
                    #     num_samples=val_samples,
                    #     num_sweeps=num_sweeps,
                    # )
                if test_val and test_dataloader is not None:
                    test_dir = checkpoint_dir / "test_renders"
                    if is_main_process(rank):
                        print("\n  Running test-set validation...")
                    validate_test_set(
                        model=raw_model,
                        test_dataloader=test_dataloader,
                        device=device,
                        step=step,
                        output_dir=test_dir,
                        compute_point_metrics=test_metrics,
                        rank=rank,
                    )
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                if is_main_process(rank):
                    print(f"  Renders saved: {val_dir}\n")
                if is_distributed:
                    dist.barrier()
                model.train()

    # Final checkpoint
    if is_main_process(rank):
        ckpt_path = checkpoint_dir / f"step_{step:06d}_final.pth"
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "train_vae": train_vae,
                "kl_weight": kl_weight,
                "kl_weight_schedule": kl_weight_schedule,
                "kl_cycle_count": kl_cycle_count,
                "kl_cycle_ramp_fraction": kl_cycle_ramp_fraction,
                "vae_warmup_steps": vae_warmup_steps,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            ckpt_path,
        )
        print(f"\nTraining complete. Final checkpoint: {ckpt_path}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
