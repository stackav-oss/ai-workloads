"""Point-cloud histogram helpers for LiDAR distribution metrics."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import torch
from dltype import FloatTensor, dltyped

DEFAULT_PC_RANGE = (-80.0, -80.0, -4.5, 80.0, 80.0, 4.5)
DEFAULT_VOXEL_SIZE = (0.15625, 0.15625, 0.140625)
DEFAULT_METRIC_SHAPE = (1, 100, 100)  # z, y, x


def as_xyz_tensor(points: torch.Tensor | np.ndarray, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(points) if isinstance(points, np.ndarray) else points
    if tensor.ndim != 2 or tensor.shape[1] < 3:
        raise ValueError(f"Expected a point cloud shaped (N, >=3); got {tuple(tensor.shape)}.")
    if device is not None:
        tensor = tensor.to(device)
    return tensor[:, :3].contiguous().float()


def _as_flat_bool_mask(mask: torch.Tensor | np.ndarray, *, device: torch.device, expected_count: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(mask, device=device).reshape(-1)
    if tensor.numel() != expected_count:
        raise ValueError(f"{name} has {tensor.numel()} values, expected {expected_count}.")
    return tensor.bool()


def prepare_metric_point_cloud(points: torch.Tensor | np.ndarray, *, did_return=None, raydrop=None, raydrop_threshold: float | None = 0.5) -> torch.Tensor:
    """Return XYZ points after applying GT return and prediction raydrop masks."""
    device = points.device if isinstance(points, torch.Tensor) else None
    xyz = as_xyz_tensor(points, device=device)
    if did_return is not None:
        xyz = xyz[_as_flat_bool_mask(did_return, device=xyz.device, expected_count=xyz.shape[0], name="did_return")]
    if raydrop is not None and raydrop_threshold is not None:
        kept = torch.as_tensor(raydrop, device=xyz.device).reshape(-1)
        if kept.numel() != xyz.shape[0]:
            raise ValueError(f"raydrop has {kept.numel()} values, expected {xyz.shape[0]}.")
        xyz = xyz[kept < raydrop_threshold]
    return xyz


@dltyped()
def point_cloud_to_histogram(
    xyz: Annotated[torch.Tensor, FloatTensor["N 3"]],
    *,
    pc_range: tuple[float, float, float, float, float, float] = DEFAULT_PC_RANGE,
    voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
    metric_shape: tuple[int, int, int] = DEFAULT_METRIC_SHAPE,
) -> Annotated[np.ndarray, FloatTensor["Z Y X"]]:
    """Voxelize points into compact z-y-x occupancy bins using train_lidar constants."""
    if len(metric_shape) != 3 or any(dim < 1 for dim in metric_shape):
        raise ValueError("metric_shape must contain three positive integers.")
    histogram = np.zeros(metric_shape, dtype=np.float32)
    if xyz.numel() == 0:
        return histogram
    bounds = xyz.new_tensor(pc_range)
    xyz_min, xyz_max = bounds[:3], bounds[3:]
    voxel = xyz.new_tensor(voxel_size)
    valid = torch.isfinite(xyz).all(dim=-1)
    valid &= ((xyz >= xyz_min) & (xyz < xyz_max)).all(dim=-1)
    xyz = xyz[valid]
    if xyz.numel() == 0:
        return histogram
    native_shape = torch.ceil((xyz_max - xyz_min) / voxel).long()
    native_indices = torch.floor((xyz - xyz_min) / voxel).long()
    native_indices = torch.minimum(native_indices, native_shape - 1)
    nx, ny, nz = native_shape.tolist()
    vx, vy, vz = native_indices[:, 0], native_indices[:, 1], native_indices[:, 2]
    native_flat = torch.unique(vx + nx * (vy + ny * vz))
    vx = native_flat % nx
    vy = (native_flat // nx) % ny
    vz = native_flat // (nx * ny)
    metric_z, metric_y, metric_x = metric_shape
    mx = torch.div(vx * metric_x, nx, rounding_mode="floor").clamp_max(metric_x - 1)
    my = torch.div(vy * metric_y, ny, rounding_mode="floor").clamp_max(metric_y - 1)
    mz = torch.div(vz * metric_z, nz, rounding_mode="floor").clamp_max(metric_z - 1)
    metric_flat = mx + metric_x * (my + metric_y * mz)
    counts = torch.bincount(metric_flat, minlength=metric_z * metric_y * metric_x)
    return counts.reshape(metric_shape).float().cpu().numpy()


def point_cloud_histogram(points: torch.Tensor | np.ndarray, *, did_return=None, raydrop=None, raydrop_threshold: float | None = 0.5, pc_range=DEFAULT_PC_RANGE, voxel_size=DEFAULT_VOXEL_SIZE, metric_shape=DEFAULT_METRIC_SHAPE) -> np.ndarray:
    xyz = prepare_metric_point_cloud(points, did_return=did_return, raydrop=raydrop, raydrop_threshold=raydrop_threshold)
    return point_cloud_to_histogram(xyz, pc_range=pc_range, voxel_size=voxel_size, metric_shape=metric_shape)
