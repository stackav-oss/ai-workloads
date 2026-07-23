"""Maximum Mean Discrepancy over LiDAR point-cloud occupancy histograms."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

try:
    from .point_cloud_histogram import (
        DEFAULT_METRIC_SHAPE,
        DEFAULT_PC_RANGE,
        DEFAULT_VOXEL_SIZE,
        point_cloud_histogram,
    )
except ImportError:  # pragma: no cover - supports direct script imports.
    from point_cloud_histogram import (
        DEFAULT_METRIC_SHAPE,
        DEFAULT_PC_RANGE,
        DEFAULT_VOXEL_SIZE,
        point_cloud_histogram,
    )

__all__ = ["MaximumMeanDiscrepancy", "compute_mmd", "gaussian"]


@dataclass(frozen=True)
class DistributionSampleState:
    """Bookkeeping for one sample added to a distribution metric."""

    sample_id: str
    pred_occupied_bins: int
    gt_occupied_bins: int
    valid: bool

    def to_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


def gaussian(x: np.ndarray, y: np.ndarray, sigma: float = 0.5) -> float:
    """Gaussian kernel used by the MMD estimator."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    support_size = max(x.shape[0], y.shape[0])
    if x.shape[0] < support_size:
        x = np.pad(x, (0, support_size - x.shape[0]))
    if y.shape[0] < support_size:
        y = np.pad(y, (0, support_size - y.shape[0]))
    dist_sq = float(np.sum((x - y) ** 2))
    return float(np.exp(-dist_sq / (2.0 * sigma * sigma)))


def _normalize_histograms(samples: list[np.ndarray]) -> np.ndarray:
    normalized = []
    for sample in samples:
        flat = np.asarray(sample, dtype=np.float64).reshape(-1)
        total = flat.sum()
        if total <= 0.0 or not np.isfinite(total):
            continue
        normalized.append(flat / total)
    if not normalized:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(normalized, axis=0)


def _mean_gaussian_kernel(
    samples1: np.ndarray,
    samples2: np.ndarray,
    *,
    sigma: float,
    chunk_size: int,
) -> float:
    if samples1.shape[0] == 0 or samples2.shape[0] == 0:
        return float("nan")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    samples2_t = samples2.T
    samples2_norm = np.sum(samples2 * samples2, axis=1)[None, :]
    total = 0.0
    count = 0
    denom = 2.0 * sigma * sigma

    for start in range(0, samples1.shape[0], chunk_size):
        chunk = samples1[start : start + chunk_size]
        chunk_norm = np.sum(chunk * chunk, axis=1)[:, None]
        dist_sq = chunk_norm + samples2_norm - 2.0 * (chunk @ samples2_t)
        np.maximum(dist_sq, 0.0, out=dist_sq)
        total += float(np.exp(-dist_sq / denom).sum())
        count += dist_sq.size

    return total / count


def compute_mmd(
    samples1: list[np.ndarray] | np.ndarray,
    samples2: list[np.ndarray] | np.ndarray,
    kernel=gaussian,
    *,
    is_hist: bool = True,
    sigma: float = 0.5,
    chunk_size: int = 32,
    **_kwargs,
) -> float:
    """Return biased MMD estimate between two histogram sample sets."""
    del kernel
    if not is_hist:
        raise ValueError("compute_mmd currently expects histogram inputs.")

    samples1_array = (
        _normalize_histograms(samples1)
        if isinstance(samples1, list)
        else _normalize_histograms([row for row in samples1])
    )
    samples2_array = (
        _normalize_histograms(samples2)
        if isinstance(samples2, list)
        else _normalize_histograms([row for row in samples2])
    )
    if samples1_array.shape[0] == 0 or samples2_array.shape[0] == 0:
        return float("nan")

    mmd = (
        _mean_gaussian_kernel(
            samples1_array,
            samples1_array,
            sigma=sigma,
            chunk_size=chunk_size,
        )
        + _mean_gaussian_kernel(
            samples2_array,
            samples2_array,
            sigma=sigma,
            chunk_size=chunk_size,
        )
        - 2.0
        * _mean_gaussian_kernel(
            samples1_array,
            samples2_array,
            sigma=sigma,
            chunk_size=chunk_size,
        )
    )
    return max(float(mmd), 0.0)


class MaximumMeanDiscrepancy:
    """Accumulate generated and GT point clouds for dataset-level MMD."""

    def __init__(
        self,
        *,
        pc_range: tuple[float, float, float, float, float, float] = DEFAULT_PC_RANGE,
        voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
        metric_shape: tuple[int, int, int] = DEFAULT_METRIC_SHAPE,
        raydrop_threshold: float | None = 0.5,
        sigma: float = 0.5,
        kernel_chunk_size: int = 32,
    ) -> None:
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.metric_shape = metric_shape
        self.raydrop_threshold = raydrop_threshold
        self.sigma = sigma
        self.kernel_chunk_size = kernel_chunk_size
        self.gt_set: list[np.ndarray] = []
        self.gen_set: list[np.ndarray] = []
        self.samples: list[DistributionSampleState] = []

    def add_sample(
        self,
        *,
        pred_xyz: torch.Tensor | np.ndarray,
        gt_xyz: torch.Tensor | np.ndarray,
        sample_id: str | None = None,
        pred_raydrop: torch.Tensor | np.ndarray | None = None,
        gt_did_return: torch.Tensor | np.ndarray | None = None,
    ) -> DistributionSampleState:
        sample_name = sample_id or f"sample_{len(self.samples):06d}"
        pred_hist = point_cloud_histogram(
            pred_xyz,
            raydrop=pred_raydrop,
            raydrop_threshold=self.raydrop_threshold,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size,
            metric_shape=self.metric_shape,
        )
        gt_hist = point_cloud_histogram(
            gt_xyz,
            did_return=gt_did_return,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size,
            metric_shape=self.metric_shape,
        )
        return self.add_histograms(
            pred_hist=pred_hist,
            gt_hist=gt_hist,
            sample_id=sample_name,
        )

    def add_histograms(
        self,
        *,
        pred_hist: np.ndarray,
        gt_hist: np.ndarray,
        sample_id: str | None = None,
    ) -> DistributionSampleState:
        sample_name = sample_id or f"sample_{len(self.samples):06d}"
        pred_flat = np.asarray(pred_hist, dtype=np.float32).reshape(-1)
        gt_flat = np.asarray(gt_hist, dtype=np.float32).reshape(-1)
        valid = bool(pred_flat.sum() > 0.0 and gt_flat.sum() > 0.0)
        state = DistributionSampleState(
            sample_id=sample_name,
            pred_occupied_bins=int(np.count_nonzero(pred_flat)),
            gt_occupied_bins=int(np.count_nonzero(gt_flat)),
            valid=valid,
        )
        self.samples.append(state)
        if valid:
            self.gen_set.append(pred_flat)
            self.gt_set.append(gt_flat)
        return state

    def update(self, data: dict) -> None:
        self.add_sample(
            pred_xyz=data["pred_xyz"],
            pred_raydrop=data.get("pred_raydrop"),
            gt_xyz=data["gt_xyz"],
            gt_did_return=data.get("gt_did_return"),
            sample_id=data.get("sample_id"),
        )

    def compute(self) -> float:
        return compute_mmd(
            self.gt_set,
            self.gen_set,
            sigma=self.sigma,
            chunk_size=self.kernel_chunk_size,
        )

    def aggregate(self) -> dict[str, float | int]:
        valid_samples = [sample for sample in self.samples if sample.valid]
        return {
            "num_samples": len(self.samples),
            "valid_samples": len(valid_samples),
            "invalid_samples": len(self.samples) - len(valid_samples),
            "mmd": self.compute(),
            "pred_occupied_bins_mean": self._mean(
                [sample.pred_occupied_bins for sample in self.samples]
            ),
            "gt_occupied_bins_mean": self._mean(
                [sample.gt_occupied_bins for sample in self.samples]
            ),
        }

    def format_aggregate(self) -> str:
        aggregate = self.aggregate()
        return "\n".join(
            [
                "MMD occupancy metrics",
                f"  num_samples: {aggregate['num_samples']}",
                f"  valid_samples: {aggregate['valid_samples']}",
                f"  invalid_samples: {aggregate['invalid_samples']}",
                f"  mmd: {aggregate['mmd']:.8e}",
                (
                    "  occupied bins: "
                    f"pred_mean={aggregate['pred_occupied_bins_mean']:.1f} "
                    f"gt_mean={aggregate['gt_occupied_bins_mean']:.1f}"
                ),
            ]
        )

    def print_aggregate(self) -> None:
        print(self.format_aggregate())

    @staticmethod
    def _mean(values: list[int]) -> float:
        if not values:
            return float("nan")
        return float(np.asarray(values, dtype=np.float64).mean())
