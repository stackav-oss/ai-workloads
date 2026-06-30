"""Jensen-Shannon distance over LiDAR point-cloud occupancy histograms."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

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

__all__ = ["JensenShannonDivergence"]


@dataclass(frozen=True)
class JSDSampleState:
    """Bookkeeping for one sample added to JSD."""

    sample_id: str
    pred_occupied_bins: int
    gt_occupied_bins: int
    valid: bool

    def to_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


class JensenShannonDivergence:
    """Accumulate generated and GT point clouds for dataset-level JSD."""

    def __init__(
        self,
        spatial_shape: tuple[int, int, int] = DEFAULT_METRIC_SHAPE,
        *,
        pc_range: tuple[float, float, float, float, float, float] = DEFAULT_PC_RANGE,
        voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
        raydrop_threshold: float | None = 0.5,
    ) -> None:
        self.spatial_shape = spatial_shape
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.raydrop_threshold = raydrop_threshold
        self.p = np.zeros(spatial_shape, dtype=np.float64)
        self.q = np.zeros(spatial_shape, dtype=np.float64)
        self.samples: list[JSDSampleState] = []

    def add_sample(
        self,
        *,
        pred_xyz: torch.Tensor | np.ndarray,
        gt_xyz: torch.Tensor | np.ndarray,
        sample_id: str | None = None,
        pred_raydrop: torch.Tensor | np.ndarray | None = None,
        gt_did_return: torch.Tensor | np.ndarray | None = None,
    ) -> JSDSampleState:
        sample_name = sample_id or f"sample_{len(self.samples):06d}"
        pred_hist = point_cloud_histogram(
            pred_xyz,
            raydrop=pred_raydrop,
            raydrop_threshold=self.raydrop_threshold,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size,
            metric_shape=self.spatial_shape,
        )
        gt_hist = point_cloud_histogram(
            gt_xyz,
            did_return=gt_did_return,
            pc_range=self.pc_range,
            voxel_size=self.voxel_size,
            metric_shape=self.spatial_shape,
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
    ) -> JSDSampleState:
        sample_name = sample_id or f"sample_{len(self.samples):06d}"
        pred_hist = np.asarray(pred_hist, dtype=np.float64).reshape(self.spatial_shape)
        gt_hist = np.asarray(gt_hist, dtype=np.float64).reshape(self.spatial_shape)
        self.p += pred_hist
        self.q += gt_hist

        valid = bool(pred_hist.sum() > 0.0 and gt_hist.sum() > 0.0)
        state = JSDSampleState(
            sample_id=sample_name,
            pred_occupied_bins=int(np.count_nonzero(pred_hist)),
            gt_occupied_bins=int(np.count_nonzero(gt_hist)),
            valid=valid,
        )
        self.samples.append(state)
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
        p = self.p.reshape(-1)
        q = self.q.reshape(-1)
        if p.sum() <= 0.0 or q.sum() <= 0.0:
            return float("nan")
        return float(jensenshannon(p, q))

    def aggregate(self) -> dict[str, float | int]:
        valid_samples = [sample for sample in self.samples if sample.valid]
        return {
            "num_samples": len(self.samples),
            "valid_samples": len(valid_samples),
            "invalid_samples": len(self.samples) - len(valid_samples),
            "jsd": self.compute(),
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
                "JSD occupancy metrics",
                f"  num_samples: {aggregate['num_samples']}",
                f"  valid_samples: {aggregate['valid_samples']}",
                f"  invalid_samples: {aggregate['invalid_samples']}",
                f"  jsd: {aggregate['jsd']:.8f}",
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
