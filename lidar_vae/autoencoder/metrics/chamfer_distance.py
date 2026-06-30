"""Per-sample and aggregate Chamfer distance metrics for point clouds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Annotated

import numpy as np
import torch
from dltype import FloatTensor, dltyped

DEFAULT_PC_RANGE = (-80.0, -80.0, -4.5, 80.0, 80.0, 4.5)


@dataclass(frozen=True)
class ChamferSampleMetric:
    """Chamfer metrics for one prediction/ground-truth point-cloud pair."""

    sample_id: str
    pred_point_count: int
    gt_point_count: int
    chamfer_pred_to_gt: float
    chamfer_gt_to_pred: float
    chamfer_symmetric: float
    chamfer_pred_to_gt_sqrt_m: float
    chamfer_gt_to_pred_sqrt_m: float
    chamfer_symmetric_sqrt_m: float
    valid: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@lru_cache(maxsize=1)
def _load_cuda_chamfer():
    try:
        from chamfer3D.dist_chamfer_3D import chamfer_3DDist
    except Exception:
        return None
    return chamfer_3DDist()


def _as_xyz_tensor(
    points: torch.Tensor | np.ndarray,
    *,
    device: torch.device | None,
) -> torch.Tensor:
    if isinstance(points, np.ndarray):
        tensor = torch.from_numpy(points)
    else:
        tensor = points

    if tensor.ndim != 2 or tensor.shape[1] < 3:
        raise ValueError(
            f"Expected a point cloud shaped (N, >=3); got {tuple(tensor.shape)}."
        )

    if device is not None:
        tensor = tensor.to(device)
    return tensor[:, :3].contiguous().float()


def _as_flat_mask(
    mask: torch.Tensor | np.ndarray,
    *,
    device: torch.device,
    expected_count: int,
    name: str,
) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        tensor = torch.from_numpy(mask)
    else:
        tensor = mask
    tensor = tensor.to(device).reshape(-1)
    if tensor.numel() != expected_count:
        raise ValueError(
            f"{name} has {tensor.numel()} values, expected {expected_count}."
        )
    return tensor.bool()


def _filter_points(
    xyz: torch.Tensor,
    *,
    pc_range: tuple[float, float, float, float, float, float] | None,
) -> torch.Tensor:
    if xyz.numel() == 0:
        return xyz.reshape(0, 3)

    valid = torch.isfinite(xyz).all(dim=-1)
    if pc_range is not None:
        bounds = xyz.new_tensor(pc_range)
        xyz_min = bounds[:3]
        xyz_max = bounds[3:]
        valid &= ((xyz >= xyz_min) & (xyz <= xyz_max)).all(dim=-1)
    return xyz[valid]


def _deterministic_cap(points: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
    ).round()
    return points.index_select(0, indices.long())


def _nearest_squared_cdist(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, source_xyz.shape[0], chunk_size):
        source_chunk = source_xyz[start : start + chunk_size]
        distances = torch.cdist(source_chunk, target_xyz).square()
        chunks.append(distances.min(dim=1).values)
    return torch.cat(chunks, dim=0)


@dltyped()
def chamfer_distance_components(
    pred_xyz: Annotated[torch.Tensor, FloatTensor["N 3"]],
    gt_xyz: Annotated[torch.Tensor, FloatTensor["M 3"]],
    *,
    use_cuda_extension: bool = True,
    fallback_chunk_size: int = 1024,
) -> tuple[
    Annotated[torch.Tensor, FloatTensor["N"]],
    Annotated[torch.Tensor, FloatTensor["M"]],
]:
    """Return squared nearest-neighbor distances in both Chamfer directions."""
    if pred_xyz.numel() == 0 or gt_xyz.numel() == 0:
        empty = pred_xyz.new_empty(0)
        return empty, empty
    if pred_xyz.device != gt_xyz.device:
        gt_xyz = gt_xyz.to(pred_xyz.device)

    chamfer = _load_cuda_chamfer() if use_cuda_extension else None
    if chamfer is not None and pred_xyz.is_cuda and gt_xyz.is_cuda:
        dist_pred_to_gt, dist_gt_to_pred, _, _ = chamfer(
            pred_xyz.contiguous().unsqueeze(0),
            gt_xyz.contiguous().unsqueeze(0),
        )
        return dist_pred_to_gt.squeeze(0), dist_gt_to_pred.squeeze(0)

    if fallback_chunk_size < 1:
        raise ValueError("fallback_chunk_size must be at least 1.")
    return (
        _nearest_squared_cdist(
            pred_xyz,
            gt_xyz,
            chunk_size=fallback_chunk_size,
        ),
        _nearest_squared_cdist(
            gt_xyz,
            pred_xyz,
            chunk_size=fallback_chunk_size,
        ),
    )


class ChamferDistanceMetrics:
    """Accumulate per-sample Chamfer metrics and print dataset aggregates."""

    def __init__(
        self,
        *,
        max_points: int | None = 65536,
        raydrop_threshold: float | None = 0.5,
        pc_range: tuple[float, float, float, float, float, float] | None = (
            DEFAULT_PC_RANGE
        ),
        use_cuda_extension: bool = True,
        fallback_chunk_size: int = 1024,
    ) -> None:
        self.max_points = max_points
        self.raydrop_threshold = raydrop_threshold
        self.pc_range = pc_range
        self.use_cuda_extension = use_cuda_extension
        self.fallback_chunk_size = fallback_chunk_size
        self.samples: list[ChamferSampleMetric] = []

    def add_sample(
        self,
        *,
        pred_xyz: torch.Tensor | np.ndarray,
        gt_xyz: torch.Tensor | np.ndarray,
        sample_id: str | None = None,
        pred_raydrop: torch.Tensor | np.ndarray | None = None,
        gt_did_return: torch.Tensor | np.ndarray | None = None,
    ) -> ChamferSampleMetric:
        """Compute and store Chamfer metrics for one point-cloud pair."""
        sample_name = sample_id or f"sample_{len(self.samples):06d}"
        device = pred_xyz.device if isinstance(pred_xyz, torch.Tensor) else None

        pred = _as_xyz_tensor(pred_xyz, device=device)
        gt = _as_xyz_tensor(gt_xyz, device=pred.device)

        if pred_raydrop is not None and self.raydrop_threshold is not None:
            pred_returned = _as_flat_mask(
                torch.as_tensor(pred_raydrop) < self.raydrop_threshold,
                device=pred.device,
                expected_count=pred.shape[0],
                name="pred_raydrop",
            )
            pred = pred[pred_returned]

        if gt_did_return is not None:
            gt_returned = _as_flat_mask(
                gt_did_return,
                device=gt.device,
                expected_count=gt.shape[0],
                name="gt_did_return",
            )
            gt = gt[gt_returned]

        pred = _deterministic_cap(
            _filter_points(pred, pc_range=self.pc_range),
            self.max_points,
        )
        gt = _deterministic_cap(
            _filter_points(gt, pc_range=self.pc_range),
            self.max_points,
        )

        metric = self._compute_sample_metric(
            sample_id=sample_name,
            pred=pred,
            gt=gt,
        )
        self.samples.append(metric)
        return metric

    def _compute_sample_metric(
        self,
        *,
        sample_id: str,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> ChamferSampleMetric:
        pred_count = int(pred.shape[0])
        gt_count = int(gt.shape[0])
        if pred_count == 0 or gt_count == 0:
            nan = float("nan")
            return ChamferSampleMetric(
                sample_id=sample_id,
                pred_point_count=pred_count,
                gt_point_count=gt_count,
                chamfer_pred_to_gt=nan,
                chamfer_gt_to_pred=nan,
                chamfer_symmetric=nan,
                chamfer_pred_to_gt_sqrt_m=nan,
                chamfer_gt_to_pred_sqrt_m=nan,
                chamfer_symmetric_sqrt_m=nan,
                valid=False,
            )

        with torch.no_grad():
            dist_pred_to_gt, dist_gt_to_pred = chamfer_distance_components(
                pred,
                gt,
                use_cuda_extension=self.use_cuda_extension,
                fallback_chunk_size=self.fallback_chunk_size,
            )
            pred_to_gt = dist_pred_to_gt.mean()
            gt_to_pred = dist_gt_to_pred.mean()
            symmetric = 0.5 * (pred_to_gt + gt_to_pred)

        return ChamferSampleMetric(
            sample_id=sample_id,
            pred_point_count=pred_count,
            gt_point_count=gt_count,
            chamfer_pred_to_gt=float(pred_to_gt.detach().cpu().item()),
            chamfer_gt_to_pred=float(gt_to_pred.detach().cpu().item()),
            chamfer_symmetric=float(symmetric.detach().cpu().item()),
            chamfer_pred_to_gt_sqrt_m=float(
                pred_to_gt.clamp_min(0.0).sqrt().detach().cpu().item()
            ),
            chamfer_gt_to_pred_sqrt_m=float(
                gt_to_pred.clamp_min(0.0).sqrt().detach().cpu().item()
            ),
            chamfer_symmetric_sqrt_m=float(
                symmetric.clamp_min(0.0).sqrt().detach().cpu().item()
            ),
            valid=True,
        )

    def aggregate(self) -> dict[str, float | int]:
        """Return sample-averaged aggregate metrics."""
        valid_samples = [sample for sample in self.samples if sample.valid]
        result: dict[str, float | int] = {
            "num_samples": len(self.samples),
            "valid_samples": len(valid_samples),
            "invalid_samples": len(self.samples) - len(valid_samples),
        }

        result["pred_point_count_mean"] = self._mean(
            [sample.pred_point_count for sample in self.samples]
        )
        result["gt_point_count_mean"] = self._mean(
            [sample.gt_point_count for sample in self.samples]
        )

        metric_names = (
            "chamfer_pred_to_gt",
            "chamfer_gt_to_pred",
            "chamfer_symmetric",
            "chamfer_pred_to_gt_sqrt_m",
            "chamfer_gt_to_pred_sqrt_m",
            "chamfer_symmetric_sqrt_m",
        )
        for name in metric_names:
            values = [getattr(sample, name) for sample in valid_samples]
            result[f"{name}_mean"] = self._mean(values)
            result[f"{name}_median"] = self._median(values)
            result[f"{name}_std"] = self._std(values)
        return result

    def format_aggregate(self) -> str:
        aggregate = self.aggregate()
        lines = [
            "Chamfer distance metrics",
            f"  num_samples: {aggregate['num_samples']}",
            f"  valid_samples: {aggregate['valid_samples']}",
            f"  invalid_samples: {aggregate['invalid_samples']}",
            "  squared distances (m^2):",
            (
                "    chamfer_pred_to_gt: "
                f"mean={aggregate['chamfer_pred_to_gt_mean']:.6f} "
                f"median={aggregate['chamfer_pred_to_gt_median']:.6f} "
                f"std={aggregate['chamfer_pred_to_gt_std']:.6f}"
            ),
            (
                "    chamfer_gt_to_pred: "
                f"mean={aggregate['chamfer_gt_to_pred_mean']:.6f} "
                f"median={aggregate['chamfer_gt_to_pred_median']:.6f} "
                f"std={aggregate['chamfer_gt_to_pred_std']:.6f}"
            ),
            (
                "    chamfer_symmetric: "
                f"mean={aggregate['chamfer_symmetric_mean']:.6f} "
                f"median={aggregate['chamfer_symmetric_median']:.6f} "
                f"std={aggregate['chamfer_symmetric_std']:.6f}"
            ),
            "  sqrt mean-squared distances (m):",
            (
                "    chamfer_pred_to_gt_sqrt_m: "
                f"mean={aggregate['chamfer_pred_to_gt_sqrt_m_mean']:.6f} "
                f"median={aggregate['chamfer_pred_to_gt_sqrt_m_median']:.6f} "
                f"std={aggregate['chamfer_pred_to_gt_sqrt_m_std']:.6f}"
            ),
            (
                "    chamfer_gt_to_pred_sqrt_m: "
                f"mean={aggregate['chamfer_gt_to_pred_sqrt_m_mean']:.6f} "
                f"median={aggregate['chamfer_gt_to_pred_sqrt_m_median']:.6f} "
                f"std={aggregate['chamfer_gt_to_pred_sqrt_m_std']:.6f}"
            ),
            (
                "    chamfer_symmetric_sqrt_m: "
                f"mean={aggregate['chamfer_symmetric_sqrt_m_mean']:.6f} "
                f"median={aggregate['chamfer_symmetric_sqrt_m_median']:.6f} "
                f"std={aggregate['chamfer_symmetric_sqrt_m_std']:.6f}"
            ),
            (
                "  point counts: "
                f"pred_mean={aggregate['pred_point_count_mean']:.1f} "
                f"gt_mean={aggregate['gt_point_count_mean']:.1f}"
            ),
        ]
        return "\n".join(lines)

    def print_aggregate(self) -> None:
        print(self.format_aggregate())

    @staticmethod
    def _mean(values: list[float | int]) -> float:
        if not values:
            return float("nan")
        return float(np.asarray(values, dtype=np.float64).mean())

    @staticmethod
    def _median(values: list[float | int]) -> float:
        if not values:
            return float("nan")
        return float(np.median(np.asarray(values, dtype=np.float64)))

    @staticmethod
    def _std(values: list[float | int]) -> float:
        if not values:
            return float("nan")
        return float(np.asarray(values, dtype=np.float64).std())
