"""Lightweight NuScenes types shared between the raw and Lance dataset modules.

This module has ZERO dependency on ``nuscenes-devkit``.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, NamedTuple

import torch

from jormungand.datastructures.sequential_data import SequentialDataFrame


def _make_rig(tuple_name: str, values: dict[str, Any]) -> NamedTuple | None:
    if not values:
        return None
    rig_type = namedtuple(tuple_name, values.keys())
    return rig_type(**values)


class CameraName(str, Enum):
    CAM_FRONT = "CAM_FRONT"
    CAM_FRONT_LEFT = "CAM_FRONT_LEFT"
    CAM_FRONT_RIGHT = "CAM_FRONT_RIGHT"
    CAM_BACK = "CAM_BACK"
    CAM_BACK_LEFT = "CAM_BACK_LEFT"
    CAM_BACK_RIGHT = "CAM_BACK_RIGHT"


@dataclass(frozen=True)
class NuScenesDataConfig:
    CAM_FRONT: bool = True
    CAM_FRONT_LEFT: bool = True
    CAM_FRONT_RIGHT: bool = True
    CAM_BACK: bool = True
    CAM_BACK_LEFT: bool = True
    CAM_BACK_RIGHT: bool = True
    lidar: bool = True
    bounding_boxes_3d: bool = True
    # Number of *previous* non-keyframe LiDAR sweeps to aggregate into the
    # current keyframe LiDAR. 0 = keyframe only (default, backward compat).
    # UniPAD/UVTR use 9 (i.e. 1 keyframe + 9 sweeps = 10 scans).
    # Aggregated points are transformed into the current keyframe's ego frame
    # so the resulting PointCloud is still in ego frame.
    lidar_sweeps_num: int = 0
    # Drop points whose XY-radius is below this threshold (metres) when
    # aggregating sweeps – matches UniPAD's ``remove_close=True`` default.
    lidar_sweep_close_radius: float = 1.0

    @property
    def requested_camera_names(self) -> tuple[CameraName, ...]:
        return tuple(
            camera_name
            for camera_name in CameraName
            if getattr(self, camera_name.value)
        )

    @classmethod
    def from_requested_cameras(
        cls,
        requested_cameras: Iterable[str],
        *,
        lidar: bool = True,
        bounding_boxes_3d: bool = True,
    ) -> NuScenesDataConfig:
        requested_camera_set = {
            CameraName(camera_name).value for camera_name in requested_cameras
        }
        config_kwargs = {
            camera_name.value: camera_name.value in requested_camera_set
            for camera_name in CameraName
        }
        config_kwargs["lidar"] = lidar
        config_kwargs["bounding_boxes_3d"] = bounding_boxes_3d
        return cls(**config_kwargs)


class NuScenesDataFrame(SequentialDataFrame):
    def __init__(
        self,
        sequence_id: str,
        sequence_idx: int,
        timestamp: int,
        global_se3_ego: Any | None = None,
        lidar_rig: NamedTuple | None = None,
        camera_rig: NamedTuple | None = None,
        bounding_boxes_3d: torch.Tensor | None = None,
        bbox_classes: list[str] | None = None,
        bbox_tracking_ids: list[str] | None = None,
        is_key_frame: bool = True,
    ) -> None:
        super().__init__(
            sequence_id=sequence_id,
            sequence_idx=sequence_idx,
            timestamp=timestamp,
            global_se3_ego=global_se3_ego,
            lidar_rig=lidar_rig,
            camera_rig=camera_rig,
        )
        self.bounding_boxes_3d = bounding_boxes_3d
        self.bbox_classes = bbox_classes
        self.bbox_tracking_ids = bbox_tracking_ids
        self.is_key_frame = is_key_frame
        self._dataset: Any = None
        self._dataset_index: int | None = None

    def _attach_dataset(self, dataset: Any, dataset_index: int) -> NuScenesDataFrame:
        self._dataset = dataset
        self._dataset_index = dataset_index
        return self

    def _adjacent_frame(
        self, step: int, key_frames_only: bool = False
    ) -> NuScenesDataFrame | None:
        """Navigate adjacency. If key_frames_only, skip non-keyframes."""
        if self._dataset is None or self._dataset_index is None:
            return None
        cursor = self._dataset.get_adjacent_frame(self._dataset_index, step)
        if not key_frames_only:
            return cursor
        # Skip until we find a keyframe (or run out of frames)
        while cursor is not None and not cursor.is_key_frame:
            cursor = cursor._dataset.get_adjacent_frame(cursor._dataset_index, step)
        return cursor

    def next_frame(
        self, key_frames_only: bool | None = None
    ) -> NuScenesDataFrame | None:
        """Next frame. key_frames_only=None uses dataset default."""
        if key_frames_only is None:
            key_frames_only = getattr(self._dataset, "key_frames_only", False)
        return self._adjacent_frame(1, key_frames_only=key_frames_only)

    def prev_frame(
        self, key_frames_only: bool | None = None
    ) -> NuScenesDataFrame | None:
        """Previous frame. key_frames_only=None uses dataset default."""
        if key_frames_only is None:
            key_frames_only = getattr(self._dataset, "key_frames_only", False)
        return self._adjacent_frame(-1, key_frames_only=key_frames_only)

    def to(self, device: torch.device) -> NuScenesDataFrame:
        sequential_data_frame = super().to(device)
        bounding_boxes_3d = (
            self.bounding_boxes_3d.to(device)
            if self.bounding_boxes_3d is not None
            else None
        )
        frame = type(self)(
            sequence_id=sequential_data_frame.sequence_id,
            sequence_idx=sequential_data_frame.sequence_idx,
            timestamp=sequential_data_frame.timestamp,
            global_se3_ego=sequential_data_frame.global_se3_ego,
            lidar_rig=sequential_data_frame.lidar_rig,
            camera_rig=sequential_data_frame.camera_rig,
            bounding_boxes_3d=bounding_boxes_3d,
            bbox_classes=self.bbox_classes,
            bbox_tracking_ids=self.bbox_tracking_ids,
            is_key_frame=self.is_key_frame,
        )
        if self._dataset is not None and self._dataset_index is not None:
            frame._attach_dataset(self._dataset, self._dataset_index)
        return frame


def collate_nuscenes_frames(
    batch: list[NuScenesDataFrame],
) -> list[NuScenesDataFrame]:
    return batch


def _points_in_obb(
    points: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Test which points fall inside oriented bounding boxes (yaw-only rotation).

    Args:
        points: (P, 3) point positions.
        centers: (B, 3) box centers.
        sizes: (B, 3) box [l, w, h].
        yaws: (B,) box yaw angles.
        margin: extra metres added to each box half-extent.

    Returns:
        (P, B) bool mask — True where point p is inside box b.
    """
    # (P, B, 3)
    local = points[:, None, :] - centers[None, :, :]
    cos, sin = torch.cos(-yaws), torch.sin(-yaws)  # (B,)
    rx = local[..., 0] * cos[None, :] - local[..., 1] * sin[None, :]
    ry = local[..., 0] * sin[None, :] + local[..., 1] * cos[None, :]
    rz = local[..., 2]
    half = sizes.T / 2 + margin  # (3, B)
    return (rx.abs() < half[0]) & (ry.abs() < half[1]) & (rz.abs() < half[2])


def _compute_box_velocities(
    frame: NuScenesDataFrame,
    ref_ego_se3_global: "SE3",
) -> dict[str, torch.Tensor]:
    """Compute per-tracked-object velocity (m/s) in reference ego frame.

    Compares bounding-box centres between *frame* (reference keyframe) and the
    previous keyframe. Returns a mapping from tracking_id → velocity [3].
    """
    if (
        frame.bounding_boxes_3d is None
        or frame.bbox_tracking_ids is None
        or not frame.bbox_tracking_ids
    ):
        return {}

    prev_key = frame.prev_frame(key_frames_only=True)
    if (
        prev_key is None
        or prev_key.bounding_boxes_3d is None
        or prev_key.bbox_tracking_ids is None
        or prev_key.global_se3_ego is None
    ):
        return {}

    dt = (frame.timestamp - prev_key.timestamp) * 1e-6  # µs → seconds
    if dt <= 0:
        return {}

    # Build look-up: tracking_id → center in reference ego frame for prev keyframe
    prev_centers_prev_ego = prev_key.bounding_boxes_3d[:, 3:6]
    prev_centers_global = prev_key.global_se3_ego.transform_points(
        prev_centers_prev_ego
    )
    prev_centers_ref = ref_ego_se3_global.transform_points(prev_centers_global)
    prev_id_to_center = {
        tid: prev_centers_ref[i] for i, tid in enumerate(prev_key.bbox_tracking_ids)
    }

    velocities: dict[str, torch.Tensor] = {}
    for i, tid in enumerate(frame.bbox_tracking_ids):
        if tid in prev_id_to_center:
            ref_center = frame.bounding_boxes_3d[i, 3:6]
            velocities[tid] = (ref_center - prev_id_to_center[tid]) / dt
    return velocities


# Categories whose instances can move and need sweep compensation.
_DYNAMIC_PREFIXES = ("vehicle.", "human.", "animal")


def aggregate_lidar_sweeps(
    frame: NuScenesDataFrame,
    num_sweeps: int,
    close_radius: float = 1.0,
) -> NuScenesDataFrame:
    """Aggregate LiDAR from N previous frames into the reference frame's ego coordinate system.

    Walks ``prev_frame()`` up to *num_sweeps* times, transforms each frame's
    LiDAR points into the reference frame's ego frame, and concatenates them
    into one thick pointcloud.  Points on *moving* objects are compensated
    using per-frame interpolated bounding boxes: for each sweep, points inside
    a dynamic object's box at the sweep timestamp are translated to the
    object's position at the reference frame timestamp.

    Args:
        frame: The reference frame (typically a keyframe with bounding boxes).
        num_sweeps: Number of *previous* frames whose LiDAR to include
            (0 = reference only, 9 = reference + 9 previous = 10 scans).
        close_radius: Drop points whose XY-distance to the ego origin is below
            this threshold (metres). Set to 0 to disable.

    Returns:
        A new NuScenesDataFrame identical to *frame* but with an aggregated
        LiDAR pointcloud in the lidar_rig.
    """
    from jormungand.datastructures.pointcloud import PointCloud
    from jormungand.datastructures.se3 import SE3

    if num_sweeps <= 0 or frame.lidar_rig is None or frame.global_se3_ego is None:
        return frame

    # Collect the reference frame's lidar first
    ref_pc: PointCloud = frame.lidar_rig.LIDAR_TOP
    ref_global_se3_ego: SE3 = frame.global_se3_ego
    ref_ego_se3_global = ref_global_se3_ego.inverse()

    # --- Reference frame dynamic box setup ---
    # Build a lookup from tracking_id → box center in ref ego frame
    ref_dyn_centers: dict[str, torch.Tensor] = {}
    if (
        frame.bounding_boxes_3d is not None
        and frame.bbox_classes is not None
        and frame.bbox_tracking_ids is not None
    ):
        for i, cls in enumerate(frame.bbox_classes):
            if any(cls.startswith(p) for p in _DYNAMIC_PREFIXES):
                tid = frame.bbox_tracking_ids[i]
                # centers are already in ref ego frame (stored that way)
                ref_dyn_centers[tid] = frame.bounding_boxes_3d[i, 3:6]

    all_points = [ref_pc.points]
    all_intensity = [ref_pc.intensity] if ref_pc.intensity is not None else []
    has_intensity = ref_pc.intensity is not None

    # --- Load previous frames: batch if possible, otherwise walk one-by-one ---
    prev_frames: list[NuScenesDataFrame] = []
    if (
        frame._dataset is not None
        and frame._dataset_index is not None
        and hasattr(frame._dataset, "get_prev_sweep_frames")
    ):
        prev_frames = frame._dataset.get_prev_sweep_frames(
            frame._dataset_index, num_sweeps
        )
    else:
        # Fallback: sequential navigation
        cur = frame.prev_frame(key_frames_only=False)
        while cur is not None and len(prev_frames) < num_sweeps:
            prev_frames.append(cur)
            cur = cur.prev_frame(key_frames_only=False)

    # Walk backwards at full sensor frequency (20Hz)
    for prev in prev_frames:
        if prev.lidar_rig is None or prev.global_se3_ego is None:
            continue

        prev_pc: PointCloud = prev.lidar_rig.LIDAR_TOP
        prev_global_se3_ego: SE3 = prev.global_se3_ego

        # Fuse all transforms: ref_ego ← global ← prev_ego ← sensor
        ref_se3_prev = ref_ego_se3_global.compose(prev_global_se3_ego)
        ref_se3_sensor = ref_se3_prev.compose(prev_pc.ego_se3_sensor)
        points_ref_ego = ref_se3_sensor.transform_points(prev_pc.points)

        # --- Compensate moving objects using per-frame interpolated boxes ---
        if (
            ref_dyn_centers
            and prev.bounding_boxes_3d is not None
            and prev.bbox_classes is not None
            and prev.bbox_tracking_ids is not None
            and len(prev.bbox_tracking_ids) > 0
        ):
            # Find dynamic boxes in the sweep frame that also exist in the reference frame
            sweep_dyn_indices = []
            sweep_dyn_tids = []
            for i, cls in enumerate(prev.bbox_classes):
                tid = prev.bbox_tracking_ids[i]
                if tid in ref_dyn_centers and any(
                    cls.startswith(p) for p in _DYNAMIC_PREFIXES
                ):
                    sweep_dyn_indices.append(i)
                    sweep_dyn_tids.append(tid)

            if sweep_dyn_indices:
                sweep_boxes = prev.bounding_boxes_3d[sweep_dyn_indices]  # (D, 9)
                # Sweep box centers are in the sweep's ego frame → transform to ref ego
                sweep_centers_sweep_ego = sweep_boxes[:, 3:6]
                sweep_centers_ref = ref_se3_prev.transform_points(
                    sweep_centers_sweep_ego
                )
                ref_centers = torch.stack(
                    [ref_dyn_centers[tid] for tid in sweep_dyn_tids]
                )  # (D, 3)

                # Skip boxes that barely moved (< 0.1m displacement)
                displacements = (ref_centers - sweep_centers_ref).norm(dim=1)
                moved_mask = displacements > 0.1
                if moved_mask.any():
                    # Filter to only significantly displaced boxes
                    moved_indices = moved_mask.nonzero(as_tuple=True)[0]
                    sweep_sizes = sweep_boxes[moved_indices, :3]  # l, w, h
                    sweep_centers_ref = sweep_centers_ref[moved_indices]
                    ref_centers = ref_centers[moved_indices]

                    # Get ref yaws for matching boxes
                    moved_tids = [sweep_dyn_tids[i] for i in moved_indices.tolist()]
                    ref_yaws = []
                    for tid in moved_tids:
                        idx = frame.bbox_tracking_ids.index(tid)
                        ref_yaws.append(frame.bounding_boxes_3d[idx, 6])
                    ref_yaws_t = torch.tensor(ref_yaws)

                    inside = _points_in_obb(
                        points_ref_ego,
                        sweep_centers_ref,
                        sweep_sizes,
                        ref_yaws_t,
                    )
                    any_inside = inside.any(dim=1)  # (P,)
                    if any_inside.any():
                        box_idx = inside.float().argmax(dim=1)  # (P,)
                        correction = ref_centers[box_idx] - sweep_centers_ref[box_idx]
                        points_ref_ego = points_ref_ego.clone()
                        points_ref_ego[any_inside] += correction[any_inside]

        all_points.append(points_ref_ego)
        if has_intensity and prev_pc.intensity is not None:
            all_intensity.append(prev_pc.intensity)

    # Concatenate
    merged_points = torch.cat(all_points, dim=0)
    merged_intensity: torch.Tensor | None = None
    if has_intensity and all_intensity:
        merged_intensity = torch.cat(all_intensity, dim=0)

    # Filter close points
    if close_radius > 0:
        xy_dist = merged_points[:, :2].norm(dim=1)
        keep = xy_dist >= close_radius
        merged_points = merged_points[keep]
        if merged_intensity is not None:
            merged_intensity = merged_intensity[keep]

    aggregated_pc = PointCloud(
        points=merged_points,
        intensity=merged_intensity,
        ego_se3_sensor=SE3.identity(),
        caption=ref_pc.caption,
        timestamp_ns=ref_pc.timestamp_ns,
    )

    lidar_rig = _make_rig("LidarRig", {"LIDAR_TOP": aggregated_pc})

    new_frame = NuScenesDataFrame(
        sequence_id=frame.sequence_id,
        sequence_idx=frame.sequence_idx,
        timestamp=frame.timestamp,
        global_se3_ego=frame.global_se3_ego,
        lidar_rig=lidar_rig,
        camera_rig=frame.camera_rig,
        bounding_boxes_3d=frame.bounding_boxes_3d,
        bbox_classes=frame.bbox_classes,
        bbox_tracking_ids=frame.bbox_tracking_ids,
        is_key_frame=frame.is_key_frame,
    )
    if frame._dataset is not None and frame._dataset_index is not None:
        new_frame._attach_dataset(frame._dataset, frame._dataset_index)
    return new_frame
