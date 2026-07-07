"""NuScenes Lance dataset reader + visualization entry point.

Usage::

    python -m jormungand.datasets.nuscenes.nuscenes_lance \
        --version v1.0-trainval --data-root /data/nuscenes \
        --index 0

    python -m jormungand.datasets.nuscenes.nuscenes_lance \
        --version v1.0-trainval --data-root /data/nuscenes \
        --index 0 --num-frames 20 --colorize-lidar --frustum-scale 2.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Sequence, cast

import os
import click
import cv2
import lance
import numpy as np
import torch

from jormungand.datasets.nuscenes.nuscenes_class_to_colors import NUSCENES_COLORS
from jormungand.datastructures.sequential_data_renderer import (
    SequentialFrameRenderer,
)
from jormungand.datastructures.camera_image import CameraImage
from jormungand.datastructures.pointcloud import PointCloud
from jormungand.datastructures.se3 import SE3
from jormungand.datasets.nuscenes.nuscenes_dataframe_utils import (
    CameraName,
    NuScenesDataConfig,
    NuScenesDataFrame,
    _make_rig,
    aggregate_lidar_sweeps,
)


class _LanceRecord(NamedTuple):
    sequence_id: str
    timestamp_us: int
    sequence_idx: int
    row_index: int
    is_key_frame: bool


def _default_lance_path(data_root: str, split: str) -> str:
    if not data_root.endswith("/"):
        data_root += "/"
    return data_root + f"{split}.lance/"


def _se3_from_list(values: Sequence[float] | None) -> SE3 | None:
    if values is None:
        return None
    return SE3.from_array(torch.tensor(values, dtype=torch.float32).reshape(4, 4))


def _tensor_from_rows(
    values: Sequence[Sequence[Any]] | bytes | None,
    *,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not values:
        return torch.empty((0, width), dtype=dtype)
    if isinstance(values, (bytes, bytearray)):
        np_dtype = {torch.float32: np.float32, torch.int32: np.int32}[dtype]
        arr = np.frombuffer(values, dtype=np_dtype).reshape(-1, width)
        return torch.from_numpy(arr.copy())
    return torch.tensor(values, dtype=dtype)


def _tensor_from_vector(
    values: Sequence[Any] | bytes | None, dtype: torch.dtype
) -> torch.Tensor | None:
    if values is None:
        return None
    if isinstance(values, (bytes, bytearray)):
        np_dtype = {torch.float32: np.float32, torch.int32: np.int32}[dtype]
        arr = np.frombuffer(values, dtype=np_dtype)
        return torch.from_numpy(arr.copy())
    return torch.tensor(values, dtype=dtype)


def _pointcloud_from_dict(values: dict[str, Any] | None) -> PointCloud | None:
    if values is None:
        return None
    raw_colors = values.get("colors")
    colors = (
        _tensor_from_rows(raw_colors, width=3, dtype=torch.int32)
        if raw_colors
        else None
    )
    return PointCloud(
        points=_tensor_from_rows(values.get("points"), width=3, dtype=torch.float32),
        intensity=_tensor_from_vector(values.get("intensity"), torch.int32),
        colors=colors,
        classes=_tensor_from_vector(values.get("classes"), torch.int32),
        ego_se3_sensor=_se3_from_list(values.get("ego_se3_sensor")),
        caption=values.get("caption"),
        timestamp_ns=values.get("timestamp_ns"),
    )


def _decode_camera_image(image_bytes: bytes) -> np.ndarray:
    encoded_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode camera image bytes from Lance row.")
    return image


def _camera_image_from_dict(values: dict[str, Any] | None) -> CameraImage | None:
    if values is None:
        return None
    img_bytes = values.get("img_bytes")
    if img_bytes is None:
        return None

    intrinsics_values = values.get("intrinsics")
    intrinsics = None
    if intrinsics_values is not None:
        intrinsics = torch.tensor(intrinsics_values, dtype=torch.float32).reshape(3, 3)

    return CameraImage.from_cv2_img(
        _decode_camera_image(img_bytes),
        ego_se3_camera=_se3_from_list(values.get("ego_se3_camera")),
        intrinsics=intrinsics,
        timestamp_ns=values.get("timestamp_ns"),
    )


def _bounding_boxes_from_rows(
    values: Sequence[Sequence[float]] | None,
) -> torch.Tensor | None:
    if values is None:
        return None
    return _tensor_from_rows(values, width=9, dtype=torch.float32)


class NuScenesLanceDataset(torch.utils.data.Dataset[NuScenesDataFrame]):
    """NuScenes dataset backed by an exported Lance table.

    The Lance dataset stores synchronized per-frame payloads, so this loader
    only needs to build a row index and reconstruct Jormungand data structures on
    demand.
    """

    def __init__(
        self,
        data_config: NuScenesDataConfig,
        version: str,
        data_root: str | Path = "/data/nuscenes",
        *,
        lance_path: str | Path | None = None,
        scene_names: Sequence[str] | None = None,
        key_frames_only: bool = True,
    ) -> None:
        self.data_config = data_config
        self.version = version
        self.data_root = str(data_root)
        if lance_path is not None:
            self.lance_path = str(lance_path)
        else:
            self.lance_path = _default_lance_path(self.data_root, version)

        print(f"Loading NuScenes Lance dataset from '{self.lance_path}'...")
        self.scene_names = set(scene_names) if scene_names is not None else None
        self.key_frames_only = key_frames_only

        if (
            not self.lance_path.startswith("s3://")
            and not Path(self.lance_path).exists()
        ):
            raise ValueError(
                f"No NuScenes Lance dataset was found for version '{self.version}' at '{self.lance_path}'."
            )

        self._lance_dataset = lance.dataset(self.lance_path)
        self._lance_pid = os.getpid()
        # _all_records: full-frequency index for adjacency navigation (prev/next frame)
        # _records: the subset exposed to the dataloader (__len__/__getitem__)
        self._all_records = self._build_full_index()
        if self.key_frames_only:
            self._records = [r for r in self._all_records if r.is_key_frame]
        else:
            self._records = self._all_records
        # Map from all_records row_index → position in _all_records
        self._all_records_pos = {
            r.row_index: i for i, r in enumerate(self._all_records)
        }
        self._index_by_key = {
            (record.sequence_id, record.timestamp_us): idx
            for idx, record in enumerate(self._records)
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_lance_dataset"] = None
        state["_lance_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _ensure_lance_dataset(self) -> None:
        current_pid = os.getpid()
        if (
            self._lance_dataset is None
            or self._lance_pid is None
            or self._lance_pid != current_pid
        ):
            self._lance_dataset = lance.dataset(self.lance_path)
            self._lance_pid = current_pid

    def _build_full_index(self) -> list[_LanceRecord]:
        self._ensure_lance_dataset()
        metadata_rows = self._lance_dataset.to_table(
            columns=["sequence_id", "sequence_idx", "timestamp", "is_key_frame"]
        ).to_pylist()

        records: list[_LanceRecord] = []
        for row_index, row in enumerate(metadata_rows):
            sequence_id = str(row["sequence_id"])
            if self.scene_names is not None and sequence_id not in self.scene_names:
                continue
            records.append(
                _LanceRecord(
                    sequence_id=sequence_id,
                    timestamp_us=int(row["timestamp"]),
                    sequence_idx=int(row["sequence_idx"]),
                    row_index=row_index,
                    is_key_frame=bool(row.get("is_key_frame", True)),
                )
            )

        if not records:
            raise ValueError(
                f"No NuScenes Lance frames found for version '{self.version}' "
                f"under '{self.lance_path}'."
            )
        return records

    def _selected_columns(self) -> list[str]:
        columns = ["sequence_id", "sequence_idx", "timestamp", "global_se3_ego"]
        if self.data_config.lidar:
            columns.append("lidar_rig")
        if self.data_config.requested_camera_names:
            columns.append("camera_rig")
        if self.data_config.bounding_boxes_3d:
            columns.extend(["bounding_boxes_3d", "bbox_classes", "bbox_tracking_ids"])
        return columns

    @staticmethod
    def _sweep_columns() -> list[str]:
        """Minimal columns needed for sweep aggregation (no cameras)."""
        return [
            "sequence_id",
            "sequence_idx",
            "timestamp",
            "global_se3_ego",
            "lidar_rig",
            "bounding_boxes_3d",
            "bbox_classes",
            "bbox_tracking_ids",
        ]

    def _load_row(self, row_index: int) -> dict[str, Any]:
        self._ensure_lance_dataset()
        return self._lance_dataset.take(
            [row_index], columns=self._selected_columns()
        ).to_pylist()[0]

    def _load_lidar_rig(self, row: dict[str, Any]) -> NamedTuple | None:
        if not self.data_config.lidar:
            return None
        lidar_rig = row.get("lidar_rig")
        if lidar_rig is None:
            return None
        pointclouds = {
            sensor_name: pointcloud
            for sensor_name, pointcloud in (
                (name, _pointcloud_from_dict(vals)) for name, vals in lidar_rig.items()
            )
            if pointcloud is not None
        }
        return _make_rig("LidarRig", pointclouds)

    def _load_camera_rig(self, row: dict[str, Any]) -> NamedTuple | None:
        if not self.data_config.requested_camera_names:
            return None
        camera_rig = row.get("camera_rig")
        if camera_rig is None:
            return None
        camera_images = {
            camera_name.value: camera_image
            for camera_name, camera_image in (
                (cn, _camera_image_from_dict(camera_rig.get(cn.value)))
                for cn in self.data_config.requested_camera_names
            )
            if camera_image is not None
        }
        return _make_rig("CameraRig", camera_images)

    def _load_bounding_boxes(
        self, row: dict[str, Any]
    ) -> tuple[torch.Tensor | None, list[str] | None, list[str] | None]:
        if not self.data_config.bounding_boxes_3d:
            return None, None, None
        return (
            _bounding_boxes_from_rows(row.get("bounding_boxes_3d")),
            list(row.get("bbox_classes") or []),
            list(row.get("bbox_tracking_ids") or []),
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> NuScenesDataFrame:
        record = self._records[index]
        row = self._load_row(record.row_index)
        bounding_boxes_3d, bbox_classes, bbox_tracking_ids = self._load_bounding_boxes(
            row
        )

        frame = NuScenesDataFrame(
            sequence_id=record.sequence_id,
            sequence_idx=record.sequence_idx,
            timestamp=record.timestamp_us,
            global_se3_ego=_se3_from_list(row.get("global_se3_ego")),
            lidar_rig=self._load_lidar_rig(row),
            camera_rig=self._load_camera_rig(row),
            bounding_boxes_3d=bounding_boxes_3d,
            bbox_classes=bbox_classes,
            bbox_tracking_ids=bbox_tracking_ids,
            is_key_frame=record.is_key_frame,
        )
        # Attach using position in _all_records so prev/next navigate at full frequency
        all_pos = self._all_records_pos[record.row_index]
        return frame._attach_dataset(cast(Any, self), all_pos)

    def get_frame(self, sequence_id: str, timestamp_us: int) -> NuScenesDataFrame:
        key = (sequence_id, timestamp_us)
        if key not in self._index_by_key:
            raise KeyError(
                f"No NuScenes Lance frame for sequence '{sequence_id}', "
                f"timestamp {timestamp_us}."
            )
        return self[self._index_by_key[key]]

    def get_adjacent_frame(
        self, dataset_index: int, step: int
    ) -> NuScenesDataFrame | None:
        """Navigate adjacency at full frequency (20Hz) via _all_records."""
        adjacent_index = dataset_index + step
        if adjacent_index < 0 or adjacent_index >= len(self._all_records):
            return None
        if (
            self._all_records[dataset_index].sequence_id
            != self._all_records[adjacent_index].sequence_id
        ):
            return None
        record = self._all_records[adjacent_index]
        row = self._load_row(record.row_index)
        bounding_boxes_3d, bbox_classes, bbox_tracking_ids = self._load_bounding_boxes(
            row
        )
        frame = NuScenesDataFrame(
            sequence_id=record.sequence_id,
            sequence_idx=record.sequence_idx,
            timestamp=record.timestamp_us,
            global_se3_ego=_se3_from_list(row.get("global_se3_ego")),
            lidar_rig=self._load_lidar_rig(row),
            camera_rig=self._load_camera_rig(row),
            bounding_boxes_3d=bounding_boxes_3d,
            bbox_classes=bbox_classes,
            bbox_tracking_ids=bbox_tracking_ids,
            is_key_frame=record.is_key_frame,
        )
        return frame._attach_dataset(cast(Any, self), adjacent_index)

    def get_prev_sweep_frames(
        self, dataset_index: int, num_prev: int
    ) -> list[NuScenesDataFrame]:
        """Batch-load up to num_prev previous frames in one Lance read.

        Only loads sweep-relevant columns (lidar + pose + bboxes, no cameras).
        Returns frames in reverse chronological order (most recent first).
        """
        # Collect valid adjacent record indices
        records_to_load: list[tuple[int, Any]] = []  # (all_records_pos, record)
        seq_id = self._all_records[dataset_index].sequence_id
        for step in range(1, num_prev + 1):
            adj_idx = dataset_index - step
            if adj_idx < 0:
                break
            rec = self._all_records[adj_idx]
            if rec.sequence_id != seq_id:
                break
            records_to_load.append((adj_idx, rec))

        if not records_to_load:
            return []

        # Batch read from Lance
        self._ensure_lance_dataset()
        row_indices = [rec.row_index for _, rec in records_to_load]
        rows = self._lance_dataset.take(
            row_indices, columns=self._sweep_columns()
        ).to_pylist()

        # Construct frames
        frames: list[NuScenesDataFrame] = []
        for (adj_idx, record), row in zip(records_to_load, rows):
            bboxes = _bounding_boxes_from_rows(row.get("bounding_boxes_3d"))
            frame = NuScenesDataFrame(
                sequence_id=record.sequence_id,
                sequence_idx=record.sequence_idx,
                timestamp=record.timestamp_us,
                global_se3_ego=_se3_from_list(row.get("global_se3_ego")),
                lidar_rig=self._load_lidar_rig(row),
                camera_rig=None,
                bounding_boxes_3d=bboxes,
                bbox_classes=list(row.get("bbox_classes") or []),
                bbox_tracking_ids=list(row.get("bbox_tracking_ids") or []),
                is_key_frame=record.is_key_frame,
            )
            frame._attach_dataset(cast(Any, self), adj_idx)
            frames.append(frame)

        return frames


def _frame_summary(frame: NuScenesDataFrame) -> str:
    lidar_points = 0
    if frame.lidar_rig is not None:
        lidar_points = len(frame.lidar_rig.LIDAR_TOP)

    camera_names: list[str] = []
    if frame.camera_rig is not None:
        camera_names = list(frame.camera_rig._asdict().keys())

    num_boxes = 0
    if frame.bounding_boxes_3d is not None:
        num_boxes = int(frame.bounding_boxes_3d.shape[0])

    return (
        f"sequence_id={frame.sequence_id} sequence_idx={frame.sequence_idx} "
        f"timestamp={frame.timestamp} is_key_frame={frame.is_key_frame} "
        f"lidar_points={lidar_points} "
        f"cameras={camera_names} boxes={num_boxes}"
    )


@click.command()
@click.option("--version", default="v1.0-trainval", show_default=True)
@click.option(
    "--data-root",
    type=str,
    default="/data/nuscenes",
    show_default=True,
)
@click.option("--index", type=int, default=0, show_default=True)
@click.option(
    "--camera",
    type=click.Choice([cn.value for cn in CameraName]),
    multiple=True,
    help="Camera to include. Repeat to load a subset.",
)
@click.option("--scene-name", multiple=True, help="Restrict to scene names.")
@click.option("--no-lidar", is_flag=True)
@click.option("--no-bounding-boxes", is_flag=True)
@click.option(
    "--sweeps",
    type=int,
    default=0,
    show_default=True,
    help="Number of previous LiDAR sweeps to aggregate (0=disabled).",
)
@click.option("--num-frames", type=int, default=None)
@click.option("--fps", type=float, default=2.0, show_default=True)
@click.option("--colorize-lidar", is_flag=True)
@click.option("--no-frustums", is_flag=True)
@click.option("--frustum-scale", type=float, default=0.5, show_default=True)
def main(
    version: str,
    data_root: str,
    index: int,
    camera: tuple[str, ...],
    scene_name: tuple[str, ...],
    no_lidar: bool,
    no_bounding_boxes: bool,
    sweeps: int,
    num_frames: int | None,
    fps: float,
    colorize_lidar: bool,
    no_frustums: bool,
    frustum_scale: float,
) -> None:
    """Read and visualize NuScenes data from a Lance dataset."""
    requested_cameras = list(camera) if camera else None
    config = (
        NuScenesDataConfig.from_requested_cameras(
            requested_cameras,
            lidar=not no_lidar,
            bounding_boxes_3d=not no_bounding_boxes,
        )
        if requested_cameras is not None
        else NuScenesDataConfig(
            lidar=not no_lidar,
            bounding_boxes_3d=not no_bounding_boxes,
        )
    )

    dataset = NuScenesLanceDataset(
        data_config=config,
        version=version,
        data_root=data_root,
        key_frames_only=False,
        scene_names=scene_name or None,
    )
    click.echo(f"dataset_size={len(dataset)}")
    frame = dataset[index]
    if sweeps > 0:
        frame = aggregate_lidar_sweeps(frame, num_sweeps=sweeps)
    click.echo(_frame_summary(frame))

    renderer = SequentialFrameRenderer(category_colors=NUSCENES_COLORS)

    if num_frames is not None:
        # Collect frames manually so we can apply sweep aggregation to each
        frames = [frame]
        cursor = frame
        for i in range(num_frames - 1):
            cursor = cursor.next_frame()
            if cursor is None:
                break
            if sweeps > 0:
                cursor = aggregate_lidar_sweeps(cursor, num_sweeps=sweeps)
            frames.append(cursor)
            click.echo(f"  loaded frame {len(frames)}")

        renderer.render(
            frames,
            server_label=f"NuScenes Lance seq @ {frame.sequence_idx} ({len(frames)} frames)",
            colorize_lidar_from_images=colorize_lidar,
            show_camera_frustums=not no_frustums,
            frustum_scale=frustum_scale,
            fps=fps,
        )
    else:
        renderer.render(
            frame,
            server_label=f"NuScenes Lance: [{frame.sequence_idx}]",
            colorize_lidar_from_images=colorize_lidar,
            show_camera_frustums=not no_frustums,
            frustum_scale=frustum_scale,
        )


if __name__ == "__main__":
    main()
