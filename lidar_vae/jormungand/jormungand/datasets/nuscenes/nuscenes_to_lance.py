"""Export NuScenes dataset to Lance format at full sensor frequency.

Each row corresponds to one LiDAR scan (keyframes at 2Hz + sweeps at ~20Hz).
For each LiDAR timestamp, the nearest camera capture (by timestamp) is matched
per camera channel.  Bounding-box annotations are stored for ALL rows:
keyframe rows use direct annotations; non-keyframe (sweep) rows use the
NuScenes ``get_boxes()`` API which linearly interpolates box centers and
slerps orientations between adjacent keyframe annotations.

Usage::

    python -m jormungand.datasets.nuscenes.nuscenes_to_lance \
        --version v1.0-trainval --data-root /data/nuscenes \
        --output-path /data/nuscenes/v1.0-trainval.lance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import cv2
import lance
import numpy as np
import pyarrow as pa
import torch
from pyquaternion import Quaternion

from jormungand.datasets.nuscenes.nuscenes_local import _NuScenes
from jormungand.datastructures.se3 import SE3
from jormungand.datasets.nuscenes.nuscenes_dataframe_utils import CameraName

# --------------------------------------------------------------------------- #
# Arrow type constants
# --------------------------------------------------------------------------- #

_SE3_TYPE = pa.list_(pa.float32(), 16)
_VEC3_FLOAT32_TYPE = pa.list_(pa.float32(), 3)
_VEC3_INT32_TYPE = pa.list_(pa.int32(), 3)
_INTRINSICS_TYPE = pa.list_(pa.float32(), 9)
_BOUNDING_BOX_TYPE = pa.list_(pa.float32(), 9)


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #


def _se3_to_list(se3: SE3 | None) -> list[float] | None:
    if se3 is None:
        return None
    return se3.to_array().detach().cpu().to(torch.float32).reshape(-1).tolist()


def _nuscenes_quat_to_se3(
    rotation: list[float],
    translation: list[float],
) -> SE3:
    q = Quaternion(rotation)
    return SE3(
        rotation_matrix=torch.tensor(q.rotation_matrix, dtype=torch.float32),
        translation=torch.tensor(translation, dtype=torch.float32),
    )


def _se3_to_xyz_yaw_pitch_roll(se3: SE3) -> torch.Tensor:
    rot = np.asarray(se3.rotation_matrix.detach().cpu().numpy(), dtype=np.float64)
    U, _, Vt = np.linalg.svd(rot)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    q = Quaternion(matrix=R)
    yaw, pitch, roll = q.yaw_pitch_roll
    tx, ty, tz = se3.translation.detach().cpu().numpy()
    return torch.tensor([tx, ty, tz, yaw, pitch, roll], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def _pointcloud_struct_type() -> pa.StructType:
    return pa.struct(
        [
            pa.field("points", pa.binary()),
            pa.field("intensity", pa.binary()),
            pa.field("colors", pa.list_(_VEC3_INT32_TYPE)),
            pa.field("classes", pa.list_(pa.int32())),
            pa.field("ego_se3_sensor", _SE3_TYPE),
            pa.field("caption", pa.string()),
            pa.field("timestamp_ns", pa.int64()),
        ]
    )


def _camera_struct_type() -> pa.StructType:
    return pa.struct(
        [
            pa.field("img_bytes", pa.binary()),
            pa.field("image_path", pa.string()),
            pa.field("ego_se3_camera", _SE3_TYPE),
            pa.field("intrinsics", _INTRINSICS_TYPE),
            pa.field("caption", pa.string()),
            pa.field("timestamp_ns", pa.int64()),
        ]
    )


def _build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sequence_id", pa.string()),
            pa.field("sequence_idx", pa.int64()),
            pa.field("timestamp", pa.int64()),
            pa.field("is_key_frame", pa.bool_()),
            pa.field("global_se3_ego", _SE3_TYPE),
            pa.field(
                "lidar_rig",
                pa.struct(
                    [pa.field("LIDAR_TOP", _pointcloud_struct_type())],
                ),
            ),
            pa.field(
                "camera_rig",
                pa.struct(
                    [
                        pa.field(camera_name.value, _camera_struct_type())
                        for camera_name in CameraName
                    ]
                ),
            ),
            pa.field("bounding_boxes_3d", pa.list_(_BOUNDING_BOX_TYPE)),
            pa.field("bbox_classes", pa.list_(pa.string())),
            pa.field("bbox_tracking_ids", pa.list_(pa.string())),
        ]
    )


# --------------------------------------------------------------------------- #
# Exporter
# --------------------------------------------------------------------------- #


class NuScenesLanceExporter:
    """Export NuScenes at full LiDAR frequency with nearest-camera matching."""

    def __init__(
        self,
        version: str,
        data_root: Path,
        output_path: Path,
        *,
        scene_names: tuple[str, ...] | None,
        batch_size: int,
        overwrite: bool,
    ) -> None:
        self.version = version
        self.data_root = data_root
        self.output_path = output_path
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.schema = _build_schema()

        self._nusc = _NuScenes(version=version, dataroot=str(data_root), verbose=False)
        self.scene_names = set(scene_names) if scene_names else None

    # ------------------------------------------------------------------ #
    # Index building
    # ------------------------------------------------------------------ #

    def _build_lidar_records(self) -> list[dict[str, Any]]:
        """Build a flat list of all LiDAR sample_data records, ordered by scene then timestamp.

        Also populates ``self._scenes_with_lidar`` so the camera index can be
        filtered to only scenes that have lidar data on disk.
        """
        records: list[dict[str, Any]] = []
        self._scenes_with_lidar: set[str] = set()

        for scene in self._nusc.scene:
            if self.scene_names and scene["name"] not in self.scene_names:
                continue

            scene_token = scene["token"]

            # Find the first LIDAR_TOP sample_data for this scene.
            first_sample = self._nusc.get("sample", scene["first_sample_token"])
            lidar_sd_token = first_sample["data"].get("LIDAR_TOP")
            if lidar_sd_token is None:
                continue

            # Walk backward to the very first LIDAR_TOP sample_data in this scene.
            lidar_sd = self._nusc.get("sample_data", lidar_sd_token)
            while lidar_sd["prev"]:
                lidar_sd = self._nusc.get("sample_data", lidar_sd["prev"])

            # Now walk forward through ALL lidar sample_data records.
            sequence_idx = 0
            scene_has_lidar = False
            while True:
                lidar_path = self.data_root / lidar_sd["filename"]
                if lidar_path.exists():
                    records.append(
                        {
                            "scene_token": scene_token,
                            "lidar_sd_token": lidar_sd["token"],
                            "timestamp": lidar_sd["timestamp"],
                            "is_key_frame": lidar_sd["is_key_frame"],
                            "sample_token": lidar_sd.get("sample_token"),
                            "sequence_idx": sequence_idx,
                        }
                    )
                    sequence_idx += 1
                    scene_has_lidar = True

                if not lidar_sd["next"]:
                    break
                lidar_sd = self._nusc.get("sample_data", lidar_sd["next"])

            if scene_has_lidar:
                self._scenes_with_lidar.add(scene_token)

        return records

    def _build_camera_index(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Build per-scene, per-camera-channel lists of sample_data sorted by timestamp.

        Only includes scenes that have lidar data on disk (populated by
        ``_build_lidar_records``).

        Returns: {camera_name: {scene_token: [sorted list of sample_data dicts]}}
        """
        camera_index: dict[str, dict[str, list[dict[str, Any]]]] = {
            cam.value: {} for cam in CameraName
        }

        for scene in self._nusc.scene:
            if self.scene_names and scene["name"] not in self.scene_names:
                continue
            if scene["token"] not in self._scenes_with_lidar:
                continue

            scene_token = scene["token"]
            first_sample = self._nusc.get("sample", scene["first_sample_token"])

            for cam_name in CameraName:
                cam_sd_token = first_sample["data"].get(cam_name.value)
                if cam_sd_token is None:
                    continue

                # Walk backward to first sample_data for this camera.
                cam_sd = self._nusc.get("sample_data", cam_sd_token)
                while cam_sd["prev"]:
                    cam_sd = self._nusc.get("sample_data", cam_sd["prev"])

                # Walk forward collecting all records for this scene.
                scene_records: list[dict[str, Any]] = []
                while True:
                    scene_records.append(cam_sd)
                    if not cam_sd["next"]:
                        break
                    cam_sd = self._nusc.get("sample_data", cam_sd["next"])

                camera_index[cam_name.value][scene_token] = scene_records

        return camera_index

    def _find_nearest_camera(
        self,
        camera_records: list[dict[str, Any]],
        target_timestamp: int,
    ) -> dict[str, Any] | None:
        """Find nearest camera sample_data by timestamp using binary search.

        camera_records must be sorted by timestamp (guaranteed since they come
        from a single scene's linked-list walk).
        """
        if not camera_records:
            return None

        lo, hi = 0, len(camera_records) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if camera_records[mid]["timestamp"] < target_timestamp:
                lo = mid + 1
            else:
                hi = mid

        best = lo
        if lo > 0:
            diff_lo = abs(camera_records[lo]["timestamp"] - target_timestamp)
            diff_prev = abs(camera_records[lo - 1]["timestamp"] - target_timestamp)
            if diff_prev < diff_lo:
                best = lo - 1

        # Reject if the time gap is too large (>100ms = 100_000 μs).
        if abs(camera_records[best]["timestamp"] - target_timestamp) > 100_000:
            return None

        return camera_records[best]

    # ------------------------------------------------------------------ #
    # Row serialization
    # ------------------------------------------------------------------ #

    def _load_and_serialize_lidar(
        self, lidar_sd: dict[str, Any]
    ) -> dict[str, Any] | None:
        lidar_path = self.data_root / lidar_sd["filename"]
        if not lidar_path.exists():
            return None

        scan = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
        points_sensor = torch.tensor(scan[:, :3], dtype=torch.float32)
        intensity = torch.tensor(scan[:, 3] * 255, dtype=torch.int32).clamp(0, 255)

        # Transform to ego frame.
        calib = self._nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_se3_sensor = _nuscenes_quat_to_se3(calib["rotation"], calib["translation"])
        points_ego = ego_se3_sensor.transform_points(points_sensor)

        return {
            "LIDAR_TOP": {
                "points": points_ego.numpy().astype(np.float32).tobytes(),
                "intensity": intensity.numpy().astype(np.int32).tobytes(),
                "colors": None,
                "classes": None,
                "ego_se3_sensor": None,
                "caption": "LIDAR_TOP",
                "timestamp_ns": lidar_sd["timestamp"],
            }
        }

    def _load_and_serialize_camera(
        self, cam_sd: dict[str, Any]
    ) -> dict[str, Any] | None:
        image_path = self.data_root / cam_sd["filename"]
        if not image_path.exists():
            return None

        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            return None

        _, img_encoded = cv2.imencode(".jpg", bgr_image)
        img_bytes = img_encoded.tobytes()

        calib = self._nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        ego_se3_camera = _nuscenes_quat_to_se3(calib["rotation"], calib["translation"])
        intrinsics_raw = calib["camera_intrinsic"]
        intrinsics_list = (
            torch.tensor(intrinsics_raw, dtype=torch.float32).reshape(-1).tolist()
            if intrinsics_raw
            else None
        )

        return {
            "img_bytes": img_bytes,
            "image_path": cam_sd["filename"],
            "ego_se3_camera": _se3_to_list(ego_se3_camera),
            "intrinsics": intrinsics_list,
            "caption": None,
            "timestamp_ns": cam_sd["timestamp"],
        }

    def _load_bounding_boxes(
        self, sample_token: str, global_se3_ego: SE3
    ) -> tuple[list[list[float]] | None, list[str] | None, list[str] | None]:
        sample = self._nusc.get("sample", sample_token)
        ann_tokens: list[str] = sample["anns"]
        if not ann_tokens:
            return [], [], []

        ego_se3_global = global_se3_ego.inverse()

        bounding_boxes: list[list[float]] = []
        bbox_classes: list[str] = []
        bbox_tracking_ids: list[str] = []

        for ann_token in ann_tokens:
            ann = self._nusc.get("sample_annotation", ann_token)
            box_w, box_l, box_h = ann["size"]
            global_se3_box = _nuscenes_quat_to_se3(ann["rotation"], ann["translation"])
            ego_se3_box = ego_se3_global @ global_se3_box
            xyz_ypr = _se3_to_xyz_yaw_pitch_roll(ego_se3_box)
            row = [box_l, box_w, box_h] + xyz_ypr.tolist()
            bounding_boxes.append(row)
            bbox_classes.append(ann["category_name"])
            bbox_tracking_ids.append(ann["instance_token"])

        return bounding_boxes, bbox_classes, bbox_tracking_ids

    def _load_interpolated_bounding_boxes(
        self, lidar_sd_token: str, global_se3_ego: SE3
    ) -> tuple[list[list[float]] | None, list[str] | None, list[str] | None]:
        """Load bounding boxes for a non-keyframe using the NuScenes get_boxes() API.

        get_boxes() linearly interpolates box centers and slerps orientations
        between adjacent keyframe annotations at the sweep's timestamp.
        """
        boxes = self._nusc.get_boxes(lidar_sd_token)
        if not boxes:
            return [], [], []

        ego_se3_global = global_se3_ego.inverse()

        bounding_boxes: list[list[float]] = []
        bbox_classes: list[str] = []
        bbox_tracking_ids: list[str] = []

        for box in boxes:
            # box.center is in global frame, box.orientation is a Quaternion
            global_se3_box = _nuscenes_quat_to_se3(
                box.orientation.elements.tolist(), box.center.tolist()
            )
            ego_se3_box = ego_se3_global @ global_se3_box
            xyz_ypr = _se3_to_xyz_yaw_pitch_roll(ego_se3_box)
            # box.wlh is [width, length, height]
            box_w, box_l, box_h = box.wlh
            row = [box_l, box_w, box_h] + xyz_ypr.tolist()
            bounding_boxes.append(row)
            bbox_classes.append(box.name)
            # Look up instance_token from the annotation record
            ann = self._nusc.get("sample_annotation", box.token)
            bbox_tracking_ids.append(ann["instance_token"])

        return bounding_boxes, bbox_classes, bbox_tracking_ids

    def _serialize_row(
        self,
        record: dict[str, Any],
        camera_index: dict[str, dict[str, list[dict[str, Any]]]],
    ) -> dict[str, Any]:
        lidar_sd = self._nusc.get("sample_data", record["lidar_sd_token"])
        timestamp = record["timestamp"]
        scene_token = record["scene_token"]

        # Ego pose at this LiDAR timestamp.
        ego_pose = self._nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        global_se3_ego = _nuscenes_quat_to_se3(
            ego_pose["rotation"], ego_pose["translation"]
        )

        # LiDAR.
        lidar_rig = self._load_and_serialize_lidar(lidar_sd)

        # Camera rig: find nearest camera for each channel within same scene.
        camera_rig: dict[str, Any] = {}
        for cam_name in CameraName:
            scene_cam_records = camera_index[cam_name.value].get(scene_token, [])
            nearest_sd = self._find_nearest_camera(scene_cam_records, timestamp)
            if nearest_sd is not None:
                camera_rig[cam_name.value] = self._load_and_serialize_camera(nearest_sd)
            else:
                camera_rig[cam_name.value] = None

        # Bounding boxes: keyframes use direct annotations, non-keyframes use
        # the NuScenes get_boxes() API which interpolates between keyframes.
        bounding_boxes_3d = None
        bbox_classes = None
        bbox_tracking_ids = None
        if record["is_key_frame"] and record["sample_token"]:
            bounding_boxes_3d, bbox_classes, bbox_tracking_ids = (
                self._load_bounding_boxes(record["sample_token"], global_se3_ego)
            )
        elif not record["is_key_frame"]:
            bounding_boxes_3d, bbox_classes, bbox_tracking_ids = (
                self._load_interpolated_bounding_boxes(
                    record["lidar_sd_token"], global_se3_ego
                )
            )

        return {
            "sequence_id": record["scene_token"],
            "sequence_idx": record["sequence_idx"],
            "timestamp": timestamp,
            "is_key_frame": record["is_key_frame"],
            "global_se3_ego": _se3_to_list(global_se3_ego),
            "lidar_rig": lidar_rig,
            "camera_rig": camera_rig,
            "bounding_boxes_3d": bounding_boxes_3d,
            "bbox_classes": bbox_classes,
            "bbox_tracking_ids": bbox_tracking_ids,
        }

    # ------------------------------------------------------------------ #
    # Main export loop
    # ------------------------------------------------------------------ #

    def export(self) -> dict[str, Any]:
        if self.output_path.exists() and not self.overwrite:
            raise click.ClickException(
                f"Output path already exists: {self.output_path}. Re-run with --overwrite."
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        click.echo("Building LiDAR record index...")
        lidar_records = self._build_lidar_records()
        num_keyframes = sum(1 for r in lidar_records if r["is_key_frame"])
        num_sweeps = len(lidar_records) - num_keyframes
        click.echo(
            f"  {len(lidar_records)} LiDAR scans "
            f"({num_keyframes} keyframes + {num_sweeps} sweeps) "
            f"across {len(self._scenes_with_lidar)} scenes"
        )

        click.echo("Building camera index (filtered to scenes with lidar on disk)...")
        camera_index = self._build_camera_index()
        for cam_name in CameraName:
            total_frames = sum(
                len(recs) for recs in camera_index[cam_name.value].values()
            )
            click.echo(
                f"  {cam_name.value}: {total_frames} frames "
                f"across {len(camera_index[cam_name.value])} scenes"
            )

        rows: list[dict[str, Any]] = []
        written_frames = 0
        write_mode = "overwrite" if self.output_path.exists() else "create"

        with click.progressbar(
            lidar_records,
            label=f"Exporting NuScenes {self.version} to Lance",
        ) as progress_bar:
            for record in progress_bar:
                rows.append(self._serialize_row(record, camera_index))
                written_frames += 1

                if len(rows) >= self.batch_size:
                    table = pa.Table.from_pylist(rows, schema=self.schema)
                    lance.write_dataset(table, self.output_path, mode=write_mode)
                    write_mode = "append"
                    rows = []

        if rows:
            table = pa.Table.from_pylist(rows, schema=self.schema)
            lance.write_dataset(table, self.output_path, mode=write_mode)

        return {
            "version": self.version,
            "output_path": str(self.output_path),
            "written_frames": written_frames,
            "keyframes": sum(1 for r in lidar_records if r["is_key_frame"]),
            "non_keyframes": sum(1 for r in lidar_records if not r["is_key_frame"]),
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_output_path(data_root: Path, version: str) -> Path:
    return data_root / f"{version}.lance"


@click.command()
@click.option(
    "--version",
    default="v1.0-trainval",
    show_default=True,
    help="NuScenes version (e.g. v1.0-trainval, v1.0-mini).",
)
@click.option(
    "--data-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/data/nuscenes"),
    show_default=True,
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit output path. Defaults to {data_root}/{version}.lance.",
)
@click.option("--scene-name", multiple=True, help="Restrict to specific scene names.")
@click.option("--batch-size", default=8, show_default=True, type=int)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
def main(
    version: str,
    data_root: Path,
    output_path: Path | None,
    scene_name: tuple[str, ...],
    batch_size: int,
    overwrite: bool,
) -> None:
    """Export NuScenes dataset to Lance format at full sensor frequency."""
    output_path = output_path or _default_output_path(data_root, version)
    exporter = NuScenesLanceExporter(
        version=version,
        data_root=data_root,
        output_path=output_path,
        scene_names=scene_name or None,
        batch_size=batch_size,
        overwrite=overwrite,
    )
    summary = exporter.export()

    click.echo(f"version={summary['version']}")
    click.echo(f"output_path={summary['output_path']}")
    click.echo(f"written_frames={summary['written_frames']}")
    click.echo(f"keyframes={summary['keyframes']}")
    click.echo(f"non_keyframes={summary['non_keyframes']}")


if __name__ == "__main__":
    main()
