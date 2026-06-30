from __future__ import annotations

from pathlib import Path
from typing import (
    NamedTuple,
    Sequence,
)

import cv2
import numpy as np
import torch
import os.path as osp
import time

from nuscenes.nuscenes import NuScenes as _NuScenesBase, NuScenesExplorer
from nuscenes.utils.color_map import get_colormap

from jormungand.datasets.nuscenes.nuscenes_dataframe_utils import (
    NuScenesDataConfig,
    NuScenesDataFrame,
    _make_rig,
)
from pyquaternion import Quaternion

from jormungand.datastructures.camera_image import CameraImage
from jormungand.datastructures.pointcloud import PointCloud
from jormungand.datastructures.se3 import SE3


class _NuScenes(_NuScenesBase):
    """NuScenes subclass that skips lidarseg/panoptic loading.

    The standard category.json shipped with nuScenes v1.0 lacks the 'index'
    field expected by lidarseg, and the lidarseg label directory may not exist.
    We override ``__init__`` to skip the entire lidarseg/panoptic block.
    """

    def __init__(
        self,
        version: str = "v1.0-trainval",
        dataroot: str = "/data/sets/nuscenes",
        verbose: bool = True,
        map_resolution: float = 0.1,
    ) -> None:
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = [
            "category",
            "attribute",
            "visibility",
            "instance",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "log",
            "scene",
            "sample",
            "sample_data",
            "sample_annotation",
            "map",
        ]

        assert osp.exists(self.table_root), (
            f"Database version not found: {self.table_root}"
        )

        start_time = time.time()
        if verbose:
            print(f"======\nLoading NuScenes tables for version {self.version}...")

        self.category = self.__load_table__("category")
        self.attribute = self.__load_table__("attribute")
        self.visibility = self.__load_table__("visibility")
        self.instance = self.__load_table__("instance")
        self.sensor = self.__load_table__("sensor")
        self.calibrated_sensor = self.__load_table__("calibrated_sensor")
        self.ego_pose = self.__load_table__("ego_pose")
        self.log = self.__load_table__("log")
        self.scene = self.__load_table__("scene")
        self.sample = self.__load_table__("sample")
        self.sample_data = self.__load_table__("sample_data")
        self.sample_annotation = self.__load_table__("sample_annotation")
        self.map = self.__load_table__("map")

        self.colormap = get_colormap()

        # Skip lidarseg / panoptic loading entirely.

        if osp.exists(osp.join(self.table_root, "image_annotations.json")):
            self.image_annotations = self.__load_table__("image_annotations")

        # Skip map mask initialization – we don't use map data.

        if verbose:
            for table in self.table_names:
                print(f"{len(getattr(self, table))} {table},")
            print(f"Done loading in {time.time() - start_time:.3f} seconds.\n======")

        self.__make_reverse_index__(verbose)
        self.explorer = NuScenesExplorer(self)


def _nuscenes_quat_to_se3(
    rotation: list[float],
    translation: list[float],
) -> SE3:
    """Convert NuScenes [w, x, y, z] quaternion + [x, y, z] translation to SE3."""
    q = Quaternion(rotation)
    return SE3(
        rotation_matrix=torch.tensor(q.rotation_matrix, dtype=torch.float32),
        translation=torch.tensor(translation, dtype=torch.float32),
    )


def _se3_to_xyz_yaw_pitch_roll(se3: SE3) -> torch.Tensor:
    """Extract [tx, ty, tz, yaw, pitch, roll] from an SE3, re-orthogonalizing first.

    After composing multiple float32 SE3 transforms, the rotation matrix may
    accumulate small numerical drift that makes pyquaternion reject it. We
    project back to SO(3) via SVD before extracting Euler angles.
    """
    rot = np.asarray(se3.rotation_matrix.detach().cpu().numpy(), dtype=np.float64)
    U, _, Vt = np.linalg.svd(rot)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    q = Quaternion(matrix=R)
    yaw, pitch, roll = q.yaw_pitch_roll
    tx, ty, tz = se3.translation.detach().cpu().numpy()
    return torch.tensor(
        [tx, ty, tz, yaw, pitch, roll], dtype=torch.float32, device=se3.device
    )


class _SampleRecord(NamedTuple):
    scene_token: str
    sample_token: str
    timestamp_us: int
    sequence_idx: int


class NuScenesLocalDataset(
    torch.utils.data.Dataset[NuScenesDataFrame],
):
    """NuScenes dataset backed by the official ``nuscenes-devkit``.

    Directory layout expected::

        <data_root>/
            <version>/          # e.g. v1.0-trainval
                *.json          # metadata tables
            samples/
                LIDAR_TOP/      # keyframe lidar .pcd.bin
                CAM_FRONT/      # keyframe camera .jpg
                ...
            sweeps/             # non-keyframe sensor data (not used here)
    """

    def __init__(
        self,
        data_config: NuScenesDataConfig,
        version: str,
        data_root: str | Path = "/data/nuscenes",
        *,
        scene_names: Sequence[str] | None = None,
    ) -> None:
        self.data_config = data_config
        self.version = version
        self.data_root = Path(data_root)
        self.scene_names = set(scene_names) if scene_names is not None else None

        self._nuscenes = _NuScenes(
            version=version, dataroot=str(self.data_root), verbose=False
        )
        self._records = self._build_sample_index()

    def _build_sample_index(self) -> list[_SampleRecord]:
        scene_token_set: set[str] | None = None
        if self.scene_names is not None:
            scene_token_set = {
                scene["token"]
                for scene in self._nuscenes.scene
                if scene["name"] in self.scene_names
            }

        records: list[_SampleRecord] = []
        for scene in self._nuscenes.scene:
            if scene_token_set is not None and scene["token"] not in scene_token_set:
                continue

            # Walk the linked list of samples for this scene
            sample_token: str | None = scene["first_sample_token"]
            sequence_idx = 0
            # sometimes there is not lidar file. Skip those samples fromt he dataframe.
            sample = self._nuscenes.get("sample", sample_token)
            lidar_sd_token = sample["data"].get("LIDAR_TOP")
            if lidar_sd_token is None:
                continue
            lidar_sd = self._nuscenes.get("sample_data", lidar_sd_token)
            lidar_path = self.data_root / lidar_sd["filename"]
            if not lidar_path.exists():
                continue
            while sample_token:
                sample = self._nuscenes.get("sample", sample_token)
                records.append(
                    _SampleRecord(
                        scene_token=scene["token"],
                        sample_token=sample_token,
                        timestamp_us=int(sample["timestamp"]),
                        sequence_idx=sequence_idx,
                    )
                )
                sequence_idx += 1
                sample_token = sample["next"] or None

        if not records:
            raise ValueError(
                f"No NuScenes samples with on-disk LIDAR_TOP data found "
                f"for version '{self.version}' under '{self.data_root}'."
            )
        return records

    def _load_global_se3_ego(self, sample_token: str) -> SE3 | None:
        sample = self._nuscenes.get("sample", sample_token)
        lidar_sd_token = sample["data"].get("LIDAR_TOP")
        if lidar_sd_token is None:
            return None
        lidar_sd = self._nuscenes.get("sample_data", lidar_sd_token)
        ego_pose = self._nuscenes.get("ego_pose", lidar_sd["ego_pose_token"])
        return _nuscenes_quat_to_se3(ego_pose["rotation"], ego_pose["translation"])

    def _load_lidar_rig(self, sample_token: str) -> NamedTuple | None:
        if not self.data_config.lidar:
            return None

        sample = self._nuscenes.get("sample", sample_token)
        lidar_sd_token = sample["data"].get("LIDAR_TOP")
        if lidar_sd_token is None:
            return None

        lidar_sd = self._nuscenes.get("sample_data", lidar_sd_token)
        lidar_path = self.data_root / lidar_sd["filename"]
        # NuScenes .pcd.bin: float32, 5 values per point (x, y, z, intensity, ring_index)
        scan = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
        points_sensor = torch.tensor(scan[:, :3], dtype=torch.float32)
        intensity = torch.tensor(scan[:, 3] * 255, dtype=torch.int32).clamp(0, 255)

        # Transform points from sensor frame → ego frame so that they are
        # consistent with Argoverse (which stores lidar already in ego frame).
        calib = self._nuscenes.get(
            "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
        )
        ego_se3_sensor = _nuscenes_quat_to_se3(calib["rotation"], calib["translation"])
        points_ego = ego_se3_sensor.transform_points(points_sensor)

        pointcloud = PointCloud(
            points=points_ego,
            intensity=intensity,
            ego_se3_sensor=SE3.identity(),
            caption="LIDAR_TOP",
        )
        return _make_rig("LidarRig", {"LIDAR_TOP": pointcloud})

    def _load_camera_rig(self, sample_token: str) -> NamedTuple | None:
        if not self.data_config.requested_camera_names:
            return None

        sample = self._nuscenes.get("sample", sample_token)
        camera_images: dict[str, CameraImage] = {}

        for camera_name in self.data_config.requested_camera_names:
            cam_sd_token = sample["data"].get(camera_name.value)
            if cam_sd_token is None:
                continue

            cam_sd = self._nuscenes.get("sample_data", cam_sd_token)
            image_path = self.data_root / cam_sd["filename"]
            if not image_path.exists():
                continue

            bgr_image = cv2.imread(str(image_path))
            if bgr_image is None:
                continue

            calib = self._nuscenes.get(
                "calibrated_sensor", cam_sd["calibrated_sensor_token"]
            )
            ego_se3_camera = _nuscenes_quat_to_se3(
                calib["rotation"], calib["translation"]
            )
            intrinsics_raw = calib["camera_intrinsic"]
            intrinsics = (
                torch.tensor(intrinsics_raw, dtype=torch.float32)
                if intrinsics_raw
                else None
            )

            camera_images[camera_name.value] = CameraImage.from_cv2_img(
                bgr_image,
                ego_se3_camera=ego_se3_camera,
                intrinsics=intrinsics,
            )

        return _make_rig("CameraRig", camera_images)

    def _load_bounding_boxes(
        self,
        sample_token: str,
        global_se3_ego: SE3 | None,
    ) -> tuple[torch.Tensor | None, list[str] | None, list[str] | None]:
        """Load 3D bounding-box annotations for ``sample_token`` in the ego frame.

        NuScenes stores annotations as ``[w, l, h]`` sizes plus a ``[w, x, y, z]``
        quaternion + ``[x, y, z]`` translation in the *global* frame. We build an
        ``SE3`` from those and compose with ``ego_se3_global`` (the inverse of the
        ego pose for the frame) to obtain ``ego_se3_box``. The resulting boxes are
        returned as a ``(N, 9)`` tensor with columns ``[l, w, h, tx, ty, tz, yaw,
        pitch, roll]``, matching :class:`NuScenesDataFrame`.
        """
        if not self.data_config.bounding_boxes_3d:
            return None, None, None

        sample = self._nuscenes.get("sample", sample_token)
        ann_tokens: list[str] = sample["anns"]
        if not ann_tokens:
            return torch.empty((0, 9), dtype=torch.float32), [], []

        ego_se3_global = (
            global_se3_ego.inverse() if global_se3_ego is not None else None
        )

        bounding_boxes: list[torch.Tensor] = []
        bbox_classes: list[str] = []
        bbox_tracking_ids: list[str] = []

        for ann_token in ann_tokens:
            ann = self._nuscenes.get("sample_annotation", ann_token)
            # NuScenes annotation size order is [width, length, height].
            box_w, box_l, box_h = ann["size"]

            # Build the box pose in the global frame, then transform to ego.
            # Reuse ``_nuscenes_quat_to_se3`` so SE3 dtype (float32)
            # matches ``global_se3_ego`` and the matmul below does not error.
            global_se3_box = _nuscenes_quat_to_se3(ann["rotation"], ann["translation"])
            ego_se3_box = (
                ego_se3_global @ global_se3_box
                if ego_se3_global is not None
                else global_se3_box
            )

            bounding_boxes.append(
                torch.cat(
                    (
                        torch.tensor([box_l, box_w, box_h], dtype=torch.float32),
                        _se3_to_xyz_yaw_pitch_roll(ego_se3_box),
                    )
                )
            )
            bbox_classes.append(ann["category_name"])
            bbox_tracking_ids.append(ann["instance_token"])

        return torch.stack(bounding_boxes, dim=0), bbox_classes, bbox_tracking_ids

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> NuScenesDataFrame:
        record = self._records[index]
        global_se3_ego = self._load_global_se3_ego(record.sample_token)
        bounding_boxes_3d, bbox_classes, bbox_tracking_ids = self._load_bounding_boxes(
            record.sample_token, global_se3_ego
        )
        frame = NuScenesDataFrame(
            sequence_id=record.scene_token,
            sequence_idx=record.sequence_idx,
            timestamp=record.timestamp_us,
            global_se3_ego=global_se3_ego,
            lidar_rig=self._load_lidar_rig(record.sample_token),
            camera_rig=self._load_camera_rig(record.sample_token),
            bounding_boxes_3d=bounding_boxes_3d,
            bbox_classes=bbox_classes,
            bbox_tracking_ids=bbox_tracking_ids,
        )
        return frame._attach_dataset(self, index)

    def get_adjacent_frame(
        self, dataset_index: int, step: int
    ) -> NuScenesDataFrame | None:
        adjacent_index = dataset_index + step
        if adjacent_index < 0 or adjacent_index >= len(self._records):
            return None
        if (
            self._records[dataset_index].scene_token
            != self._records[adjacent_index].scene_token
        ):
            return None
        return self[adjacent_index]
