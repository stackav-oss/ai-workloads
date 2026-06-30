from __future__ import annotations

from typing import NamedTuple
import torch
from jormungand.datastructures.se3 import SE3
from jormungand.datastructures.camera_image import CameraImage
from jormungand.datastructures.pointcloud import PointCloud
from abc import ABC, abstractmethod


class SequentialDataFrame(ABC):
    def __init__(
        self,
        sequence_id: str,
        sequence_idx: int,
        timestamp: int,
        global_se3_ego: SE3 | None = None,
        lidar_rig: NamedTuple[str, PointCloud] | None = None,
        camera_rig: NamedTuple[str, CameraImage] | None = None,
    ) -> None:
        """A SequentialDataFrame is a container for all the data associated with a single timestamp in a sequence.
        It can be used to store point clouds, images, and other sensor data, as well as the SE3 transformations this frame and some canonical "global" frame for the sequence or dataset.

        Args:
            sequence_id: A string identifier for the sequence this frame belongs to.
            sequence_idx: An integer index for this frame within the sequence, starting from 0.
            timestamp: An integer timestamp for this frame, typically in microseconds, but the unit is not important as long as it is consistent across frames in the sequence. If no timestamp information is available, this can be set to the sequence_idx.
            global_se3_ego: An SE3 transformation from the ego frame to a global frame, if available. This can be used to transform all data in this frame to the global frame. If the ego does not move this may be the identity matrix.
            lidar_rig: An optional NamedTuple containing PointClouds from the lidar sensors in this frame. The keys of the NamedTuple should be the sensor names, and the values should be PointCloud instances.
            camera_rig: An optional NamedTuple containing CameraImages from the camera sensors in this frame. The keys of the NamedTuple should be the sensor names, and the values should be CameraImage instances.
        """
        self.sequence_id: str = sequence_id
        self.sequence_idx: int = sequence_idx
        self.timestamp: int = timestamp
        self.global_se3_ego: SE3 | None = global_se3_ego
        self.lidar_rig: NamedTuple[str, PointCloud] | None = lidar_rig
        self.camera_rig: NamedTuple[str, CameraImage] | None = camera_rig

    def to(self, device: torch.device) -> "SequentialDataFrame":
        """Move all data in the frame to the specified device."""
        global_se3_ego = self.global_se3_ego.to(device) if self.global_se3_ego else None
        lidar_rig = (
            type(self.lidar_rig)(
                **{k: v.to(device) for k, v in self.lidar_rig._asdict().items()}
            )
            if self.lidar_rig
            else None
        )
        camera_rig = (
            type(self.camera_rig)(
                **{k: v.to(device) for k, v in self.camera_rig._asdict().items()}
            )
            if self.camera_rig
            else None
        )
        return type(self)(
            sequence_id=self.sequence_id,
            sequence_idx=self.sequence_idx,
            timestamp=self.timestamp,
            global_se3_ego=global_se3_ego,
            lidar_rig=lidar_rig,
            camera_rig=camera_rig,
        )

    @abstractmethod
    def next_frame(self) -> "SequentialDataFrame":
        """Returns the next frame in the sequence, or None if this is the last frame."""
        pass

    @abstractmethod
    def prev_frame(self) -> "SequentialDataFrame":
        """Returns the previous frame in the sequence, or None if this is the first frame."""
        pass

    def render(self) -> None:
        """Render the data in this frame using viser."""
        pass
