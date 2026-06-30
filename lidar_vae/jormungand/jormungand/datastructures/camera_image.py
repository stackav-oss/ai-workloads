import torch

from jormungand.datastructures.pointcloud import PointCloud
from jormungand.datastructures.se3 import SE3
from dltype import UInt8Tensor, FloatTensor, BoolTensor, dltyped
from typing import Annotated
import numpy as np
import einops


class CameraImage:
    @dltyped()
    def __init__(
        self,
        image: Annotated[torch.Tensor, UInt8Tensor["C H W"]],
        ego_se3_camera: SE3 | None = None,
        intrinsics: Annotated[torch.Tensor, FloatTensor["3 3"]] | None = None,
        caption: str | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        """
        Args:
            image: Torch tensor of shape (C, H, W), stores RGB pixels in the range [0, 255]
            ego_se3_camera: SE3 transformation from the camera frame to the ego frame, is identity by default.
            intrinsics: Optional torch tensor of shape (3, 3) representing the camera intrinsics matrix if it is known.
            caption: Optional string caption for the image.
            timestamp_ns: Optional timestamp in nanoseconds for this image.
        """
        self.device = image.device
        self.image = image
        self.ego_se3_camera = (
            SE3.identity().to(self.device)
            if ego_se3_camera is None
            else ego_se3_camera.to(self.device)
        )
        self.intrinsics: Annotated[torch.Tensor | None, FloatTensor["3 3"]] = (
            intrinsics.to(self.device) if intrinsics is not None else None
        )
        self.caption: str | None = caption

        self.timestamp_ns: int | None = timestamp_ns

    def to(self, device: torch.device) -> "CameraImage":
        return CameraImage(
            image=self.image.to(device),
            ego_se3_camera=self.ego_se3_camera.to(device),
            intrinsics=self.intrinsics.to(device)
            if self.intrinsics is not None
            else None,
            caption=self.caption,
            timestamp_ns=self.timestamp_ns,
        )

    def __repr__(self) -> str:
        return f"CameraImage(image_shape={self.image.shape}, ego_se3_camera={self.ego_se3_camera}, caption={self.caption})"

    @classmethod
    def from_pillow_img(
        cls, pil_image, ego_se3_camera=None, intrinsics=None, timestamp_ns=None
    ) -> "CameraImage":
        """Create a CameraImage from a PIL image."""
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                einops.rearrange(np.array(pil_image), "h w c -> c h w")
            )
        ).to(torch.uint8)
        return cls(
            image=image_tensor,
            ego_se3_camera=ego_se3_camera,
            intrinsics=intrinsics,
            timestamp_ns=timestamp_ns,
        )

    @classmethod
    def from_cv2_img(
        cls, cv2_image, ego_se3_camera=None, intrinsics=None, timestamp_ns=None
    ) -> "CameraImage":
        """Create a CameraImage from a cv2 image (which is in BGR format)."""
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                einops.rearrange(cv2_image[:, :, ::-1], "h w c -> c h w")
            )
        ).to(torch.uint8)
        return cls(
            image=image_tensor,
            ego_se3_camera=ego_se3_camera,
            intrinsics=intrinsics,
            timestamp_ns=timestamp_ns,
        )

    @dltyped()
    def normalize(self) -> Annotated[torch.Tensor, FloatTensor["C H W"]]:
        """Normalize the image to be in the range [0, 1]"""
        return self.image.float() / 255.0

    @dltyped()
    def imgnet_normalize(self) -> Annotated[torch.Tensor, FloatTensor["C H W"]]:
        """Normalize the image using ImageNet mean and torch.std"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.image.device).view(
            -1, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=self.image.device).view(
            -1, 1, 1
        )
        return (self.normalize() - mean) / std

    @dltyped()
    def resize(self, new_height: int, new_width: int) -> "CameraImage":
        """Resize the image to the given height and width using bilinear interpolation."""
        resized_image = torch.nn.functional.interpolate(
            self.image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        scaled_intrinsics = self.intrinsics
        # adjust intrinsics if they exist
        if self.intrinsics is not None:
            height, width = self.image.shape[1:]
            scale_x = new_width / width
            scale_y = new_height / height
            scaled_intrinsics = self.intrinsics.clone()
            scaled_intrinsics[0, 0] *= scale_x  # fx
            scaled_intrinsics[1, 1] *= scale_y  # fy
            scaled_intrinsics[0, 2] *= scale_x  # cx
            scaled_intrinsics[1, 2] *= scale_y  # cy
        return CameraImage(
            image=resized_image,
            ego_se3_camera=self.ego_se3_camera,
            intrinsics=scaled_intrinsics,
            caption=self.caption,
        )

    @dltyped()
    def crop(self, top: int, left: int, height: int, width: int) -> "CameraImage":
        """Crop the image to the given bounding box."""
        cropped_image = self.image[:, top : top + height, left : left + width]
        cropped_intrinsics = self.intrinsics
        # adjust intrinsics if they exist
        if self.intrinsics is not None:
            cropped_intrinsics = self.intrinsics.clone()
            cropped_intrinsics[0, 2] -= left  # cx
            cropped_intrinsics[1, 2] -= top  # cy
        return CameraImage(
            image=cropped_image,
            ego_se3_camera=self.ego_se3_camera,
            intrinsics=cropped_intrinsics,
            caption=self.caption,
        )

    @dltyped()
    def as_pointcloud(
        self, depthbuffer: Annotated[torch.Tensor, FloatTensor["H W"]]
    ) -> "PointCloud":
        """Convert the camera image to a point cloud using the depth buffer and intrinsics."""
        assert self.intrinsics is not None, (
            "Intrinsics must be provided to convert to point cloud"
        )
        device = self.image.device
        height, width = depthbuffer.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device)
        )
        x = x.flatten()
        y = y.flatten()
        depth = depthbuffer.flatten()
        valid_mask = depth > 0
        x = x[valid_mask]
        y = y[valid_mask]
        depth = depth[valid_mask]
        intrinsics_inv = torch.linalg.inv(self.intrinsics)
        points_homogeneous = (
            torch.stack([x * depth, y * depth, depth], dim=1) @ intrinsics_inv.T
        )
        points_homogeneous = torch.cat(
            [
                points_homogeneous,
                torch.ones((points_homogeneous.shape[0], 1), device=device),
            ],
            dim=1,
        )
        points_ego = (self.ego_se3_camera.transform_matrix @ points_homogeneous.T).T[
            :, :3
        ]
        return PointCloud(
            points_ego, ego_se3_sensor=self.ego_se3_camera, caption=self.caption
        )

    @dltyped()
    def project_points_to_image(
        self, points_ego: Annotated[torch.Tensor, FloatTensor["N 3"]]
    ) -> tuple[
        Annotated[torch.Tensor, FloatTensor["N 2"]],
        Annotated[torch.Tensor, BoolTensor["N"]],
    ]:
        """Project 3D points in the ego frame to 2D pixel coordinates in the image plane using the camera intrinsics and extrinsics."""
        assert self.intrinsics is not None, (
            "Intrinsics must be provided to project points to image"
        )
        device = self.image.device
        num_points = points_ego.shape[0]
        points_homogeneous = torch.cat(
            [points_ego, torch.ones((num_points, 1), device=device)], dim=1
        )
        points_camera = (
            torch.linalg.inv(self.ego_se3_camera.transform_matrix)
            @ points_homogeneous.T
        ).T[:, :3]
        # Only points with positive depth are valid
        valid_mask = (
            points_camera[:, 2] > 1e-3
        )  # to avoid numerical issues with points that are very close to the camera plane
        projected_points = points_camera @ self.intrinsics.T
        pixel_coords = projected_points[:, :2] / projected_points[:, 2:3]
        return pixel_coords, valid_mask
