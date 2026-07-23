from typing import Annotated

import torch
from dltype import FloatTensor, dltyped
from pyquaternion import Quaternion


class SE3:
    """An SE3 class allows point cloud rotation and translation operations.

    SE3s are typically named src_SE3_dst, where src and dst are frames of reference.
    For example, if we have a point cloud in the sensor frame and want to transform it to the ego frame, we can use ego_SE3_sensor to perform the transformation.
    To chain transformations we can right multiply. For example, ego_se3_camera = ego_SE3_sensor * sensor_SE3_camera.
    """

    @dltyped()
    def __init__(
        self,
        rotation_matrix: Annotated[torch.Tensor, FloatTensor["3 3"]],
        translation: Annotated[torch.Tensor, FloatTensor["3"]],
    ) -> None:
        """Initialize an SE3 instance with its rotation and translation matrices.
        Args:
            rotation: Array of shape (3, 3)
            translation: Array of shape (3,)
        """
        self.device = rotation_matrix.device
        rotation_matrix = torch.as_tensor(rotation_matrix)
        translation = torch.as_tensor(
            translation,
            dtype=rotation_matrix.dtype,
            device=rotation_matrix.device,
        )
        assert rotation_matrix.shape == (3, 3), (
            f"Rotation matrix must have shape (3, 3), got {rotation_matrix.shape}"
        )
        assert translation.shape == (3,), (
            f"Translation must be a 3D vector, got {translation.shape}"
        )

        self.transform_matrix = torch.eye(
            4,
            dtype=rotation_matrix.dtype,
            device=rotation_matrix.device,
        )
        self.transform_matrix[:3, :3] = rotation_matrix
        self.transform_matrix[:3, 3] = translation

    def to(self, device: torch.device) -> "SE3":
        return SE3(
            rotation_matrix=self.rotation_matrix.to(device),
            translation=self.translation.to(device),
        )

    @property
    def rotation_matrix(self) -> Annotated[torch.Tensor, FloatTensor["3 3"]]:
        return self.transform_matrix[:3, :3]

    @property
    def translation(self) -> Annotated[torch.Tensor, FloatTensor["3"]]:
        return self.transform_matrix[:3, 3]

    @staticmethod
    def identity() -> "SE3":
        """Return the identity transformation."""
        return SE3(rotation_matrix=torch.eye(3), translation=torch.zeros(3))

    @staticmethod
    def from_rot_x_y_z_translation_x_y_z(rx, ry, rz, tx, ty, tz) -> "SE3":
        rotation_matrix = torch.as_tensor(
            Quaternion(axis=[1, 0, 0], angle=rx).rotation_matrix
            @ Quaternion(axis=[0, 1, 0], angle=ry).rotation_matrix
            @ Quaternion(axis=[0, 0, 1], angle=rz).rotation_matrix
        )
        translation = torch.tensor([tx, ty, tz], dtype=rotation_matrix.dtype)
        return SE3(rotation_matrix, translation)

    @staticmethod
    def from_rot_w_x_y_z_translation_x_y_z(rw, rx, ry, rz, tx, ty, tz) -> "SE3":
        rotation_matrix = torch.as_tensor(
            Quaternion(w=rw, x=rx, y=ry, z=rz).rotation_matrix
        )
        translation = torch.tensor([tx, ty, tz], dtype=rotation_matrix.dtype)
        return SE3(rotation_matrix, translation)

    def as_xyz_yaw_pitch_roll(self) -> Annotated[torch.Tensor, FloatTensor["6"]]:
        q = Quaternion(matrix=self.rotation_matrix.cpu().numpy())
        yaw, pitch, roll = q.yaw_pitch_roll
        tx, ty, tz = self.translation.cpu().numpy()
        return torch.tensor([tx, ty, tz, yaw, pitch, roll], device=self.device)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, SE3):
            return False
        return torch.allclose(
            self.rotation_matrix, __value.rotation_matrix
        ) and torch.allclose(self.translation, __value.translation)

    def translate(
        self, translation: Annotated[torch.Tensor, FloatTensor["3"]]
    ) -> "SE3":
        """Return a new SE3 instance with the given translation applied."""
        translation = torch.as_tensor(
            translation,
            dtype=self.translation.dtype,
            device=self.translation.device,
        )
        assert translation.shape == (3,), (
            f"Translation must be a 3D vector, got {translation.shape}"
        )
        return SE3(
            rotation_matrix=self.rotation_matrix,
            translation=self.translation + translation,
        )

    def scale(self, scale: float) -> "SE3":
        """Return a new SE3 instance with the given scale applied."""
        return SE3(
            rotation_matrix=self.rotation_matrix * scale,
            translation=self.translation * scale,
        )

    def transform_points(
        self, point_cloud: Annotated[torch.Tensor, FloatTensor["N 3"]]
    ) -> Annotated[torch.Tensor, FloatTensor["N 3"]]:
        """Apply the SE(3) transformation to this point cloud.
        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`
        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        point_cloud = torch.as_tensor(
            point_cloud,
            dtype=self.rotation_matrix.dtype,
            device=self.rotation_matrix.device,
        )
        return point_cloud @ self.rotation_matrix.T + self.translation

    def inverse(self) -> "SE3":
        """Return the inverse of the current SE3 transformation.
        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.
        Returns:
            src_SE3_target: instance of SE3 class, representing
                inverse of SE3 transformation target_SE3_src
        """
        inverse_rotation = self.rotation_matrix.T
        return SE3(
            rotation_matrix=inverse_rotation,
            translation=inverse_rotation @ (-self.translation),
        )

    def compose(self, right_se3: "SE3") -> "SE3":
        """Compose (right multiply) this class' transformation matrix T with another SE3 instance.
        Algebraic representation: chained_se3 = T * right_se3
        Args:
            right_se3: another instance of SE3 class
        Returns:
            chained_se3: new instance of SE3 class
        """
        return SE3.from_array(self.transform_matrix @ right_se3.transform_matrix)

    def __matmul__(self, right_se3: "SE3") -> "SE3":
        return self.compose(right_se3)

    def to_array(self) -> torch.Tensor:
        """Return the SE3 transformation matrix as a tensor."""
        return self.transform_matrix

    @staticmethod
    def from_array(transform_matrix: torch.Tensor) -> "SE3":
        """Initialize an SE3 instance from a tensor."""
        transform_matrix = torch.as_tensor(transform_matrix)
        return SE3(
            rotation_matrix=transform_matrix[:3, :3],
            translation=transform_matrix[:3, 3],
        )

    def __repr__(self) -> str:
        return f"SE3(rotation_matrix={self.rotation_matrix}, translation={self.translation})"
