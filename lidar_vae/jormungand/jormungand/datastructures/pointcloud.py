from einops import repeat
import torch

from jormungand.datastructures.se3 import SE3
from dltype import FloatTensor, BoolTensor, dltyped, IntTensor
from typing import Annotated


@dltyped()
def to_fixed_array(
    array: Annotated[torch.Tensor, FloatTensor["N 3"]],
    max_len: int,
    pad_val: float = float("nan"),
    intensity: Annotated[torch.Tensor, IntTensor["N"]] | None = None,
    colors: Annotated[torch.Tensor, IntTensor["N 3"]] | None = None,
    classes: Annotated[torch.Tensor, IntTensor["N"]] | None = None,
) -> tuple[
    Annotated[torch.Tensor, FloatTensor["max_len 3"]],
    Annotated[torch.Tensor, IntTensor["max_len"]] | None,
    Annotated[torch.Tensor, IntTensor["max_len 3"]] | None,
    Annotated[torch.Tensor, IntTensor["max_len"]] | None,
]:
    array = torch.as_tensor(array)
    if len(array) > max_len:
        raise ValueError(
            f"array has length {len(array)}, which is greater than max_len {max_len}"
        )

    pad_shape = (max_len - len(array), *array.shape[1:])
    padding = torch.full(
        pad_shape,
        pad_val,
        dtype=array.dtype,
        device=array.device,
    )
    fixed_points = torch.cat((array, padding), dim=0)

    def _pad_optional(
        values: torch.Tensor | None,
        pad: int,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        values = torch.as_tensor(values, device=array.device)
        if len(values) != len(array):
            raise ValueError(
                f"optional field has length {len(values)}, expected {len(array)}"
            )
        extra_shape = values.shape[1:]
        values_padding = torch.full(
            (max_len - len(values), *extra_shape),
            pad,
            dtype=values.dtype,
            device=values.device,
        )
        return torch.cat((values, values_padding), dim=0)

    fixed_intensity = _pad_optional(intensity, pad=-1)
    fixed_colors = _pad_optional(colors, pad=0)
    fixed_classes = _pad_optional(classes, pad=-1)
    return fixed_points, fixed_intensity, fixed_colors, fixed_classes


@dltyped()
def from_fixed_array(
    array: Annotated[torch.Tensor, FloatTensor["max_len 3"]],
    intensity: Annotated[torch.Tensor, IntTensor["max_len"]] | None = None,
    colors: Annotated[torch.Tensor, IntTensor["max_len 3"]] | None = None,
    classes: Annotated[torch.Tensor, IntTensor["max_len"]] | None = None,
) -> tuple[
    Annotated[torch.Tensor, FloatTensor["N 3"]],
    Annotated[torch.Tensor, IntTensor["N"]] | None,
    Annotated[torch.Tensor, IntTensor["N 3"]] | None,
    Annotated[torch.Tensor, IntTensor["N"]] | None,
]:
    if not isinstance(array, torch.Tensor):
        raise ValueError(f"expected array to be a torch tensor, got {type(array)}")
    are_valid_points = torch.logical_not(torch.isnan(array).any(dim=1))
    points = array[are_valid_points]

    def _extract_optional(values: torch.Tensor | None) -> torch.Tensor | None:
        if values is None:
            return None
        values = torch.as_tensor(values, device=array.device)
        if len(values) != len(array):
            raise ValueError(
                f"optional fixed array has length {len(values)}, expected {len(array)}"
            )
        return values[are_valid_points]

    return (
        points,
        _extract_optional(intensity),
        _extract_optional(colors),
        _extract_optional(classes),
    )


class PointCloud:
    def __init__(
        self,
        points: Annotated[torch.Tensor, FloatTensor["N 3"]],
        intensity: Annotated[torch.Tensor, IntTensor["N"]] | None = None,
        colors: Annotated[torch.Tensor, IntTensor["N 3"]] | None = None,
        classes: Annotated[torch.Tensor, IntTensor["N"]] | None = None,
        ego_se3_sensor: SE3 | None = None,
        caption: str | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        """
        Args:
            points: Torch tensor of shape (N, 3)
            intensity: Optional torch tensor of shape (N,) representing the intensity of each point, with values in [0, 255].
            colors: Optional torch tensor of shape (N, 3) representing the RGB color of each point, with values in [0, 255].
                If no colors are set, colorizes by intensity. If no intensity, is set to blue.
            classes: Optional torch tensor of shape (N,) representing the class label of each point.
            ego_se3_sensor: SE3 transformation from the sensor frame to the ego frame, is identity by default.
            caption: Optional string caption for the point cloud.
            timestamp_ns: Optional timestamp in nanoseconds for this point cloud.
        """
        self.device = points.device
        self.points = points
        self.intensity = intensity
        self.classes = classes
        if colors is None:
            if self.intensity is not None:
                self.colors = self.colorize_by_intensity()
            else:
                self.colors = repeat(
                    torch.tensor([[0, 0, 255]], device=self.device),
                    "1 c -> n c",
                    n=len(points),
                )
        else:
            self.colors = colors
        self.ego_se3_sensor = (
            SE3.identity().to(self.device)
            if ego_se3_sensor is None
            else ego_se3_sensor.to(self.device)
        )
        self.caption: str | None = caption
        self.timestamp_ns: int | None = timestamp_ns

    def to(self, device: torch.device) -> "PointCloud":
        return PointCloud(
            self.points.to(device),
            intensity=self.intensity.to(device) if self.intensity is not None else None,
            colors=self.colors.to(device) if self.colors is not None else None,
            classes=self.classes.to(device) if self.classes is not None else None,
            ego_se3_sensor=self.ego_se3_sensor.to(device),
            caption=self.caption,
            timestamp_ns=self.timestamp_ns,
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PointCloud):
            return False
        points_match = torch.allclose(self.points, o.points)
        intensity_match = (self.intensity is None and o.intensity is None) or (
            self.intensity is not None
            and o.intensity is not None
            and torch.equal(self.intensity, o.intensity)
        )
        classes_match = (self.classes is None and o.classes is None) or (
            self.classes is not None
            and o.classes is not None
            and torch.equal(self.classes, o.classes)
        )
        return (
            points_match
            and intensity_match
            and classes_match
            and self.ego_se3_sensor == o.ego_se3_sensor
            and self.caption == o.caption
        )

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self) -> str:
        return f"PointCloud with {len(self)} points"

    def __getitem__(self, idx):
        return self.points[idx]

    def colorize_by_intensity(self, colormap="viridis") -> torch.Tensor:
        import matplotlib.pyplot as plt

        if self.intensity is None:
            raise ValueError("intensity is not set for this point cloud")
        normed_intensity = (self.intensity - self.intensity.min()) / (
            self.intensity.max() - self.intensity.min() + 1e-8
        )
        cmap = plt.get_cmap(colormap)
        colors = cmap(normed_intensity.cpu().numpy())[:, :3] * 255
        return torch.from_numpy(colors).to(self.points.device).to(torch.int)

    def transform(self, se3: SE3) -> "PointCloud":
        assert isinstance(se3, SE3), (
            f"se3 must be an SE3, got {type(se3)}, expected {SE3}"
        )
        return PointCloud(
            se3.to(self.points.device).transform_points(self.points),
            intensity=self.intensity,
            colors=self.colors,
            classes=self.classes,
            ego_se3_sensor=self.ego_se3_sensor,
            caption=self.caption,
            timestamp_ns=self.timestamp_ns,
        )

    @dltyped()
    def transform_masked(
        self, se3: SE3, mask: Annotated[torch.Tensor, BoolTensor["N"]]
    ) -> "PointCloud":
        assert isinstance(se3, SE3)
        mask = torch.as_tensor(mask, device=self.points.device)
        updated_points = self.points.clone()
        updated_points[mask] = se3.to(self.points.device).transform_points(
            self.points[mask]
        )
        return PointCloud(
            updated_points,
            intensity=self.intensity,
            colors=self.colors,
            classes=self.classes,
            ego_se3_sensor=self.ego_se3_sensor,
            caption=self.caption,
            timestamp_ns=self.timestamp_ns,
        )

    def translate(
        self, translation: Annotated[torch.Tensor, FloatTensor["3"]]
    ) -> "PointCloud":
        translation = torch.as_tensor(
            translation,
            dtype=self.points.dtype,
            device=self.points.device,
        )
        return PointCloud(
            self.points + translation,
            intensity=self.intensity,
            colors=self.colors,
            classes=self.classes,
            ego_se3_sensor=self.ego_se3_sensor,
            caption=self.caption,
            timestamp_ns=self.timestamp_ns,
        )

    def to_fixed_array(
        self, max_points: int
    ) -> tuple[
        Annotated[torch.Tensor, FloatTensor["max_points 3"]],
        Annotated[torch.Tensor, IntTensor["max_points"]] | None,
        Annotated[torch.Tensor, IntTensor["max_points 3"]] | None,
        Annotated[torch.Tensor, IntTensor["max_points"]] | None,
    ]:
        fixed = to_fixed_array(
            self.points,
            max_points,
            intensity=self.intensity,
            colors=self.colors,
            classes=self.classes,
        )
        return fixed

    def matched_point_diffs(self, other: "PointCloud") -> torch.Tensor:
        assert len(self) == len(other)
        return self.points - other.points

    def matched_point_distance(self, other: "PointCloud") -> torch.Tensor:
        assert len(self) == len(other)
        return torch.linalg.norm(self.matched_point_diffs(other), dim=1)

    @staticmethod
    @dltyped()
    def from_fixed_array(
        points: Annotated[torch.Tensor, FloatTensor["max_len 3"]],
        intensity: Annotated[torch.Tensor, IntTensor["max_len"]] | None = None,
        colors: Annotated[torch.Tensor, IntTensor["max_len 3"]] | None = None,
        classes: Annotated[torch.Tensor, IntTensor["max_len"]] | None = None,
    ) -> "PointCloud":
        points, intensity, colors, classes = from_fixed_array(
            points,
            intensity=intensity,
            colors=colors,
            classes=classes,
        )
        return PointCloud(
            points,
            intensity=intensity,
            colors=colors,
            classes=classes,
        )

    @dltyped()
    def to_array(self) -> Annotated[torch.Tensor, FloatTensor["N 3"]]:
        return self.points

    def copy(self) -> "PointCloud":
        return PointCloud(
            self.points.clone(),
            intensity=self.intensity.clone() if self.intensity is not None else None,
            colors=self.colors.clone() if self.colors is not None else None,
            classes=self.classes.clone() if self.classes is not None else None,
            ego_se3_sensor=self.ego_se3_sensor,
            caption=self.caption,
        )

    @dltyped()
    def mask_points(
        self, mask: Annotated[torch.Tensor, BoolTensor["N"]]
    ) -> "PointCloud":
        mask = torch.as_tensor(mask, device=self.points.device)
        assert mask.ndim == 1
        if mask.dtype == torch.bool:
            assert mask.shape[0] == len(self)
        else:
            in_bounds = torch.logical_and(mask >= 0, mask < len(self))
            assert torch.all(in_bounds), (
                f"mask values must be in bounds, got {(~in_bounds).sum().item()} indices not in bounds out of {len(self)} points"
            )

        return PointCloud(
            self.points[mask],
            intensity=self.intensity[mask] if self.intensity is not None else None,
            colors=self.colors[mask] if self.colors is not None else None,
            classes=self.classes[mask] if self.classes is not None else None,
            ego_se3_sensor=self.ego_se3_sensor,
            caption=self.caption,
        )

    @dltyped()
    def within_region_mask(
        self, x_min, x_max, y_min, y_max, z_min, z_max
    ) -> Annotated[torch.Tensor, BoolTensor["N"]]:
        mask = torch.logical_and(self.points[:, 0] < x_max, self.points[:, 0] > x_min)
        mask = torch.logical_and(mask, self.points[:, 1] < y_max)
        mask = torch.logical_and(mask, self.points[:, 1] > y_min)
        mask = torch.logical_and(mask, self.points[:, 2] < z_max)
        mask = torch.logical_and(mask, self.points[:, 2] > z_min)
        return mask

    def within_region(self, x_min, x_max, y_min, y_max, z_min, z_max) -> "PointCloud":
        mask = self.within_region_mask(x_min, x_max, y_min, y_max, z_min, z_max)
        return self.mask_points(mask)

    @property
    def shape(self) -> tuple[int, int]:
        return self.points.shape
