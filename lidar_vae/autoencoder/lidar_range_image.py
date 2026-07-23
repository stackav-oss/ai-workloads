from dataclasses import dataclass
from pathlib import Path
import struct
import zlib

import numpy as np


class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""

    EXTENSIONS_SCAN = [".bin"]

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """Reset scan members."""
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros(
            (self.proj_H, self.proj_W), dtype=np.int32
        )  # [H,W] mask

    def size(self):
        """Return the size of the point cloud."""
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """Open raw scan and fill in attributes"""
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError(
                "Filename should be string type, but was {type}".format(
                    type=str(type(filename))
                )
            )

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """Set scan attributes (instead of opening from file)"""
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)


@dataclass
class RangeProjection:
    proj_range: np.ndarray
    proj_xyz: np.ndarray
    proj_remission: np.ndarray
    proj_idx: np.ndarray
    proj_mask: np.ndarray
    unproj_range: np.ndarray
    proj_x: np.ndarray
    proj_y: np.ndarray


def project_range_image(
    points: np.ndarray,
    remissions: np.ndarray | None = None,
    *,
    height: int = 32,
    width: int = 1070,
    fov_up_deg: float = 10.0,
    fov_down_deg: float = -30.0,
) -> RangeProjection:
    """Project XYZ points to a spherical range-view image.

    Empty pixels are set to -1. If several points land in one pixel, the nearest
    point is kept by writing farther points first and nearer points last.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Range image height and width must be positive.")
    if fov_up_deg <= fov_down_deg:
        raise ValueError("fov_up_deg must be greater than fov_down_deg.")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}.")

    if remissions is None:
        remissions = np.zeros(points.shape[0], dtype=np.float32)
    else:
        remissions = np.asarray(remissions, dtype=np.float32).reshape(-1)
    if remissions.shape[0] != points.shape[0]:
        raise ValueError(
            "remissions must have one value per point, got "
            f"{remissions.shape[0]} remissions for {points.shape[0]} points."
        )

    scan = LaserScan(
        project=True,
        H=height,
        W=width,
        fov_up=fov_up_deg,
        fov_down=fov_down_deg,
    )
    scan.set_points(points=points, remissions=remissions)

    return RangeProjection(
        proj_range=scan.proj_range,
        proj_xyz=scan.proj_xyz,
        proj_remission=scan.proj_remission,
        proj_idx=scan.proj_idx,
        proj_mask=(scan.proj_idx >= 0).astype(np.float32),
        unproj_range=np.asarray(scan.unproj_range, dtype=np.float32),
        proj_x=np.asarray(scan.proj_x, dtype=np.int32),
        proj_y=np.asarray(scan.proj_y, dtype=np.int32),
    )


def colorize_range_image(
    range_image: np.ndarray,
    *,
    invalid_value: float = -1.0,
    cmap: str = "viridis",
) -> np.ndarray:
    """Convert a range image to an RGB uint8 image for visualization."""
    ranges = np.asarray(range_image, dtype=np.float32)
    valid = ranges > invalid_value
    normalized = np.zeros_like(ranges, dtype=np.float32)

    if np.any(valid):
        valid_ranges = ranges[valid]
        min_range = float(valid_ranges.min())
        max_range = float(valid_ranges.max())
        if max_range > min_range:
            normalized[valid] = (valid_ranges - min_range) / (max_range - min_range)
        else:
            normalized[valid] = 1.0

    rgb = _apply_colormap(normalized, cmap=cmap)
    rgb[~valid] = 0
    return rgb


def _apply_colormap(values: np.ndarray, *, cmap: str) -> np.ndarray:
    if cmap == "gray":
        gray = (np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    if cmap == "turbo":
        stops = np.array(
            [
                [48, 18, 59],
                [38, 110, 180],
                [70, 190, 110],
                [250, 230, 65],
                [180, 30, 40],
            ],
            dtype=np.float32,
        )
    else:
        stops = np.array(
            [
                [68, 1, 84],
                [59, 82, 139],
                [33, 145, 140],
                [94, 201, 98],
                [253, 231, 37],
            ],
            dtype=np.float32,
        )

    clipped = np.clip(values, 0.0, 1.0)
    scaled = clipped * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    weight = (scaled - lower)[..., None]
    rgb = stops[lower] * (1.0 - weight) + stops[upper] * weight
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum)
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", checksum & 0xFFFFFFFF)
    )


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"PNG image must have shape [H, W, 3], got {rgb.shape}.")

    height, _, _ = rgb.shape
    width = rgb.shape[1]
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        header
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def save_range_projection(
    *,
    points: np.ndarray,
    remissions: np.ndarray | None,
    output_prefix: Path,
    height: int = 32,
    width: int = 1070,
    fov_up_deg: float = 10.0,
    fov_down_deg: float = -30.0,
    cmap: str = "viridis",
) -> tuple[Path, Path]:
    """Save a range projection as both a colorized PNG and tensor NPZ."""
    projection = project_range_image(
        points=points,
        remissions=remissions,
        height=height,
        width=width,
        fov_up_deg=fov_up_deg,
        fov_down_deg=fov_down_deg,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    npz_path = output_prefix.with_suffix(".npz")

    _save_rgb_png(png_path, colorize_range_image(projection.proj_range, cmap=cmap))
    np.savez_compressed(
        npz_path,
        proj_range=projection.proj_range,
        proj_xyz=projection.proj_xyz,
        proj_remission=projection.proj_remission,
        proj_idx=projection.proj_idx,
        proj_mask=projection.proj_mask,
        unproj_range=projection.unproj_range,
        proj_x=projection.proj_x,
        proj_y=projection.proj_y,
    )
    return png_path, npz_path
