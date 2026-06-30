"""Dataset-agnostic viser renderer for :class:`SequentialDataFrame`.

This module provides a single class, :class:`SequentialFrameRenderer`, that can
visualize one or more sequential frames in a viser server. It works with any
:class:`SequentialDataFrame` subclass (NuScenes, AV2, Waymo, ...). All
dataset-specific knobs (e.g. per-category bounding-box colors) are passed in
explicitly so that the renderer itself does not depend on any particular
dataset module.

A frame is expected to have the standard :class:`SequentialDataFrame`
attributes (``lidar_rig``, ``camera_rig``, ``global_se3_ego``, etc.). The
renderer additionally looks up two *optional* attributes via :func:`getattr`:

* ``bounding_boxes_3d`` — a tensor of shape ``(N, 9)`` with columns
  ``[l, w, h, tx, ty, tz, yaw, pitch, roll]`` in the ego frame.
* ``bbox_classes``      — list of length ``N`` with the per-box category name.

If those attributes are absent the bounding-box overlay is simply skipped.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch
import viser
from pyquaternion import Quaternion

from jormungand.datastructures.camera_image import CameraImage
from jormungand.datastructures.pointcloud import PointCloud
from jormungand.datastructures.se3 import SE3
from jormungand.datastructures.sequential_data import SequentialDataFrame


_DEFAULT_BOX_COLOR: tuple[int, int, int] = (255, 255, 255)
_DEFAULT_LIDAR_COLORIZE_CAMERA_PRIORITY: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)


_UNIT_BOX_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)
_UNIT_BOX_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=np.uint32,
)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _camera_image_to_numpy(camera_image: CameraImage) -> np.ndarray:
    return _tensor_to_numpy(
        camera_image.image.permute(1, 2, 0).contiguous().to(torch.uint8)
    )


def _rotation_matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion.

    Re-orthogonalizes via SVD so that ``pyquaternion`` does not reject
    matrices with small numerical drift from float32 operations.
    """
    rot = np.asarray(rot, dtype=np.float64)
    U, _, Vt = np.linalg.svd(rot)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return np.asarray(Quaternion(matrix=R).elements, dtype=np.float32)


def _camera_frustum_args(
    camera_image: CameraImage,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Build (image, fov_y, aspect, position, wxyz) for ``add_camera_frustum``."""
    image = _camera_image_to_numpy(camera_image)
    H, W = image.shape[:2]
    if camera_image.intrinsics is not None:
        fy = float(camera_image.intrinsics[1, 1].item())
        fov_y = 2.0 * math.atan(H / (2.0 * fy))
    else:
        fov_y = math.radians(60.0)
    aspect = float(W) / float(H)
    position = _tensor_to_numpy(
        camera_image.ego_se3_camera.translation.to(torch.float32)
    )
    wxyz = _rotation_matrix_to_wxyz(
        _tensor_to_numpy(camera_image.ego_se3_camera.rotation_matrix)
    )
    return image, fov_y, aspect, position, wxyz


def _colorize_lidar_from_camera_images(
    point_cloud: PointCloud,
    camera_rig: NamedTuple,
    camera_priority: Sequence[str],
) -> np.ndarray | None:
    """Project lidar points (in ego frame) into each camera image and sample
    a per-point RGB color.

    Cameras are visited in ``camera_priority`` order; each lidar point gets
    its color from the first camera in that order which sees it. Points that
    do not project into any image keep their existing color (typically the
    intensity-based fallback). Returns a ``(N, 3)`` uint8 array, or ``None``
    if no usable cameras with intrinsics were found.
    """
    available = camera_rig._asdict()
    ordered: list[tuple[str, CameraImage]] = []
    seen: set[str] = set()
    for name in camera_priority:
        cam = available.get(name)
        if cam is not None and cam.intrinsics is not None:
            ordered.append((name, cam))
            seen.add(name)
    for name, cam in available.items():
        if name not in seen and cam.intrinsics is not None:
            ordered.append((name, cam))
    if not ordered:
        return None

    points_ego = point_cloud.points
    n = points_ego.shape[0]
    if n == 0:
        return None

    colors = point_cloud.colors.to(torch.uint8).clone()
    assigned = torch.zeros(n, dtype=torch.bool, device=points_ego.device)

    for _, camera in ordered:
        device = camera.image.device
        points_on_device = points_ego.to(device)
        pixel_coords, valid_mask = camera.project_points_to_image(points_on_device)

        H, W = camera.image.shape[1], camera.image.shape[2]
        u = pixel_coords[:, 0]
        v = pixel_coords[:, 1]
        in_bounds = valid_mask & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        in_bounds_cpu = in_bounds.to(colors.device)
        if not torch.any(in_bounds_cpu):
            continue

        to_assign = in_bounds_cpu & (~assigned)
        if not torch.any(to_assign):
            continue

        u_int = u[in_bounds].to(torch.long).clamp_(0, W - 1)
        v_int = v[in_bounds].to(torch.long).clamp_(0, H - 1)
        sampled = camera.image[:, v_int, u_int].permute(1, 0).to(colors.device)

        target_indices = torch.nonzero(to_assign, as_tuple=False).squeeze(1)
        in_bounds_indices = torch.nonzero(in_bounds_cpu, as_tuple=False).squeeze(1)
        new_mask = ~assigned[in_bounds_indices]
        new_samples = sampled[new_mask]
        colors[target_indices] = new_samples.to(colors.dtype)
        assigned[target_indices] = True

    return _tensor_to_numpy(colors.to(torch.uint8))


@dataclass
class _FrameHandles:
    """Bookkeeping for all viser handles created for a single rendered frame."""

    point_clouds: list[Any] = field(default_factory=list)
    boxes: list[Any] = field(default_factory=list)
    frustums: list[Any] = field(default_factory=list)
    box_category_counts: dict[str, int] = field(default_factory=dict)

    @property
    def all(self) -> list[Any]:
        return [*self.point_clouds, *self.boxes, *self.frustums]

    def remove_all(self) -> None:
        for handle in self.all:
            try:
                handle.remove()
            except Exception:
                # Some handles may already have been removed by a parent.
                pass


class SequentialFrameRenderer:
    """Renders one or many :class:`SequentialDataFrame`(s) in a viser server.

    The same renderer instance is reused for both single-frame and sequence
    visualization. When more than one frame is provided, the GUI grows a
    timeline slider, a Play/Pause toggle, and an FPS slider; a background
    thread advances the slider for playback.

    Args:
        category_colors: Optional mapping from bounding-box category name to
            an ``(R, G, B)`` tuple in ``[0, 255]``. Categories absent from the
            map fall back to ``default_box_color``.
        default_box_color: Fallback color for box categories without an entry
            in ``category_colors``.
        lidar_colorize_camera_priority: When the user enables "Colorize lidar
            from images", lidar points are colored by the first camera in this
            list that sees them. Cameras not present in the rig are skipped;
            cameras present but missing from this list are appended at the end.
    """

    def __init__(
        self,
        category_colors: dict[str, tuple[int, int, int]] | None = None,
        default_box_color: tuple[int, int, int] = _DEFAULT_BOX_COLOR,
        lidar_colorize_camera_priority: Sequence[str] = (
            _DEFAULT_LIDAR_COLORIZE_CAMERA_PRIORITY
        ),
    ) -> None:
        self.category_colors = category_colors or {}
        self.default_box_color = default_box_color
        self.lidar_colorize_camera_priority = tuple(lidar_colorize_camera_priority)

    def render(
        self,
        frames: SequentialDataFrame | Sequence[SequentialDataFrame],
        *,
        server_label: str = "Sequential frame viewer",
        colorize_lidar_from_images: bool = False,
        show_camera_frustums: bool = True,
        frustum_scale: float = 0.5,
        fps: float = 2.0,
    ) -> None:
        """Open a viser server and render ``frames`` (blocks until Ctrl+C).

        Args:
            frames: A single frame or a sequence of frames. With more than one
                frame, the renderer enables timeline + playback controls.
            server_label: Window/tab label shown in viser.
            colorize_lidar_from_images: Initial state of the
                "Colorize lidar from images" checkbox.
            show_camera_frustums: Initial state of the "Show camera frustums"
                checkbox; also controls whether frustum geometry is built.
            frustum_scale: Initial size (meters) of the camera frustums.
            fps: Initial playback FPS for sequences.
        """
        if isinstance(frames, SequentialDataFrame):
            frames = [frames]
        else:
            frames = list(frames)
        if not frames:
            raise ValueError("render() requires at least one frame.")

        self._run_viser_loop(
            frames,
            server_label=server_label,
            colorize_lidar_from_images=colorize_lidar_from_images,
            show_camera_frustums=show_camera_frustums,
            frustum_scale=frustum_scale,
            fps=fps,
        )

    def render_sequence(
        self,
        start_frame: SequentialDataFrame,
        num_frames: int | None = None,
        *,
        server_label: str | None = None,
        colorize_lidar_from_images: bool = False,
        show_camera_frustums: bool = True,
        frustum_scale: float = 0.5,
        fps: float = 2.0,
    ) -> None:
        """Walk ``start_frame.next_frame()`` to collect a sequence, then render.

        Args:
            start_frame: The first frame in the sequence.
            num_frames: Maximum number of frames to collect. ``None`` means walk
                until the end of the scene.
            server_label: Optional server label; defaults to a summary string.
            colorize_lidar_from_images: Initial state of the colorize checkbox.
            show_camera_frustums: Initial state of the frustums checkbox.
            frustum_scale: Initial frustum size.
            fps: Initial playback FPS.
        """
        frames: list[SequentialDataFrame] = [start_frame]
        cursor: SequentialDataFrame | None = start_frame
        while True:
            if num_frames is not None and len(frames) >= num_frames:
                break
            cursor = cursor.next_frame()
            if cursor is None:
                break
            frames.append(cursor)
            print(f"  loaded frame {len(frames)}")

        if server_label is None:
            server_label = (
                f"Sequence @ idx {start_frame.sequence_idx} ({len(frames)} frames)"
            )

        self.render(
            frames,
            server_label=server_label,
            colorize_lidar_from_images=colorize_lidar_from_images,
            show_camera_frustums=show_camera_frustums,
            frustum_scale=frustum_scale,
            fps=fps,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _populate_frame(
        self,
        server: viser.ViserServer,
        frame: SequentialDataFrame,
        *,
        prefix: str,
        colorize_lidar_from_images: bool,
        show_camera_frustums: bool,
        point_size: float,
        frustum_scale: float,
    ) -> _FrameHandles:
        """Create scene-graph nodes for one frame and return their handles."""
        handles = _FrameHandles()

        if frame.lidar_rig is not None:
            for sensor_name, point_cloud in frame.lidar_rig._asdict().items():
                if len(point_cloud) == 0:
                    continue
                points = _tensor_to_numpy(point_cloud.points.to(torch.float32))

                colors_np: np.ndarray | None = None
                if colorize_lidar_from_images and frame.camera_rig is not None:
                    colors_np = _colorize_lidar_from_camera_images(
                        point_cloud,
                        frame.camera_rig,
                        self.lidar_colorize_camera_priority,
                    )
                if colors_np is None:
                    colors_np = _tensor_to_numpy(point_cloud.colors.to(torch.uint8))

                handles.point_clouds.append(
                    server.scene.add_point_cloud(
                        name=f"{prefix}/lidar/{sensor_name}",
                        points=points,
                        colors=colors_np,
                        point_size=point_size,
                        point_shape="rounded",
                    )
                )

        bounding_boxes_3d = getattr(frame, "bounding_boxes_3d", None)
        bbox_classes = getattr(frame, "bbox_classes", None)
        if bounding_boxes_3d is not None and bounding_boxes_3d.numel() > 0:
            boxes = _tensor_to_numpy(bounding_boxes_3d.to(torch.float32))
            grouped: dict[str, dict[str, list[np.ndarray]]] = {}

            for i, row in enumerate(boxes):
                category = (
                    bbox_classes[i]
                    if bbox_classes is not None and i < len(bbox_classes)
                    else "UNLABELED"
                )
                handles.box_category_counts[category] = (
                    handles.box_category_counts.get(category, 0) + 1
                )
                length, width, height, tx, ty, tz, yaw, pitch, roll = row.tolist()
                box_se3 = SE3.from_rot_x_y_z_translation_x_y_z(
                    rx=roll,
                    ry=pitch,
                    rz=yaw,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                )
                bucket = grouped.setdefault(
                    category, {"positions": [], "wxyzs": [], "scales": []}
                )
                bucket["positions"].append(
                    _tensor_to_numpy(box_se3.translation.to(torch.float32))
                )
                bucket["wxyzs"].append(
                    _rotation_matrix_to_wxyz(_tensor_to_numpy(box_se3.rotation_matrix))
                )
                bucket["scales"].append(
                    np.array([length, width, height], dtype=np.float32)
                )

            for category, bucket in grouped.items():
                handles.boxes.append(
                    server.scene.add_batched_meshes_simple(
                        name=f"{prefix}/boxes/{category}",
                        vertices=_UNIT_BOX_VERTICES,
                        faces=_UNIT_BOX_FACES,
                        batched_wxyzs=np.asarray(bucket["wxyzs"], dtype=np.float32),
                        batched_positions=np.asarray(
                            bucket["positions"], dtype=np.float32
                        ),
                        batched_scales=np.asarray(bucket["scales"], dtype=np.float32),
                        batched_colors=self.category_colors.get(
                            category, self.default_box_color
                        ),
                        opacity=0.5,
                        material="toon3",
                        flat_shading=True,
                        side="double",
                    )
                )

        if show_camera_frustums and frame.camera_rig is not None:
            for camera_name, camera_image in frame.camera_rig._asdict().items():
                try:
                    image, fov_y, aspect, position, wxyz = _camera_frustum_args(
                        camera_image
                    )
                except Exception as exc:
                    print(f"  [warn] could not build frustum for {camera_name}: {exc}")
                    continue
                frustum_handle = server.scene.add_camera_frustum(
                    name=f"{prefix}/cameras/{camera_name}",
                    fov=fov_y,
                    aspect=aspect,
                    scale=frustum_scale,
                    line_width=2.0,
                    color=(40, 200, 255),
                    image=image,
                )
                # Some viser versions reject position/wxyz as kwargs to
                # ``add_camera_frustum``; assigning on the handle is portable.
                frustum_handle.position = tuple(position.tolist())
                frustum_handle.wxyz = tuple(wxyz.tolist())
                handles.frustums.append(frustum_handle)

        return handles

    def _scene_extent(
        self, frames: list[SequentialDataFrame]
    ) -> tuple[np.ndarray, float]:
        """Return ``(scene_center, scene_extent)`` aggregated over all frames."""
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for frame in frames:
            if frame.lidar_rig is None:
                continue
            for pc in frame.lidar_rig._asdict().values():
                if len(pc) == 0:
                    continue
                pts = _tensor_to_numpy(pc.points.to(torch.float32))
                mins.append(pts.min(axis=0))
                maxs.append(pts.max(axis=0))
        if mins:
            scene_min = np.min(np.stack(mins, axis=0), axis=0)
            scene_max = np.max(np.stack(maxs, axis=0), axis=0)
            return (
                0.5 * (scene_min + scene_max),
                max(float(np.max(scene_max - scene_min)), 1.0),
            )
        return np.zeros(3, dtype=np.float64), 50.0

    def _run_viser_loop(
        self,
        frames: list[SequentialDataFrame],
        *,
        server_label: str,
        colorize_lidar_from_images: bool,
        show_camera_frustums: bool,
        frustum_scale: float,
        fps: float,
    ) -> None:
        n = len(frames)
        is_sequence = n > 1

        server = viser.ViserServer(label=server_label)
        server.scene.set_up_direction("+z")

        scene_center, scene_extent = self._scene_extent(frames)
        server.initial_camera.look_at = tuple(scene_center.tolist())
        server.initial_camera.position = tuple(
            (
                scene_center
                + np.array(
                    [1.5 * scene_extent, -1.5 * scene_extent, scene_extent],
                    dtype=np.float64,
                )
            ).tolist()
        )
        server.scene.add_frame(
            "/ego",
            axes_length=max(1.0, 0.1 * scene_extent),
            axes_radius=max(0.02, 0.0025 * scene_extent),
        )

        folder_label = "Sequence" if is_sequence else "Frame"
        with server.gui.add_folder(folder_label):
            gui_summary = server.gui.add_markdown("")
            gui_index = (
                server.gui.add_slider(
                    "Frame", min=0, max=n - 1, step=1, initial_value=0
                )
                if is_sequence
                else None
            )
            gui_play = server.gui.add_checkbox("Play", False) if is_sequence else None
            gui_fps = (
                server.gui.add_slider(
                    "FPS", min=0.5, max=10.0, step=0.5, initial_value=fps
                )
                if is_sequence
                else None
            )
            gui_show_lidar = server.gui.add_checkbox("Show lidar", True)
            gui_point_size = server.gui.add_slider(
                "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.03
            )
            gui_show_boxes = server.gui.add_checkbox("Show boxes", True)
            gui_show_frustums = server.gui.add_checkbox(
                "Show camera frustums", show_camera_frustums
            )
            gui_frustum_scale = server.gui.add_slider(
                "Frustum scale",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=frustum_scale,
            )
            gui_colorize = server.gui.add_checkbox(
                "Colorize lidar from images", colorize_lidar_from_images
            )

        # Per-camera GUI image panels (updated when stepping through frames).
        image_panel_handles: dict[str, Any] = {}
        with server.gui.add_folder("Images"):
            cam_names: list[str] = []
            for frame in frames:
                if frame.camera_rig is None:
                    continue
                for cam_name in frame.camera_rig._asdict().keys():
                    if cam_name not in cam_names:
                        cam_names.append(cam_name)
            if not cam_names:
                server.gui.add_markdown("_No camera images loaded._")
            else:
                first_imgs = (
                    frames[0].camera_rig._asdict()
                    if frames[0].camera_rig is not None
                    else {}
                )
                for cam_name in cam_names:
                    cam_img = first_imgs.get(cam_name)
                    if cam_img is None:
                        continue
                    np_img = _camera_image_to_numpy(cam_img)
                    image_panel_handles[cam_name] = server.gui.add_image(
                        np_img,
                        label=f"{cam_name} ({np_img.shape[1]}x{np_img.shape[0]})",
                    )

        current: dict[str, _FrameHandles] = {}
        handles_lock = threading.Lock()

        def _current_index() -> int:
            return gui_index.value if gui_index is not None else 0

        def _render(index: int) -> None:
            with handles_lock, server.atomic():
                old = current.pop("h", None)
                if old is not None:
                    old.remove_all()
                frame = frames[index]
                new = self._populate_frame(
                    server,
                    frame,
                    prefix="/frame",
                    colorize_lidar_from_images=gui_colorize.value,
                    show_camera_frustums=gui_show_frustums.value,
                    point_size=gui_point_size.value,
                    frustum_scale=gui_frustum_scale.value,
                )
                for h in new.point_clouds:
                    h.visible = gui_show_lidar.value
                for h in new.boxes:
                    h.visible = gui_show_boxes.value
                for h in new.frustums:
                    h.visible = gui_show_frustums.value
                current["h"] = new

                header = f"- frame: **{index + 1} / {n}**\n" if is_sequence else ""
                gui_summary.content = (
                    f"{header}"
                    f"- sequence_id: {frame.sequence_id}\n"
                    f"- sequence_idx: {frame.sequence_idx}\n"
                    f"- timestamp: {frame.timestamp}\n"
                    f"- lidar_sensors: {len(new.point_clouds)}\n"
                    f"- boxes: {sum(new.box_category_counts.values())}\n"
                )

                if frame.camera_rig is not None:
                    rig = frame.camera_rig._asdict()
                    for cam_name, panel in image_panel_handles.items():
                        cam_img = rig.get(cam_name)
                        if cam_img is not None:
                            panel.image = _camera_image_to_numpy(cam_img)

        if gui_index is not None:

            @gui_index.on_update
            def _(_) -> None:
                _render(gui_index.value)

        @gui_show_lidar.on_update
        def _(_) -> None:
            with handles_lock, server.atomic():
                h = current.get("h")
                if h is None:
                    return
                for pc in h.point_clouds:
                    pc.visible = gui_show_lidar.value

        @gui_point_size.on_update
        def _(_) -> None:
            with handles_lock, server.atomic():
                h = current.get("h")
                if h is None:
                    return
                for pc in h.point_clouds:
                    pc.point_size = gui_point_size.value

        @gui_show_boxes.on_update
        def _(_) -> None:
            with handles_lock, server.atomic():
                h = current.get("h")
                if h is None:
                    return
                for b in h.boxes:
                    b.visible = gui_show_boxes.value

        @gui_show_frustums.on_update
        def _(_) -> None:
            with handles_lock:
                h = current.get("h")
            if h is None or not h.frustums:
                _render(_current_index())
                return
            with handles_lock, server.atomic():
                for f in h.frustums:
                    f.visible = gui_show_frustums.value

        @gui_frustum_scale.on_update
        def _(_) -> None:
            with handles_lock:
                h = current.get("h")
            if h is None:
                return
            if h.frustums:
                with handles_lock, server.atomic():
                    for f in h.frustums:
                        f.scale = gui_frustum_scale.value
            else:
                _render(_current_index())

        @gui_colorize.on_update
        def _(_) -> None:
            _render(_current_index())

        _render(0)

        stop_event = threading.Event() if is_sequence else None
        if is_sequence:

            def _playback_loop() -> None:
                while not stop_event.is_set():
                    if gui_play.value:
                        next_idx = (gui_index.value + 1) % n
                        # Setting ``gui_index.value`` from the server side does
                        # NOT trigger ``on_update`` callbacks; re-render directly.
                        gui_index.value = next_idx
                        try:
                            _render(next_idx)
                        except Exception as exc:
                            print(f"  [warn] playback render failed: {exc}")
                    stop_event.wait(1.0 / max(gui_fps.value, 0.1))

            threading.Thread(target=_playback_loop, daemon=True).start()

        print(
            f"Viser {'sequence' if is_sequence else 'frame'} viewer started"
            f"{f' ({n} frames)' if is_sequence else ''}. Press Ctrl+C to stop."
        )
        try:
            server.sleep_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if stop_event is not None:
                stop_event.set()
