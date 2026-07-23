"""NuScenes viser visualization entry point.

Usage::

    python -m jormungand.datasets.nuscenes.visualize \\
        --version v1.0-trainval --data-root /data/nuscenes \\
        --index 0 --num-frames 20 --frustum-scale 2.0 --colorize-lidar --fps 2
"""

from __future__ import annotations

import argparse
from pathlib import Path


from jormungand.jormungand.datastructures.sequential_data_renderer import (
    SequentialFrameRenderer,
)
from jormungand.datasets.nuscenes.nuscenes_class_to_colors import NUSCENES_COLORS
from jormungand.datasets.nuscenes.nuscenes_local import (
    CameraName,
    NuScenesDataConfig,
    NuScenesDataFrame,
    NuScenesLocalDataset,
)


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
        f"timestamp={frame.timestamp} lidar_points={lidar_points} "
        f"cameras={camera_names} boxes={num_boxes}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the NuScenes sequential dataset in viser."
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="NuScenes version folder name (e.g. v1.0-trainval, v1.0-mini).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/nuscenes"),
        help="NuScenes dataset root.",
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset index to load.")
    parser.add_argument(
        "--camera",
        action="append",
        choices=[camera_name.value for camera_name in CameraName],
        help="Camera to include. Repeat to load a subset. Defaults to all cameras.",
    )
    parser.add_argument(
        "--scene-name",
        action="append",
        help="Restrict to specific scene names (e.g. scene-0001). Repeat for multiple.",
    )
    parser.add_argument(
        "--no-lidar", action="store_true", help="Disable lidar loading."
    )
    parser.add_argument(
        "--no-bounding-boxes",
        action="store_true",
        help="Disable 3D bounding box loading.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=(
            "If set, render a sequence (video) of this many frames starting "
            "at --index instead of a single frame."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Initial playback FPS for sequence rendering.",
    )
    parser.add_argument(
        "--colorize-lidar",
        action="store_true",
        help="Colorize lidar points by projecting them into the camera images.",
    )
    parser.add_argument(
        "--no-frustums",
        action="store_true",
        help="Disable per-camera frustum rendering.",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.5,
        help="Initial size of the camera frustums (in meters).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = (
        NuScenesDataConfig.from_requested_cameras(
            args.camera,
            lidar=not args.no_lidar,
            bounding_boxes_3d=not args.no_bounding_boxes,
        )
        if args.camera is not None
        else NuScenesDataConfig(
            lidar=not args.no_lidar,
            bounding_boxes_3d=not args.no_bounding_boxes,
        )
    )

    dataset = NuScenesLocalDataset(
        data_config=config,
        version=args.version,
        data_root=args.data_root,
        scene_names=args.scene_name,
    )
    print(f"dataset_size={len(dataset)}")

    frame = dataset[args.index]
    print(_frame_summary(frame))

    renderer = SequentialFrameRenderer(category_colors=NUSCENES_COLORS)

    if args.num_frames is not None:
        renderer.render_sequence(
            frame,
            num_frames=args.num_frames,
            server_label=f"NuScenes seq @ {frame.sequence_idx}",
            colorize_lidar_from_images=args.colorize_lidar,
            show_camera_frustums=not args.no_frustums,
            frustum_scale=args.frustum_scale,
            fps=args.fps,
        )
    else:
        renderer.render(
            frame,
            server_label=f"NuScenes: [{frame.sequence_idx}]",
            colorize_lidar_from_images=args.colorize_lidar,
            show_camera_frustums=not args.no_frustums,
            frustum_scale=args.frustum_scale,
        )


if __name__ == "__main__":
    main()
