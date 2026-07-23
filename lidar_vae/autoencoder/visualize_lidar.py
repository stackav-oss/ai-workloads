"""Viser visualization of GT vs predicted LiDAR point clouds.

Loads .npz files saved during validation and displays them as a sequence
in a 3D interactive viewer using viser.

Usage (inside lidar_vae docker):
    python autoencoder/visualize_lidar.py \
        --points-dir autoencoder/checkpoints/lidar_only/val_renders/points

Controls:
    - Use the slider to step through samples
    - Toggle between GT and Prediction views
    - Points are colored by intensity (if available) or height
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import viser

try:
    from .lidar_range_image import save_range_projection
except ImportError:
    from lidar_range_image import save_range_projection


def height_colormap(points: np.ndarray) -> np.ndarray:
    """Color points by height (z-coordinate): blue (low) → red (high)."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-6:
        return np.full((len(points), 3), 128, dtype=np.uint8)
    normalized = (z - z_min) / (z_max - z_min)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[:, 0] = (normalized * 255).astype(np.uint8)  # Red
    colors[:, 1] = ((1 - np.abs(normalized - 0.5) * 2) * 200).astype(np.uint8)  # Green
    colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)  # Blue
    return colors


def intensity_colormap(intensity: np.ndarray) -> np.ndarray:
    """Color points by intensity: dark → bright yellow."""
    intensity = np.asarray(intensity).reshape(-1)
    if len(intensity) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    i_norm = np.clip(intensity, 0, 1)
    colors = np.zeros((len(intensity), 3), dtype=np.uint8)
    colors[:, 0] = (i_norm * 255).astype(np.uint8)
    colors[:, 1] = (i_norm * 255).astype(np.uint8)
    colors[:, 2] = (i_norm * 80).astype(np.uint8)
    return colors


def _pred_intensity_for_points(
    sample: dict[str, np.ndarray], pred_key: str, pred_xyz: np.ndarray
) -> np.ndarray | None:
    """Return predicted intensities aligned with the point array being displayed."""
    if pred_key == "pred_xyz_filtered":
        if "pred_intensity_filtered" in sample:
            intensity = np.asarray(sample["pred_intensity_filtered"]).reshape(-1)
            if len(intensity) == len(pred_xyz):
                return intensity

        if "pred_intensity" in sample and "pred_raydrop" in sample:
            intensity = np.asarray(sample["pred_intensity"]).reshape(-1)
            raydrop = np.asarray(sample["pred_raydrop"]).reshape(-1)
            if len(intensity) == len(raydrop):
                filtered_intensity = intensity[raydrop < 0.5]
                if len(filtered_intensity) == len(pred_xyz):
                    return filtered_intensity

    if "pred_intensity" in sample:
        intensity = np.asarray(sample["pred_intensity"]).reshape(-1)
        if len(intensity) == len(pred_xyz):
            return intensity

    return None


def _gt_intensity_for_points(
    sample: dict[str, np.ndarray], gt_xyz: np.ndarray
) -> np.ndarray | None:
    if "gt_intensity" not in sample:
        return None
    intensity = np.asarray(sample["gt_intensity"]).reshape(-1)
    if len(intensity) != len(gt_xyz):
        return None
    return intensity


def _raydrop_filtered_prediction(
    sample: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None] | None:
    if "pred_xyz_filtered" in sample:
        pred_xyz = np.asarray(sample["pred_xyz_filtered"], dtype=np.float32)
        return pred_xyz, _pred_intensity_for_points(
            sample,
            "pred_xyz_filtered",
            pred_xyz,
        )

    if "pred_xyz" not in sample:
        return None

    pred_xyz = np.asarray(sample["pred_xyz"], dtype=np.float32)
    pred_intensity = _pred_intensity_for_points(sample, "pred_xyz", pred_xyz)
    if "pred_raydrop" not in sample:
        return pred_xyz, pred_intensity

    raydrop = np.asarray(sample["pred_raydrop"]).reshape(-1)
    if len(raydrop) != len(pred_xyz):
        return pred_xyz, pred_intensity

    keep = raydrop < 0.5
    filtered_xyz = pred_xyz[keep]
    filtered_intensity = pred_intensity[keep] if pred_intensity is not None else None
    return filtered_xyz, filtered_intensity


def _nonempty_xyz(sample: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key not in sample:
        return None
    xyz = np.asarray(sample[key], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or xyz.shape[0] == 0:
        return None
    return xyz[:, :3]


def _first_nonempty_xyz(sample: dict[str, np.ndarray]) -> np.ndarray | None:
    for key in ("gt_xyz", "pred_xyz_filtered", "pred_xyz", "gt_all_xyz"):
        xyz = _nonempty_xyz(sample, key)
        if xyz is not None:
            return xyz
    return None


def _prediction_display_cloud(
    sample: dict[str, np.ndarray],
) -> tuple[str, np.ndarray, np.ndarray | None] | None:
    pred_key = "pred_xyz_filtered"
    pred_xyz = _nonempty_xyz(sample, pred_key)
    if pred_xyz is None:
        pred_key = "pred_xyz"
        pred_xyz = _nonempty_xyz(sample, pred_key)
    if pred_xyz is None:
        return None
    return pred_key, pred_xyz, _pred_intensity_for_points(sample, pred_key, pred_xyz)


def _scalar_metadata(sample: dict[str, np.ndarray], key: str) -> float | None:
    if key not in sample:
        return None
    value = np.asarray(sample[key])
    if value.size != 1:
        return None
    return float(value.reshape(-1)[0])


def _save_validation_range_images(
    *,
    sample_files: list[Path],
    output_dir: Path,
    height: int,
    width: int,
    fov_up_deg: float,
    fov_down_deg: float,
) -> None:
    saved_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with click.progressbar(sample_files, label="Saving range images") as files:
        for sample_path in files:
            sample = dict(np.load(str(sample_path), allow_pickle=True))
            stem = sample_path.stem

            if "gt_xyz" in sample:
                gt_xyz = np.asarray(sample["gt_xyz"], dtype=np.float32)
                save_range_projection(
                    points=gt_xyz[:, :3],
                    remissions=_gt_intensity_for_points(sample, gt_xyz),
                    output_prefix=output_dir / f"{stem}_gt_range",
                    height=height,
                    width=width,
                    fov_up_deg=fov_up_deg,
                    fov_down_deg=fov_down_deg,
                )
                saved_count += 1

            filtered_pred = _raydrop_filtered_prediction(sample)
            if filtered_pred is not None:
                pred_xyz, pred_intensity = filtered_pred
                save_range_projection(
                    points=pred_xyz[:, :3],
                    remissions=pred_intensity,
                    output_prefix=output_dir / f"{stem}_pred_raydrop_range",
                    height=height,
                    width=width,
                    fov_up_deg=fov_up_deg,
                    fov_down_deg=fov_down_deg,
                )
                saved_count += 1

            if "pred_xyz" in sample and "pred_xyz_filtered" in sample:
                pred_xyz = np.asarray(sample["pred_xyz"], dtype=np.float32)
                save_range_projection(
                    points=pred_xyz[:, :3],
                    remissions=_pred_intensity_for_points(
                        sample,
                        "pred_xyz",
                        pred_xyz,
                    ),
                    output_prefix=output_dir / f"{stem}_pred_no_raydrop_range",
                    height=height,
                    width=width,
                    fov_up_deg=fov_up_deg,
                    fov_down_deg=fov_down_deg,
                )
                saved_count += 1

    click.echo(f"Saved {saved_count} range-image PNG/NPZ pairs to {output_dir}")


@click.command()
@click.option(
    "--points-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing .npz point cloud files from validation.",
)
@click.option("--point-size", default=0.03, help="Point size in the viewer.")
@click.option(
    "--range-image-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional directory for saved range-image PNG/NPZ pairs.",
)
@click.option("--range-image-height", type=int, default=32)
@click.option("--range-image-width", type=int, default=1070)
@click.option("--range-fov-up", type=float, default=10.0)
@click.option("--range-fov-down", type=float, default=-30.0)
@click.option(
    "--range-images-only/--no-range-images-only",
    default=False,
    help="Save range images and exit without starting the viser viewer.",
)
def main(
    points_dir: Path,
    point_size: float,
    range_image_dir: Path | None,
    range_image_height: int,
    range_image_width: int,
    range_fov_up: float,
    range_fov_down: float,
    range_images_only: bool,
) -> None:
    """Visualize GT vs predicted LiDAR point clouds from validation."""

    # Discover all unique step numbers from filenames (step{N:06d}_sample{M:02d}.npz)
    all_npz = sorted(points_dir.glob("*.npz"))
    if not all_npz:
        print(f"No .npz files found in {points_dir}")
        return

    if range_image_dir is not None:
        _save_validation_range_images(
            sample_files=all_npz,
            output_dir=range_image_dir,
            height=range_image_height,
            width=range_image_width,
            fov_up_deg=range_fov_up,
            fov_down_deg=range_fov_down,
        )
        if range_images_only:
            return

    step_numbers = sorted(
        {int(f.stem.split("_")[1].replace("step", "")) for f in all_npz}
    )
    print(f"Found steps: {step_numbers}")

    def load_samples(step: int) -> list[dict]:
        files = sorted(points_dir.glob(f"*{step:06d}_*.npz"))
        result = []
        for f in files:
            data = dict(np.load(str(f), allow_pickle=True))
            data["filename"] = f.stem
            result.append(data)
        return result

    samples = load_samples(step_numbers[0])
    current_step = [step_numbers[0]]
    is_posterior_std_sweep = any(
        str(np.asarray(sample.get("source_mode", "")).item()) == "posterior_std_sweep"
        for sample in samples
        if "source_mode" in sample
    )

    def step_display_text(step: int, step_samples: list[dict]) -> str:
        if not is_posterior_std_sweep:
            return f"**Step: {step}**"
        scale = None
        for sample in step_samples:
            scale = _scalar_metadata(sample, "posterior_std_scale")
            if scale is not None:
                break
        if scale is None:
            scale = step / 1000.0
        return f"**Std scale: {scale:g} x posterior std**"

    server = viser.ViserServer(label="LiDAR GT vs Prediction")
    server.scene.set_up_direction("+z")

    # Compute scene extent from the first non-empty point cloud for camera positioning.
    first_xyz = None
    for sample in samples:
        first_xyz = _first_nonempty_xyz(sample)
        if first_xyz is not None:
            break
    if first_xyz is not None:
        scene_center = first_xyz.mean(axis=0)
        scene_extent = float(np.max(first_xyz.max(axis=0) - first_xyz.min(axis=0)))
        if not np.isfinite(scene_extent) or scene_extent < 1e-6:
            scene_extent = 50.0
    else:
        scene_center = np.zeros(3)
        scene_extent = 50.0

    server.initial_camera.look_at = tuple(scene_center.tolist())
    server.initial_camera.position = tuple(
        (
            scene_center
            + np.array([1.5 * scene_extent, -1.5 * scene_extent, 0.5 * scene_extent])
        ).tolist()
    )

    print("\nViser server running — open the viewer in your browser.")

    # State
    current_idx = [0]
    show_gt = [True]
    show_pred = [True]
    show_pred_no_raydrop = [False]
    use_intensity_color = [True]
    handles: dict = {}

    def update_scene():
        """Update the 3D scene based on current state."""
        # Remove old handles
        for key in list(handles.keys()):
            handles.pop(key).remove()

        sample = samples[current_idx[0]]

        # GT point cloud
        if show_gt[0] and "gt_xyz" in sample:
            gt_xyz = _nonempty_xyz(sample, "gt_xyz")
            if gt_xyz is None:
                gt_xyz = np.empty((0, 3), dtype=np.float32)
            gt_intensity = (
                np.asarray(sample["gt_intensity"]).reshape(-1)
                if "gt_intensity" in sample
                else None
            )
            if (
                use_intensity_color[0]
                and gt_intensity is not None
                and len(gt_intensity) == len(gt_xyz)
            ):
                gt_colors = intensity_colormap(gt_intensity)
            elif len(gt_xyz) > 0:
                gt_colors = height_colormap(gt_xyz)
            if len(gt_xyz) > 0:
                handles["gt"] = server.scene.add_point_cloud(
                    name="/gt",
                    points=gt_xyz,
                    colors=gt_colors,
                    point_size=point_size,
                    point_shape="rounded",
                )

        # Predicted point cloud (filtered by raydrop)
        if show_pred[0]:
            pred_display = _prediction_display_cloud(sample)
            if pred_display is not None:
                pred_key, pred_xyz, pred_intensity = pred_display
                pred_intensity = _pred_intensity_for_points(sample, pred_key, pred_xyz)
                if use_intensity_color[0] and pred_intensity is not None:
                    pred_colors = intensity_colormap(pred_intensity)
                else:
                    pred_colors = height_colormap(pred_xyz)
                handles["pred"] = server.scene.add_point_cloud(
                    name="/pred_raydrop",
                    points=pred_xyz,
                    colors=pred_colors,
                    point_size=point_size,
                    point_shape="rounded",
                )

        # Predicted point cloud before raydrop filtering
        if show_pred_no_raydrop[0] and "pred_xyz" in sample:
            pred_xyz = _nonempty_xyz(sample, "pred_xyz")
            if pred_xyz is not None:
                pred_intensity = _pred_intensity_for_points(
                    sample, "pred_xyz", pred_xyz
                )
                if use_intensity_color[0] and pred_intensity is not None:
                    pred_colors = intensity_colormap(pred_intensity)
                else:
                    pred_colors = height_colormap(pred_xyz)
                handles["pred_no_raydrop"] = server.scene.add_point_cloud(
                    name="/pred_no_raydrop",
                    points=pred_xyz,
                    colors=pred_colors,
                    point_size=point_size,
                    point_shape="rounded",
                )

    # Add GUI controls
    with server.gui.add_folder("Controls"):
        step_slider = server.gui.add_slider(
            "Std Scale" if is_posterior_std_sweep else "Training Step",
            min=0,
            max=len(step_numbers) - 1,
            step=1,
            initial_value=0,
        )
        step_label = server.gui.add_markdown(
            step_display_text(step_numbers[0], samples)
        )
        slider = server.gui.add_slider(
            "Sample Index",
            min=0,
            max=len(samples) - 1,
            step=1,
            initial_value=0,
        )
        gt_checkbox = server.gui.add_checkbox("Show GT", initial_value=True)
        pred_checkbox = server.gui.add_checkbox(
            "Show Prediction (Raydrop)", initial_value=True
        )
        pred_no_raydrop_checkbox = server.gui.add_checkbox(
            "Show Prediction (No Raydrop)", initial_value=False
        )
        intensity_checkbox = server.gui.add_checkbox(
            "Color by Intensity", initial_value=True
        )
        info_label = server.gui.add_markdown(f"**{samples[0]['filename']}**")

    @step_slider.on_update
    def _on_step_slider(event: viser.GuiEvent) -> None:
        step_idx = int(step_slider.value)
        new_step = step_numbers[step_idx]
        current_step[0] = new_step
        new_samples = load_samples(new_step)
        samples.clear()
        samples.extend(new_samples)
        step_label.content = step_display_text(new_step, samples)
        # Clamp sample index in case new step has fewer samples
        current_idx[0] = min(current_idx[0], len(samples) - 1)
        slider.max = len(samples) - 1
        slider.value = current_idx[0]
        info_label.content = f"**{samples[current_idx[0]]['filename']}**"
        update_scene()

    @slider.on_update
    def _on_slider(event: viser.GuiEvent) -> None:
        current_idx[0] = int(slider.value)
        info_label.content = f"**{samples[current_idx[0]]['filename']}**"
        update_scene()

    @gt_checkbox.on_update
    def _on_gt(event: viser.GuiEvent) -> None:
        show_gt[0] = gt_checkbox.value
        update_scene()

    @pred_checkbox.on_update
    def _on_pred(event: viser.GuiEvent) -> None:
        show_pred[0] = pred_checkbox.value
        update_scene()

    @pred_no_raydrop_checkbox.on_update
    def _on_pred_no_raydrop(event: viser.GuiEvent) -> None:
        show_pred_no_raydrop[0] = pred_no_raydrop_checkbox.value
        update_scene()

    @intensity_checkbox.on_update
    def _on_intensity(event: viser.GuiEvent) -> None:
        use_intensity_color[0] = intensity_checkbox.value
        update_scene()

    # Initial render
    update_scene()

    print("\nControls:")
    print("  - Training Step slider: switch between checkpoints")
    print("  - Sample Index slider: navigate between validation samples within a step")
    print("  - Show GT: toggle ground truth point cloud")
    print("  - Show Prediction (Raydrop): toggle raydrop-filtered prediction")
    print("  - Show Prediction (No Raydrop): toggle full prediction before raydrop")
    print("  - Color by Intensity: toggle intensity vs height coloring")

    # Keep server alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
