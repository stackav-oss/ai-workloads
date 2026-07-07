# LiDAR VAE

This repository contains a LiDAR-only VAE training bundle for reconstructing
NuScenes point clouds from learned neural feature grids. It includes the model
code, the embedded Jormungand NuScenes dataloader, Docker/Pixi environment
files, evaluation metrics, and saved GT/PRED renders.

## Repository Layout

- `autoencoder/`: LiDAR encoder, BEV VAE, neural render head, training script,
  metrics, checkpoints, and render previews.
- `autoencoder/train_lidar_only.py`: main training and validation entry point.
- `autoencoder/renders/`: GT vs PRED render images used in this README.
- `jormungand/`: local copy of the Jormungand dataset/runtime code needed by
  the training script.
- `jormungand/jormungand/datasets/nuscenes/`: NuScenes Lance conversion,
  loading, visualization, and utility code.
- `Dockerfile`, `pixi.toml`, `build_pig.sh`, `launch.sh`: reproducible local
  GPU environment.

## Setting Up The Repository

Run the commands below from the `lidar_vae` directory unless noted otherwise.

### Local Docker

Build the local GPU image:

```bash
bash build_pig.sh
```

Launch the container with GPU access, the NuScenes mount, and port `8080`
forwarded for Viser:

```bash
bash launch.sh
```

The launch script mounts this repository into the container at
`/project/ai-workloads/lidar_vae` and mounts the local NuScenes dataset at
`/data/nuscenes`. Override the Viser port with `LIDAR_VAE_HOST_PORT` if `8080`
is already in use:

```bash
LIDAR_VAE_HOST_PORT=8081 bash launch.sh
```

## Dataset

The repository uses a local copy of the Jormungand dataloader. NuScenes records
are exported to Lance tables, and the training script reads those tables through
`NuScenesLanceDataset`. Each Lance row represents one LiDAR scan. The exporter
keeps the full LiDAR frequency, matches the nearest camera frame per camera
channel, and stores poses, point clouds, image bytes, camera intrinsics, and box
metadata in one table.

The NuScenes release files included here are:

- `nuscenes_to_lance.py`: converts raw NuScenes data into Lance.
- `nuscenes_lance.py`: reads Lance tables and provides the visualization entry
  point.
- `nuscenes_local.py`: local NuScenes API wrapper used by the exporter.
- `nuscenes_dataframe_utils.py`: shared frame, sweep, rig, and aggregation
  utilities.
- `visualize.py`: visualization helper code.
- `nuscenes_class_to_colors.py`: semantic color map for NuScenes classes.

### Convert NuScenes To Lance

Download NuScenes locally so that the raw dataset is available under
`/data/nuscenes`, then export the train/val table:

```bash
python -m jormungand.datasets.nuscenes.nuscenes_to_lance \
  --version v1.0-trainval \
  --data-root /data/nuscenes \
  --output-path /data/nuscenes/v1.0-trainval.lance
```

If you run full test-set validation locally, export the test split as well:

```bash
python -m jormungand.datasets.nuscenes.nuscenes_to_lance \
  --version v1.0-test \
  --data-root /data/nuscenes \
  --output-path /data/nuscenes/v1.0-test.lance
```

The exporter overwrites existing output by default. Use `--no-overwrite` when
you want it to fail instead of replacing an existing Lance table.

### Visualize The Lance Dataset

Use the Lance reader to inspect frames in Viser:

```bash
python -m jormungand.datasets.nuscenes.nuscenes_lance \
  --version v1.0-trainval \
  --data-root /data/nuscenes \
  --index 0 \
  --num-frames 20 \
  --colorize-lidar \
  --frustum-scale 2.0
```

This opens a Viser server on `localhost:8080` inside the container. If you are
using `launch.sh`, that port is forwarded to the host.

## Model And Latent Shapes

The LiDAR-only path is:

```text
VoxelNet encoder -> 3D FPN -> sparse VFE decoder -> dense voxel feature grid
-> BEV pooling -> BEV encoder -> VAE resampling -> BEV decoder
-> neural feature grid -> ray sampling -> LiDAR reconstruction
```

Shape summary:

- Sparse LiDAR encoder/FPN produces a bottleneck around
  `(B, 256, 8, 128, 128)`.
- The VFE decoder reconstructs a dense feature grid shaped
  `(B, 16, 64, 1024, 1024)`. In BEV-friendly layout, this is
  `B x 1024 x 1024 x 16 x 64`.
- BEV pooling converts that grid to `(B, 64, 1024, 1024)`.
- The BEV encoder downsamples to the VAE latent:
  `(B, 32, 128, 128)`, or `128 x 128 x 32` per sample.
- The VAE samples `z` from `mu` and `log_var`, then the BEV decoder upsamples
  back to occupancy logits `(B, 64, 1024, 1024)` and a neural feature grid
  `(B, 16, 64, 1024, 1024)`.
- The render head samples rays through the neural feature grid and reconstructs
  depth, intensity, raydrop, occupancy, and SDF-related outputs.

## Training

`train_lidar_only.py` is already LiDAR-only, so no camera-disable flag is
required.

### Multi GPU

From `lidar_vae/autoencoder`, set the run name and Python path:

```bash
export RUN_NAME=ai_workloads_rec
mkdir -p logs checkpoints/${RUN_NAME}/lidar_only
export PYTHONPATH=$(pwd):$(pwd)/../jormungand:$(pwd)/../third_party/OpenSceneFlow:${PYTHONPATH}
```

Start four-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_lidar_only.py \
  --max-steps 200000 \
  --batch-size 2 \
  --checkpoint-dir checkpoints/${RUN_NAME}/lidar_only \
  --vae-warmup-steps 10000 \
  --kl-weight 5e-1 \
  --kl-weight-schedule cyclic \
  --kl-cycle-count 4 \
  --kl-cycle-ramp-fraction 0.5 \
  --lr 1e-3 \
  > logs/${RUN_NAME}.log 2>&1 < /dev/null &
```

Follow progress with `tail -f logs/${RUN_NAME}.log`.

### Single GPU

From `lidar_vae/autoencoder`:

```bash
export RUN_NAME=ai_workloads_rec
mkdir -p logs checkpoints/${RUN_NAME}/lidar_only
export PYTHONPATH=$(pwd):$(pwd)/../jormungand:$(pwd)/../third_party/OpenSceneFlow:${PYTHONPATH}

python train_lidar_only.py \
  --max-steps 200000 \
  --batch-size 2 \
  --checkpoint-dir checkpoints/${RUN_NAME}/lidar_only \
  --vae-warmup-steps 10000 \
  --kl-weight 5e-1 \
  --kl-weight-schedule cyclic \
  --kl-cycle-count 4 \
  --kl-cycle-ramp-fraction 0.5 \
  --lr 1e-3 \
  > logs/${RUN_NAME}.log 2>&1 < /dev/null &

echo $! > logs/${RUN_NAME}.pid
echo logs/${RUN_NAME}.log
```

Useful knobs:

- `--batch-size`: per-GPU batch size.
- `--num-sweeps`: number of LiDAR sweeps aggregated for encoder input.
- `--max-train-rays 0`: use all rays during training.
- `--test-val/--no-test-val`: enable or disable full test-set validation.
- `--test-metrics/--no-test-metrics`: enable or disable dense point-cloud
  validation metrics.
- `--resume <checkpoint>`: resume from a saved checkpoint.
- `--overfit-samples <N>`: debug on a small fixed subset.

## Results

Checkpoint:

```text
checkpoints/ai_workloads_rec/lidar_only/step_080000.pth
```

Training snapshot at step `80000/200000`, epoch `3`, learning rate
`6.57e-04`:

| Metric | Value |
| --- | ---: |
| depth_loss | 0.0576 |
| eikonal_loss | 0.0022 |
| free_space_loss | 0.0000 |
| intensity_loss | 0.1073 |
| kl_loss | 0.0096 |
| kl_loss_raw | 0.1030 |
| kl_weight | 0.0937 |
| kl_weight_scale | 0.9368 |
| latent_channel_mean_abs_p95 | 0.0251 |
| latent_channel_std_error_p95 | 0.0205 |
| latent_gaussian_score | 0.1486 |
| latent_mu_mean | 0.0039 |
| latent_mu_std | 0.3012 |
| latent_std_mean | 0.9341 |
| latent_std_std | 0.1737 |
| latent_true_kl | 0.1030 |
| latent_z_mean | 0.0039 |
| latent_z_std | 0.9970 |
| occupancy_loss | 0.0096 |
| raydrop_loss | 0.0274 |
| sdf_loss | 0.0001 |

Test-set validation at step `80000` over `3004` batches:

| Metric | Value |
| --- | ---: |
| depth_loss | 0.0615 |
| intensity_loss | 0.0897 |
| kl_loss | 0.0108 |
| kl_weight | 0.0947 |
| kl_weight_scale | 0.9473 |
| latent_channel_mean_abs_p95 | 0.2462 |
| latent_channel_std_error_p95 | 0.0639 |
| latent_gaussian_score | 0.4243 |
| latent_true_kl | 0.1142 |
| latent_z_mean | 0.0226 |
| latent_z_std | 1.0263 |
| occupancy_loss | 0.0101 |
| raydrop_loss | 0.0278 |
| point_metric_samples | 6008 |
| point_metric_valid_samples | 6008 |
| chamfer_symmetric_mean | 3.006666 |
| chamfer_symmetric_sqrt_m_mean | 1.675572 |
| jsd | 0.12198742 |
| mmd | 2.24676057e-04 |

## Visualizations

GT vs PRED point-cloud renders from `autoencoder/renders/`:

| Sample | GT | PRED |
| --- | --- | --- |
| 1 | <img src="autoencoder/renders/gt_1.png" width="360" alt="Ground truth sample 1"> | <img src="autoencoder/renders/pred_1.png" width="360" alt="Prediction sample 1"> |
| 2 | <img src="autoencoder/renders/gt_2.png" width="360" alt="Ground truth sample 2"> | <img src="autoencoder/renders/pred_2.png" width="360" alt="Prediction sample 2"> |
| 3 | <img src="autoencoder/renders/gt_3.png" width="360" alt="Ground truth sample 3"> | <img src="autoencoder/renders/pred_3.png" width="360" alt="Prediction sample 3"> |
| 4 | <img src="autoencoder/renders/gt_4.png" width="360" alt="Ground truth sample 4"> | <img src="autoencoder/renders/pred_4.png" width="360" alt="Prediction sample 4"> |
