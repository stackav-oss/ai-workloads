#!/bin/bash
# Launch the lidar_vae Docker container with NuScenes data and GPU access.
# Run from this repo root: bash launch.sh

USER_NAME="$(id -un)"
DATASET_ROOT="/ephemeral/datasets/nuscenes/nuscenes/nuscenes"
HOST_PORT="${LIDAR_VAE_HOST_PORT:-8080}"

touch "$(pwd)/docker_history.txt"
docker run --gpus=all --rm -it \
  --shm-size=16gb \
  -v "$(pwd)":/project/ai-workloads/lidar_vae \
  -v /tmp:/tmp \
  -v "$(pwd)/docker_history.txt":/home/"$USER_NAME"/.bash_history \
  -v ~/.gitconfig:/home/"$USER_NAME"/.gitconfig:ro \
  -v ~/.ssh:/home/"$USER_NAME"/.ssh:ro \
  -v "$HOME/.aws:/home/$USER_NAME/.aws:ro" \
  -v "$HOME/.volt:/home/$USER_NAME/.volt" \
  -v "$DATASET_ROOT":/data/nuscenes \
  -e DISPLAY="$DISPLAY" \
  -e HISTFILE="/home/$USER_NAME/.bash_history" \
  -e VOLT_CONFIG_DIR="/home/$USER_NAME/.volt/config" \
  -e AWS_PROFILE="magic" \
  -p "$HOST_PORT":8080 \
  -h "$HOSTNAME" \
  lidar_vae:latest
