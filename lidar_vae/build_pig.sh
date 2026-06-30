#!/bin/bash
set -e

# Build the lidar_vae Docker image.
# Run from this repo root: bash build_pig.sh

docker build \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  --build-arg USERNAME="$(id -un)" \
  -f Dockerfile . \
  -t lidar_vae:latest
