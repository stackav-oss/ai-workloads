#!/bin/bash
# COSMOS Transfer Inference Script
# Handles GPU-specific setup and distributed inference for control types.

set -euo pipefail

# --- Configuration & Validation ---

source "$HOME/.local/bin/env"

# Ensure environment is ready
[[ -z "${HF_TOKEN:-}" ]] && { echo "Error: HF_TOKEN is not set."; exit 1; }

# Validate control type argument
CONTROL_TYPE="${1:-}"
VALID_TYPES=("edge" "vis" "depth" "seg" "all")

if [[ ! " ${VALID_TYPES[@]} " =~ " ${CONTROL_TYPE} " ]]; then
    echo "Usage: $0 [${VALID_TYPES[*]// /|}]"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_BASE="/results/transfer/$CONTROL_TYPE"
INFERENCE_DIR="$RESULT_BASE/inference"

# --- Hardware Detection ---

GPU_MODEL=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected $NUM_GPUS x '$GPU_MODEL'"
[[ $NUM_GPUS -lt 1 ]] && { echo "Error: NVIDIA GPU required."; exit 1; }

# Determine CUDA versions based on GPU model
if [[ "$GPU_MODEL" == *"GB200"* ]]; then
    CUDA_VER="13"
    CUDA_TOOLKIT="cuda-toolkit-13-0"
    UV_EXTRA="cu130"
else
    CUDA_VER="12"
    CUDA_TOOLKIT="cuda-toolkit-12-8"
    UV_EXTRA="cu128"
fi

#CUDA_DIR="/usr/local/cuda-$CUDA_VER"

# --- Environment Setup ---

log() { echo -e "\n--- $1 ---"; }

log "Setting up system directories"
mkdir -p /datasets/results
ln -sfn /datasets/results /results
mkdir -p "$INFERENCE_DIR"
#rm -rf "${INFERENCE_DIR:?}"/*

log "Ensuring CUDA $CUDA_VER and UV environment"
#if [[ ! -d "$CUDA_DIR" ]]; then
#rm -rf /usr/local/cuda-12; rm -rf /usr/local/cuda-12.8/;
#rm -rf /usr/local/cuda-13; rm -rf /usr/local/cuda-13.0/;

# uninstall existing CUDA installations to avoid conflicts
apt-get purge -y --remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
apt-get autoremove -y  --purge
apt-get autoclean -y 

# install cuda toolkit
ARCH=$(uname -m)
REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH/aarch64/sbsa}/cuda-keyring_1.1-1_all.deb"
wget -q "$REPO_URL" -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt update
apt install -y "$CUDA_TOOLKIT"
rm cuda-keyring.deb

#fi

# Update system-wide CUDA symlink
#ln -sfn "$CUDA_DIR" /etc/alternatives/cuda

# --- Inference Execution ---

log "Syncing UV environment ($UV_EXTRA)"
cd /cosmos-transfer2.5
uv sync --python 3.10 --extra="$UV_EXTRA"
source .venv/bin/activate

log "Launching distributed inference: $CONTROL_TYPE"
torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 "$SCRIPT_DIR/inference.py" --disable-guardrails -o "$INFERENCE_DIR/caption_0" --caption-variant 0 "control:$CONTROL_TYPE" || true
torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 "$SCRIPT_DIR/inference.py" --disable-guardrails -o "$INFERENCE_DIR/caption_1" --caption-variant 1 "control:$CONTROL_TYPE" || true
torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 "$SCRIPT_DIR/inference.py" --disable-guardrails -o "$INFERENCE_DIR/caption_2" --caption-variant 2 "control:$CONTROL_TYPE" || true
torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 "$SCRIPT_DIR/inference.py" --disable-guardrails -o "$INFERENCE_DIR/caption_3" --caption-variant 3 "control:$CONTROL_TYPE" || true
torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 "$SCRIPT_DIR/inference.py" --disable-guardrails -o "$INFERENCE_DIR/caption_4" --caption-variant 4 "control:$CONTROL_TYPE" || true
torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 "$SCRIPT_DIR/inference.py" --disable-guardrails -o "$INFERENCE_DIR/caption_5" --caption-variant 5 "control:$CONTROL_TYPE" || true

deactivate
echo -e "\nInference completed. Results in: $INFERENCE_DIR"
