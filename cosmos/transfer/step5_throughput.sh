#!/bin/bash
# COSMOS Transfer Runtime Benchmarking Inference Script
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


# --- Environment Setup ---

log() { echo -e "\n--- $1 ---"; }

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


# --- Inference Execution ---

log "Syncing UV environment ($UV_EXTRA)"
cd /cosmos-transfer2.5
uv sync --python 3.10 --extra="$UV_EXTRA"
source .venv/bin/activate

# NUM_GPUS
#CONTROL_TYPE
log "Launching distributed inference: $CONTROL_TYPE"
torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 "$SCRIPT_DIR/inference.py" --disable-guardrails -o /tmp "control:$CONTROL_TYPE"

deactivate
