#!/bin/bash
# COSMOS Transfer Runtime Benchmarking Inference Script
# Handles GPU-specific setup and distributed inference for control types.

set -euo pipefail

# --- Configuration & Validation ---

source "$HOME/.local/bin/env"

# Ensure environment is ready
[[ -z "${HF_TOKEN:-}" ]] && { echo "Error: HF_TOKEN is not set."; exit 1; }

# Paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Hardware Detection ---

GPU_MODEL=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected $NUM_GPUS x '$GPU_MODEL'"
[[ $NUM_GPUS -lt 1 ]] && { echo "Error: NVIDIA GPU required."; exit 1; }
GPU_CONFIGURATIONS=(4 2 1)
if [[ $NUM_GPUS -gt 4 ]]; then
    GPU_CONFIGURATIONS=(8 4 2 1)
fi
echo "GPU configurations to test: ${GPU_CONFIGURATIONS[*]}"

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

# --- Environment Setup ---

log() { echo -e "\n--- $1 ---"; }

# Output directory for results
tmp_dir="/tmp/throughput"


# --- Inference Execution ---

log "Syncing UV environment ($UV_EXTRA)"
cd /cosmos-transfer2.5
uv sync --python 3.10 --extra="$UV_EXTRA"
source .venv/bin/activate


run_throughput_test() {
    local root_dir="$1"
    local nproc="$2"
    local output_dir="$3"
    mkdir -p "$output_dir"
    
    # Clean up intermediate outputs
    rm -rf "$output_dir"/*.mp4
    torchrun --nproc_per_node="$nproc" --master_port=12341 "$root_dir/throughput.py" --disable-guardrails -o "$output_dir" "control:edge"
    rm -rf "$output_dir"/*.mp4
    torchrun --nproc_per_node="$nproc" --master_port=12341 "$root_dir/throughput.py" --disable-guardrails -o "$output_dir" "control:vis"
    rm -rf "$output_dir"/*.mp4
    torchrun --nproc_per_node="$nproc" --master_port=12341 "$root_dir/throughput.py" --disable-guardrails -o "$output_dir" "control:depth"
    rm -rf "$output_dir"/*.mp4
    torchrun --nproc_per_node="$nproc" --master_port=12341 "$root_dir/throughput.py" --disable-guardrails -o "$output_dir" "control:seg"
    rm -rf "$output_dir"/*.mp4
    torchrun --nproc_per_node="$nproc" --master_port=12341 "$root_dir/throughput.py" --disable-guardrails -o "$output_dir" "control:all"
    # Clean up final outputs
    rm -rf "$output_dir"/*.mp4
}

for nproc in "${GPU_CONFIGURATIONS[@]}"; do
    if ! run_throughput_test "$ROOT_DIR" "$nproc" "$tmp_dir"; then
        echo "Warning: Test with $nproc GPUs failed"
    fi
done

log "Generating throughput summary"
python "$ROOT_DIR/generate_throughput_summary.py" "$tmp_dir"

deactivate
