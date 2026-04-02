set -euo pipefail  # Exit on error, undefined variables, and pipe failures

source $HOME/.local/bin/env

if [ $# -eq 0 ]; then
    echo "Error: Please provide an control type [edge|vis|depth|seg]"
    exit 1
fi

control_type=$1
if [ "$control_type" != "edge" ] && [ "$control_type" != "vis" ] && [ "$control_type" != "depth" ] && [ "$control_type" != "seg" ] && [ "$control_type" != "all" ]; then
    echo "Error: Invalid control type. Must be 'edge', 'vis', 'depth', 'seg', or 'all'"
    exit 1
fi
echo "Control type: $control_type"


# Configuration
PAI_DIR="/physical-ai-bench/conditional_generation"

# Get script directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script located in: $ROOT_DIR"

cd "$PAI_DIR"

# Functions
check_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "Error: HF_TOKEN is either unset or empty" >&2
        echo "Please set your Hugging Face token: export HF_TOKEN=your_token_here" >&2
        exit 1
    fi
}

get_gpu_info() {
    local gpu_model
    gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0)
    echo "Detected GPU: '$gpu_model'"
    export GPU_MODEL="$gpu_model"
}


# Setup and validation
check_hf_token
get_gpu_info

# remove existing CUDA installations to avoid conflicts
apt-get purge -y --remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
apt-get autoremove -y  --purge
apt-get autoclean -y 

# Install CUDA toolkit 12.8
arch=$(uname -m)
if [ "$arch" == "aarch64" ]; then
    echo "Running on ARM architecture (aarch64)"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
else
    echo "Running on x86_64 architecture"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
fi
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt -y install cuda-toolkit-12-8

# Handle aarch64 specific requirements
if [ "$arch" == "aarch64" ]; then
    # decord is not compatible with aarch64 architecture, use decord2 instead which is a fork of decord with aarch64 support
    sed -i -e 's/"decord"/"decord2"/g'  "$PAI_DIR/pyproject.toml"
fi

uv sync --python 3.10
uv pip install setuptools
uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --reinstall -e third_party/Grounded-SAM-2
uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --reinstall --no-build-isolation -e third_party/Grounded-SAM-2/grounding_dino
uv pip install --reinstall huggingface-hub==0.36.0
uv pip install --reinstall xformers==0.0.35
uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

source .venv/bin/activate
bash get_checkpoint.sh

uv pip install --reinstall huggingface-hub

available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: $available_gpus"
if [ $available_gpus -lt 1 ]; then
    echo "Error: At least 1 NVIDIA GPUs are required"
    exit 1
fi

# Fix for transformers image processing utils
sed -i -e 's/torch.from_numpy(image).contiguous()/torch.from_numpy(image.copy()).contiguous()/g'  "/physical-ai-bench/conditional_generation/.venv/lib/python3.10/site-packages/transformers/image_processing_utils_fast.py"


for caption_id in {0..5}; do
    for video_prefix in {000..059}; do
        mkdir -p "/results/transfer/${control_type}/evaluation/caption_${caption_id}"
        metrics_file="/results/transfer/${control_type}/evaluation/caption_${caption_id}/metrics_${video_prefix}.json"
        if [ ! -f "$metrics_file" ]; then
            mkdir -p /batches/videos/
            rm -rf /batches/videos/*
            cp /results/transfer/${control_type}/inference/caption_${caption_id}/task_${video_prefix}?.mp4 /batches/videos/ || true

            # Only run if mp4 files exist
            if ls /batches/videos/*.mp4 1> /dev/null 2>&1; then
                python -m torch.distributed.run --standalone --nproc_per_node $available_gpus compute_metrics.py calculate-metrics \
                --gt_path /datasets/physical-ai-bench-conditional-generation \
                --videos_path  /batches/ --output_path "$metrics_file"
                sleep 60 # Sleep to ensure that previous processes have released GPU memory before starting the next one
            fi
        else
            echo "Metrics file $metrics_file already exists, skipping."
        fi
    done
done


python "/$ROOT_DIR/generate_evaluation_results.py" --metrics_dir "/results/transfer/${control_type}/evaluation/"
deactivate