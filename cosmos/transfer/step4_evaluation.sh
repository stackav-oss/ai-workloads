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

if [ "$GPU_MODEL" == "NVIDIA GB200" ]; then
    echo "Configuring for NVIDIA GB200"
    sed -i -e 's/"decord"/"decord2"/g'  "$PAI_DIR/pyproject.toml"
fi

uv sync --python 3.10
uv pip install setuptools
if [ "$GPU_MODEL" == "NVIDIA GB200" ]; then
    echo "Using PyTorch with CUDA 13.0 for NVIDIA GB200"
    #uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
    uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
else
    echo "Using PyTorch with CUDA 12.8 for NVIDIA H100"
    uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
fi

uv pip install -e third_party/Grounded-SAM-2
uv pip install --no-build-isolation -e third_party/Grounded-SAM-2/grounding_dino
uv pip install --reinstall huggingface-hub==0.36.0
source .venv/bin/activate
bash get_checkpoint.sh

uv pip install --reinstall xformers==0.0.35
if [ "$GPU_MODEL" == "NVIDIA GB200" ]; then
    echo "Using PyTorch with CUDA 13.0 for NVIDIA GB200"
    #uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
else
    echo "Using PyTorch with CUDA 12.8 for NVIDIA H100"
    uv pip install --reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
fi
uv pip install --reinstall huggingface-hub


available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: $available_gpus"
if [ $available_gpus -lt 1 ]; then
    echo "Error: At least 1 NVIDIA GPUs are required"
    exit 1
fi

sed -i -e 's/torch.from_numpy(image).contiguous()/torch.from_numpy(image.copy()).contiguous()/g'  "/physical-ai-bench/conditional_generation/.venv/lib/python3.10/site-packages/transformers/image_processing_utils_fast.py"


for video_prefix in {00..05}; do
    mkdir -p /batches/videos/
    rm -rf /batches/videos/*
    cp /results/transfer/${control_type}/inference/task_${video_prefix}??.mp4 /batches/videos/
    
    #cp /results/transfer/${control_type}/inference/task_${video_prefix}??_caption?.mp4 /batches/videos/
    #rename -v 's/(task_[0-9A-Za-z]+)_caption([0-9]+)\.mp4/$1__$2_caption$2.mp4/' /batches/videos/*.mp4
    
    python -m torch.distributed.run --standalone --nproc_per_node 4 compute_metrics.py calculate-metrics \
    --gt_path /datasets/physical-ai-bench-conditional-generation \
    --videos_path  /batches/ --output_path "/results/transfer/${control_type}/metrics_${video_prefix}.json"
done


python generate_evaluation_results.py --metrics_dir "/results/transfer/${control_type}"
