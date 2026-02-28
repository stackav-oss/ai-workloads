source $HOME/.local/bin/env

# Check if HF_TOKEN is set and not empty
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN is either unset or empty"
  exit 1
fi

gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0)
echo "Detected GPU: '$gpu_model'"

available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: $available_gpus"
if [ $available_gpus -lt 1 ]; then
    echo "Error: At least 1 NVIDIA GPUs are required"
    exit 1
fi


# Get the absolute path of the directory containing this script
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "The script is located in: ${ROOT_DIR}"


echo  "Running inference and benchmarking for inference type: $inference_type"

# Create result directories
mkdir -p /results/transfer/depth/inference

# clean previous results
rm /results/transfer/depth/inference/*  > /dev/null 2>&1  || true


# Run inference using custom pyton script mounted from repo
cd /cosmos-transfer2.5

if [ "$gpu_model" == "NVIDIA GB200" ]; then
    echo "Using PyTorch with CUDA 13.0 for NVIDIA GB200"
    uv sync --python 3.10 --extra=cu130   > /dev/null 2>&1  || true
else
    echo "Using PyTorch with CUDA 12.9 for NVIDIA H100"
    uv sync --python 3.10 --extra=cu128   > /dev/null 2>&1  || true
fi
source .venv/bin/activate

torchrun --nproc_per_node=$available_gpus "/$ROOT_DIR/inference.py" --inference-type $inference_type --disable-guardrails -o "/results/transfer/depth/inference"

deactivate
