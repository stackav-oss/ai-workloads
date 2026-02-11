source $HOME/.local/bin/env

# Check if HF_TOKEN is set and not empty
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN is either unset or empty"
fi


if [ $# -eq 0 ]; then
    echo "Error: Please provide an inference type [text2world|image2world]"
    exit 1
fi

inference_type=$1
if [ "$inference_type" != "text2world" ] && [ "$inference_type" != "image2world" ]; then
    echo "Error: Invalid inference type. Must be 'text2world' or 'image2world'"
    exit 1
fi
echo "Inference type: $inference_type"

gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0)
echo "Detected GPU: $gpu_model"

available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: $available_gpus"
if [ $available_gpus -lt 4 ]; then
    echo "Error: At least 4 NVIDIA GPUs are required"
    exit 1
fi


# Get the absolute path of the directory containing this script
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "The script is located in: ${ROOT_DIR}"


echo  "Running inference and benchmarking for inference type: $inference_type"

# Create result directories
mkdir -p /results/predict/$inference_type/inference
mkdir -p /results/predict/$inference_type/benchmark

# clean previous results
rm /results/predict/$inference_type/inference/*  > /dev/null 2>&1  || true
rm /results/predict/$inference_type/benchmark/*  > /dev/null 2>&1  || true


# Run inference using custom pyton script mounted from repo
cd /cosmos-predict2.5

if [ "$GPU_MODEL" == "NVIDIA GB200" ]; then
    uv sync --python 3.10 --extra=cu130   > /dev/null 2>&1  || true
else
    uv sync --python 3.10 --extra=cu128   > /dev/null 2>&1  || true
fi
source .venv/bin/activate

torchrun --nproc_per_node=$available_gpus "/$ROOT_DIR/inference.py" --inference-type $inference_type --disable-guardrails -o "/results/predict/$inference_type/inference"

deactivate
