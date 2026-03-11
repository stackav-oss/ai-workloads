source $HOME/.local/bin/env

# Check if HF_TOKEN is set and not empty
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN is either unset or empty"
  exit 1
fi

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


echo  "Running inference and benchmarking for control type: $control_type"

# Create result directories
mkdir -p /results/transfer/$control_type/inference

# clean previous results
rm /results/transfer/$control_type/inference/*  > /dev/null 2>&1  || true


# Run inference using custom pyton script mounted from repo
cd /cosmos-transfer2.5

mkdir -p /datasets/physical-ai-bench-conditional-generation/videos
cp /cosmos-transfer2.5/assets/robot_example/robot_input.mp4 /datasets/physical-ai-bench-conditional-generation/videos/task_0000.mp4
cp /cosmos-transfer2.5/assets/robot_example/robot_input.mp4 /datasets/physical-ai-bench-conditional-generation/videos/task_0001.mp4
cp /cosmos-transfer2.5/assets/robot_example/robot_input.mp4 /datasets/physical-ai-bench-conditional-generation/videos/task_0002.mp4
cp /cosmos-transfer2.5/assets/robot_example/robot_input.mp4 /datasets/physical-ai-bench-conditional-generation/videos/task_0003.mp4

mkdir -p /datasets/physical-ai-bench-conditional-generation/canny
cp /cosmos-transfer2.5/assets/robot_example/seg/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0000.mp4
cp /cosmos-transfer2.5/assets/robot_example/seg/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0001.mp4
cp /cosmos-transfer2.5/assets/robot_example/seg/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0002.mp4
cp /cosmos-transfer2.5/assets/robot_example/seg/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0003.mp4

if [ "$gpu_model" == "NVIDIA GB200" ]; then
    echo "Using PyTorch with CUDA 13.0 for NVIDIA GB200"
    uv sync --python 3.10 --extra=cu130
    #rm /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12 || true
    #rm  /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so || true
    #ln -s /cosmos-transfer2.5/.venv/lib/python3.10/site-packages/nvidia/cu13/lib/libcudart.so.13 /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so
    #rm /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12 
    #/usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12 -> libcudart.so.12.8.90
else
    echo "Using PyTorch with CUDA 12.8 for NVIDIA H100"
    uv sync --python 3.10 --extra=cu128   > /dev/null 2>&1  || true
fi
source .venv/bin/activate

torchrun --nproc_per_node=$available_gpus --master_port=12341 "/$ROOT_DIR/inference.py" --disable-guardrails -o "/results/transfer/$control_type/inference" control:$control_type

deactivate
