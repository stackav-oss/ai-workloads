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
cp /cosmos-transfer2.5/assets/robot_example/edge/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0000.mp4
cp /cosmos-transfer2.5/assets/robot_example/edge/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0001.mp4
cp /cosmos-transfer2.5/assets/robot_example/edge/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0002.mp4
cp /cosmos-transfer2.5/assets/robot_example/edge/robot_edge.mp4 /datasets/physical-ai-bench-conditional-generation/canny/task_0003.mp4

mkdir -p /datasets/physical-ai-bench-conditional-generation/captions
cat > /datasets/physical-ai-bench-conditional-generation/captions/task_0000.json <<'EOF'
{"caption": "The video is a first-person perspective, possibly from a robotic or mechanical point of view, focusing on a small, round wooden table in a cozy living room setting. The table is neatly arranged with a few items: a white mug with a cute design, a folded yellow cloth, a box of tissues, and a small decorative vase with artificial flowers. The background features a dark television screen mounted on a wooden cabinet, and a blue armchair is visible to the right. The robotic arms, which are black with metallic joints, are positioned in front of the camera, suggesting an interaction with the objects on the table. Throughout the video, the arms remain mostly static, hovering over the table, indicating a potential setup for a demonstration or test of the robotic arms capabilities. The lighting is soft, creating a warm and inviting atmosphere. The camera remains fixed, providing a stable view of the scene, allowing the viewer to focus on the details of the objects and the robotic arms. The setting suggests a domestic environment, possibly for a vlog or a demonstration video, emphasizing the interaction between technology and everyday life."}
EOF
cp /datasets/physical-ai-bench-conditional-generation/captions/task_0000.json /datasets/physical-ai-bench-conditional-generation/captions/task_0001.json
cp /datasets/physical-ai-bench-conditional-generation/captions/task_0000.json /datasets/physical-ai-bench-conditional-generation/captions/task_0002.json
cp /datasets/physical-ai-bench-conditional-generation/captions/task_0000.json /datasets/physical-ai-bench-conditional-generation/captions/task_0003.json



if [ -d "/usr/local/cuda-13" ]; then
    echo "/usr/local/cuda-13 exists"
else
    echo "/usr/local/cuda-13 does not exist"
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
    #apt -y install cuda-toolkit-12-8
    apt -y install cuda-toolkit-13-0
fi


#ln -s /etc/alternatives/cuda-12 /usr/local/cuda-12 || true
#ln -s /etc/alternatives/cuda-13 /usr/local/cuda-13 || true
#rm /usr/local/cuda-12

rm  /etc/alternatives/cuda
ln -s /usr/local/cuda-13.0 /etc/alternatives/cuda




if [ "$gpu_model" == "NVIDIA GB200" ]; then
    echo "Using PyTorch with CUDA 13.0 for NVIDIA GB200"
    uv sync --python 3.10 --extra=cu130
    
    # /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12 -> /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12.8.90
    # /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so -> libcudart.so.12

    #unset $(env | cut -d= -f1 | egrep "(CUDA|NV_|LIBRARY_PATH)")
    #rm /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so
    #ln -s /cosmos-transfer2.5/.venv/lib/python3.10/site-packages/nvidia/cu13/lib/libcudart.so.13 /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so

    #rm /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12 || true
    #ln -s /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12.8.90 /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12
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
