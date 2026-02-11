# SYSTEM PACKAGES
export DEBIAN_FRONTEND=noninteractive
export TZ=EAmerica/New_York
apt update
apt install -y git curl vim git-lfs curl ffmpeg libx11-dev tree wget tmux
git lfs install

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version

# Required repos
cd /
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
git checkout 173b0fe

cd /
git clone https://github.com/SHI-Labs/physical-ai-bench.git
cd physical-ai-bench
git checkout a72c2e9
