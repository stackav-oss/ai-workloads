#!/bin/bash
# COSMOS Setup Script
# Installs system packages and required repositories

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
COSMOS_COMMIT="9557c1a"
PHYSICAL_AI_BENCH_COMMIT="a72c2e9"
INSTALL_DIR="/"

echo "Starting COSMOS environment setup..."

# SYSTEM PACKAGES
export DEBIAN_FRONTEND=noninteractive
export TZ=America/New_York

echo "Updating system packages..."
apt update
apt install -y git curl vim git-lfs curl ffmpeg libx11-dev tree wget tmux

echo "Configuring Git LFS..."
git lfs install

echo "Installing UV package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version


echo "Cloning cosmos-predict2.5 repository..."
cd "$INSTALL_DIR"
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
git checkout "$COSMOS_COMMIT"

echo "Cloning physical-ai-bench repository..."
cd "$INSTALL_DIR"
git clone https://github.com/SHI-Labs/physical-ai-bench.git
cd physical-ai-bench
git checkout "$PHYSICAL_AI_BENCH_COMMIT"

echo "COSMOS environment setup completed successfully!"
echo "Repositories installed in: $INSTALL_DIR"
