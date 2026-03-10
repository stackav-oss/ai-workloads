#!/bin/bash
# COSMOS Setup Script
# Installs system packages and required repositories

set -euo pipefail

# Configuration
readonly COSMOS_COMMIT="9557c1a"
readonly COSMOS_TRANSFER_COMMIT="67b6d48"
readonly PHYSICAL_AI_BENCH_COMMIT="a72c2e9"
readonly INSTALL_DIR="/"

# System-level configurations
export DEBIAN_FRONTEND=noninteractive
export TZ=America/New_York

# Helper: Clone repo and checkout specific commit
clone_and_checkout() {
	local repo_url=$1
	local commit_hash=$2
	local repo_name=$(basename "$repo_url" .git)

	echo "--- Setting up $repo_name ---"
	cd "$INSTALL_DIR"
	if [ ! -d "$repo_name" ]; then
		git clone "$repo_url"
	fi
	cd "$repo_name"
	git checkout "$commit_hash"
	echo "Finished setting up $repo_name at $commit_hash"
	echo
}

# --- Main Setup Process ---

echo "Starting COSMOS environment setup..."
echo

# 1. System Packages
echo "Updating system packages..."
apt-get update
apt-get install -y --no-install-recommends \
	curl \
	ffmpeg \
	git \
	git-lfs \
	libx11-dev \
	tmux \
	tree \
	vim \
	wget \
	rename

# 2. Git LFS Initial Configuration
echo "Configuring Git LFS..."
git lfs install

# 3. Package Manager: uv
if ! command -v uv &> /dev/null; then
	echo "Installing UV package manager..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	source "$HOME/.local/bin/env"
fi
uv --version

# 4. Clone Repositories
clone_and_checkout "https://github.com/nvidia-cosmos/cosmos-predict2.5.git" "$COSMOS_COMMIT"
clone_and_checkout "https://github.com/SHI-Labs/physical-ai-bench.git" "$PHYSICAL_AI_BENCH_COMMIT"
clone_and_checkout "https://github.com/nvidia-cosmos/cosmos-transfer2.5" "$COSMOS_TRANSFER_COMMIT"


echo "COSMOS environment setup completed successfully!"
echo "Repositories installed in: $INSTALL_DIR"
