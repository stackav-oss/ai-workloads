#!/bin/bash
# Data Download Script
# Downloads required datasets from Hugging Face

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
DATASETS_DIR="/datasets"
PHYSICAL_AI_BENCH_DATASET="shi-labs/physical-ai-bench-generation"
PHYSICAL_AI_BENCH_CONDITIONAL="shi-labs/physical-ai-bench-conditional-generation"

# Functions
check_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "Error: HF_TOKEN is either unset or empty" >&2
        echo "Please set your Hugging Face token: export HF_TOKEN=your_token_here" >&2
        exit 1
    fi
    echo "HF_TOKEN is configured"
}

setup_python_env() {
    echo "Setting up Python environment..."
    source "$HOME/.local/bin/env"
    
    mkdir -p "$DATASETS_DIR"
    cd "$DATASETS_DIR"
    
    uv venv --clear
    source .venv/bin/activate
    uv pip install huggingface_hub
}

download_dataset() {
    local dataset_name="$1"
    local local_dir="$2"
    
    if [ -d "$local_dir" ]; then
        echo "Dataset already exists: $local_dir"
        return 0
    fi
    
    echo "Downloading dataset: $dataset_name -> $local_dir"
    if ! hf download "$dataset_name" --repo-type dataset --local-dir "$local_dir" > /dev/null; then
        echo "Error: Failed to download dataset $dataset_name" >&2
        return 1
    fi
    echo "Successfully downloaded: $dataset_name"
}

cleanup_env() {
    if [[ "${VIRTUAL_ENV:-}" ]]; then
        deactivate
        echo "Python environment deactivated"
    fi
}

# Main execution
echo "Starting data download process..."

check_hf_token
setup_python_env

# Download datasets
download_dataset "$PHYSICAL_AI_BENCH_DATASET" "$DATASETS_DIR/physical-ai-bench-generation"

# Uncomment to download conditional generation dataset if needed
# download_dataset "$PHYSICAL_AI_BENCH_CONDITIONAL" "$DATASETS_DIR/physical-ai-bench-conditional-generation"

cleanup_env

echo "Data download completed successfully!"
echo "Datasets available in: $DATASETS_DIR"
