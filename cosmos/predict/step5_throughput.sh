#!/bin/bash
# COSMOS Throughput Evaluation Script
# Measures inference throughput for text2world or image2world generation

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
MINIMUM_GPU_COUNT=4
COSMOS_DIR="/cosmos-predict2.5"
GPU_CONFIGURATIONS=(8 4 2 1)  # Test different GPU counts

# Functions
check_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "Warning: HF_TOKEN is either unset or empty" >&2
        echo "Throughput evaluation may still work without authentication" >&2
    else
        echo "HF_TOKEN is configured"
    fi
}

validate_inference_type() {
    local inference_type="$1"
    if [[ "$inference_type" != "text2world" && "$inference_type" != "image2world" ]]; then
        echo "Error: Invalid inference type '$inference_type'. Must be 'text2world' or 'image2world'" >&2
        exit 1
    fi
}

get_gpu_info() {
    local gpu_model
    local available_gpus
    
    gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0)
    available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    
    echo "Detected GPU: '$gpu_model'"
    echo "Available GPUs: $available_gpus"
    
    if [ "$available_gpus" -lt $MINIMUM_GPU_COUNT ]; then
        echo "Error: At least $MINIMUM_GPU_COUNT NVIDIA GPUs are required, but only $available_gpus found" >&2
        exit 1
    fi
    
    # Export for use by other functions
    export GPU_MODEL="$gpu_model"
    export AVAILABLE_GPUS="$available_gpus"
}

clean_throughput_results() {
    local inference_type="$1"
    local throughput_dir="/results/predict/$inference_type/throughput"
    
    echo "Cleaning previous throughput results..."
    
    if [ -d "$throughput_dir" ]; then
        rm -rf "$throughput_dir"/*
        echo "Cleaned throughput directory: $throughput_dir"
    else
        echo "Creating new throughput directory: $throughput_dir"
    fi
    
    mkdir -p "$throughput_dir"
}

setup_python_environment() {
    echo "Setting up Python environment for throughput evaluation..."
    
    if [ "$GPU_MODEL" == "NVIDIA GB200" ]; then
        echo "Using PyTorch with CUDA 13.0 for NVIDIA GB200"
        if ! uv sync --python 3.10 --extra=cu130 > /dev/null 2>&1; then
            echo "Warning: Failed to sync with cu130, continuing..."
        fi
    else
        echo "Using PyTorch with CUDA 12.8 for other GPUs"
        if ! uv sync --python 3.10 --extra=cu128 > /dev/null 2>&1; then
            echo "Warning: Failed to sync with cu128, continuing..."
        fi
    fi
    
    source .venv/bin/activate
    echo "Python environment activated"
}

run_throughput_test() {
    local inference_type="$1"
    local root_dir="$2"
    local nproc="$3"
    local results_base_dir="$4"
    
    echo
    echo "=== Testing with $nproc GPUs ==="
    
    if [ "$nproc" -gt "$AVAILABLE_GPUS" ]; then
        echo "Skipping $nproc GPU test (only $AVAILABLE_GPUS available)"
        return 0
    fi
    
    # Create individual results directory for this nproc
    local nproc_results_dir="$results_base_dir/${nproc}gpu"
    mkdir -p "$nproc_results_dir"
    
    if ! torchrun --nproc_per_node="$nproc" \
        "$root_dir/throughput.py" \
        --inference-type "$inference_type" \
        --disable-guardrails \
        -o "/tmp/throughput_${nproc}gpu" \
        --save-results-to "$nproc_results_dir"; then
        echo "Error: Throughput test failed for $nproc GPUs" >&2
        return 1
    fi
    
    echo "Throughput test completed for $nproc GPUs"
}

run_all_throughput_tests() {
    local inference_type="$1"
    local root_dir="$2"
    
    echo "Running comprehensive throughput evaluation for: $inference_type"
    echo "Available GPUs: $AVAILABLE_GPUS"
    
    # Create main results directory
    local results_dir="/results/predict/$inference_type/throughput"
    mkdir -p "$results_dir"
    
    local failed_tests=0
    
    for nproc in "${GPU_CONFIGURATIONS[@]}"; do
        if ! run_throughput_test "$inference_type" "$root_dir" "$nproc" "$results_dir"; then
            ((failed_tests++))
            echo "Warning: Test with $nproc GPUs failed"
        fi
    done
    
    # Generate consolidated summary
    python "$root_dir/generate_throughput_summary.py" \
        --inference_type "$inference_type" \
        --results_dir "$results_dir"
    
    # Print final summary
    if [ -f "$results_dir/throughput_results.txt" ]; then
        echo
        cat "$results_dir/throughput_results.txt"
    fi
    
    if [ $failed_tests -gt 0 ]; then
        echo "Warning: $failed_tests throughput tests failed"
    else
        echo "All throughput tests completed successfully"
    fi
}

cleanup_environment() {
    if [[ "${VIRTUAL_ENV:-}" ]]; then
        deactivate
        echo "Python environment deactivated"
    fi
}

# Main execution
echo "Starting COSMOS throughput evaluation..."

# Validate inputs
if [ $# -eq 0 ]; then
    echo "Error: Please provide an inference type [text2world|image2world]" >&2
    echo "Usage: $0 <inference_type>" >&2
    exit 1
fi

inference_type="$1"
validate_inference_type "$inference_type"

# Get script directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script located in: $ROOT_DIR"

# Source UV environment
source "$HOME/.local/bin/env"

# Setup and validation
check_hf_token
get_gpu_info
clean_throughput_results "$inference_type"

echo "Running throughput evaluation for: $inference_type"

# Setup Python environment in COSMOS directory
cd "$COSMOS_DIR"
setup_python_environment

# Trap to ensure cleanup
trap cleanup_environment EXIT

run_all_throughput_tests "$inference_type" "$ROOT_DIR"

echo "\nCOSMOS throughput evaluation completed!"
echo "Check the console output above for detailed timing statistics"


