#!/bin/bash
# COSMOS Benchmark Script
# Runs evaluation benchmarks for inference results

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
INFRASTRUCTURE_DIR="/physical-ai-bench/generation"
RESULTS_BASE_DIR="/results/predict"
DATASETS_DIR="/datasets"

# VQA evaluation dimensions
EVAL_DIMENSIONS="aesthetic_quality background_consistency imaging_quality motion_smoothness overall_consistency subject_consistency i2v_background i2v_subject"

# Functions
check_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "Error: HF_TOKEN is either unset or empty" >&2
        echo "Please set your Hugging Face token: export HF_TOKEN=your_token_here" >&2
        exit 1
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
    gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0)
    echo "Detected GPU: '$gpu_model'"
    export GPU_MODEL="$gpu_model"
}

setup_python_environment() {
    echo "Setting up Python environment..."
    
    # Deactivate any existing environment
    deactivate 2>/dev/null || true
    
    # Handle GB200 specific requirements
    if [ "$GPU_MODEL" == "NVIDIA GB200" ]; then
        echo "Configuring for NVIDIA GB200"
        sed -i -e 's/"decord"/"decord2"/g' -e 's/qwen-vl-utils\[decord\]/qwen-vl-utils/g' "$INFRASTRUCTURE_DIR/pyproject.toml"
    fi
    
    # Sync environment
    if ! uv sync; then
        echo "Error: Failed to sync Python environment" >&2
        exit 1
    fi
    
    source .venv/bin/activate
    
    # Install additional dependencies
    if ! uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"; then
        echo "Warning: Failed to install detectron2, continuing..."
    fi
    
    # Handle GB200 specific PyTorch installation
    if [ "$GPU_MODEL" == "NVIDIA GB200" ]; then
        echo "Installing PyTorch with CUDA 13.0 for NVIDIA GB200"
        uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
        uv pip install --reinstall numpy==1.26.4
    fi
    
    # Verify PyTorch installation
    python -m torch.utils.collect_env | grep "PyTorch" || echo "Warning: Could not verify PyTorch installation"
}

optimize_vllm_config() {
    echo "Optimizing VLLM configuration for quality evaluation..."
    local vqa_file="$INFRASTRUCTURE_DIR/pbench/vqa_evaluation.py"
    
    if [ -f "$vqa_file" ]; then
        sed -i 's/"gpu_memory_utilization": 0.55,/"gpu_memory_utilization": 0.75,/g' "$vqa_file"
        sed -i 's/"enable_expert_parallel": True,/"enable_expert_parallel": False, "enforce_eager": True,/g' "$vqa_file"
        echo "VLLM configuration optimized"
    else
        echo "Warning: VQA evaluation file not found: $vqa_file"
    fi
}

run_domain_evaluation() {
    local inference_type="$1"
    local videos_path="$RESULTS_BASE_DIR/$inference_type/inference"
    local output_path="$RESULTS_BASE_DIR/$inference_type/evaluation"
    
    echo "Running domain score evaluation..."
    
    if ! python -m torch.distributed.run --standalone --nproc_per_node 1 evaluate.py \
        --mode custom_input \
        --prompt_file "$DATASETS_DIR/physical-ai-bench-generation/cosmos_predict2_bench_full_info.json" \
        --dimension $EVAL_DIMENSIONS \
        --custom_image_folder "$DATASETS_DIR/physical-ai-bench-generation/condition_image" \
        --videos_path "$videos_path" \
        --output_path "$output_path"; then
        echo "Error: Domain evaluation failed" >&2
        exit 1
    fi
    
    echo "Domain evaluation completed"
}

run_quality_evaluation() {
    local inference_type="$1"
    local video_dir="$RESULTS_BASE_DIR/$inference_type/inference"
    local output_dir="$RESULTS_BASE_DIR/$inference_type/evaluation"
    
    echo "Running quality score evaluation..."
    
    if ! python evaluate_vqa.py \
        --tensor_parallel_size 4 \
        --prompt_file "$DATASETS_DIR/physical-ai-bench-generation/cosmos_predict2_bench_full_info.json" \
        --vqa_questions_dir "$DATASETS_DIR/physical-ai-bench-generation/vqa" \
        --video_dir "$video_dir" \
        --output_dir "$output_dir"; then
        echo "Error: Quality evaluation failed" >&2
        exit 1
    fi
    
    echo "Quality evaluation completed"
}

aggregate_results() {
    local inference_type="$1"
    local root_dir="$2"
    local input_dir="$RESULTS_BASE_DIR/$inference_type/evaluation"
    local output_dir="$RESULTS_BASE_DIR/$inference_type/evaluation"
    
    echo "Aggregating results..."
    
    if ! python "$root_dir/aggregate_pai_bench_results.py" \
        --inference_type="$inference_type" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir"; then
        echo "Error: Results aggregation failed" >&2
        exit 1
    fi
    
    # Display final results
    local final_results_file="$output_dir/final_results.txt"
    if [ -f "$final_results_file" ]; then
        echo
        echo "=== FINAL PAI-BENCH EVALUATION RESULTS ==="
        cat "$final_results_file"
        echo "========================================="
    fi
    
    echo "Results aggregation completed"
}

cleanup_environment() {
    if [[ "${VIRTUAL_ENV:-}" ]]; then
        deactivate
        echo "Python environment deactivated"
    fi
}

# Main execution
echo "Starting COSMOS benchmark evaluation..."

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

# Setup and validation
check_hf_token
get_gpu_info

echo "Running benchmark evaluation for: $inference_type"

# Setup Python environment
cd "$INFRASTRUCTURE_DIR"
setup_python_environment

# Trap to ensure cleanup
trap cleanup_environment EXIT

# Run evaluations
run_domain_evaluation "$inference_type"
optimize_vllm_config
run_quality_evaluation "$inference_type"

# Aggregate and display results
aggregate_results "$inference_type" "$ROOT_DIR"

echo "COSMOS evaluation completed successfully!"
echo "Results available in: $RESULTS_BASE_DIR/$inference_type/evaluation"