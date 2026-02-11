source $HOME/.local/bin/env

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

# Get the absolute path of the directory containing this script
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "The script is located in: ${ROOT_DIR}"

# Run domain score benchmarking script
cd /physical-ai-bench/generation

deactivate || true

uv sync 
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"



uv run python -m torch.distributed.run --standalone --nproc_per_node 1 evaluate.py \
--mode custom_input \
--prompt_file /datasets/physical-ai-bench-generation/cosmos_predict2_bench_full_info.json \
--dimension aesthetic_quality background_consistency imaging_quality motion_smoothness overall_consistency subject_consistency i2v_background i2v_subject \
--custom_image_folder /datasets/physical-ai-bench-generation/condition_image \
--videos_path /results/predict/$inference_type/inference \
--output_path /results/predict/$inference_type/benchmark 

# Fix VLLM config for quality score evaluation
sed -i 's/"gpu_memory_utilization": 0.55,/"gpu_memory_utilization": 0.75,/g' /physical-ai-bench/generation/pbench/vqa_evaluation.py
sed -i 's/"enable_expert_parallel": True,/"enable_expert_parallel": False, "enforce_eager": True,/g' /physical-ai-bench/generation/pbench/vqa_evaluation.py

# Run quality score benchmarking script
uv run python evaluate_vqa.py \
--tensor_parallel_size 4 \
--prompt_file /datasets/physical-ai-bench-generation/cosmos_predict2_bench_full_info.json \
--vqa_questions_dir /datasets/physical-ai-bench-generation/vqa \
--video_dir /results/predict/$inference_type/inference  \
--output_dir /results/predict/$inference_type/benchmark  


## Aggregate results and print summary
uv run python "$ROOT_DIR/aggregate_results.py" --inference_type=$inference_type --input_dir /results/predict/$inference_type/benchmark --output_dir /results/predict/$inference_type/benchmark
