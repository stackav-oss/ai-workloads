"""Throughput evaluation script for Cosmos inference models.

This script evaluates the throughput of text2world and image2world inference
models using the Physical AI Benchmark dataset.
"""

import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pydantic
import tyro

from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from cosmos_predict2.inference import Inference
from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATASET_BASE_PATH = Path("/datasets/physical-ai-bench-generation")
BENCH_DATA_FILE = DATASET_BASE_PATH / "cosmos_predict2_bench_full_info.json"
CONDITION_IMAGE_DIR = DATASET_BASE_PATH / "condition_image"

DEFAULT_NUM_OUTPUT_FRAMES = 33
DEFAULT_NUM_STEPS = 35
DEFAULT_SEED = 0
DEFAULT_GUIDANCE = 4
NUM_WARMUP_RUNS = 1
NUM_BENCHMARK_RUNS = 3

SUPPORTED_INFERENCE_TYPES = {"text2world", "image2world"}


def get_nproc_per_node() -> int:
    """Get the number of processes per node from environment variable.
    
    Returns:
        Number of processes per node, or -1 if not available or invalid.
    """
    if 'LOCAL_WORLD_SIZE' in os.environ:
        try:
            return int(os.environ['LOCAL_WORLD_SIZE'])
        except ValueError:
            logger.error("LOCAL_WORLD_SIZE is not a valid integer: %s", 
                        os.environ['LOCAL_WORLD_SIZE'])
    
    return -1


class Args(pydantic.BaseModel):
    """Command-line arguments for throughput evaluation.
    
    Attributes:
        setup: Setup arguments that can only be provided via CLI
        overrides: Inference parameter overrides that can be provided via CLI or JSON file
    """
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    setup: SetupArguments
    overrides: InferenceOverrides


def load_benchmark_data() -> List[Dict[str, Any]]:
    """Load benchmark data from the Physical AI Benchmark dataset.
    
    Returns:
        List of benchmark data items containing video_id and prompt_en.
        
    Raises:
        FileNotFoundError: If the benchmark data file doesn't exist.
        json.JSONDecodeError: If the data file is not valid JSON.
    """
    try:
        with open(BENCH_DATA_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError as e:
        logger.error("Benchmark data file not found: %s", BENCH_DATA_FILE)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in benchmark data file: %s", e)
        raise


def create_inference_sample(video_id: str, prompt: str, inference_type: str) -> InferenceArguments:
    """Create an inference sample based on the inference type.
    
    Args:
        video_id: Unique identifier for the video
        prompt: Text prompt for generation
        inference_type: Type of inference ('text2world' or 'image2world')
        
    Returns:
        Configured InferenceArguments instance
        
    Raises:
        ValueError: If inference_type is not supported
        FileNotFoundError: If image file doesn't exist for image2world type
    """
    if inference_type not in SUPPORTED_INFERENCE_TYPES:
        raise ValueError(f"Unsupported inference type: {inference_type}. "
                        f"Supported types: {SUPPORTED_INFERENCE_TYPES}")
    
    base_args = {
        "name": video_id,
        "prompt": prompt,
        "inference_type": inference_type,
        "num_output_frames": DEFAULT_NUM_OUTPUT_FRAMES,
        "num_steps": DEFAULT_NUM_STEPS,
        "seed": DEFAULT_SEED,
        "guidance": DEFAULT_GUIDANCE
    }
    
    if inference_type == "image2world":
        image_path = CONDITION_IMAGE_DIR / f"{video_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Condition image not found: {image_path}")
        base_args["input_path"] = str(image_path)
    
    return InferenceArguments(**base_args)


def prepare_inference_samples(inference_type: str) -> List[InferenceArguments]:
    """Prepare inference samples from benchmark data.
    
    Args:
        inference_type: Type of inference to prepare samples for
        
    Returns:
        List of prepared InferenceArguments
    """
    pai_data = load_benchmark_data()
    inference_samples = []
    
    for item in pai_data:
        try:
            video_id = item["video_id"]
            prompt_en = item["prompt_en"]
            inference_sample = create_inference_sample(video_id, prompt_en, inference_type)
            inference_samples.append(inference_sample)
        except (KeyError, FileNotFoundError) as e:
            logger.warning("Skipping item %s: %s", item.get("video_id", "unknown"), e)
            continue
    
    logger.info("Prepared %d inference samples for %s", len(inference_samples), inference_type)
    return inference_samples


def run_warmup(inference: Inference, samples: List[InferenceArguments], 
               output_dir: str, inference_type: str) -> None:
    """Run warmup inference to initialize the model.
    
    Args:
        inference: Initialized Inference object
        samples: List of inference samples
        output_dir: Output directory for results
        inference_type: Type of inference being performed
    """
    if not samples:
        logger.warning("No samples available for warmup")
        return
        
    start_time = time.perf_counter()
    inference.generate(samples[0:NUM_WARMUP_RUNS], output_dir=output_dir)
    elapsed_time = time.perf_counter() - start_time
    
    if is_rank0():
        logger.info("%s throughput warmup time: %.4f seconds using %d GPUs", 
                   inference_type, elapsed_time, get_nproc_per_node())


def run_benchmark(inference: Inference, samples: List[InferenceArguments], 
                 output_dir: str, inference_type: str, nproc_per_node: int) -> List[float]:
    """Run benchmark iterations and collect timing data.
    
    Args:
        inference: Initialized Inference object
        samples: List of inference samples
        output_dir: Output directory for results
        inference_type: Type of inference being performed
        nproc_per_node: Number of processes per node for this test
        
    Returns:
        List of elapsed times for each benchmark run
    """
    if len(samples) < NUM_BENCHMARK_RUNS:
        logger.warning("Not enough samples for benchmark runs. Have %d, need %d", 
                      len(samples), NUM_BENCHMARK_RUNS)
        return []
    
    elapsed_times = []
    for i in range(NUM_BENCHMARK_RUNS):
        start_time = time.perf_counter()
        inference.generate(samples[i:i+1], output_dir=output_dir)
        elapsed_time = time.perf_counter() - start_time
        elapsed_times.append(elapsed_time)
        
        if is_rank0():
            logger.info("%s throughput step %d time: %.4f seconds using %d GPUs", 
                       inference_type, i+1, elapsed_time, nproc_per_node)
    
    return elapsed_times


def save_throughput_results(results_data: List[Dict[str, Any]], inference_type: str, output_dir: str) -> str:
    """Save throughput results to a nicely formatted file.
    
    Args:
        results_data: List of throughput results for different GPU configurations
        inference_type: Type of inference being benchmarked
        output_dir: Base output directory
        
    Returns:
        Path to the saved results file
    """
    results_dir = Path(output_dir) / inference_type / "throughput"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "throughput_results.txt"
    
    # Create formatted results content
    content_lines = [
        "=" * 60,
        f"COSMOS {inference_type.upper()} THROUGHPUT BENCHMARK RESULTS",
        "=" * 60,
        f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Inference Type: {inference_type}",
        f"Dataset: Physical AI Benchmark",
        "",
        "GPU Configuration and Performance Results:",
        "-" * 40,
    ]
    
    for result in results_data:
        nproc = result['nproc_per_node']
        times = result['elapsed_times']
        if not times:
            content_lines.extend([
                f"\n🔸 {nproc} GPU{'s' if nproc > 1 else ''}:",
                f"   Status: FAILED or SKIPPED",
            ])
            continue
            
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = nproc / avg_time  # samples per second per GPU
        
        content_lines.extend([
            f"\n🔸 {nproc} GPU{'s' if nproc > 1 else ''}:",
            f"   Average Time: {avg_time:.4f} seconds",
            f"   Std Deviation: {std_time:.4f} seconds",
            f"   Throughput: {throughput:.4f} samples/sec/GPU",
            f"   Runs Completed: {len(times)}",
        ])
    
    content_lines.extend([
        "",
        "-" * 40,
        "Summary:",
        f"✅ Total Configurations Tested: {len([r for r in results_data if r['elapsed_times']])}",
        f"❌ Failed Configurations: {len([r for r in results_data if not r['elapsed_times']])}",
        "",
        "=" * 60,
    ])
    
    content = "\n".join(content_lines)
    
    try:
        with open(results_file, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info("Throughput results saved to %s", results_file)
        return str(results_file)
    except IOError as e:
        logger.error("Failed to save throughput results: %s", e)
        raise


def print_final_throughput_results(results_data: List[Dict[str, Any]], inference_type: str) -> None:
    """Print final throughput results summary to console.
    
    Args:
        results_data: List of throughput results for different GPU configurations
        inference_type: Type of inference being benchmarked
    """
    print("\n" + "=" * 60)
    print(f"🚀 FINAL THROUGHPUT RESULTS - {inference_type.upper()}")
    print("=" * 60)
    
    successful_tests = [r for r in results_data if r['elapsed_times']]
    failed_tests = [r for r in results_data if not r['elapsed_times']]
    
    if successful_tests:
        print("\n📊 Performance Summary:")
        for result in successful_tests:
            nproc = result['nproc_per_node']
            times = result['elapsed_times']
            avg_time = statistics.mean(times)
            throughput = nproc / avg_time
            print(f"   {nproc} GPU{'s' if nproc > 1 else ''}: {avg_time:.4f}s avg, {throughput:.4f} samples/sec/GPU")
    
    if failed_tests:
        print(f"\n❌ Failed Tests: {len(failed_tests)} configurations")
    
    print(f"\n✅ Overall: {len(successful_tests)}/{len(results_data)} configurations completed successfully")
    print("=" * 60)


def main(args: Args) -> None:
    """Main throughput evaluation function.
    
    Args:
        args: Parsed command-line arguments
    """
    inference_type = args.overrides.inference_type
    logger.info("Starting throughput evaluation for %s", inference_type)
    
    # Get number of processes for this specific run
    nproc_per_node = get_nproc_per_node()
    
    try:
        # Prepare inference samples
        inference_samples = prepare_inference_samples(inference_type)
        if not inference_samples:
            logger.error("No valid inference samples prepared")
            return
        
        # Initialize output directory and inference engine
        init_output_dir(args.setup.output_dir, profile=args.setup.profile)
        inference = Inference(args.setup)
        
        # Run warmup
        run_warmup(inference, inference_samples, args.setup.output_dir, inference_type)
        
        # Run benchmark for this specific nproc configuration
        elapsed_times = run_benchmark(inference, inference_samples, args.setup.output_dir, inference_type, nproc_per_node)
        
        # Calculate statistics for this run
        if elapsed_times:
            avg_time = statistics.mean(elapsed_times)
            std_time = statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0
            throughput = nproc_per_node / avg_time
            
            if is_rank0():
                logger.info("\n\n=== THROUGHPUT RESULTS FOR %d GPUs ===", nproc_per_node)
                logger.info("Average Time: %.4f seconds", avg_time)
                logger.info("Std Deviation: %.4f seconds", std_time)
                logger.info("Throughput: %.4f samples/sec/GPU", throughput)
                logger.info("============================================\n")
            
    except Exception as e:
        logger.error("Error during throughput evaluation: %s", e)
        raise


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
