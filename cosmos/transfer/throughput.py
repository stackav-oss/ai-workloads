import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent))

import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from cosmos_transfer2.config import handle_tyro_exception, is_rank0
from inference import Args, CONTROL_TYPE_TO_HINTS, prepare_samples

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CONTROL_TYPE_NAMES = {
    "EdgeConfig": "edge",
    "DepthConfig": "depth",
    "BlurConfig": "vis",
    "SegConfig": "seg",
    "AllConfig": "all",
}


def get_control_type_name(control) -> str:
    """Return a stable control type name for logs and report filenames."""
    class_name = type(control).__name__
    if class_name in CONTROL_TYPE_NAMES:
        return CONTROL_TYPE_NAMES[class_name]
    return class_name.replace("Config", "").lower()


def run_benchmark(inference_engine, samples, output_dir):
    """Run warmup and benchmark iterations, then return computed metrics.
    
    Args:
        inference_engine: Initialized Inference engine
        samples: List of inference samples
        output_dir: Directory for inference outputs
        
    Returns:
        Dictionary containing elapsed times and computed metrics
    """
    if len(samples) < 2:
        raise ValueError("At least 2 samples are required (1 warmup + 1 benchmark).")

    # Warm up with the first sample.
    logger.info("Warming up the model with the first sample...")
    inference_engine.generate(samples[0:1], output_dir=output_dir)

    elapsed_times = []
    logger.info("Starting benchmark runs...")
    for i in range(1, len(samples)):
        start_time = time.perf_counter()
        inference_engine.generate(samples[i:i+1], output_dir=output_dir)
        elapsed_time = time.perf_counter() - start_time
        elapsed_times.append(elapsed_time)
        logger.info("Run %d completed. Elapsed time: %.4f seconds", i, elapsed_time)

    avg_time = statistics.mean(elapsed_times)
    std_time = statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0.0
    nproc_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    throughput = nproc_per_node / avg_time if avg_time > 0 else 0.0

    return {
        "elapsed_times": elapsed_times,
        "avg_time": avg_time,
        "std_time": std_time,
        "throughput": throughput,
        "total_runs": len(elapsed_times),
        "nproc_per_node": nproc_per_node,
        "evaluation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def save_metrics_json(metrics, output_dir, control_type):
    """Save benchmark metrics to a JSON file for aggregation.
    
    Args:
        metrics: Dictionary containing benchmark metrics
        output_dir: Directory for inference outputs
        control_type: Name of the control type (e.g., 'edge', 'vis')
    """
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    nproc_per_node = metrics["nproc_per_node"]
    results_file = results_dir / f"throughput_{control_type}_{nproc_per_node}gpu.json"

    # Prepare data for JSON storage
    json_data = {
        "control_type": control_type,
        "nproc_per_node": nproc_per_node,
        "avg_time": metrics["avg_time"],
        "std_time": metrics["std_time"],
        "throughput": metrics["throughput"],
        "total_runs": metrics["total_runs"],
        "evaluation_time": metrics["evaluation_time"],
        "elapsed_times": metrics["elapsed_times"],
    }

    with open(results_file, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=2)

    logger.info("Metrics saved to %s", results_file)


def main(args: Args):
    from cosmos_transfer2.inference import Control2WorldInference

    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    control_type = get_control_type_name(args.control)
    # Get batch hint keys based on active control type
    batch_hint_keys = CONTROL_TYPE_TO_HINTS.get(type(args.control), [])
    logger.info("Using batch hint keys: %s for control type: %s", batch_hint_keys, control_type)

    # Init engine.
    inference_engine = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)

    # Prepare samples.
    inference_samples = prepare_samples(args, 0, 5)
    if len(inference_samples) < 2:
        logger.error("Not enough samples for benchmark. Need at least 2, found %d.", len(inference_samples))
        return

    # Run benchmark and return computed metrics.
    benchmark_metrics = run_benchmark(inference_engine, inference_samples, output_dir=args.setup.output_dir)

    # Rank 0 saves benchmark results as JSON.
    if is_rank0():
        save_metrics_json(benchmark_metrics, args.setup.output_dir, control_type)


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

    main(args)

    cleanup_environment()
