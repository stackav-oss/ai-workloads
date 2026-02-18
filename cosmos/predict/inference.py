"""Inference script for Cosmos models.

This script runs inference using the Physical AI Benchmark dataset
for both text2world and image2world generation tasks.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

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
logger = logging.getLogger(__name__)

# Constants
DATASET_BASE_PATH = Path("/datasets/physical-ai-bench-generation")
BENCH_DATA_FILE = DATASET_BASE_PATH / "cosmos_predict2_bench_full_info.json"
CONDITION_IMAGE_DIR = DATASET_BASE_PATH / "condition_image"

DEFAULT_NUM_OUTPUT_FRAMES = 33
DEFAULT_NUM_STEPS = 35
DEFAULT_SEED = 0
DEFAULT_GUIDANCE = 4

SUPPORTED_INFERENCE_TYPES = {"text2world", "image2world"}


class Args(pydantic.BaseModel):
    """Command-line arguments for inference.
    
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
    ugur_i = 9
    for item in pai_data:
        ugur_i += 1
        if ugur_i % 50 != 0:
            continue
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


def main(args: Args) -> None:
    """Main inference function.
    
    Args:
        args: Parsed command-line arguments
    """
    inference_type = args.overrides.inference_type
    logger.info("Starting inference for %s", inference_type)
    
    try:
        # Prepare inference samples
        inference_samples = prepare_inference_samples(inference_type)
        if not inference_samples:
            logger.error("No valid inference samples prepared")
            return
        
        # Initialize output directory and inference engine
        init_output_dir(args.setup.output_dir, profile=args.setup.profile)
        inference = Inference(args.setup)
        
        # Run inference
        logger.info("Running inference on %d samples", len(inference_samples))
        inference.generate(inference_samples, output_dir=args.setup.output_dir)
        
        if is_rank0():
            logger.info("Inference completed successfully")
            
    except Exception as e:
        logger.error("Error during inference: %s", e)
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
