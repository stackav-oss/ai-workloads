from pathlib import Path
from typing import Annotated, Union
import json
import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2.config import (
    BlurConfig,
    DepthConfig,
    EdgeConfig,
    InferenceArguments,
    InferenceOverrides,
    SegConfig,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)
import logging
import sys

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AllConfig(pydantic.BaseModel):
    pass

ControlUnion = Annotated[
    Union[
        Annotated[EdgeConfig, tyro.conf.subcommand("edge")],
        Annotated[DepthConfig, tyro.conf.subcommand("depth")],
        Annotated[BlurConfig, tyro.conf.subcommand("vis")],
        Annotated[SegConfig, tyro.conf.subcommand("seg")],
        Annotated[AllConfig, tyro.conf.subcommand("all")],
    ],
    tyro.conf.ConsolidateSubcommandArgs,
]

# Map control types to their corresponding batch hint keys
CONTROL_TYPE_TO_HINTS = {
    EdgeConfig: ['edge'],
    DepthConfig: ['depth'],
    BlurConfig: ['vis'],
    SegConfig: ['seg'],
    AllConfig: ['edge', 'depth', 'vis', 'seg']
}


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""

    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""

    control: ControlUnion = EdgeConfig()
    """Control help. Run control:edge --help for more information about edge etc."""

    caption_variant: int = pydantic.Field(
        default=0,
        description="Caption variant to use for inference [0-5].",
        ge=0,
        le=5,
    )
    """Caption variant to use for inference."""

def read_prompt_file(prompt_path):
    with open(prompt_path, 'r') as f:
        data = json.load(f)
    return data["caption"]

def prepare_samples(args, offset, size):
    inference_samples = []
    for i in range(offset, offset + size):
        task_id = f"task_{i:04d}"
        variant_id = task_id
        # main output variant
        caption_file_name = task_id if args.caption_variant == 0 else f"{task_id}_caption{args.caption_variant}"
        prompt = read_prompt_file(f"/datasets/physical-ai-bench-conditional-generation/captions/{caption_file_name}.json")
        output_video = args.setup.output_dir / f"{variant_id}.mp4"
        if not output_video.exists():
            original_video = f"/datasets/physical-ai-bench-conditional-generation/videos/{task_id}.mp4"
            edge_config = EdgeConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/canny/{task_id}.mp4") if isinstance(args.control, (AllConfig, EdgeConfig)) else None
            depth_config = DepthConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/depth_vids/{task_id}.mp4") if isinstance(args.control, (AllConfig, DepthConfig)) else None
            blur_config = BlurConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/blur/{task_id}.mp4") if isinstance(args.control, (AllConfig, BlurConfig)) else None
            seg_config = SegConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/sam2_vids/{task_id}.mp4") if isinstance(args.control, (AllConfig, SegConfig)) else None
            base_args = {
                "name": variant_id,
                "prompt": prompt,
                "video_path": original_video,
                "edge": edge_config,
                "depth": depth_config,
                "vis": blur_config,
                "seg": seg_config,
            }
            sample = InferenceArguments(**base_args)
            inference_samples.append(sample)
        else:
            print(f"Output for {variant_id} already exists, skipping inference.")
    return inference_samples

NUM_BENCHMARK_RUNS = 5
def run_benchmark(inference_engine, samples, output_dir):
    """Run benchmark iterations and collect timing data.
    
    Args:
        inference_engine: Initialized Inference engine
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
        inference_engine.generate(samples[i:i+1], output_dir=output_dir)
        elapsed_time = time.perf_counter() - start_time
        elapsed_times.append(elapsed_time)
        
        if is_rank0():
            logger.info("%s throughput step %d time: %.4f seconds using %d GPUs", 
                       inference_type, i+1, elapsed_time, nproc_per_node)
    
    return elapsed_times


def main(args: Args):
    BATCH_SIZE = 5

    from cosmos_transfer2.inference import Control2WorldInference
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    # Get batch hint keys based on active control type
    batch_hint_keys = CONTROL_TYPE_TO_HINTS.get(type(args.control), [])
    print(f"Using batch hint keys: {batch_hint_keys} for control type: {type(args.control).__name__}")
    inference_engine = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)

    inference_samples = prepare_samples(args, 0, BATCH_SIZE)
    #warm up
    inference_engine.generate(inference_samples, output_dir=args.setup.output_dir)
    run_benchmark(inference_engine, inference_samples, output_dir=args.setup.output_dir)
    
    



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
