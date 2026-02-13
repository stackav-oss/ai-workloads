"""Throughput evaluation script."""

import time
import json
import os
import statistics
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


def get_nproc_per_node():
    # 'LOCAL_WORLD_SIZE' environment variable holds the value passed to --nproc-per-node
    if 'LOCAL_WORLD_SIZE' in os.environ:
        try:
            return int(os.environ['LOCAL_WORLD_SIZE'])
        except ValueError:
            print("Error: LOCAL_WORLD_SIZE is not a valid integer.")
    else:
        # Fallback or handle cases where the script is not launched via torchrun
        print("Warning: LOCAL_WORLD_SIZE environment variable not found. Returning None.")
        return -1


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    ###input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s).
    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""


def main(
    args: Args,
):
    inference_type = args.overrides.inference_type
    inference_samples = []
    pai_data = json.load(open("/datasets/physical-ai-bench-generation/cosmos_predict2_bench_full_info.json", "r"))
    for item in pai_data:
        video_id = item["video_id"]
        prompt_en = item["prompt_en"]
        if inference_type == "text2world":
            inference_sample = InferenceArguments(name=video_id, prompt=prompt_en, inference_type="text2world", num_output_frames=33, num_steps=35, seed=0, guidance=4)
            inference_samples.append(inference_sample)
        elif inference_type == "image2world":
            image_path = f"/datasets/physical-ai-bench-generation/condition_image/{video_id}.jpg"
            inference_sample = InferenceArguments(input_path=image_path, name=video_id, prompt=prompt_en, inference_type="image2world", num_output_frames=33, num_steps=35, seed=0, guidance=4)
            inference_samples.append(inference_sample)
        else:
            raise ValueError(f"Unsupported inference type: {inference_type}")

    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    inference = Inference(args.setup)

    start_time = time.perf_counter()
    inference.generate(inference_samples[0:1], output_dir=args.setup.output_dir)
    elapsed_time = time.perf_counter() - start_time
    if is_rank0():
        print(f"\n\n {inference_type} throughput warmup time: {elapsed_time:.4f} seconds using", get_nproc_per_node(), "GPUs")

    elapsed_times = []
    for i in range(10):
        start_time = time.perf_counter()
        inference.generate(inference_samples[i:i+1], output_dir=args.setup.output_dir)
        elapsed_time = time.perf_counter() - start_time
        elapsed_times.append(elapsed_time)
        if is_rank0():
            print(f"\n\n {inference_type} throughput step {i+1} time: {elapsed_time:.4f} seconds using", get_nproc_per_node(), "GPUs")

    if is_rank0():
        avg_time = statistics.mean(elapsed_times)
        std_time = statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0
        
        results = f"Inference Statistics:\nAverage: {avg_time:.4f} seconds\nStd Dev: {std_time:.4f} seconds\n"
        print(f"\n\n{results}")
        
        #with open("/tmp/inference_results.txt", "w") as f:
        #    f.write(results)


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
