"""Inference script."""

from pathlib import Path
from typing import Annotated
import json
import os
import sys
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

import os
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

        # UGUR DEBUG
        #if len(inference_samples)>20:
        #    break
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    inference = Inference(args.setup)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


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
