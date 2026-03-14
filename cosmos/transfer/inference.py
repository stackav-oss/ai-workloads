from pathlib import Path
from typing import Annotated, Union

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


def prepare_samples(args, offset, size):
    inference_samples = []
    for i in range(offset, offset + size):
        task_id = f"task_{i:04d}"
        # main output variant
        variant_id = task_id if args.caption_variant == 0 else f"{task_id}_caption{args.caption_variant}"
        output_video = args.setup.output_dir / f"{variant_id}.mp4"
        if not output_video.exists():
            original_video = f"/datasets/physical-ai-bench-conditional-generation/videos/{task_id}.mp4"
            edge_config = EdgeConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/canny/{task_id}.mp4") if isinstance(args.control, (AllConfig, EdgeConfig)) else None
            depth_config = DepthConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/depth_vids/{task_id}.mp4") if isinstance(args.control, (AllConfig, DepthConfig)) else None
            blur_config = BlurConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/blur/{task_id}.mp4") if isinstance(args.control, (AllConfig, BlurConfig)) else None
            seg_config = SegConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/sam2_vids/{task_id}.mp4") if isinstance(args.control, (AllConfig, SegConfig)) else None

            base_args = {
                "name": variant_id,
                "prompt_path": f"/datasets/physical-ai-bench-conditional-generation/captions/{variant_id}.json",
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

def main(args: Args):
    NUMBER_OF_TASKS = 600
    BATCH_SIZE = 40

    from cosmos_transfer2.inference import Control2WorldInference
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    # Get batch hint keys based on active control type
    batch_hint_keys = CONTROL_TYPE_TO_HINTS.get(type(args.control), [])
    print(f"Using batch hint keys: {batch_hint_keys} for control type: {type(args.control).__name__}")
    inference = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)

    for batch_offset in range(0, NUMBER_OF_TASKS, BATCH_SIZE):
        inference_samples = prepare_samples(args, batch_offset, BATCH_SIZE)
        if not inference_samples:
            print(f"All outputs for batch starting at {batch_offset} already exist, skipping this batch.", flush=True)
            continue
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

    print("\n" + "="*50)
    print("INFERENCE CONFIGURATION")
    print("="*50)
    for field, value in args.__dict__.items():
        print(f"[{field.upper()}]: {type(value)}")
        print(value)
        print("-" * 30)
    print("="*50 + "\n")
    
    #from cosmos_transfer2.inference import Control2WorldInference
    #init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    ## Get batch hint keys based on active control type
    #batch_hint_keys = CONTROL_TYPE_TO_HINTS.get(type(args.control), [])
    #print(f"Using batch hint keys: {batch_hint_keys} for control type: {type(args.control).__name__}")
    #inference_engine = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)
    #for key, value in inference_engine.__dict__.items():
    #    print(f"[{key.upper()}]: {(value)}")


    main(args)

    cleanup_environment()
