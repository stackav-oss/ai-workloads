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


def main(
    args: Args,
):
    print("\n" + "="*50)
    print("INFERENCE CONFIGURATION")
    print("="*50)
    for field, value in args.__dict__.items():
        print(f"[{field.upper()}]:")
        print(value)
        print("-" * 30)
    print("="*50 + "\n")

    control_video_mapping = {
        "blur": "blur",
        "edge": "canny",
        "depth": "depth_vids",
        "seg": "sam2_vids",
    }

    #DEFAULT_NUM_OUTPUT_FRAMES = 33
    DEFAULT_NUM_STEPS = 35
    DEFAULT_SEED = 0
    DEFAULT_GUIDANCE = 4

    inference_samples = []
    for i in range(0, 4):
        task_id = f"task_{i:04d}"
        original_video = f"/datasets/physical-ai-bench-conditional-generation/videos/{task_id}.mp4"
        edge_config = EdgeConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/canny/{task_id}.mp4") if isinstance(args.control, (AllConfig, EdgeConfig)) else None
        depth_config = DepthConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/depth_vids/{task_id}.mp4") if isinstance(args.control, (AllConfig, DepthConfig)) else None
        blur_config = BlurConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/blur/{task_id}.mp4") if isinstance(args.control, (AllConfig, BlurConfig)) else None
        seg_config = SegConfig(control_path=f"/datasets/physical-ai-bench-conditional-generation/sam2_vids/{task_id}.mp4") if isinstance(args.control, (AllConfig, SegConfig)) else None

        # main output variant
        variant_id = task_id
        output_video = args.setup.output_dir / f"{variant_id}.mp4"
        if not output_video.exists():
            base_args = {
                "name": variant_id,
                "prompt_path": f"/datasets/physical-ai-bench-conditional-generation/captions/{variant_id}.json",
                "video_path": original_video,
                "edge": edge_config,
                "depth": depth_config,
                "vis": blur_config,
                "seg": seg_config,
                "num_steps": DEFAULT_NUM_STEPS,
                "seed": DEFAULT_SEED,
                "guidance": DEFAULT_GUIDANCE
                #"num_output_frames": DEFAULT_NUM_OUTPUT_FRAMES
            }
            sample = InferenceArguments(**base_args)
            inference_samples.append(sample)
        else:
            print(f"Output for {variant_id} already exists, skipping inference.")

        # caption variations
        #for j in range(1, 6):
        #    variant_id = f"{task_id}_caption{j}"
        #    output_video = args.setup.output_dir / f"{variant_id}.mp4"
        #    if output_video.exists():
        #        print(f"Output for {variant_id} already exists, skipping inference.")
        #        continue
        #    base_args = {
        #        "name": variant_id,
        #        "prompt_path": f"/datasets/physical-ai-bench-conditional-generation/captions/{variant_id}.json",
        #        "video_path": original_video,
        #        "edge": edge_config if isinstance(args.control, (AllConfig, EdgeConfig)) else None,
        #        "depth": depth_config if isinstance(args.control, (AllConfig, DepthConfig)) else None,
        #        "vis": blur_config if isinstance(args.control, (AllConfig, BlurConfig)) else None,
        #        "seg": seg_config if isinstance(args.control, (AllConfig, SegConfig)) else None
        #    }
        #    sample = InferenceArguments(**base_args)
        #    inference_samples.append(sample)
        
    from cosmos_transfer2.inference import Control2WorldInference
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    # Get batch hint keys based on active control type
    batch_hint_keys = CONTROL_TYPE_TO_HINTS.get(type(args.control), [])
    print(f"Using batch hint keys: {batch_hint_keys} for control type: {type(args.control).__name__}")
    inference = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)
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
