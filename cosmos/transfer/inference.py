# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Annotated, Union

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2._src.imaginaire.utils import log
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
    print(args)
    print("\n" + "="*50)
    print("INFERENCE CONFIGURATION")
    print("="*50)
    for field, value in args.__dict__.items():
        print(f"[{field.upper()}]:")
        print(value)
        print("-" * 30)
    print("="*50 + "\n")

    
    inference_samples = []
    for i in range(600):
        output_mp4 = args.setup.output_dir / f"task_{i:04d}.mp4"
        print(f"{output_mp4=}")
        return
        task_id = f"task_{i:04d}"
        base_args = {
            "name": task_id,
            "prompt_path": f"/datasets/physical-ai-bench-conditional-generation/captions/{task_id}.json",
            "video_path": f"/datasets/physical-ai-bench-conditional-generation/videos/{task_id}.mp4",
            "edge": EdgeConfig() if isinstance(args.control, (AllConfig, EdgeConfig)) else None,
            "depth": DepthConfig() if isinstance(args.control, (AllConfig, DepthConfig)) else None,
            "vis": BlurConfig() if isinstance(args.control, (AllConfig, BlurConfig)) else None,
            "seg": SegConfig() if isinstance(args.control, (AllConfig, SegConfig)) else None
        }
        sample = InferenceArguments(**base_args)
        inference_samples.append(sample)
        print(sample)
        continue
    
        # caption variations
        for j in range(1, 6):
            base_args = {
                "name": f"{task_id}_caption{j}",
                "prompt_path": f"/datasets/physical-ai-bench-conditional-generation/captions/{task_id}_caption{j}.json",
                "video_path": f"/datasets/physical-ai-bench-conditional-generation/videos/{task_id}.mp4",
                "edge": EdgeConfig() if isinstance(args.control, (AllConfig, EdgeConfig)) else None,
                "depth": DepthConfig() if isinstance(args.control, (AllConfig, DepthConfig)) else None,
                "vis": BlurConfig() if isinstance(args.control, (AllConfig, BlurConfig)) else None,
                "seg": SegConfig() if isinstance(args.control, (AllConfig, SegConfig)) else None
            }
            sample = InferenceArguments(**base_args)
            inference_samples.append(sample)
        

    from cosmos_transfer2.inference import Control2WorldInference
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    #CONTROL_KEYS = ["edge", "vis", "depth", "seg"]
    batch_hint_keys = []
    if isinstance(args.control, EdgeConfig):
        batch_hint_keys.append('edge')
    if isinstance(args.control, DepthConfig):
        batch_hint_keys.append('depth')
    if isinstance(args.control, BlurConfig):
        batch_hint_keys.append('vis')
    if isinstance(args.control, SegConfig):
        batch_hint_keys.append('seg')
    if isinstance(args.control, AllConfig):
        batch_hint_keys.extend(['edge', 'depth', 'vis', 'seg'])
    print(f"Batch hint keys: {batch_hint_keys}")

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
