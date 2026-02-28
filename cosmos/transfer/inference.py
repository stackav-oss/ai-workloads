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

ControlUnion = Annotated[
    Union[
        Annotated[EdgeConfig, tyro.conf.subcommand("edge")],
        Annotated[DepthConfig, tyro.conf.subcommand("depth")],
        Annotated[BlurConfig, tyro.conf.subcommand("vis")],
        Annotated[SegConfig, tyro.conf.subcommand("seg")],
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
    
    #args.setup.disable_guardrails=True
    #args.setup.offload_guardrail_models=True

    print("="*50, "args")
    print(args)
    #print("="*50, "input_files")
    #print(args.input_files)
    print("="*50, "setup")
    print(args.setup)
    print("="*50, "overrides")
    print(args.overrides)
    
    #inference_samples, batch_hint_keys = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    #print("="*50, "inference_samples")
    #print(inference_samples)
    #print("="*50, "batch_hint_keys")
    #print(batch_hint_keys)
    #print("="*50)
    #return

    inference_samples = []
    for i in range(600):
        task_id = f"task_{i:04d}"
        base_args = {
            "name": task_id,
            "prompt_path": f"/datasets/physical-ai-bench-conditional-generation/captions/{task_id}.json",
            "video_path": f"/datasets/physical-ai-bench-conditional-generation/videos/{task_id}.mp4",
            "edge": None,
            "depth": DepthConfig(),
            "vis": None,
            "seg": None
        }
        sample = InferenceArguments(**base_args)
        inference_samples.append(sample)
        print(sample)

        for j in range(1, 6):
            base_args = {
                "name": f"{task_id}_caption{j}",
                "prompt_path": f"/datasets/physical-ai-bench-conditional-generation/captions/{task_id}_caption{j}.json",
                "video_path": f"/datasets/physical-ai-bench-conditional-generation/videos/{task_id}.mp4",
                "edge": None,
                "depth": DepthConfig(),
                "vis": None,
                "seg": None
            }
            sample = InferenceArguments(**base_args)
            inference_samples.append(sample)
            print(sample)
        break

        
    return

    from cosmos_transfer2.inference import Control2WorldInference
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    batch_hint_keys=['depth']
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
