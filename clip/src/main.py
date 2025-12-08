"""Main Entrypoint code."""

from pathlib import Path

import click
import numpy as np
from benchmarker import print_benchmark_report
from imagenet_eval import EvaluationConfig, ImagenetEvaluator, print_results
from PIL import Image
from vllm import LLM

speed_limit_classification_prompts = {
    "cat": "A photo of a cat.",
    "dog": "A photo of a dog.",
    "elephant": "A photo of an elephant.",
    "tiger": "A photo of a tiger.",
    "giraffe": "A photo of a giraffe.",
}


def demo(model: str) -> None:
    """Run a simple demo of CLIP evaluation."""
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.97,
        runner="pooling",
        limit_mm_per_prompt={"image": 1},
    )
    # Parse images folder and embed those images
    print("Running demo...")
    demo_image_folder = Path("data/images")
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    image_paths = [p for p in demo_image_folder.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]
    text_prompts = speed_limit_classification_prompts.values()
    for image_path in image_paths:
        label_class_name = image_path.stem
        # read PIL image
        image = Image.open(image_path).convert("RGB")
        image_prompt = [{"prompt": "", "multi_modal_data": {"image": image}}]
        similarities = []
        image_outputs = llm.embed(image_prompt)
        text_outputs = llm.embed(list(text_prompts))
        image_embed = np.array([output.outputs.embedding for output in image_outputs])
        text_embed = np.array([output.outputs.embedding for output in text_outputs])
        similarities = image_embed @ text_embed.T
        predicted_indices = np.argmax(similarities, axis=1)
        prediction = list(speed_limit_classification_prompts.keys())[predicted_indices[0]]
        print(f"Label class name: {label_class_name}, Predicted: {prediction}")
        print("Correct prediction!" if label_class_name == prediction else "Incorrect prediction.")
        print("-" * 40)


@click.command()
@click.option(
    "--imagenet-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to ImageNet root directory (parent of train/val folders)",
)
@click.option(
    "--model",
    default="openai/clip-vit-base-patch32",
    help="CLIP model name",
)
@click.option(
    "--prompt-template",
    default="a photo of a {}",
    help="Template for text prompts (use {} for class name placeholder)",
)
@click.option(
    "--batch-size",
    default=512,
    type=int,
    help="Batch size for processing images",
)
@click.option(
    "--ensemble-templates",
    is_flag=True,
    help="Ensemble the results of multiple templates per class",
)
@click.option("--demo", "is_demo", is_flag=True, default=False, help="Run in demo mode.")
def main(  # noqa: PLR0913
    imagenet_root: Path,
    model: str,
    prompt_template: str,
    batch_size: int,
    ensemble_templates: bool,
    is_demo: bool,
) -> None:
    """CLIP evaluation on image classification."""
    if is_demo:
        demo(model)
        return
    split = "val"
    config = EvaluationConfig(
        imagenet_root=imagenet_root,
        split=split,
        model=model,
        prompt_template=prompt_template,
        batch_size=batch_size,
        ensemble_templates=ensemble_templates,
        num_workers=8,
        prefetch_factor=2,
    )
    run_evaluation(config)


def run_evaluation(config: EvaluationConfig) -> None:
    """Run CLIP evaluation with the given parameters.

    Args:
        config: Evaluation configuration
    """
    print(f"Initializing CLIP model: {config.model}")
    evaluator = ImagenetEvaluator(config)
    results = evaluator.evaluate()
    print_results(results)
    print_benchmark_report()


if __name__ == "__main__":
    main()
