"""CLIP evaluation script for image classification."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from benchmarker import benchmark_vllm
from imagenet_classes import IMAGENET_CLASSES, imagenet_templates
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


@dataclass
class EvaluationConfig:
    """Configuration for CLIP evaluation."""

    imagenet_root: Path
    split: str
    model: str
    prompt_template: str
    batch_size: int
    ensemble_templates: bool
    num_workers: int = 4  # Number of data loading workers
    prefetch_factor: int = 1  # Batches to prefetch per worker
    pin_memory: bool = True  # Pin memory for faster GPU transfer


@dataclass
class EvaluationResults:
    """Results from CLIP evaluation."""

    accuracy: float
    correct: int
    total: int


class ImagenetEvaluator:
    """Main CLIP evaluation orchestrator."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize CLIP evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.llm = LLM(
            model=config.model,
            gpu_memory_utilization=0.97,
            runner="pooling",
            limit_mm_per_prompt={"image": 1},
        )

    def load_dataset(self) -> ImageNet:
        """Load ImageNet dataset with transform to convert PIL Images to tensors.

        Returns:
            ImageNet dataset instance
        """
        print(f"Loading ImageNet {self.config.split} dataset...")

        # # Define transform to convert PIL Images to tensors
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
                transforms.Resize((384, 384)),
            ]
        )

        dataset = ImageNet(
            root=str(self.config.imagenet_root),
            split=self.config.split,
            transform=transform,
        )
        print(f"Loaded {len(dataset)} images from {len(dataset.classes)} classes")
        return dataset

    @benchmark_vllm(name="vllm.embed_text", track_throughput=True)
    def _embed_text_batch(self, prompts: list[str] | list[dict[str, list[int]]]) -> NDArray[np.float32]:
        """Embed a batch of text prompts using vLLM.

        Args:
            prompts: List of text prompts or tokenized prompts

        Returns:
            Text embeddings array
        """
        outputs = self.llm.embed(prompts)
        return np.array([output.outputs.embedding for output in outputs])

    @benchmark_vllm(name="vllm.embed_images", track_throughput=True)
    def _embed_images(self, images: list[Image.Image] | torch.Tensor) -> NDArray[np.float32]:
        """Generate and normalize image embeddings.

        Args:
            images: List of PIL images or torch.Tensor of shape (B, C, H, W)

        Returns:
            Normalized image embeddings array
        """
        # Handle tensor input - convert to list of PIL images for VLLM
        if isinstance(images, torch.Tensor):
            to_pil = transforms.ToPILImage()
            images = [to_pil(img) for img in images]

        inputs = [{"prompt": "", "multi_modal_data": {"image": img}} for img in images]
        outputs = self.llm.embed(inputs)
        embeddings = np.array([output.outputs.embedding for output in outputs])
        # Normalize for zero-shot classification
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def get_siglip_tokenized(self, text_prompts: list[list[str]]) -> list[list[int]]:
        """Tokenize using huggingface because VLLM to mainintain padding = max_length."""
        hf_tokenizer = AutoTokenizer.from_pretrained(self.config.model.lower())
        tokenized_inputs = []
        is_siglip2 = "siglip2" in self.config.model.lower()
        for class_prompts in text_prompts:
            # Tokenize all prompts for this class
            class_tokens = []
            for prompt in class_prompts:
                # For SigLIP 2, lowercase and remove common punctuation
                processed_prompt = prompt
                if is_siglip2:
                    processed_prompt = prompt.lower()
                    punctuation = ".,!?;:'\"-"
                    processed_prompt = processed_prompt.translate(str.maketrans("", "", punctuation))

                token_ids = hf_tokenizer.encode(processed_prompt, padding="max_length", max_length=64)
                class_tokens.append(token_ids)

            # Store as list of dicts - vLLM expects this format
            tokenized_inputs.append([{"prompt_token_ids": token_ids} for token_ids in class_tokens])

        return tokenized_inputs

    def prepare_text_embeddings(
        self,
        class_names: list[str],
    ) -> NDArray[np.float32]:
        """Prepare text embeddings for all classes.

        Args:
            class_names: List of class names

        Returns:
            Text embeddings array
        """
        # Generate text prompts
        if self.config.ensemble_templates:
            print("Using ensemble of templates for text prompts.")
            text_prompts = [
                [template.format(class_name) for template in imagenet_templates] for class_name in class_names
            ]
        else:
            text_prompts = [[self.config.prompt_template.format(class_name)] for class_name in class_names]

        # Tokenize for SigLIP models
        if "siglip" in self.config.model.lower():
            text_prompts = self.get_siglip_tokenized(text_prompts)

        # Embed text prompts
        embeddings_list = []
        for prompts in tqdm(text_prompts, desc="Embedding text prompts"):
            embeddings = self._embed_text_batch(prompts)
            # Average across templates and normalize
            mean_embedding = np.mean(embeddings, axis=0)
            normalized_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            embeddings_list.append(normalized_embedding)

        text_embeddings = np.array(embeddings_list)
        # Final normalization for zero-shot classification
        return text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    def process_batch(
        self,
        batch_images: torch.Tensor,
        text_embeddings: NDArray[np.float32],
        class_names: list[str],
    ) -> list[str]:
        """Process a batch of images and predict classes.

        Args:
            batch_images: Tensor of images with shape (B, C, H, W)
            text_embeddings: Text embeddings for all classes
            class_names: List of class names

        Returns:
            List of predicted class names
        """
        image_embeddings = self._embed_images(batch_images)
        similarities = image_embeddings @ text_embeddings.T
        predicted_indices = np.argmax(similarities, axis=1)
        return [class_names[idx] for idx in predicted_indices]

    def evaluate(self) -> EvaluationResults:
        """Run full evaluation pipeline with DataLoader.

        Returns:
            Evaluation results
        """
        # Load dataset and create dataloader
        dataset = self.load_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.num_workers > 0,
            drop_last=False,
        )
        class_names = [IMAGENET_CLASSES[i] for i in range(len(dataset.classes))]
        print(f"Classes: {len(class_names)}")

        # Prepare text embeddings
        text_embeddings = self.prepare_text_embeddings(class_names)

        # Process images in batches with DataLoader
        all_predictions = []
        all_true_labels = []

        print(
            f"Processing {len(dataset)} images with batch_size={self.config.batch_size}, "
            + f"num_workers={self.config.num_workers}, prefetch_factor={self.config.prefetch_factor}"
        )

        for batch_images, batch_label_indices in tqdm(dataloader, desc="Processing images"):
            # batch_images is a tensor of shape (B, C, H, W)
            # batch_label_indices is a tensor of shape (B,)

            # Convert label indices to class names
            batch_labels = [class_names[idx.item()] for idx in batch_label_indices]

            # Process batch (batch_images is a tensor)
            predictions = self.process_batch(batch_images, text_embeddings, class_names)
            all_predictions.extend(predictions)
            all_true_labels.extend(batch_labels)

        # Calculate metrics
        correct = sum(pred == true for pred, true in zip(all_predictions, all_true_labels, strict=False))
        accuracy = correct / len(all_true_labels) if all_true_labels else 0.0

        return EvaluationResults(
            accuracy=accuracy,
            correct=correct,
            total=len(all_true_labels),
        )


def print_results(results: EvaluationResults) -> None:
    """Print evaluation results.

    Args:
        results: Evaluation results to print
    """
    print("\n" + "=" * 80)
    print("OVERALL ACCURACY SUMMARY")
    print("=" * 80)
    print(f"Total Samples:        {results.total:>10d}")
    print(f"Correct Predictions:  {results.correct:>10d}")
    print(f"Incorrect:            {results.total - results.correct:>10d}")
    print(f"\nOverall Accuracy:     {results.accuracy:>10.4f} ({results.accuracy * 100:.2f}%)")
    print("=" * 80)
