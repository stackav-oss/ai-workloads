"""Accuracy and throughput benchmarking whisper model using vllm."""

import logging
import pathlib
import time
from typing import Any, Final

import click
import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchaudio
import whisper
from tqdm import tqdm
from typing_extensions import override
from vllm import LLM, SamplingParams
from whisper.normalizers import EnglishTextNormalizer

_logger: Final = logging.getLogger(__name__)


_EXPECTED_SAMPLE_RATE = 16000
_GPU_MEMORY_UTILIZATION = 0.95
_DEFAULT_MAX_NUM_SEQS = 128
_SEED = 32
_SPLIT = "test-clean"
_SAMPLING_PARAMS = SamplingParams(temperature=0, top_p=1.0, max_tokens=200)
_CURRENT_DIR = pathlib.Path(__file__).parent
_ASSETS_DIR = _CURRENT_DIR / "media"
_AUDIO_FILES = ["demo_001.flac", "demo_002.flac", "demo_003.flac"]


def format_prompt(audio: np.ndarray, sample_rate: int = 16000) -> Any:  # pyright: ignore[reportUnknownParameterType, reportMissingTypeArgument] # noqa: ANN401
    """Format prompt."""
    return {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": (audio, sample_rate),
        },
    }


class LibriSpeech(torch.utils.data.Dataset[tuple[torch.Tensor, str]]):
    """A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.

    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, split: str) -> None:
        """Initialize the object."""
        cache_path = pathlib.Path(f"~/.cache/{split}").expanduser()
        cache_path.mkdir(exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=cache_path.expanduser(),
            url=split,
            download=True,
        )

    def __len__(self) -> int:
        """Return length."""
        return len(self.dataset)

    @override
    def __getitem__(self, item: int) -> tuple[Any, str]:
        """Return item."""
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == _EXPECTED_SAMPLE_RATE
        audio = whisper.pad_or_trim(audio.flatten())
        return audio, text


def calculate_results(start_time: float, end_time: float, hypotheses: list[str], references: list[str]) -> tuple[float, float]:
    """Calculate wer and throughput."""
    throughput = len(references) / (end_time - start_time)

    data = pd.DataFrame({"hypothesis": hypotheses, "reference": references})

    normalizer = EnglishTextNormalizer()

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]

    return jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"])) * 100, throughput


def benchmark_vllm(llm: LLM) -> tuple[float, float]:
    """Calculate and return wer and throughput using vLLM backend."""
    dataset = LibriSpeech(_SPLIT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    hypotheses = []
    references = []
    start_time = time.perf_counter()
    for mels, texts in tqdm(loader, leave=True):
        prompts = [format_prompt(mel.cpu().numpy().squeeze()) for mel in mels]
        results = llm.generate(prompts, _SAMPLING_PARAMS, use_tqdm=False)
        hypotheses.extend([result.outputs[0].text for result in results])
        references.extend(texts)
    end_time = time.perf_counter()

    return calculate_results(start_time, end_time, hypotheses, references)


def benchmark_openai(model: whisper.Whisper, batch_size: int) -> tuple[float, float]:
    """Calculate and return wer and throughput using OpenAI Whisper backend."""
    dataset = LibriSpeech(_SPLIT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    hypotheses = []
    references = []
    start_time = time.perf_counter()
    for mels, texts in tqdm(loader, leave=True):
        batched_audio = torch.stack([mel for mel in mels])
        batched_mel = whisper.log_mel_spectrogram(batched_audio, n_mels=model.dims.n_mels).to(model.device)

        options = whisper.DecodingOptions()
        results = whisper.decode(model, batched_mel, options)

        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    end_time = time.perf_counter()

    return calculate_results(start_time, end_time, hypotheses, references)


def demo_vllm(llm: LLM) -> None:
    """Run demo with sample audio files using vLLM backend."""
    prompts = []
    for file_name in _AUDIO_FILES:
        file_path = _ASSETS_DIR / file_name
        audio, sample_rate = librosa.load(file_path, sr=None)
        assert sample_rate == _EXPECTED_SAMPLE_RATE, (
            f"Expected sample rate {_EXPECTED_SAMPLE_RATE}, but got {sample_rate}"
        )
        prompts.append(format_prompt(audio))  # pyright: ignore[reportArgumentType]

    results = llm.generate(prompts, _SAMPLING_PARAMS, use_tqdm=False)
    _logger.info("\n%sDEMO TRANSCRIPTIONS%s", "=" * 70, "=" * 70)
    for result, audio_file in zip(results, _AUDIO_FILES, strict=False):
        text = result.outputs[0].text
        _logger.info("%s: transcription: %s", audio_file, text)


def demo_openai(model: whisper.Whisper) -> None:
    """Run demo with sample audio files using OpenAI Whisper backend."""
    _logger.info("\n%sDEMO TRANSCRIPTIONS%s", "=" * 70, "=" * 70)

    for file_name in _AUDIO_FILES:
        file_path = _ASSETS_DIR / file_name
        audio, sample_rate = librosa.load(file_path, sr=None)
        assert sample_rate == _EXPECTED_SAMPLE_RATE, (
            f"Expected sample rate {_EXPECTED_SAMPLE_RATE}, but got {sample_rate}"
        )
        audio = whisper.pad_or_trim(audio)
        result = model.transcribe(audio)
        _logger.info("%s: transcription: %s", file_name, result["text"])


def initialize_vllm(model_name: str, kv_cache_type: str, max_num_seqs: int) -> LLM:
    """Initialize vLLM model."""
    return LLM(
        model=model_name,
        max_model_len=448,
        kv_cache_dtype=kv_cache_type,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=_GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,
        disable_log_stats=False,
        seed=_SEED,
    )


def initialize_openai(model: str) -> whisper.Whisper:
    """Initialize OpenAI Whisper model."""
    return whisper.load_model(model)


@click.command()
@click.option(
    "--model",
    type=click.Choice(["large-v3", "large-v3-turbo"]),
    required=True,
)
@click.option(
    "--backend",
    type=click.Choice(["vllm", "openai"]),
    default="vllm",
    help="Backend to use for inference (default: vllm).",
)
@click.option(
    "--kv-cache-type",
    default="auto",
    type=click.Choice(["fp8", "auto"]),
    help="KV cache type (vLLM only).",
)
@click.option(
    "--max-num-seqs",
    default=_DEFAULT_MAX_NUM_SEQS,
    type=int,
    help="Maximum number of sequences (vLLM only).",
)
@click.option(
    "--batch-size",
    default=16,
    type=int,
    help="Batch size for processing audio files (OpenAI backend only).",
)
@click.option("--demo", "is_demo", is_flag=True, default=False, help="Run in demo mode.")
def main(model: str, backend: str, kv_cache_type: str, max_num_seqs: int, batch_size: int, is_demo: bool) -> None:
    """Main method."""
    logging.basicConfig(level=logging.INFO)

    if backend == "openai":
        _logger.info("Using OpenAI Whisper backend with model: %s", model)
        openai_model = initialize_openai(model=model)
        if is_demo:
            _logger.info("Running in demo mode with reduced dataset size.")
            demo_openai(openai_model)
            return

        wer, throughput = benchmark_openai(model=openai_model, batch_size=batch_size)
        _logger.info("\n%sBENCHMARK RESULTS%s", "=" * 70, "=" * 70)
        _logger.info(
                "  WER: %.2f   |   throughput: %.2f reqs/sec  |   model: %-10s   |   batch_size: %s   |   backend: openai\n\n",
            wer,
            throughput,
            model,
            batch_size,
        )
    else:
        model_name = f"openai/whisper-{model}"
        llm = initialize_vllm(model_name=model_name, kv_cache_type=kv_cache_type, max_num_seqs=max_num_seqs)
        if is_demo:
            _logger.info("Running in demo mode with reduced dataset size.")
            demo_vllm(llm)
            return

        wer, throughput = benchmark_vllm(llm=llm)
        _logger.info("\n%sBENCHMARK RESULTS%s", "=" * 70, "=" * 70)
        _logger.info(
            "  WER: %.2f   |   throughput: %.2f reqs/sec  |   model: %-10s   |   kv_cache_type: %-4s  |  max_num_seqs: %s\n\n",
            wer,
            throughput,
            model,
            kv_cache_type,
            max_num_seqs,
        )


if __name__ == "__main__":
    main()
