"""Opt-in runtime, memory, and power benchmarking helpers."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])
BYTES_PER_GIB = 1024**3
MIB_PER_GIB = 1024


def _bytes_to_gib(value: float) -> float:
    return float(value) / BYTES_PER_GIB


def _mib_to_gib(value: float) -> float:
    return float(value) / MIB_PER_GIB


def _current_rss_bytes() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", "r", encoding="utf-8") as statm:
            rss_pages = int(statm.read().split()[1])
        return rss_pages * page_size
    except (OSError, IndexError, ValueError):
        return 0


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _physical_gpu_index(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    local_index = device.index
    if local_index is None:
        local_index = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if 0 <= local_index < len(entries) and entries[local_index].isdigit():
            return int(entries[local_index])
    return int(local_index)


@dataclass
class BenchmarkSample:
    duration_s: float
    items: int
    cpu_rss_end_gib: float
    gpu_peak_alloc_gib: float


@dataclass
class PowerSample:
    power_w: float
    gpu_memory_used_gib: float
    gpu_utilization_pct: float


class NvidiaSmiProbe:
    def __init__(self, device: torch.device) -> None:
        self.gpu_index = _physical_gpu_index(device)
        self.available = self.gpu_index is not None

    def sample(self) -> PowerSample | None:
        if not self.available:
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=power.draw,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            self.available = False
            return None
        if result.returncode != 0:
            self.available = False
            return None
        try:
            power_w, memory_mib, utilization_pct = (
                float(part.strip())
                for part in result.stdout.strip().splitlines()[0].split(",")[:3]
            )
        except (IndexError, ValueError):
            return None
        return PowerSample(
            power_w=power_w,
            gpu_memory_used_gib=_mib_to_gib(memory_mib),
            gpu_utilization_pct=utilization_pct,
        )


class BenchmarkRecorder:
    """Collect runtime, memory, and power samples for named benchmark phases."""

    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        phase_limits: dict[str, int],
        rank: int = 0,
        power_interval_s: float = 0.5,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.rank = rank
        self.phase_limits = {
            key: max(0, int(value)) for key, value in phase_limits.items()
        }
        self.phase_counts = {key: 0 for key in self.phase_limits}
        self.samples: dict[str, list[BenchmarkSample]] = {
            key: [] for key in self.phase_limits
        }
        self.power_samples: dict[str, list[PowerSample]] = {
            key: [] for key in self.phase_limits
        }
        self._probe = NvidiaSmiProbe(device)
        self._power_interval_s = max(0.1, float(power_interval_s))
        self._active_phase: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.enabled and self._probe.available:
            self._thread = threading.Thread(target=self._sample_power, daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def should_record(self, phase: str) -> bool:
        return (
            self.enabled
            and self.phase_limits.get(phase, 0) > 0
            and self.phase_counts.get(phase, 0) < self.phase_limits[phase]
        )

    @contextlib.contextmanager
    def measure(self, phase: str, *, items: int = 0) -> Iterator[None]:
        if not self.should_record(phase):
            yield
            return

        _cuda_sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started_at = time.perf_counter()

        with self._lock:
            self._active_phase = phase
        try:
            yield
        finally:
            _cuda_sync(self.device)
            duration_s = time.perf_counter() - started_at
            with self._lock:
                self._active_phase = None

            cpu_rss_end = _current_rss_bytes()
            if self.device.type == "cuda":
                gpu_peak_alloc = torch.cuda.max_memory_allocated(self.device)
            else:
                gpu_peak_alloc = 0

            sample = BenchmarkSample(
                duration_s=duration_s,
                items=max(0, int(items)),
                cpu_rss_end_gib=_bytes_to_gib(cpu_rss_end),
                gpu_peak_alloc_gib=_bytes_to_gib(gpu_peak_alloc),
            )
            self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
            if self.rank == 0:
                self.samples.setdefault(phase, []).append(sample)

    def measure_decorator(
        self,
        phase: str,
        *,
        items_getter: Callable[..., int] | None = None,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                items = items_getter(*args, **kwargs) if items_getter else 0
                with self.measure(phase, items=items):
                    return fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def _sample_power(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                phase = self._active_phase
            if phase:
                sample = self._probe.sample()
                if sample is not None:
                    with self._lock:
                        self.power_samples.setdefault(phase, []).append(sample)
            self._stop.wait(self._power_interval_s)

    @staticmethod
    def _max_field(samples: list[BenchmarkSample], field: str) -> float:
        return max(getattr(sample, field) for sample in samples)

    def _phase_summary(self, phase: str) -> dict[str, float | int | None]:
        samples = self.samples.get(phase, [])
        if not samples:
            return {"steps": 0}

        durations = [sample.duration_s for sample in samples]
        total_duration_s = sum(durations)
        total_items = sum(sample.items for sample in samples)
        power_samples = self.power_samples.get(phase, [])
        powers = [sample.power_w for sample in power_samples]
        gpu_memory = [sample.gpu_memory_used_gib for sample in power_samples]
        gpu_utilization = [sample.gpu_utilization_pct for sample in power_samples]

        return {
            "steps": len(samples),
            "samples": total_items,
            "mean_s_per_batch": statistics.fmean(durations),
            "samples_per_s": (
                total_items / total_duration_s if total_duration_s > 0 else 0.0
            ),
            "pytorch_peak_alloc_max_gib": self._max_field(
                samples, "gpu_peak_alloc_gib"
            ),
            "cpu_rss_max_gib": self._max_field(samples, "cpu_rss_end_gib"),
            "power_mean_w": statistics.fmean(powers) if powers else None,
            "power_max_w": max(powers) if powers else None,
            "gpu_device_memory_max_gib": max(gpu_memory) if gpu_memory else None,
            "gpu_utilization_mean_pct": (
                statistics.fmean(gpu_utilization) if gpu_utilization else None
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "phases": {
                phase: self._phase_summary(phase)
                for phase in self.phase_limits
                if self.samples.get(phase)
            },
        }

    def print_summary(self, title: str = "LiDAR VAE Benchmark Summary") -> None:
        if not self.enabled or self.rank != 0:
            return
        self.close()
        summary = self.summary()
        print(f"\n=== {title} ===", flush=True)
        for phase, values in summary["phases"].items():
            print(
                f"{phase}: "
                f"samples={values['samples']} "
                f"mean_s_per_batch={values['mean_s_per_batch']:.4f} "
                f"samples_per_s={values['samples_per_s']:.3f}",
                flush=True,
            )
        print(
            "BENCHMARK_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True),
            flush=True,
        )


def benchmark_phase(
    recorder: BenchmarkRecorder,
    phase: str,
    *,
    items_getter: Callable[..., int] | None = None,
) -> Callable[[F], F]:
    return recorder.measure_decorator(phase, items_getter=items_getter)


def print_benchmark_summary(
    recorder: BenchmarkRecorder,
    *,
    title: str = "LiDAR VAE Benchmark Summary",
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            finally:
                recorder.print_summary(title)

        return wrapper  # type: ignore[return-value]

    return decorator
