"""Benchmarking utilities for vLLM inference metrics.

This module provides decorators and classes for tracking vLLM inference performance,
focusing on overall throughput calculation.
"""

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from tabulate import tabulate

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class _BenchmarkMetrics:
    """Metrics for a single benchmark measurement."""

    duration_samples: list[float] = field(default_factory=list)
    total_items: int = 0
    call_count: int = 0


# Global metrics storage
_METRICS: dict[str, _BenchmarkMetrics] = {}


def _print_benchmark_report() -> None:
    """Print formatted benchmark report."""
    if not _METRICS:
        return

    print("\n" + "=" * 80)
    print("VLLM INFERENCE BENCHMARK REPORT")
    print("=" * 80)

    headers = ["Operation", "Calls", "Overall Throughput\n(items/s)"]
    table_data = []

    for name in sorted(_METRICS.keys()):
        metric = _METRICS[name]
        total_time = sum(metric.duration_samples)
        throughput = metric.total_items / total_time if total_time > 0 else 0.0
        table_data.append([name, metric.call_count, f"{throughput:.2f}"])

    print(tabulate(table_data, headers=headers, tablefmt="pretty", numalign="right", stralign="left"))
    print("=" * 80)


def benchmark_vllm(
    name: str | None = None,
    *,
    track_throughput: bool = False,
) -> Callable[[F], F]:
    """Decorator to benchmark vLLM inference calls.

    Args:
        name: Custom name for the benchmark (defaults to function name)
        track_throughput: Whether to track throughput based on return value length

    Returns:
        Decorated function that tracks vLLM inference metrics
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            benchmark_name = name or f"{func.__module__}.{func.__qualname__}"

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            # Calculate item count if requested
            num_items = 0
            if track_throughput:
                try:
                    num_items = len(result) if hasattr(result, "__len__") else 0
                except TypeError:
                    num_items = 0

            # Add or update metrics
            if benchmark_name not in _METRICS:
                _METRICS[benchmark_name] = _BenchmarkMetrics()

            metric = _METRICS[benchmark_name]
            metric.duration_samples.append(duration)
            metric.total_items += num_items
            metric.call_count += 1

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def print_benchmark_report() -> None:
    """Print the benchmark report and return it."""
    _print_benchmark_report()
