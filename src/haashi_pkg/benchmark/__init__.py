

# benchmark/__init__.py


"""Benchmarking and performance profiling utilities."""

from .benchmark import Benchmark
from .exceptions import (
    BenchmarkError,
    InvalidFunctionError,
    BenchmarkTimeoutError
)

__all__ = [
    'Benchmark',
    'BenchmarkError',
    'InvalidFunctionError',
    'BenchmarkTimeoutError'
]


__version__ = '1.2.0'
