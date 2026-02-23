

# benchmark/benchmark.py


"""Benchmarking and performance profiling utilities."""

import timeit
import logging
from typing import Callable, Any, Optional

from haashi_pkg.utility import Logger
from .exceptions import BenchmarkError, InvalidFunctionError


class Benchmark:
    """
    Performance benchmarking and timing utilities.

    Provides methods to measure execution time and profile functions.

    Example:
        >>> from haashi_pkg.benchmark import Benchmark
        >>> 
        >>> bench = Benchmark()
        >>> 
        >>> def my_function():
        ...     return sum(range(1000000))
        >>> 
        >>> avg_time = bench.measure_time(my_function, run_times=10)
        [INFO] Average execution time: 0.0234 seconds
        >>> print(f"Function took {avg_time:.4f}s on average")
        Function took 0.0234s on average
    """

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize Benchmark with optional logger.

        Args:
            logger: Optional Logger instance (default: creates INFO logger)
        """
        self.logger = logger or Logger(level=logging.INFO)

    def measure_time(
        self,
        func: Callable[[], Any],
        run_times: int = 5
    ) -> float:
        """
        Measure average execution time of a function.

        Runs the function multiple times and returns the average execution time.
        Useful for performance testing and optimization.

        Args:
            func: Callable function to measure (must take no arguments)
            run_times: Number of times to run for averaging (default: 5)

        Returns:
            Average execution time in seconds (float)

        Raises:
            ValueError: If run_times is less than 1
            InvalidFunctionError: If func is not callable
            BenchmarkError: If function execution fails

        Example:
            >>> bench = Benchmark()
            >>> def expensive_operation():
            ...     result = 0
            ...     for i in range(1000000):
            ...         result += i
            ...     return result
            >>> 
            >>> # Measure with default 5 runs
            >>> time_taken = bench.measure_time(expensive_operation)
            [INFO] Average execution time: 0.0456 seconds
            >>> 
            >>> # Measure with 10 runs for better accuracy
            >>> time_taken = bench.measure_time(expensive_operation, run_times=10)
            [INFO] Average execution time: 0.0449 seconds
        """
        # Validate inputs
        if run_times < 1:
            raise ValueError(f"run_times must be at least 1, got {run_times}")

        if not callable(func):
            raise InvalidFunctionError(
                f"Expected callable function, got {type(func).__name__}"
            )

        try:
            self.logger.debug(f"Running benchmark {run_times} times...")

            # Use timeit for accurate timing
            total_time = timeit.timeit(func, number=run_times)
            average_time = total_time / run_times

            self.logger.info(
                f"Average execution time: {average_time:.4f} seconds")

            return average_time

        except InvalidFunctionError:
            # Re-raise our custom exception
            raise

        except Exception as e:
            error_msg = f"Failed to benchmark function: {str(e)}"
            self.logger.error(error_msg)
            raise BenchmarkError(error_msg) from e

