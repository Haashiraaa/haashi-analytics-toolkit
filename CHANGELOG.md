

# Changelog

All notable changes to haashi-analytics-toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Memory profiling utilities
- Benchmark decorators for automatic timing
- Database connection benchmarking

---

## [1.1.0] - 2026-02-23

### Added
- **New module**: `haashi_pkg.benchmark` for performance profiling
- `Benchmark` class with `measure_time()` method for timing function execution
- `ScreenUtil.wait_and_enter()` method for interactive pausing in pipelines
- Custom exceptions for benchmark module:
  - `BenchmarkError` - Base exception for benchmark operations
  - `InvalidFunctionError` - Raised when function is not callable
  - `BenchmarkTimeoutError` - For future timeout handling

### Example Usage

**Performance Timing:**
```python
from haashi_pkg.benchmark import Benchmark

bench = Benchmark()

def my_function():
    return sum(range(1000000))

# Measure with default 5 runs
avg_time = bench.measure_time(my_function)

# Measure with custom runs for better accuracy
avg_time = bench.measure_time(my_function, run_times=10)
print(f"Average: {avg_time:.4f}s")
```

**Interactive Pausing:**
```python
from haashi_pkg.utility import ScreenUtil as su



# Simple pause
su.wait_and_enter()

# Custom message
su.wait_and_enter("Review the output before continuing...")

# Use in pipeline
def run_pipeline():
    step_1()
    su.wait_and_enter("Check output. Press Enter for step 2...")
    step_2()
```

---

## [1.0.0] - 2026-02-10

### Added
- Initial stable release
- **utility module**:
  - `Logger` class for structured logging with JSON error support
  - `FileHandler` class for file operations (read/write JSON, CSV, etc.)
- **data_engine module**:
  - `DataLoader` for loading CSV, Parquet, Excel files
  - `DataAnalyzer` for data validation, inspection, and analysis
  - `DataSaver` for saving data in multiple formats
- **plot_engine module**:
  - `PlotEngine` for creating professional data visualizations
  - Custom color palettes and themes
  - Support for line plots, bar charts, scatter plots
  - Statistics text boxes and legends

### Changed
- Migrated from deprecated `Utility` class to `Logger`/`FileHandler`
- Improved module organization with clear separation of concerns

### Fixed
- Logger conflicts when initializing `PlotEngine`
- `PlotEngine` now accepts logger parameter to prevent logging hijacking
- Proper error handling in all file operations

---

## Notes

### Version Numbering
- **MAJOR** version: Incompatible API changes
- **MINOR** version: New features (backward compatible)
- **PATCH** version: Bug fixes (backward compatible)

### Contributing
When making changes, add them to the `[Unreleased]` section first.
Move to a versioned section when ready to release.
