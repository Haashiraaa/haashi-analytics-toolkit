# haashi_pkg

A modular Python toolkit for analytics workflows — focused on pragmatic, reusable building blocks for data ingestion, validation & cleaning, saving, and visualization.

Version: 0.2.0  
Author: Haashiraaa  
Requires: Python >= 3.10  
Core dependencies: pandas, numpy, pyarrow, matplotlib, seaborn

---

## Overview

haashi_pkg provides lightweight, well-scoped utilities that make common analytics tasks easier to compose and reuse across projects. Its design separates concerns into three primary areas:

- Data ingestion (reading CSV / Parquet)
- Data inspection, validation, cleaning, and aggregation
- Data output (CSV / Parquet) and plotting helpers
- Utility helpers for common supporting tasks (logging, error handling, small helpers)

The package aims to be minimal but practical — suitable for prototypes, ETL scripts, and exploratory data work where you want consistent, testable helpers rather than full-blown frameworks.

---

## Package layout (high level)

- haashi_pkg/
  - data_engine/
    - dataengine.py — core DataEngine class: inspection, validation, normalization, aggregation, missing-data helpers
    - dataloader.py — DataLoader: convenient CSV and Parquet loading helpers (single, multiple, chunked)
    - datasaver.py — DataSaver: write DataFrames to CSV and Parquet (including compressed Parquet)
  - plot_engine/
    - plotengine.py — plotting helpers built on matplotlib / seaborn to standardize common charting patterns
  - utility/
    - utils.py — utility helpers used across the package (logging, simple error handling, small helpers)

---

## Key concepts & highlights

- DataLoader
  - Lightweight I/O helpers for CSV and Parquet
  - Supports single-file, many-files, and chunked CSV reading
  - Uses pandas with pyarrow for Parquet

- DataEngine
  - Non-mutating inspection helpers (head, dtypes, shape)
  - Missing / duplicate counting and summaries
  - Validation helpers (required columns, numeric non-negative checks, datetime checks)
  - Type conversion helpers (numeric coercion, flexible datetime parsing)
  - Normalization (column name standardization, text normalization)
  - Missing-data handling (drop rows, forward/backward fill) and aggregation utilities

- DataSaver
  - Validate save paths and ensure expected extensions
  - Save CSV and Parquet (with optional compression)
  - Simple confirmation output after saves

- PlotEngine
  - Collection of standardized plotting functions (built on matplotlib / seaborn) to make charts consistent across analyses

- Utilities
  - Centralized helpers for logging and graceful error handling used by other modules

---

## Installation

Install the package and dependencies (example with pip) — adjust to your environment as needed:

```bash
git clone https://github.com/Haashiraaa/my-packages.git
cd haashiraaa_pkg


```
# then install the package (editable/installable from this folder):
```bash
pip install -e .
```


---

## Quickstart examples

Note: imports shown are explicit; adjust if you expose wrappers in package-level __init__.

- Load a CSV and inspect

```python
from haashi_pkg.data_engine.dataloader import DataLoader
from haashi_pkg.data_engine.dataengine import DataEngine

dl = DataLoader("data/my-file.csv")
df = dl.load_csv_single()
engine = DataEngine()
engine.inspect_dataframe(df)
print(engine.count_missing(df, "some_column"))
```

- Validate and aggregate

```python
# Ensure required columns exist
engine.validate_columns_exist(df, ["id", "value", "date"])

# Convert and aggregate
df["value"] = engine.convert_numeric(df["value"])
agg = engine.aggregate(df, value_col="value", group_cols="category", op="sum")
```

- Save a cleaned DataFrame

```python
from haashi_pkg.data_engine.datasaver import DataSaver
saver = DataSaver()
saver.save_csv(df, "output/cleaned.csv")
saver.save_parquet_compressed(df, "output/cleaned.parquet")
```

- Plotting (example)

```python
# Example usage of plotting helpers (API depends on the plot_engine module)
from haashi_pkg.plot_engine.plotengine import PlotEngine
pe = PlotEngine()
pe.simple_line(df, x="date", y="value", title="Value over time")
```

---

## Contributing

Contributions and improvements are welcome. A few suggestions to keep the project coherent:

- Keep functions focused and well-documented
- Add unit tests for validation and I/O edge cases (missing columns, bad types, large files / chunking)
- Prefer explicit and deterministic behavior (do not mutate inputs unless documented)
- Keep plotting helpers configurable (palette, figsize, labels) while offering sensible defaults

---

## Roadmap / TODOs (high-level)

- Add additional readers/writers (Excel, JSON)
- Expand validation rules and custom validators
- Add more plot types and examples
- Improve packaging metadata and CI (unit tests, linting)
- Provide higher-level examples / notebooks demonstrating common ETL flows

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
---

## Acknowledgements

Built with pandas, numpy, matplotlib, seaborn, and pyarrow as the core stack for data manipulation and visualization.
