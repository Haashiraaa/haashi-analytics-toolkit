# haashi_pkg

**A professional Python toolkit for data analytics workflows** — providing modular, well-documented utilities for data ingestion, validation, transformation, visualization, and common development tasks.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-green)](https://github.com/Haashiraaa/haashi_pkg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Version:** 1.0.0  
**Author:** Haashiraaa  
**Python:** ≥ 3.10  
**Core Stack:** pandas, numpy, matplotlib, seaborn, pyarrow

---

## Overview

`haashi_pkg` provides production-ready, well-documented utilities that streamline common analytics tasks. Designed with clean architecture principles, the package separates concerns into distinct modules while maintaining cohesive workflows.

**Perfect for:**
- Data pipelines and ETL workflows
- Exploratory data analysis
- Data validation and quality assurance
- Automated reporting and visualization
- Prototype development
- Analytics scripts and notebooks

**Key Principles:**
- **Comprehensive Documentation**: Every function fully documented with examples
- **Robust Error Handling**: Custom exceptions with clear, actionable messages
- **Backward Compatible**: Deprecated features supported with migration warnings
- **Type-Safe**: Full type hints throughout
- **Production-Ready**: Professional code suitable for enterprise use

---

## Package Structure

```
haashi_pkg/
├── data_engine/          # Data loading, analysis, and saving
│   ├── data_engine.py    # Main module (DataAnalyzer, DataLoader, DataSaver)
│   ├── dataengine.py     # Deprecated wrapper (backward compatibility)
│   ├── dataloader.py     # Deprecated wrapper (backward compatibility)
│   └── datasaver.py      # Deprecated wrapper (backward compatibility)
│
├── plot_engine/          # Visualization utilities
│   └── plotengine.py     # PlotEngine class for matplotlib/seaborn workflows
│
└── utility/              # Core utilities
    └── utils.py          # Logger, FileHandler, ScreenUtil, DateTimeUtil, etc.
```

---

## Features by Module

### **Data Engine** (`haashi_pkg.data_engine`)

#### **DataAnalyzer** (formerly DataEngine)
Core data analysis, validation, and transformation utilities.

**Capabilities:**
- **Inspection**: Non-mutating data exploration (head, dtypes, shape, missing counts, duplicates)
- **Validation**: Column existence, numeric ranges, datetime types, data quality checks
- **Type Conversion**: Flexible numeric/datetime parsing with error handling
- **Normalization**: Column names, text values (case/whitespace)
- **Missing Data**: Summary stats, forward/backward fill, row dropping
- **Aggregation**: Group-by operations with single or multiple functions
- **Joins**: Validated merges with relationship checks

#### **DataLoader**
Lightweight I/O for loading tabular data.

**Supported Formats:**
- CSV (single, multiple, chunked streaming)
- Excel (.xlsx via openpyxl)
- Parquet (single, multiple via pyarrow)

**Features:**
- Automatic delimiter detection (CSV)
- Memory-efficient chunked reading
- Path validation and error handling
- Flexible header/skip row handling

#### **DataSaver**
Save DataFrames with validation and compression.

**Formats:**
- CSV (with index control)
- Parquet (standard and gzip-compressed)

**Features:**
- Path and extension validation
- Automatic directory creation
- Compression options for Parquet
- Save confirmation logging

---

### **Plot Engine** (`haashi_pkg.plot_engine`)

#### **PlotEngine**
High-level plotting interface built on matplotlib and seaborn.

**Workflow:**
1. **Setup**: Create figures and configure layouts
2. **Draw**: Add data visualizations
3. **Decorate**: Apply styling, labels, legends
4. **Finalize**: Save and/or display

**Features:**
- **4 Color Palettes**: Professional, vibrant, soft, deep
- **Flexible Layouts**: Simple grids or complex custom ratios
- **Plot Types**: Line, bar, scatter, pie
- **Reference Lines**: Horizontal/vertical targets and thresholds
- **Value Labels**: Automatic labeling on bar charts
- **Stats Boxes**: Create summary statistics displays
- **Formatting**: Currency, percentages, date axes
- **Theming**: Seaborn themes with custom backgrounds

---

### **Utilities** (`haashi_pkg.utility`)

#### **Modern Classes** (Recommended)

**Logger**
- Console logging with multiple levels (debug, info, warning, error)
- JSON error persistence with automatic rotation
- Integration with ErrorLogger for long-term error tracking

**FileHandler**
- JSON and text file I/O with validation
- Automatic path creation and permission checks
- Readable/writable path validation

**ScreenUtil**
- Terminal screen clearing
- Line clearing with delays
- Loading animations
- Text wrapping utilities

**DateTimeUtil**
- Current time with UTC offset
- Flexible date formatting

**ClipboardUtil** (Termux/Android only)
- Copy/paste operations via termux-api

#### **Legacy Class** (Deprecated)

**Utility**
- Backward-compatible wrapper combining all utilities
- **⚠️ Deprecated**: Will be removed in v2.0.0
- Use modern classes instead for new code

---

## Installation

### From Git Repository

```bash
# Clone the repository
git clone https://github.com/Haashiraaa/haashi-analytics-toolkit.git
cd haashi-analytics-toolkit/packages

# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

### Dependencies

Core dependencies are automatically installed:
- `pandas >= 2.0.0`
- `seaborn >= 0.12.0`
- `matplotlib >= 3.7.0`
- `numpy >= 1.24.0`
- `pyarrow >= 12.0.0`
- `openpyxl >= 3.1.0`

Optional dev dependencies:
```bash
pip install -e ".[dev]"  # Includes pytest, black, mypy, ruff
```

---

## Quick Start

### Complete Data Pipeline Example

```python
from haashi_pkg.data_engine import DataAnalyzer, DataLoader, DataSaver
from haashi_pkg.plot_engine import PlotEngine
from haashi_pkg.utility import Logger, FileHandler
import logging

# Setup logging and utilities
logger = Logger(level=logging.INFO)
file_handler = FileHandler(logger=logger)

# Load data
loader = DataLoader("sales_data.csv", logger=logger, file_handler=file_handler)
df = loader.load_csv_single()

# Analyze and validate
analyzer = DataAnalyzer(logger=logger)

# Validate structure
analyzer.validate_columns_exist(df, ['customer_id', 'order_date', 'amount'])

# Inspect data quality
print(f"Missing values: {analyzer.count_missing(df, 'amount')}")
print(f"Duplicates: {analyzer.count_duplicates(df, 'customer_id')}")

# Clean and transform
df = analyzer.normalize_column_names(df)
df['order_date'] = analyzer.convert_datetime(df['order_date'])
df['amount'] = analyzer.convert_numeric(df['amount'])

# Validate data quality
analyzer.validate_dates(df, 'order_date')
analyzer.validate_numeric_non_negative(df, 'amount', allow_zero=False)

# Handle missing data
df = analyzer.drop_rows_with_missing(df, ['customer_id', 'order_date', 'amount'])

# Aggregate results
monthly_sales = analyzer.aggregate(
    df,
    value_col='amount',
    group_cols='month',
    op='sum'
)

# Create visualization
pe = PlotEngine()
fig, ax = pe.create_figure(figsize=(12, 8))

pe.draw(
    ax,
    x=monthly_sales.index,
    y=monthly_sales.values,
    plot_type='bar',
    color=pe.colors_01[0],
    label='Monthly Sales'
)

pe.decorate(
    ax,
    title='Monthly Sales Performance',
    xlabel='Month',
    ylabel='Revenue',
    ylim='zero'
)

pe.format_y_axis(ax, currency='$', decimals=0)
pe.add_reference_line(ax, y=50000, label='Target', color='red', linestyle='--')
pe.set_legend(ax)

# Save results
saver = DataSaver(logger=logger, file_handler=file_handler)
saver.save_parquet_compressed(df, "sales_cleaned.parquet")
pe.save_or_show(fig, save_path="sales_chart.png", dpi=300)

# Report
print(f"Rows dropped: {analyzer.dropped_row_count}")
print(f"Total missing values: {analyzer.cumulative_missing}")
```

---

## Detailed Usage Examples

### Data Loading

```python
from haashi_pkg.data_engine import DataLoader
from haashi_pkg.utility import Logger, FileHandler

logger = Logger()
file_handler = FileHandler(logger=logger)

# Single CSV file
loader = DataLoader("data.csv", logger=logger, file_handler=file_handler)
df = loader.load_csv_single()

# Multiple files
loader = DataLoader(
    "jan.csv", "feb.csv", "mar.csv",
    logger=logger,
    file_handler=file_handler
)
dfs = loader.load_csv_many()
combined = pd.concat(dfs, ignore_index=True)

# Large file in chunks (memory efficient)
loader = DataLoader("huge_file.csv", logger=logger, file_handler=file_handler)
for chunk in loader.load_csv_chunk(chunk_size=10000):
    process(chunk)

# Excel file
loader = DataLoader("report.xlsx", logger=logger, file_handler=file_handler)
df = loader.load_excel_single(sheet_name='Sales Data')

# Parquet files
loader = DataLoader("data.parquet", logger=logger, file_handler=file_handler)
df = loader.load_parquet_single()
```

### Data Analysis & Validation

```python
from haashi_pkg.data_engine import DataAnalyzer
from haashi_pkg.utility import Logger

logger = Logger()
analyzer = DataAnalyzer(logger=logger)

# Inspect data
analyzer.inspect_dataframe(df, rows=10)

# Check data quality
missing = analyzer.count_missing(df, 'salary')
duplicates = analyzer.count_duplicates(df, 'email')

# Text quality inspection
report = analyzer.inspect_text_formatting(df, 'company_name')
print(report)  # JSON report of whitespace/case issues

# Validate requirements
analyzer.validate_columns_exist(df, ['id', 'name', 'email'])
analyzer.validate_numeric_non_negative(df, 'age', allow_zero=False)
analyzer.validate_dates(df, 'birth_date')

# Transform data
df = analyzer.normalize_column_names(df)
df['name'] = analyzer.normalize_text_values(df['name'], method='title')
df['price'] = analyzer.convert_numeric(df['price_string'])
df['date'] = analyzer.convert_datetime(df['date_string'])

# Handle missing data
total, missing, percent = analyzer.missing_summary(df, 'optional_field')
df = analyzer.drop_rows_with_missing(df, ['required1', 'required2'])
df['price'] = analyzer.fill_missing_forward(df['price'])

# Aggregate
revenue_by_region = analyzer.aggregate(
    df,
    value_col='revenue',
    group_cols='region',
    op='sum'
)

stats = analyzer.aggregate(
    df,
    value_col='revenue',
    group_cols=['year', 'quarter'],
    op=['sum', 'mean', 'count']
)
```

### Visualization

```python
from haashi_pkg.plot_engine import PlotEngine

pe = PlotEngine()

# Simple plot
fig, ax = pe.create_figure(figsize=(10, 6))
pe.draw(ax, x=[1,2,3,4], y=[10,15,13,17], plot_type='line', color='blue')
pe.decorate(ax, title='Trend', xlabel='Time', ylabel='Value')
pe.save_or_show(fig, save_path='trend.png')

# Dashboard with custom grid
fig, gs = pe.create_custom_grid(
    rows=2, cols=3,
    height_ratios=[2, 1],
    width_ratios=[2, 1, 1],
    figsize=(20, 12)
)

# Main plot (top, spanning all columns)
ax_main = fig.add_subplot(gs[0, :])
pe.draw(ax_main, x=dates, y=revenue, plot_type='line', linewidth=2)
pe.decorate(ax_main, title='Revenue Trend', ylabel='Revenue ($)')
pe.format_y_axis(ax_main, currency='$')
pe.add_reference_line(ax_main, y=100000, label='Target', color='red')

# Stats box
ax_stats = fig.add_subplot(gs[1, 0])
pe.create_stats_text_box(
    ax_stats,
    stats={'Total': 1250000, 'Average': 104167, 'Growth': '23.5%'},
    title='Key Metrics'
)

# Bar chart
ax_bar = fig.add_subplot(gs[1, 1])
pe.draw(ax_bar, x=categories, y=counts, plot_type='bar')
pe.add_value_labels_on_bars(ax_bar, format_string='{:.0f}')

# Pie chart
ax_pie = fig.add_subplot(gs[1, 2])
pe.draw(ax_pie, x=None, y=distribution, plot_type='pie', labels=labels)

pe.save_or_show(fig, save_path='dashboard.png', dpi=300)
```

### Custom Exceptions & Error Handling

```python
from haashi_pkg.data_engine import (
    DataAnalyzer,
    DataValidationError,
    DataTypeError,
    FileLoadError
)
from haashi_pkg.utility import Logger

logger = Logger()
analyzer = DataAnalyzer(logger=logger)

try:
    analyzer.validate_columns_exist(df, ['missing_column'])
except DataValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle missing columns

try:
    analyzer.validate_numeric_non_negative(df, 'age', allow_zero=False)
except DataValidationError as e:
    logger.error(f"Data quality issue: {e}")
    # Clean invalid data

try:
    df['amount'] = analyzer.convert_numeric(df['amount_str'])
except DataTypeError as e:
    logger.error(f"Type conversion failed: {e}")
    # Handle conversion errors
```

---

## Migration Guide (v0.x → v1.0)

### What Changed

1. **Consolidated Modules**: Three separate files merged into `data_engine.py`
2. **Class Renamed**: `DataEngine` → `DataAnalyzer` (more descriptive)
3. **Modern Utilities**: Now uses `Logger` and `FileHandler` instead of deprecated `Utility`
4. **Custom Exceptions**: Replaced generic errors with specific exception types
5. **Full Documentation**: 2000+ lines of comprehensive docstrings added

### Backward Compatibility

Deprecated modules remain with warnings:

```python
# OLD CODE - Still works but shows deprecation warning
from haashi_pkg.data_engine.dataengine import DataEngine

de = DataEngine()
de.validate_columns_exist(df, ['id', 'name'])
# ⚠️ DeprecationWarning: Use DataAnalyzer from data_engine instead
```

### Recommended Updates

**Option 1: Quick Update (Keep Old Class Name)**
```python
# Use new import, keep old name
from haashi_pkg.data_engine import DataEngine  # Actually imports DataAnalyzer

de = DataEngine()  # No deprecation warning when importing from package
```

**Option 2: Full Modern Update (Recommended)**
```python
from haashi_pkg.data_engine import DataAnalyzer, DataLoader, DataSaver
from haashi_pkg.utility import Logger, FileHandler
import logging

logger = Logger(level=logging.INFO)
file_handler = FileHandler(logger=logger)

analyzer = DataAnalyzer(logger=logger)
loader = DataLoader("data.csv", logger=logger, file_handler=file_handler)
saver = DataSaver(logger=logger, file_handler=file_handler)
```

### Deprecation Timeline

- **v1.0** (Current): Deprecated features supported with warnings
- **v2.0** (Future): Deprecated modules removed
  - `dataengine.py` wrapper removed
  - `dataloader.py` wrapper removed
  - `datasaver.py` wrapper removed
  - `Utility` class removed

**Action Required:** Update imports before v2.0 release.

---

## Exception Hierarchy

```python
# Data Engine Exceptions
DataEngineError                 # Base exception
├── DataValidationError         # Validation failures
├── DataTypeError              # Type conversion errors
├── FileLoadError              # File loading failures
└── FileSaveError              # File saving failures

# Plot Engine Exceptions
PlotEngineError                # Base exception
├── InvalidPlotTypeError       # Invalid plot type
├── InvalidDataError           # Data validation failures
└── ConfigurationError         # Setup/config failures

# Utility Exceptions
UtilityError                   # Base exception
├── FileOperationError         # File operation failures
└── ClipboardError            # Clipboard failures (Termux)
```

---

## Contributing

Contributions welcome! Please ensure:

- **Documentation**: Add docstrings for all public functions
- **Type Hints**: Use full type annotations
- **Error Handling**: Raise appropriate custom exceptions
- **Tests**: Add unit tests for new features
- **Non-Mutating**: Don't modify inputs unless explicitly documented
- **Examples**: Include usage examples in docstrings

---

## Roadmap

### Planned Features
- [ ] Additional file formats (Excel support)
- [ ] Enhanced validation rules and custom validators
- [ ] More plot types (heatmaps, box plots, violin plots)
- [ ] Data profiling utilities
- [ ] Integration with cloud storage (S3, GCS)
- [ ] Performance benchmarks and optimization
- [ ] Comprehensive test suite
- [ ] Example notebooks and tutorials

### In Progress
- [x] Custom exception hierarchy
- [x] Comprehensive documentation
- [x] Modern utility classes
- [x] Backward compatibility layer

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Built with the Python data science stack:
- [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [NumPy](https://numpy.org/) - Numerical computing
- [matplotlib](https://matplotlib.org/) - Visualization
- [seaborn](https://seaborn.pydata.org/) - Statistical visualization
- [PyArrow](https://arrow.apache.org/docs/python/) - Columnar data format

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Haashiraaa/haashi-analytics-toolkit/issues)
- **Documentation**: [README](https://github.com/Haashiraaa/haashi-analytics-toolkit/blob/main/README.md)
- **Repository**: [GitHub](https://github.com/Haashiraaa/haashi-analytics-toolkit)

---

**Made with ❤️ by Haashiraaa**
