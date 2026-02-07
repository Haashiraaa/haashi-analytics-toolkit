# Data Engine Package - Refactoring Summary

## Overview

The data_engine package has been consolidated and modernized. All three separate modules (`dataengine.py`, `dataloader.py`, `datasaver.py`) have been combined into a single `data_engine.py` file with three distinct classes.

## What Changed

### 1. **Consolidated Architecture**
- **Before**: 3 separate files (dataengine.py, dataloader.py, datasaver.py)
- **After**: 1 main file (data_engine.py) containing 3 classes

### 2. **Class Renaming**
- `DataEngine` → `DataAnalyzer` (more descriptive name)
- `DataLoader` → `DataLoader` (unchanged)
- `DataSaver` → `DataSaver` (unchanged)

### 3. **Modern Utility Usage**
- **Before**: Used deprecated `Utility` class
- **After**: Uses modern `Logger` and `FileHandler` classes
- Better separation of concerns and improved maintainability

### 4. **Custom Exception Hierarchy**
New custom exceptions replace generic errors and print statements:
- `DataEngineError` - Base exception
- `DataValidationError` - Validation failures
- `DataTypeError` - Type conversion issues
- `FileLoadError` - File loading failures
- `FileSaveError` - File saving failures

### 5. **Comprehensive Documentation**
- Every class has detailed docstrings
- Every method has complete documentation including:
  - Purpose and behavior
  - All parameters with types and defaults
  - Return values
  - Exceptions that can be raised
  - Multiple usage examples

### 6. **Improved Error Handling**
- **Before**: Printed errors or relied on pandas exceptions
- **After**: Raises descriptive custom exceptions with context
- Better error messages that include what failed and why

### 7. **Better Logging**
- Uses proper logging levels (debug, info, warning, error)
- **Before**: Mixed print statements and debug calls
- **After**: Consistent use of logger throughout
  - `logger.debug()` - Internal operations
  - `logger.info()` - Important milestones
  - Exceptions raised for errors (not logged)

## Migration Guide

### Option 1: Quick Migration (Backward Compatible)

Your old code will continue to work with deprecation warnings:

```python
# OLD CODE - Still works but shows deprecation warning
from haashi_pkg.data_engine.dataengine import DataEngine

de = DataEngine()
de.validate_columns_exist(df, ['id', 'name'])
```

### Option 2: Recommended Migration

Update to the new consolidated module:

```python
# NEW CODE - Recommended
from haashi_pkg.data_engine import DataAnalyzer, DataLoader, DataSaver
from haashi_pkg.utility.utils import Logger, FileHandler
import logging

# Initialize modern utilities
logger = Logger(level=logging.INFO)
file_handler = FileHandler(logger=logger)

# Use new class name with logger support
analyzer = DataAnalyzer(logger=logger)
analyzer.validate_columns_exist(df, ['id', 'name'])
```

### DataLoader Migration

**Before:**
```python
from haashi_pkg.data_engine.dataloader import DataLoader

loader = DataLoader("data.csv")
df = loader.load_csv_single()
```

**After:**
```python
from haashi_pkg.data_engine import DataLoader
from haashi_pkg.utility.utils import Logger, FileHandler
import logging

logger = Logger(level=logging.INFO)
file_handler = FileHandler(logger=logger)

loader = DataLoader("data.csv", logger=logger, file_handler=file_handler)
df = loader.load_csv_single()
```

### DataSaver Migration

**Before:**
```python
from haashi_pkg.data_engine.datasaver import DataSaver

saver = DataSaver(save_path="output.csv")
saver.save_csv(df)
```

**After:**
```python
from haashi_pkg.data_engine import DataSaver
from haashi_pkg.utility.utils import Logger, FileHandler
import logging

logger = Logger(level=logging.INFO)
file_handler = FileHandler(logger=logger)

saver = DataSaver(save_path="output.csv", logger=logger, file_handler=file_handler)
saver.save_csv(df)
```

## New Features

### 1. DataAnalyzer Improvements
- Better validation with descriptive errors
- Column existence validation before operations
- Type checking with helpful error messages
- Tracked statistics (dropped_row_count, cumulative_missing)

### 2. DataLoader Improvements
- Added `sheet_name` parameter to `load_excel_single()`
- Better error messages with file paths
- Path validation before attempting to load
- Consistent logging across all methods

### 3. DataSaver Improvements
- Public `validate_save_path()` method for custom use cases
- Automatic logging of save operations
- Better error messages with file paths
- Removed deprecated `confirm_saved()` (now done automatically via logging)

## File Structure

```
data_engine/
├── data_engine.py          # NEW - Main consolidated module
├── __init__.py             # NEW - Package initialization with exports
├── dataengine.py           # DEPRECATED - Backward compatibility wrapper
├── dataloader.py           # DEPRECATED - Backward compatibility wrapper
└── datasaver.py            # DEPRECATED - Backward compatibility wrapper
```

## Exception Handling Examples

### Before (Old Way)
```python
# Validation might fail silently or raise pandas exceptions
de = DataEngine()
de.validate_columns_exist(df, ['missing_col'])
# Generic error, hard to catch specifically
```

### After (New Way)
```python
from haashi_pkg.data_engine import DataAnalyzer, DataValidationError

analyzer = DataAnalyzer(logger=logger)

try:
    analyzer.validate_columns_exist(df, ['missing_col'])
except DataValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Specific exception, clear error message
    # "Missing required columns: ['missing_col']"
```

## Complete Example Pipeline

```python
from haashi_pkg.data_engine import DataAnalyzer, DataLoader, DataSaver
from haashi_pkg.utility.utils import Logger, FileHandler
import logging

# Setup
logger = Logger(level=logging.INFO)
file_handler = FileHandler(logger=logger)

# Load
loader = DataLoader("raw_data.csv", logger=logger, file_handler=file_handler)
df = loader.load_csv_single()

# Analyze & Clean
analyzer = DataAnalyzer(logger=logger)

# Validate structure
analyzer.validate_columns_exist(df, ['customer_id', 'order_date', 'amount'])

# Normalize
df = analyzer.normalize_column_names(df)

# Convert types
df['order_date'] = analyzer.convert_datetime(df['order_date'])
df['amount'] = analyzer.convert_numeric(df['amount'])

# Validate data quality
analyzer.validate_dates(df, 'order_date')
analyzer.validate_numeric_non_negative(df, 'amount', allow_zero=False)

# Clean
df = analyzer.drop_rows_with_missing(df, ['customer_id', 'order_date', 'amount'])

# Save
saver = DataSaver(logger=logger, file_handler=file_handler)
saver.save_parquet_compressed(df, "clean_data.parquet")

# Report
print(f"Rows dropped: {analyzer.dropped_row_count}")
print(f"Cumulative missing: {analyzer.cumulative_missing}")
```

## Breaking Changes

None! All old code continues to work via the deprecated wrapper modules. However:

1. Deprecation warnings will be shown
2. The wrapper modules will be removed in version 3.0.0
3. Please migrate to the new consolidated module before then

## Benefits of Migration

1. **Single import**: Everything in one place
2. **Better errors**: Custom exceptions with clear messages  
3. **Modern logging**: Proper use of Logger instead of prints
4. **Full documentation**: Comprehensive docstrings for all methods
5. **Type safety**: Better type hints throughout
6. **Maintainability**: Easier to maintain a single cohesive module
7. **Future-proof**: Built on modern utility classes that won't be deprecated

## Questions or Issues?

The deprecation warnings include the exact import statements you need. If you see:
```
DeprecationWarning: The 'dataengine' module is deprecated...
Please use 'from haashi_pkg.data_engine.data_engine import DataAnalyzer' instead.
```

Simply update your import to the suggested one!
