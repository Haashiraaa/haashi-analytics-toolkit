# __init__.py
"""
Data Engine Package
===================

A comprehensive toolkit for working with tabular data in pandas, providing utilities
for loading, analyzing, validating, cleaning, and saving data.

Main Components:
    DataAnalyzer: Core data analysis, validation, and transformation utilities
    DataLoader: Load data from CSV, Excel, and Parquet files
    DataSaver: Save DataFrames to CSV and Parquet formats

Custom Exceptions:
    DataEngineError: Base exception for all data engine errors
    DataValidationError: Raised when data validation fails
    DataTypeError: Raised when type conversion/validation fails
    FileLoadError: Raised when file loading fails
    FileSaveError: Raised when file saving fails

Recommended Usage:
    >>> from haashi_pkg.data_engine import DataAnalyzer, DataLoader, DataSaver
    >>> from haashi_pkg.utility.utils import Logger, FileHandler
    >>> import logging
    >>> 
    >>> # Initialize
    >>> logger = Logger(level=logging.INFO)
    >>> file_handler = FileHandler(logger=logger)
    >>> 
    >>> # Load
    >>> loader = DataLoader("data.csv", logger=logger, file_handler=file_handler)
    >>> df = loader.load_csv_single()
    >>> 
    >>> # Analyze
    >>> analyzer = DataAnalyzer(logger=logger)
    >>> analyzer.validate_columns_exist(df, ['id', 'value'])
    >>> df_clean = analyzer.normalize_column_names(df)
    >>> 
    >>> # Save
    >>> saver = DataSaver(logger=logger, file_handler=file_handler)
    >>> saver.save_parquet_compressed(df_clean, "output.parquet")

Backward Compatibility:
    The old separate modules (dataengine, dataloader, datasaver) are still available
    but deprecated. They will be removed in version 2.0.0.
    
    Old imports that still work (with deprecation warnings):
        from haashi_pkg.data_engine.dataengine import DataEngine
        from haashi_pkg.data_engine.dataloader import DataLoader
        from haashi_pkg.data_engine.datasaver import DataSaver
"""

# Import from consolidated module
from haashi_pkg.data_engine.data_engine import (
    # Main classes
    DataAnalyzer,
    DataLoader,
    DataSaver,

    # Exceptions
    DataEngineError,
    DataValidationError,
    DataTypeError,
    FileLoadError,
    FileSaveError,
)

# For backward compatibility, also expose DataEngine as alias to DataAnalyzer
# (Users importing from __init__ won't get deprecation warning)
DataEngine = DataAnalyzer

__all__ = [
    # Main classes (new names)
    'DataAnalyzer',
    'DataLoader',
    'DataSaver',

    # Backward compatibility
    'DataEngine',

    # Exceptions
    'DataEngineError',
    'DataValidationError',
    'DataTypeError',
    'FileLoadError',
    'FileSaveError',
]

__version__ = '1.0.0'
