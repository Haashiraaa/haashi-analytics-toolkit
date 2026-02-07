# dataloader.py
"""
**DEPRECATED - This module is deprecated and will be removed in version 3.0.0**

This file exists only for backward compatibility. Please update your imports to use
the consolidated data_engine module instead.

Old way (DEPRECATED):
    from haashi_pkg.data_engine.dataloader import DataLoader
    
New way (RECOMMENDED):
    from haashi_pkg.data_engine.data_engine import DataLoader

Changes:
    - Module has been consolidated into data_engine.py
    - DataLoader class functionality is preserved
    - Now uses modern Logger and FileHandler instead of deprecated Utility class
    - Improved error handling with custom exceptions (FileLoadError)
    - Comprehensive docstrings added
    - Added sheet_name parameter to load_excel_single
"""

from haashi_pkg.data_engine.data_engine import (
    DataEngineError,
    FileLoadError
)
from haashi_pkg.data_engine.data_engine import DataLoader
import warnings

# Issue deprecation warning
warnings.warn(
    "The 'dataloader' module is deprecated and will be removed in version 2.0.0. "
    "Please use 'from haashi_pkg.data_engine.data_engine import DataLoader' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility

__all__ = [
    'DataLoader',
    'DataEngineError',
    'FileLoadError'
]
