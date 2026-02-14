# dataengine.py
"""
**DEPRECATED - This module is deprecated and will be removed in version 3.0.0**

This file exists only for backward compatibility. Please update your imports to use
the consolidated data_engine module instead.

Old way (DEPRECATED):
    from haashi_pkg.data_engine.dataengine import DataEngine
    
New way (RECOMMENDED):
    from haashi_pkg.data_engine.data_engine import DataAnalyzer

Changes:
    - DataEngine class has been renamed to DataAnalyzer
    - All functionality is preserved
    - Now uses modern Logger and FileHandler instead of deprecated Utility class
    - Improved error handling with custom exceptions
    - Comprehensive docstrings added
"""

from haashi_pkg.data_engine.data_engine import (
    DataEngineError,
    DataValidationError,
    DataTypeError
)
from haashi_pkg.data_engine.data_engine import DataAnalyzer as DataEngine
import warnings

# Issue deprecation warning
warnings.warn(
    "The 'dataengine' module is deprecated and will be removed in version 2.0.0. "
    "Please use 'from haashi_pkg.data_engine.data_engine import DataAnalyzer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility

__all__ = [
    'DataEngine',
    'DataEngineError',
    'DataValidationError',
    'DataTypeError'
]
