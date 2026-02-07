# datasaver.py
"""
**DEPRECATED - This module is deprecated and will be removed in version 3.0.0**

This file exists only for backward compatibility. Please update your imports to use
the consolidated data_engine module instead.

Old way (DEPRECATED):
    from haashi_pkg.data_engine.datasaver import DataSaver
    
New way (RECOMMENDED):
    from haashi_pkg.data_engine.data_engine import DataSaver

Changes:
    - Module has been consolidated into data_engine.py
    - DataSaver class functionality is preserved
    - Now uses modern Logger and FileHandler instead of deprecated Utility class
    - Improved error handling with custom exceptions (FileSaveError)
    - Comprehensive docstrings added
    - validate_save_path is now a public method (still works the same way)
    - confirm_saved removed (now handled via logger.info automatically)
"""

from haashi_pkg.data_engine.data_engine import (
    DataEngineError,
    FileSaveError
)
from haashi_pkg.data_engine.data_engine import DataSaver
import warnings

# Issue deprecation warning
warnings.warn(
    "The 'datasaver' module is deprecated and will be removed in version 2.0.0. "
    "Please use 'from haashi_pkg.data_engine.data_engine import DataSaver' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location for backward compatibility

__all__ = [
    'DataSaver',
    'DataEngineError',
    'FileSaveError'
]
