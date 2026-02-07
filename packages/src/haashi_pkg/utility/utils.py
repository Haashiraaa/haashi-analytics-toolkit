
# utils.py

"""
Utility Module for haashi_pkg
==============================

Provides utility classes for:
- Logging (console + JSON error persistence)
- File operations (JSON, TXT)
- Screen manipulation (clear, animations)
- Datetime helpers
- Clipboard operations (Termux)

Author: Haashiraaa
"""

from __future__ import annotations

import os
import textwrap
import logging
import sys
import json
import time
import subprocess
from typing import Any, List, Union, Optional, Dict
from pathlib import Path
from datetime import datetime, timedelta, timezone


# Type aliases
PathLike = Union[str, Path]
DictFormat = Dict[Any, Any]


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class UtilityError(Exception):
    """Base exception for all utility-related errors."""
    pass


class FileOperationError(UtilityError):
    """Raised when file operations fail."""
    pass


class ClipboardError(UtilityError):
    """Raised when clipboard operations fail (Termux-specific)."""
    pass


# ============================================================================
# ERROR LOGGER (JSON Persistence)
# ============================================================================

class ErrorLogger:
    """
    Handles logging errors to both console and JSON file for persistence.

    This logger automatically saves errors to a JSON file with timestamps,
    error types, and full tracebacks for debugging.

    Attributes:
        error_log_path (Path): Path to the JSON error log file.
        max_entries (int): Maximum number of error entries to keep in JSON.

    Example:
        >>> error_logger = ErrorLogger("logs/errors.json")
        >>> error_logger.log_error(ValueError("Invalid input"), context="data_processing")
    """

    def __init__(
        self,
        error_log_path: PathLike = "logs/errors.json",
        max_entries: int = 100
    ) -> None:
        """
        Initialize the error logger.

        Args:
            error_log_path: Path where error JSON will be saved.
            max_entries: Maximum number of errors to keep (auto-rotates old entries).
        """
        self.error_log_path = Path(error_log_path)
        self.max_entries = max_entries
        self._ensure_log_directory()

    def _ensure_log_directory(self) -> None:
        """Create log directory if it doesn't exist."""
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize empty error log if doesn't exist
        if not self.error_log_path.exists():
            self._write_errors([])

    def _read_errors(self) -> List[DictFormat]:
        """Read existing errors from JSON file."""
        try:
            with open(self.error_log_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_errors(self, errors: List[DictFormat]) -> None:
        """Write errors to JSON file."""
        with open(self.error_log_path, "w") as f:
            json.dump(errors, f, indent=2)

    def log_error(
        self,
        exception: Exception,
        context: Optional[str] = None,
        extra_data: Optional[DictFormat] = None
    ) -> None:
        """
        Log an error to JSON file with full details.

        Args:
            exception: The exception object that was raised.
            context: Optional context string (e.g., "data_processing", "api_call").
            extra_data: Optional dictionary with additional debugging info.

        Example:
            >>> try:
            ...     result = 1 / 0
            ... except Exception as e:
            ...     error_logger.log_error(e, context="calculation", extra_data={"input": 0})
        """
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "context": context or "unknown",
            "traceback": self._get_traceback_string(exception)
        }

        if extra_data:
            error_entry["extra_data"] = extra_data

        # Read existing errors
        errors = self._read_errors()

        # Add new error
        errors.append(error_entry)

        # Rotate if exceeds max entries (keep most recent)
        if len(errors) > self.max_entries:
            errors = errors[-self.max_entries:]

        # Write back to file
        self._write_errors(errors)

    def _get_traceback_string(self, exception: Exception) -> str:
        """Extract traceback as string from exception."""
        import traceback
        return ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))

    def get_recent_errors(self, n: int = 10) -> List[DictFormat]:
        """
        Retrieve the N most recent errors.

        Args:
            n: Number of recent errors to retrieve.

        Returns:
            List of error dictionaries, most recent first.
        """
        errors = self._read_errors()
        return errors[-n:][::-1]  # Reverse to show most recent first

    def clear_errors(self) -> None:
        """Clear all logged errors."""
        self._write_errors([])


# ============================================================================
# LOGGER UTILITY
# ============================================================================

class Logger:
    """
    Console logging utility with multiple log levels.

    Provides clean, formatted console output for info, debug, warning, and error messages.
    Integrates with ErrorLogger for persistent error tracking.

    Attributes:
        logger (logging.Logger): Internal Python logger instance.
        error_logger (ErrorLogger): JSON error logger for persistence.

    Example:
        >>> logger = Logger(level=logging.INFO)
        >>> logger.info("Processing started")
        >>> logger.error("Failed to load file", save_to_json=True)
    """

    def __init__(
        self,
        level: int = logging.WARNING,
        error_log_path: Optional[PathLike] = None
    ) -> None:
        """
        Initialize the logger.

        Args:
            level: Logging level (logging.INFO, logging.DEBUG, etc.).
            error_log_path: Optional path for JSON error logging. If None, uses default.
        """
        self.logger = logging.getLogger("haashi_pkg")
        self.logger.setLevel(level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize error logger for JSON persistence
        log_path = error_log_path or "logs/errors.json"
        self.error_logger = ErrorLogger(log_path)

    def info(self, message: Any) -> None:
        """
        Log informational message to console.

        Args:
            message: The message to log (will be converted to string).
        """
        self.logger.info(str(message))

    def debug(self, message: Any) -> None:
        """
        Log debug message to console (only visible if level=DEBUG).

        Args:
            message: The message to log (will be converted to string).
        """
        self.logger.debug(str(message))

    def warning(self, message: Any) -> None:
        """
        Log warning message to console.

        Args:
            message: The message to log (will be converted to string).
        """
        self.logger.warning(str(message))

    def error(
        self,
        message: Any,
        exception: Optional[Exception] = None,
        save_to_json: bool = False,
        context: Optional[str] = None
    ) -> None:
        """
        Log error message to console and optionally to JSON.

        Args:
            message: The error message to log.
            exception: Optional exception object to log to JSON.
            save_to_json: If True, saves error to JSON file for persistence.
            context: Optional context for JSON logging (e.g., "api_call").

        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.error("Operation failed", exception=e, save_to_json=True, context="data_load")
        """
        self.logger.error(str(message))

        if save_to_json and exception:
            self.error_logger.log_error(exception, context=context)


# ============================================================================
# FILE OPERATIONS
# ============================================================================

class FileHandler:
    """
    Handles file operations with proper error handling and path validation.

    Supports:
    - JSON read/write
    - TXT read/write
    - Path validation and creation

    Example:
        >>> file_handler = FileHandler()
        >>> file_handler.save_json({"key": "value"}, "data/output.json")
        >>> data = file_handler.read_json("data/output.json")
    """

    def __init__(self, logger: Optional[Logger] = None) -> None:
        """
        Initialize file handler.

        Args:
            logger: Optional Logger instance for logging operations.
        """
        self.logger = logger or Logger()

    def ensure_writable_path(self, path: PathLike) -> Path:
        """
        Ensure a file path is writable by creating parent directories.

        Args:
            path: File path to validate/create.

        Returns:
            Path object with guaranteed parent directories.

        Raises:
            FileOperationError: If path creation fails.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as e:
            raise FileOperationError(
                f"Failed to create path {path}: {e}") from e

    def ensure_readable_file(self, path: PathLike) -> Path:
        """
        Ensure a file exists and is readable.

        Args:
            path: File path to validate.

        Returns:
            Path object if file exists and is readable.

        Raises:
            FileNotFoundError: If file doesn't exist.
            FileOperationError: If path exists but is not a file.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise FileOperationError(f"Not a file: {path}")

        return path

    def save_json(
        self,
        data: DictFormat,
        path: PathLike,
        mode: str = "w",
        indent: int = 4
    ) -> None:
        """
        Save dictionary to JSON file.

        Args:
            data: Dictionary to save.
            path: Destination file path.
            mode: File open mode ('w' for overwrite, 'a' for append - note: 'a' creates invalid JSON).
            indent: JSON indentation level.

        Raises:
            FileOperationError: If save operation fails.
        """
        try:
            path = self.ensure_writable_path(path)
            with open(path, mode) as f:
                json.dump(data, f, indent=indent)
            self.logger.info(f"JSON saved → {path}")
        except Exception as e:
            raise FileOperationError(
                f"Failed to save JSON to {path}: {e}") from e

    def read_json(self, path: PathLike) -> DictFormat:
        """
        Read JSON file into dictionary.

        Args:
            path: Path to JSON file.

        Returns:
            Dictionary containing JSON data.

        Raises:
            FileNotFoundError: If file doesn't exist.
            FileOperationError: If JSON is invalid or read fails.
        """
        try:
            path = self.ensure_readable_file(path)
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise FileOperationError(f"Invalid JSON in {path}: {e}") from e
        except Exception as e:
            raise FileOperationError(
                f"Failed to read JSON from {path}: {e}") from e

    def save_txt(
        self,
        data: str,
        path: PathLike,
        mode: str = "w",
        add_newline_prefix: bool = True
    ) -> None:
        """
        Save string to text file.

        Args:
            data: String content to save.
            path: Destination file path.
            mode: File open mode ('w' for overwrite, 'a' for append).
            add_newline_prefix: If True, adds newline before content.

        Raises:
            FileOperationError: If save operation fails.
        """
        try:
            path = self.ensure_writable_path(path)
            with open(path, mode) as f:
                if add_newline_prefix:
                    f.write("\n")
                f.write(data)
            self.logger.info(f"TXT saved → {path}")
        except Exception as e:
            raise FileOperationError(
                f"Failed to save TXT to {path}: {e}") from e

    def read_txt(self, path: PathLike) -> str:
        """
        Read text file into string.

        Args:
            path: Path to text file.

        Returns:
            String containing file contents.

        Raises:
            FileNotFoundError: If file doesn't exist.
            FileOperationError: If read fails.
        """
        try:
            path = self.ensure_readable_file(path)
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            raise FileOperationError(
                f"Failed to read TXT from {path}: {e}") from e


# ============================================================================
# UI/UX UTILITIES
# ============================================================================

class ScreenUtil:
    """
    Terminal/console screen manipulation utilities.

    Provides:
    - Screen clearing (cross-platform)
    - Line clearing (ANSI escape codes)
    - Loading animations
    - Text wrapping/formatting

    Example:
        >>> screen = ScreenUtil()
        >>> screen.animate(text="Loading data", cycles=3)
        >>> screen.clear_screen(timeout=1.0)
    """

    @staticmethod
    def clear_screen(timeout: float = 0) -> None:
        """
        Clear the terminal screen (cross-platform).

        Args:
            timeout: Seconds to wait before clearing.
        """
        time.sleep(timeout)
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def clear_line(n: int = 1, timeout: float = 0.5) -> None:
        """
        Clear the previous N lines in the terminal using ANSI codes.

        Args:
            n: Number of lines to erase upward.
            timeout: Delay before clearing (for visual effect).

        Note:
            Uses ANSI escape codes:
            - \\033[1A → move cursor up one line
            - \\033[2K → clear entire current line
        """
        time.sleep(timeout)
        for _ in range(n):
            sys.stdout.write("\033[1A")    # Move cursor up
            sys.stdout.write("\r\033[2K")  # Clear line
        sys.stdout.flush()

    @staticmethod
    def animate(
        text: str = "Loading",
        cycles: int = 2,
        delay: float = 0.5
    ) -> None:
        """
        Display a simple CLI loading animation with dots.

        Args:
            text: Prefix text to display (e.g., "Loading").
            cycles: Number of animation cycles (1 cycle = 3 dots).
            delay: Seconds between each dot.

        Example:
            >>> ScreenUtil.animate("Processing", cycles=3, delay=0.3)
            # Output: Processing... Processing... Processing...
        """
        for _ in range(cycles):
            for dots in range(1, 4):
                sys.stdout.write(f'\r{text}{"." * dots}')
                sys.stdout.flush()
                time.sleep(delay)

    @staticmethod
    def format_text(text: str, width: int = 70) -> str:
        """
        Wrap long text to fit within specified character width.

        Args:
            text: Text to wrap.
            width: Maximum characters per line.

        Returns:
            Wrapped text with newlines inserted appropriately.

        Example:
            >>> long_text = "This is a very long line that needs wrapping."
            >>> wrapped = ScreenUtil.format_text(long_text, width=20)
        """
        wrapper = textwrap.TextWrapper(width=width)
        formatted: List[str] = []

        for line in text.split("\n"):
            if not line.strip():
                formatted.append("")
            else:
                formatted.append(wrapper.fill(line))

        return "\n".join(formatted)


# ============================================================================
# DATETIME UTILITIES
# ============================================================================

class DateTimeUtil:
    """
    Datetime helper utilities.

    Provides UTC-based time with timezone offset support.

    Example:
        >>> dt_util = DateTimeUtil()
        >>> current_date = dt_util.get_current_time(utc_offset_hours=1, only_date=True)
        >>> timestamp = dt_util.get_current_time(only_date=False)
    """

    @staticmethod
    def get_current_time(
        utc_offset_hours: int = 0,
        only_date: bool = True
    ) -> str:
        """
        Get current time with optional UTC offset.

        Args:
            utc_offset_hours: Hours to offset from UTC (e.g., +1 for WAT/Nigeria).
            only_date: If True, returns only date (YYYY-MM-DD). 
                       If False, returns datetime (YYYY-MM-DD HH:MM:SS).

        Returns:
            Formatted datetime string.

        Example:
            >>> DateTimeUtil.get_current_time(utc_offset_hours=1, only_date=False)
            '2025-02-04 15:30:45'
        """
        current_time = (
            datetime.now(timezone.utc) + timedelta(hours=utc_offset_hours)
        )

        if only_date:
            return current_time.strftime("%Y-%m-%d")

        return current_time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# CLIPBOARD UTILITIES (TERMUX-SPECIFIC)
# ============================================================================

class ClipboardUtil:
    """
    Termux clipboard operations.

    WARNING: Only works in Termux environment on Android.
    Will raise ClipboardError if termux-api is not installed.

    Example:
        >>> clipboard = ClipboardUtil()
        >>> clipboard.copy("Hello World")
        >>> text = clipboard.paste()
    """

    @staticmethod
    def copy(text: str) -> None:
        """
        Copy text to Termux clipboard.

        Args:
            text: String to copy to clipboard.

        Raises:
            ClipboardError: If termux-clipboard-set command fails.

        Example:
            >>> ClipboardUtil.copy("Hello World")
        """
        try:
            subprocess.run(
                ["termux-clipboard-set"],
                input=text,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise ClipboardError(
                "Failed to copy to clipboard. Is termux-api installed?"
            ) from e
        except FileNotFoundError as e:
            raise ClipboardError(
                "termux-clipboard-set not found. Install termux-api package."
            ) from e

    @staticmethod
    def paste() -> str:
        """
        Paste text from Termux clipboard.

        Returns:
            String content from clipboard.

        Raises:
            ClipboardError: If termux-clipboard-get command fails.

        Example:
            >>> text = ClipboardUtil.paste()
        """
        try:
            result = subprocess.run(
                ["termux-clipboard-get"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise ClipboardError(
                "Failed to paste from clipboard. Is termux-api installed?"
            ) from e
        except FileNotFoundError as e:
            raise ClipboardError(
                "termux-clipboard-get not found. Install termux-api package."
            ) from e


# ============================================================================
# MAIN UTILITY CLASS (BACKWARDS COMPATIBILITY)
# ============================================================================

class Utility:
    """
    **LEGACY WRAPPER - For backwards compatibility with existing code.**

    This class maintains the original API while delegating to new specialized classes.

    **DEPRECATION WARNING:**
    This class exists for backwards compatibility. New code should use:
    - Logger() instead of Utility() for logging
    - FileHandler() for file operations
    - ScreenUtil() for screen operations
    - DateTimeUtil() for datetime operations
    - ClipboardUtil() for clipboard operations

    This wrapper will be removed in version 2.0.0.

    Example (OLD WAY - Still works):
        >>> util = Utility()
        >>> util.info("Processing...")
        >>> util.save_json(data, "output.json")

    Example (NEW WAY - Recommended):
        >>> logger = Logger()
        >>> file_handler = FileHandler(logger=logger)
        >>> logger.info("Processing...")
        >>> file_handler.save_json(data, "output.json")
    """

    def __init__(self, level: int = logging.WARNING) -> None:
        """
        Initialize Utility (legacy mode).

        Args:
            level: Logging level (logging.INFO, logging.DEBUG, etc.).
        """
        # Predefined text messages (legacy)
        self.text: Dict[str, str] = {
            "ERROR": "\nOops! Something went wrong.",
            "END": "\n[Program finished]",
            "MISSING_FILE": "\nFile path not found.",
        }

        # Initialize new modular components
        self.logger = Logger(level=level)
        self._file_handler = FileHandler(logger=self.logger)
        self._screen_util = ScreenUtil()
        self._datetime_util = DateTimeUtil()
        self._clipboard_util = ClipboardUtil()

        # Expose logger's logger for backwards compatibility
        self.logger_instance = self.logger.logger

    # -------- Error handling methods (DEPRECATED) --------

    def handle_error(self, exc: Exception) -> None:
        """
        **DEPRECATED:** Use try/except with custom exceptions instead.

        Legacy error handler that prints error and exits.
        """
        import warnings
        warnings.warn(
            "handle_error() is deprecated and will be removed in v2.0 "
            "Use try/except with custom exceptions instead.",
            DeprecationWarning,
            stacklevel=2
        )
        print(self.text["ERROR"])
        self.debug(exc)
        sys.exit(1)

    def handle_file_not_found(self, fnf: FileNotFoundError) -> None:
        """
        **DEPRECATED:** Use try/except with FileNotFoundError instead.

        Legacy file not found handler that prints error and exits.
        """
        import warnings
        warnings.warn(
            "handle_file_not_found() is deprecated and will be removed in v2.0 "
            "Use try/except with FileNotFoundError instead.",
            DeprecationWarning,
            stacklevel=2
        )
        print(self.text["MISSING_FILE"])
        self.debug(fnf)
        sys.exit(1)
    # -------- Logging methods (DELEGATES to Logger) --------

    def info(self, message: Any) -> None:
        """Log informational message."""
        self.logger.info(message)

    def debug(self, message: Any) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: Any) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: Any) -> None:
        """Log error message."""
        self.logger.error(message)

    # -------- UI/UX methods (DELEGATES to ScreenUtil) --------

    def clear_screen(self, timeout: float = 0) -> None:
        """Clear terminal screen."""
        self._screen_util.clear_screen(timeout)

    def clear_line(self, n: int = 1, timeout: float = 0.5) -> None:
        """Clear N previous lines."""
        self._screen_util.clear_line(n, timeout)

    def animate(
        self, r: int = 2, text: str = 'Signing in', sec: float = 0.5
    ) -> None:
        """
        Display loading animation.

        Note: Parameter 'r' renamed to 'cycles' in new API.
        """
        self._screen_util.animate(text=text, cycles=r, delay=sec)

    def format_text(self, text: str, width: int = 70) -> str:
        """Wrap text to specified width."""
        return self._screen_util.format_text(text, width)

    # -------- File handling methods (DELEGATES to FileHandler) --------

    def ensure_writable_path(self, path: PathLike) -> Path:
        """Ensure path is writable."""
        return self._file_handler.ensure_writable_path(path)

    def ensure_readable_file(self, path: PathLike) -> Path:
        """Ensure file exists and is readable."""
        return self._file_handler.ensure_readable_file(path)

    def save_json(
        self, data: DictFormat, path: PathLike, operation: str = "w"
    ) -> None:
        """Save dictionary to JSON file."""
        self._file_handler.save_json(data, path, mode=operation)

    def read_json(self, path: PathLike) -> DictFormat:
        """Read JSON file."""
        return self._file_handler.read_json(path)

    def save_txt(
        self, data: str, path: PathLike, operation: str = "w"
    ) -> None:
        """Save string to text file."""
        self._file_handler.save_txt(data, path, mode=operation)

    def read_txt(self, path: PathLike) -> str:
        """Read text file."""
        return self._file_handler.read_txt(path)

    # -------- Datetime methods (DELEGATES to DateTimeUtil) --------

    def get_current_time(
        self, utc_offset_hours: int = 0, only_date: bool = True
    ) -> str:
        """Get current time with UTC offset."""
        return self._datetime_util.get_current_time(utc_offset_hours, only_date)

    # -------- Clipboard methods (DELEGATES to ClipboardUtil) --------

    def copy(self, text: str) -> None:
        """Copy text to Termux clipboard."""
        self._clipboard_util.copy(text)

    def paste(self) -> str:
        """Paste text from Termux clipboard."""
        return self._clipboard_util.paste()
