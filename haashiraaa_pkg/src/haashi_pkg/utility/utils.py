# utils.py

from __future__ import annotations

import os
import textwrap
import logging
import sys
import json
import time
from typing import Any, List, Union, Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone


PathLike = Union[str, Path]
DictFormat = dict[Any, Any]


class Utility:
    """
    General-purpose utility class.

    Responsibilities:
    - Logging (info / debug / error)
    - Screen clearing
    - Text formatting for display
    """

    def __init__(self, level: int = logging.WARNING) -> None:
        """
        Initialize the Utility object.

        Parameters:
            level (int): Logging level (e.g. logging.INFO, logging.DEBUG).
        """

        # Predefined text messages you may want to reuse
        self.text: dict[str, str] = {
            "ERROR": "\nOops! Something went wrong.",
            "END": "\n[Program finished]",
            "MISSING_FILE": "\nFile path not found.",
        }

        # Create (or retrieve) a logger with a stable name
        self.logger = logging.getLogger("hashi_pkg")

        # Set the minimum log level this logger will handle
        self.logger.setLevel(level)

        # Prevent adding multiple handlers if Utility is instantiated many times
        if not self.logger.handlers:
            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                "[%(levelname)s] \n%(message)s"
            )

            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # -------- Error handling methods --------

    def handle_error(self, exc: Exception) -> None:
        print(self.text["ERROR"])
        self.debug(exc)
        sys.exit(1)

    def handle_file_not_found(self, fnf: FileNotFoundError) -> None:
        print(self.text["MISSING_FILE"])
        self.debug(fnf)
        sys.exit(1)

    # -------- Logging methods --------

    def info(self, message: Any) -> None:
        """Log a normal informational message."""
        self.logger.info(message)

    def debug(self, message: Any) -> None:
        """Log detailed debug information."""
        self.logger.debug(message)

    def warning(self, message: Any) -> None:
        """Log a warning message"""
        self.logger.warning(message)

    def error(self, message: Any) -> None:
        """Log an error message."""
        self.logger.error(message)

    # -------- UI/UX methods --------

    def clear_screen(self, timeout: float = 0) -> None:
        """Clear the terminal screen (cross-platform)."""
        time.sleep(timeout)
        os.system("cls" if os.name == "nt" else "clear")

    def clear_line(self, n: int = 1, timeout: float = 0.5) -> None:
        """
        Clear the previous N lines in the terminal.
                                                                                    Parameters:
            n (int): Number of lines to erase. Useful for animations.

        Uses basic ANSI escape codes:
            - \033[1A → move cursor up by one line
            - \033[2K → clear the entire current line
        """
        time.sleep(timeout)
        for _ in range(n):
            sys.stdout.write("\033[1A")     # Move cursor up
            sys.stdout.write("\r\033[2K")   # Clear line
        sys.stdout.flush()

    def format_text(self, text: str, width: int = 70) -> str:
        """
        Wrap long text to fit nicely on smaller screens.

        Parameters:
            text (str): Text to wrap.
            width (int): Max characters per line.

        Returns:
            str: Wrapped text.
        """
        wrapper = textwrap.TextWrapper(width=width)
        formatted: List[str] = []

        for line in text.split("\n"):
            if not line.strip():
                formatted.append("")
            else:
                formatted.append(wrapper.fill(line))

        return "\n".join(formatted)

    # -------- File handling methods -------- #

    def ensure_writable_path(self, path: PathLike) -> Path:
        """
        Ensure a file path is writable by creating parent directories.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_readable_file(self, path: PathLike) -> Path:
        """
        Ensure a file exists for reading.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        return path

    def save_json(
        self, data: DictFormat, path: Path, operation: str = "w"
    ) -> None:
        with open(path, operation) as file:
            json.dump(data, file, indent=4)
        self.info(f"Data saved -> {path}")

    def read_json(self, path: Path) -> DictFormat:
        with open(path, "r") as file_content:
            return json.load(file_content)

    def save_txt(
        self, data: str, path: Path, operation: str = "w"
    ) -> None:
        with open(path, operation) as file:
            file.write("\n")
            file.write(data)
        self.info(f"Data saved -> {path}")

    def read_txt(self, path: Path) -> str:
        with open(path, "r") as file_content:
            return file_content.read()

    # ---------------- Datetime methods -------------------- #

    def get_current_time(self, utc_offset_hours: int = 0) -> str:
        current_time = (
            datetime.now(timezone.utc) + timedelta(hours=utc_offset_hours)
        )
        return current_time.strftime("%Y-%m-%d %H:%M:%S")
