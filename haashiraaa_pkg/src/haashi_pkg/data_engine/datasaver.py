# datasaver.py

from pandas import DataFrame
from typing import Optional

from haashi_pkg.utility.utils import Utility


class DataSaver:
    """Save pandas DataFrames to disk in multiple formats."""

    def __init__(self, save_path: Optional[str] = None) -> None:
        """Initialize with an optional default save path."""
        self.save_path = save_path
        self.ut = Utility()

    # ========================
    # Validate save path
    # ========================

    def validate_save_path(
        self,
        path: Optional[str],
        file_type: str
    ) -> str:
        """Validate save path and enforce file extension."""
        path = path or self.save_path

        if not path:
            raise ValueError("No save path provided!")

        if not path.endswith(file_type):
            raise ValueError(
                f"Save path must end with '{file_type}', got '{path}'"
            )

        return str(self.ut.ensure_writable_path(path))

    # ========================
    # Confirm file saved
    # ========================

    def confirm_saved(self, path: str) -> None:
        """Print confirmation after a successful save."""
        print(f"File saved → {path}")

    # ========================
    # Save to CSV
    # ========================

    def save_csv(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        """Save a DataFrame as a CSV file."""
        path = self.validate_save_path(path, ".csv")
        df.to_csv(path, index=False)
        self.confirm_saved(path)

    # ========================
    # Save to Excel
    # ========================
    # COMING SOON

    # ========================
    # Save to Parquet
    # ========================

    def save_parquet_default(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        """Save a DataFrame as a Parquet file."""
        path = self.validate_save_path(path, ".parquet")
        df.to_parquet(path, index=False)
        self.confirm_saved(path)

    def save_parquet_compressed(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        """Save a compressed Parquet file using Gzip."""
        path = self.validate_save_path(path, ".parquet")
        df.to_parquet(
            path,
            engine="pyarrow",
            compression="gzip",
            index=False
        )
        self.confirm_saved(path)
