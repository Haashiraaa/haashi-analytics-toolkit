# dataloader.py

import logging
import pandas as pd


from haashi_pkg.utility.utils import Utility

from pathlib import Path
from pandas import DataFrame
from typing import Iterable, List


class DataLoader:
    """
    Lightweight data ingestion utility for loading tabular data formats.
    Focused strictly on I/O, not validation or transformation.
    """

    def __init__(self, *file_paths: str) -> None:
        """Initialize loader with one or more file paths."""
        self.file_paths: List[str] = list(file_paths)
        self.ut = Utility()

    # ==========================
    # Validate paths
    # ==========================

    def _validate_path(self, p: str) -> str:
        """Validate a single file path."""
        path = Path(p)

        return str(self.ut.ensure_readable_file(path))

    def _validate_paths(self, paths: List[str]) -> List[str]:
        """Validate multiple file paths."""
        return [self._validate_path(p) for p in paths]

    # ==========================
    # Load CSV
    # ==========================

    def load_csv_single(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> DataFrame:
        """Load a single CSV file into a DataFrame."""

        path = self._validate_path(self.file_paths[0])
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            skiprows=skip_rows,
            header=header_row,
        )

    def load_csv_many(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> List[DataFrame]:
        """Load multiple CSV files into a list of DataFrames."""

        paths = self._validate_paths(self.file_paths)
        return [
            pd.read_csv(
                path,
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
            )
            for path in paths
        ]

    def load_csv_chunk(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
        chunk_size: int = 1000,
    ) -> Iterable[DataFrame]:
        """Stream a large CSV file in iterable chunks."""

        path = self._validate_path(self.file_paths[0])
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            skiprows=skip_rows,
            header=header_row,
            chunksize=chunk_size,
        )

    # ==========================
    # Load Excel
    # ==========================

    def load_excel_single(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> DataFrame:
        """Load a single CSV file into a DataFrame."""

        path = self._validate_path(self.file_paths[0])
        return pd.read_excel(
            path,
            engine="openpyxl",
            skiprows=skip_rows,
            header=header_row,
        )

    # ==========================
    # Load JSON
    # ==========================
    # COMING SOON

    # ==========================
    # Load Parquet
    # ==========================

    def load_parquet_single(self) -> DataFrame:
        """Load a single Parquet file into a DataFrame."""

        path = self._validate_path(self.file_paths[0])
        return pd.read_parquet(
            path, engine="pyarrow"
        )

    def load_parquet_many(self) -> List[DataFrame]:
        """Load multiple Parquet files into a list of DataFrames."""

        paths = self._validate_paths(self.file_paths)
        return [
            pd.read_parquet(
                path, engine="pyarrow"
            )
            for path in paths
        ]
