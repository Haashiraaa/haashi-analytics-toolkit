from __future__ import annotations

import sys
import pandas as pd
import logging
from pandas import DataFrame, Series
from haashi_pkg.utility.utils import Utility
from typing import (
    List,
    Optional,
    Sequence,
    Union,
    Iterable,
    Any,
    Tuple,
    Dict,
    Literal,
)

# =========================
# Global config
# =========================

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

ut = Utility(level=logging.INFO)

# =========================
# Type aliases
# =========================

Column = Union[str, Sequence[str]]
AggOp = Union[str, Sequence[str]]
MissingStats = Tuple[int, int, float]

# =========================
# Validation Error
# =========================


class DataValidationError(Exception):
    """Raised when a data asset fails validation."""


class DataEngine:
    """
    DataEngine provides reusable utilities for loading, inspecting,
    validating, cleaning, transforming, and summarizing tabular data.

    Philosophy:
    - Inspection does NOT mutate data
    - Validation FAILS FAST
    - Cleaning returns new objects
    - Conversions are explicit
    """

    def __init__(self, *file_paths: str, save_path: Optional[str] = None) -> None:
        self.file_paths: List[str] = list(file_paths)
        self.save_path: Optional[str] = save_path
        self.dropped_row_count: int = 0
        self.cummulative_missing: int = 0

    # =====================================================
    # Loading
    # =====================================================

    def load_csv_single(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> DataFrame:
        try:
            return pd.read_csv(
                self.file_paths[0],
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
            )
        except FileNotFoundError as fnf:
            print(ut.text["MISSING_FILE"])
            ut.debug(fnf)
            sys.exit(1)
        except Exception as exc:
            print(ut.text["ERROR"])
            ut.debug(exc)
            sys.exit(1)

    def load_csv_many(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> List[DataFrame]:
        try:
            return [
                pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    skiprows=skip_rows,
                    header=header_row,
                )
                for path in self.file_paths
            ]
        except FileNotFoundError as fnf:
            print(ut.text["MISSING_FILE"])
            ut.debug(fnf)
            sys.exit(1)
        except Exception as exc:
            print(ut.text["ERROR"])
            ut.debug(exc)
            sys.exit(1)

    def load_csv_chunk(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
        chunk_size: int = 1000,
    ) -> Iterable[DataFrame]:
        try:
            return pd.read_csv(
                self.file_paths[0],
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
                chunksize=chunk_size,
            )
        except FileNotFoundError as fnf:
            print(ut.text["MISSING_FILE"])
            ut.debug(fnf)
            sys.exit(1)
        except Exception as exc:
            print(ut.text["ERROR"])
            ut.debug(exc)
            sys.exit(1)

    # =====================================================
    # Inspection (NO mutation)
    # =====================================================

    def inspect_dataframe(
        self,
        df: DataFrame,
        rows: int = 5,
        verbose: bool = True,
    ) -> None:
        if verbose:
            print(df.head(rows))
            print(df.dtypes)
            print(df.shape)

    def count_missing(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        if isinstance(columns, (list, tuple)):
            return [int(df[col].isna().sum()) for col in columns]
        return int(df[columns].isna().sum())

    def count_duplicates(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        if isinstance(columns, (list, tuple)):
            return [int(df[col].duplicated().sum()) for col in columns]
        return int(df[columns].duplicated().sum())

    def inspect_text_formatting(
        self,
        df: DataFrame,
        column: str,
    ) -> Dict[str, Any]:
        s = df[column].astype(str)
        lowered = s.str.lower()

        return {
            "total_values_checked": len(s),
            "has_leading_trailing_whitespace": s.str.match(r"^\s|\s$").any(),
            "has_multiple_internal_spaces": s.str.contains(r"\s{2,}").any(),
            "has_tabs_or_newlines": s.str.contains(r"[\t\n\r]").any(),
            "has_case_inconsistency": lowered.nunique() < s.nunique(),
        }

    # =====================================================
    # VALIDATION (ASSET CHECKS)
    # =====================================================

    def validate_columns_exist(
        self,
        df: DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"Missing required columns: {missing}"
            )

    def validate_numeric_non_negative(
        self,
        df: DataFrame,
        column: str,
        allow_zero: bool = False,
    ) -> None:

        if column not in df.columns:
            raise DataValidationError(f"Column '{column}' does not exist")

        s = df[column]

        if not pd.api.types.is_numeric_dtype(s):
            raise DataValidationError(
                f"Column '{column}' is not numeric"
            )

        if s.isna().all():
            raise DataValidationError(
                f"Column '{column}' is entirely missing"
            )

        if allow_zero:
            invalid = (s < 0).any()
        else:
            invalid = (s <= 0).any()

        if invalid:
            raise DataValidationError(
                f"Invalid values found in '{column}'"
            )

    def validate_dates(
        self,
        df: DataFrame,
        column: str,
    ) -> None:
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            raise DataValidationError(
                f"Column '{column}' is not datetime"
            )
        if df[column].isna().any():
            raise DataValidationError(
                f"Missing values found in date column '{column}'"
            )

    # =====================================================
    # Type Conversion
    # =====================================================

    def convert_numeric(
        self,
        series: Series,
        integer: bool = False,
    ) -> Series:
        cleaned = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)
        if integer:
            return pd.to_numeric(cleaned, errors="coerce").astype("Int64")
        return pd.to_numeric(cleaned, errors="coerce")

    def convert_datetime(
        self,
        series: Series,
    ) -> Series:
        return pd.to_datetime(series, errors="coerce", format="mixed")

    # =====================================================
    # Normalization
    # =====================================================

    def normalize_column_names(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        return df

    def normalize_text_values(
        self,
        series: Series,
        method: str = "lower",
    ) -> Series:
        s = series.astype(str).str.strip()
        if method == "lower":
            return s.str.lower()
        if method == "upper":
            return s.str.upper()
        if method == "title":
            return s.str.title()
        raise ValueError("Invalid normalization method")

    # =====================================================
    # Missing Data
    # =====================================================

    def missing_summary(
        self,
        df: DataFrame,
        column: str,
    ) -> MissingStats:
        total = len(df)
        missing = int(df[column].isna().sum())
        percent = (missing / total) * 100
        self.cummulative_missing += missing
        return total, missing, percent

    def drop_rows_with_missing(
        self,
        df: DataFrame,
        columns: Sequence[str],
    ) -> DataFrame:
        mask = df[columns].isna().any(axis=1)
        self.dropped_row_count += int(mask.sum())
        return df.loc[~mask].copy()

    def fill_missing_forward(self, series: Series) -> Series:
        return series.ffill()

    def fill_missing_backward(self, series: Series) -> Series:
        return series.bfill()

    # =====================================================
    # Aggregation
    # =====================================================

    def aggregate(
        self,
        df: DataFrame,
        value_col: str,
        group_cols: Union[str, List[str]],
        op: AggOp = "sum",
    ) -> Union[Series, DataFrame]:
        gb = df.groupby(group_cols, observed=True)[value_col]
        if isinstance(op, (list, tuple)):
            return gb.agg([o.lower() for o in op])
        return getattr(gb, op.lower())()

    # =====================================================
    # Saving
    # =====================================================

    def save_csv(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        path = path or self.save_path
        if not path:
            raise ValueError("No save path provided")
        df.to_csv(path, index=False)
        print(f"File saved → {path}")

