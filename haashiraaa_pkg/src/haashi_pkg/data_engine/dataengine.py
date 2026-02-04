# dataengine.py

from __future__ import annotations

import pandas as pd
import json
from pandas import DataFrame, Series
from typing import (
    List,
    Sequence,
    Union,
    Any,
    Tuple,
    Dict
)

# =========================
# Global config
# =========================

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


# =========================
# Type aliases
# ========================

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
    Core utilities for inspecting, validating, cleaning, and transforming tabular data.
    Inspection is non-mutating; cleaning returns new objects.
    """

    def __init__(self) -> None:
        """Initialize engine state"""
        self.dropped_row_count: int = 0
        self.cummulative_missing: int = 0

    # =====================================================
    # Inspection (NO mutation)
    # =====================================================

    def inspect_dataframe(
        self,
        df: DataFrame,
        rows: int = 5,
        verbose: bool = True,
    ) -> None:
        """Print a quick structural snapshot of the dataframe."""
        if verbose:
            print(df.head(rows))
            print(df.dtypes)
            print(df.shape)

    def count_missing(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        """Count missing values for one or more columns."""
        if isinstance(columns, (list, tuple)):
            return [int(df[col].isna().sum()) for col in columns]
        return int(df[columns].isna().sum())

    def count_duplicates(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        """Count duplicated values within one or more columns."""
        if isinstance(columns, (list, tuple)):
            return [int(df[col].duplicated().sum()) for col in columns]
        return int(df[columns].duplicated().sum())

    def inspect_text_formatting(
        self,
        df: DataFrame,
        column: str,
    ) -> str:
        """Detect common text hygiene issues in a string column."""
        s = df[column].astype(str)
        lowered = s.str.lower()

        text_format = {
            "total_values_checked": len(s),
            "has_leading_trailing_whitespace": s.str.match(
                r"^\s|\s$"
            ).any(),
            "has_multiple_internal_spaces": s.str.contains(r"\s{2,}").any(),
            "has_tabs_or_newlines": s.str.contains(r"[\t\n\r]").any(),
            "has_case_inconsistency": lowered.nunique() < s.nunique(),
        }

        return self.inspect_text_formatting_json(text_format)

    def inspect_text_formatting_json(self, data: Dict[str, Any]) -> str:
        """Serialize text inspection results as formatted JSON."""
        cleaned = {k: bool(v) for k, v in data.items()}
        return json.dumps(cleaned, indent=4)

    # =====================================================
    # VALIDATION (ASSET CHECKS)
    # =====================================================

    def validate_columns_exist(
        self,
        df: DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        """Ensure all required columns exist in the dataframe."""
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
        """Validate numeric column contains only valid non-negative values."""

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
        """Validate datetime dtype and absence of missing values."""
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
        """Coerce mixed-format numeric strings into numeric dtype."""
        cleaned = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)
        if integer:
            return pd.to_numeric(cleaned, errors="coerce").astype("Int64")
        return pd.to_numeric(cleaned, errors="coerce")

    def convert_datetime(
        self,
        series: Series,
    ) -> Series:
        """Convert a series to datetime with flexible parsing."""
        return pd.to_datetime(series, errors="coerce", format="mixed")

    # =====================================================
    # Normalization
    # =====================================================

    def normalize_column_names(self, df: DataFrame) -> DataFrame:
        """Standardize column names to lowercase and trimmed format."""
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        return df

    def normalize_text_values(
        self,
        series: Series,
        method: str = "lower",
    ) -> Series:
        """Normalize text casing and whitespace in a series."""
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
        """Return total, missing count, and missing percentage."""
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
        """Drop rows missing values in specified columns."""
        mask = df[columns].isna().any(axis=1)
        self.dropped_row_count += int(mask.sum())
        return df.loc[~mask].copy()

    def fill_missing_forward(self, series: Series) -> Series:
        """Forward-fill missing values in a series."""
        return series.ffill()

    def fill_missing_backward(self, series: Series) -> Series:
        """Backward-fill missing values in a series."""
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
        """Aggregate values by group using one or more operations."""
        gb = df.groupby(group_cols, observed=True)[value_col]
        if isinstance(op, (list, tuple)):
            return gb.agg([o.lower() for o in op])
        return getattr(gb, op.lower())()

    # ==============
    # Joins
    # ==============

    def merge(
        self,
        df1: DataFrame,
        df2: DataFrame,
        merge_col: list[str] | str,
        how: str = "left",
        validate: str = "1:1"
    ) -> DataFrame:

        return pd.merge(df1, df2, on=merge_col, how=how, validate=validate)
