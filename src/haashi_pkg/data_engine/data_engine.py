

# data_engine.py


"""
Data Engine: Comprehensive data loading, analysis, and saving utilities.

This module provides a complete toolkit for working with tabular data in pandas,
including loading from various formats, data inspection and validation, cleaning
and transformation, and saving to multiple output formats.

Classes:
    DataEngineError: Base exception for all data engine errors
    DataValidationError: Raised when data validation fails
    DataTypeError: Raised when data type conversion or validation fails
    FileLoadError: Raised when file loading operations fail
    FileSaveError: Raised when file saving operations fail
    
    DataAnalyzer: Core utilities for inspecting, validating, cleaning, and transforming data
    DataLoader: Lightweight data ingestion for loading tabular data formats
    DataSaver: Save pandas DataFrames to disk in multiple formats

Example:
    >>> from haashi_pkg.utility.utils import Logger, FileHandler
    >>> import logging
    >>> 
    >>> # Initialize components
    >>> logger = Logger(level=logging.INFO)
    >>> file_handler = FileHandler(logger=logger)
    >>> 
    >>> # Load data
    >>> loader = DataLoader("data.csv", logger=logger)
    >>> df = loader.load_csv_single()
    >>> 
    >>> # Analyze and clean
    >>> analyzer = DataAnalyzer(logger=logger)
    >>> analyzer.validate_columns_exist(df, ['name', 'age', 'salary'])
    >>> df_clean = analyzer.normalize_column_names(df)
    >>> 
    >>> # Save results
    >>> saver = DataSaver(save_path="output.parquet", logger=logger, file_handler=file_handler)
    >>> saver.save_parquet_compressed(df_clean)
"""

from __future__ import annotations

import logging
import pandas as pd
import json

from pandas import DataFrame, Series
from pathlib import Path
from typing import (
    List,
    Sequence,
    Union,
    Any,
    Tuple,
    Dict,
    Optional,
    Iterable
)

from haashi_pkg.utility.utils import Logger, FileHandler

# =========================
# Global pandas config
# =========================

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


# =========================
# Type aliases
# ========================

Column = Union[str, Sequence[str]]
AggOp = Union[str, Sequence[str]]
MissingStats = Tuple[int, int, float]

DataLike = Union[
    Dict[str, List[Any]],
    List[Dict[Any, Any]],
    List[List[Any]]
]
ColumnLike = Optional[List[str]]
IndexLike = Optional[List[Any]]


# =========================
# CUSTOM EXCEPTIONS
# =========================

class DataEngineError(Exception):
    """Base exception class for all Data Engine errors.

    All custom exceptions in the data engine inherit from this class,
    allowing for easy catching of all data-engine-specific errors.
    """
    pass


class DataValidationError(DataEngineError):
    """Raised when data validation fails.

    This occurs when data doesn't meet expected constraints such as:
    - Missing required columns
    - Invalid data types
    - Out-of-range values
    - Missing values where not allowed

    Args:
        message: Description of the validation error
    """
    pass


class DataTypeError(DataEngineError):
    """Raised when data type conversion or validation fails.

    This occurs when:
    - Type conversion fails
    - Data is not of expected type
    - Numeric data contains non-numeric values

    Args:
        message: Description of the type error
    """
    pass


class FileLoadError(DataEngineError):
    """Raised when file loading operations fail.

    This occurs when:
    - File doesn't exist
    - File format is invalid
    - File cannot be read
    - Path is not accessible

    Args:
        message: Description of the load error
        path: Optional file path that caused the error
    """

    def __init__(self, message: str, path: Optional[str] = None):
        self.path = path
        if path:
            message = f"{message} (path: {path})"
        super().__init__(message)


class FileSaveError(DataEngineError):
    """Raised when file saving operations fail.

    This occurs when:
    - Path is not writable
    - Disk is full
    - Invalid file format
    - Permission denied

    Args:
        message: Description of the save error
        path: Optional file path that caused the error
    """

    def __init__(self, message: str, path: Optional[str] = None):
        self.path = path
        if path:
            message = f"{message} (path: {path})"
        super().__init__(message)


# =========================
# DATA ANALYZER
# =========================

class DataAnalyzer:
    """
    Core utilities for inspecting, validating, cleaning, and transforming tabular data.

    This class provides a comprehensive set of tools for data quality assurance,
    including inspection methods (non-mutating), validation methods (raising errors
    on failure), and transformation methods (returning new objects).

    The analyzer maintains internal state tracking dropped rows and cumulative
    missing values for reporting purposes.

    Attributes:
        logger (Logger): Logger instance for debug/info messages
        dropped_row_count (int): Count of rows dropped during cleaning operations
        cumulative_missing (int): Cumulative count of missing values encountered

    Example:
        >>> from haashi_pkg.utility.utils import Logger
        >>> import logging
        >>> 
        >>> logger = Logger(level=logging.DEBUG)
        >>> analyzer = DataAnalyzer(logger=logger)
        >>> 
        >>> # Inspect data
        >>> analyzer.inspect_dataframe(df, rows=10)
        >>> missing = analyzer.count_missing(df, 'salary')
        >>> 
        >>> # Validate
        >>> analyzer.validate_columns_exist(df, ['name', 'age', 'salary'])
        >>> analyzer.validate_numeric_non_negative(df, 'salary', allow_zero=False)
        >>> 
        >>> # Clean
        >>> df_clean = analyzer.normalize_column_names(df)
        >>> df_clean = analyzer.drop_rows_with_missing(df_clean, ['salary'])
    """

    def __init__(
        self,
        logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize the DataAnalyzer.

        Args:
            logger: Optional Logger instance. If None, creates a default logger
                   at DEBUG level.
        """
        self.logger = logger or Logger(level=logging.DEBUG)
        self.dropped_row_count: int = 0
        self.cumulative_missing: int = 0

    # =====================================================
    # INSPECTION (NO mutation)
    # =====================================================

    def inspect_dataframe(
        self,
        df: DataFrame,
        rows: int = 5,
        verbose: bool = True,
    ) -> None:
        """
        Print a quick structural snapshot of the dataframe.

        This method provides a non-mutating overview of the dataframe structure,
        showing sample rows, data types, and shape.

        Args:
            df: DataFrame to inspect
            rows: Number of rows to display. Default is 5.
            verbose: If True, prints the inspection output. If False, silent.
                    Default is True.

        Example:
            >>> analyzer.inspect_dataframe(df, rows=10)
            # Prints first 10 rows, data types, and shape

            >>> # Silent inspection (useful when logging at DEBUG level)
            >>> analyzer.inspect_dataframe(df, verbose=False)
        """
        if verbose:
            self.logger.info(f"\n{df.head(rows)}")
            self.logger.info(f"\nData types:\n{df.dtypes}")
            self.logger.info(f"\nShape: {df.shape}")

    def count_missing(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        """
        Count missing values for one or more columns.

        Args:
            df: DataFrame to analyze
            columns: Single column name (str) or sequence of column names

        Returns:
            int: If single column, returns count of missing values
            List[int]: If multiple columns, returns list of counts

        Raises:
            DataValidationError: If column(s) don't exist in DataFrame

        Example:
            >>> # Single column
            >>> missing = analyzer.count_missing(df, 'salary')
            >>> print(f"Missing salaries: {missing}")

            >>> # Multiple columns
            >>> missing_counts = analyzer.count_missing(df, ['age', 'salary', 'bonus'])
            >>> for col, count in zip(['age', 'salary', 'bonus'], missing_counts):
            ...     print(f"{col}: {count} missing")
        """
        try:
            if isinstance(columns, (list, tuple)):
                # Validate all columns exist
                missing_cols = [c for c in columns if c not in df.columns]
                if missing_cols:
                    raise DataValidationError(
                        f"Columns not found in DataFrame: {missing_cols}"
                    )
                return [int(df[col].isna().sum()) for col in columns]
            else:
                # Validate single column exists
                if columns not in df.columns:
                    raise DataValidationError(
                        f"Column '{columns}' not found in DataFrame"
                    )
                return int(df[columns].isna().sum())
        except DataValidationError:
            raise
        except Exception as e:
            raise DataEngineError(
                f"Failed to count missing values: {str(e)}"
            ) from e

    def count_duplicates(
        self,
        df: DataFrame,
        columns: Column,
    ) -> Union[int, List[int]]:
        """
        Count duplicated values within one or more columns.

        This counts the number of duplicate occurrences (not unique values).
        For example, if a value appears 3 times, it contributes 2 to the count.

        Args:
            df: DataFrame to analyze
            columns: Single column name (str) or sequence of column names

        Returns:
            int: If single column, returns count of duplicate values
            List[int]: If multiple columns, returns list of duplicate counts

        Raises:
            DataValidationError: If column(s) don't exist in DataFrame

        Example:
            >>> # Check for duplicate IDs
            >>> dup_count = analyzer.count_duplicates(df, 'customer_id')
            >>> if dup_count > 0:
            ...     print(f"Warning: {dup_count} duplicate customer IDs found")

            >>> # Check multiple columns
            >>> dup_counts = analyzer.count_duplicates(df, ['email', 'phone', 'ssn'])
        """
        try:
            if isinstance(columns, (list, tuple)):
                # Validate all columns exist
                missing_cols = [c for c in columns if c not in df.columns]
                if missing_cols:
                    raise DataValidationError(
                        f"Columns not found in DataFrame: {missing_cols}"
                    )
                return [int(df[col].duplicated().sum()) for col in columns]
            else:
                # Validate single column exists
                if columns not in df.columns:
                    raise DataValidationError(
                        f"Column '{columns}' not found in DataFrame"
                    )
                return int(df[columns].duplicated().sum())
        except DataValidationError:
            raise
        except Exception as e:
            raise DataEngineError(
                f"Failed to count duplicates: {str(e)}"
            ) from e

    def inspect_text_formatting(
        self,
        df: DataFrame,
        column: str,
    ) -> str:
        """
        Detect common text hygiene issues in a string column.

        This method scans a text column for common formatting problems that
        may need cleaning, including whitespace issues and case inconsistencies.

        Args:
            df: DataFrame containing the column
            column: Name of the text column to inspect

        Returns:
            str: JSON-formatted string with inspection results including:
                - total_values_checked: Number of values inspected
                - has_leading_trailing_whitespace: Whitespace at start/end
                - has_multiple_internal_spaces: Multiple consecutive spaces
                - has_tabs_or_newlines: Tab or newline characters
                - has_case_inconsistency: Same text with different casing

        Raises:
            DataValidationError: If column doesn't exist

        Example:
            >>> report = analyzer.inspect_text_formatting(df, 'company_name')
            >>> print(report)
            {
                "total_values_checked": 1000,
                "has_leading_trailing_whitespace": true,
                "has_multiple_internal_spaces": false,
                "has_tabs_or_newlines": false,
                "has_case_inconsistency": true
            }
        """
        if column not in df.columns:
            raise DataValidationError(
                f"Column '{column}' not found in DataFrame"
            )

        try:
            s = df[column].astype(str)
            lowered = s.str.lower()

            text_format = {
                "total_values_checked": len(s),
                "has_leading_trailing_whitespace": bool(
                    s.str.match(r"^\s|\s$").any()
                ),
                "has_multiple_internal_spaces": bool(
                    s.str.contains(r"\s{2,}").any()
                ),
                "has_tabs_or_newlines": bool(
                    s.str.contains(r"[\t\n\r]").any()
                ),
                "has_case_inconsistency": lowered.nunique() < s.nunique(),
            }

            return json.dumps(text_format, indent=4)
        except Exception as e:
            raise DataEngineError(
                f"Failed to inspect text formatting for column '{column}': {str(e)}"
            ) from e

    # =====================================================
    # VALIDATION (ASSET CHECKS)
    # =====================================================

    def validate_columns_exist(
        self,
        df: DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        """
        Ensure all required columns exist in the dataframe.

        This is typically the first validation step in a data pipeline,
        verifying that the data source has the expected structure.

        Args:
            df: DataFrame to validate
            required_columns: List of column names that must be present

        Raises:
            DataValidationError: If any required columns are missing,
                                with a list of the missing column names

        Example:
            >>> # Validate required columns for analysis
            >>> required = ['customer_id', 'order_date', 'amount', 'status']
            >>> analyzer.validate_columns_exist(df, required)
            >>> # Raises error if any columns missing

            >>> # Use in pipeline with error handling
            >>> try:
            ...     analyzer.validate_columns_exist(df, ['id', 'name', 'email'])
            ...     # Continue processing
            ... except DataValidationError as e:
            ...     logger.error(f"Data validation failed: {e}")
            ...     # Handle missing columns
        """
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"Missing required columns: {missing}"
            )

        self.logger.debug(
            f"Validated all required columns present: {list(required_columns)}"
        )

    def validate_numeric_non_negative(
        self,
        df: DataFrame,
        column: str,
        allow_zero: bool = False,
    ) -> None:
        """
        Validate numeric column contains only valid non-negative values.

        This ensures a numeric column has appropriate values for metrics
        like prices, quantities, ages, etc. that shouldn't be negative.

        Args:
            df: DataFrame to validate
            column: Name of the numeric column to validate
            allow_zero: If True, zero is valid. If False, only positive values
                       are valid. Default is False.

        Raises:
            DataValidationError: If:
                - Column doesn't exist
                - Column is not numeric type
                - Column is entirely missing/null
                - Column contains negative values (or zero if allow_zero=False)

        Example:
            >>> # Validate price column (positive only)
            >>> analyzer.validate_numeric_non_negative(df, 'price', allow_zero=False)

            >>> # Validate quantity (zero allowed)
            >>> analyzer.validate_numeric_non_negative(df, 'quantity', allow_zero=True)

            >>> # Use in validation pipeline
            >>> for col in ['price', 'cost', 'tax']:
            ...     analyzer.validate_numeric_non_negative(df, col)
        """
        if column not in df.columns:
            raise DataValidationError(
                f"Column '{column}' does not exist in DataFrame"
            )

        s = df[column]

        if not pd.api.types.is_numeric_dtype(s):
            raise DataValidationError(
                f"Column '{column}' is not numeric (dtype: {s.dtype})"
            )

        if s.isna().all():
            raise DataValidationError(
                f"Column '{column}' is entirely missing (all NaN values)"
            )

        # Check for invalid values (ignoring NaN)
        valid_values = s.dropna()
        if allow_zero:
            invalid = (valid_values < 0).any()
            if invalid:
                raise DataValidationError(
                    f"Negative values found in column '{column}' "
                    f"(min: {valid_values.min()})"
                )
        else:
            invalid = (valid_values <= 0).any()
            if invalid:
                raise DataValidationError(
                    f"Non-positive values found in column '{column}' "
                    f"(min: {valid_values.min()})"
                )

        self.logger.debug(
            f"Validated column '{column}': all values are "
            f"{'non-negative' if allow_zero else 'positive'}"
        )

    def validate_dates(
        self,
        df: DataFrame,
        column: str,
    ) -> None:
        """
        Validate datetime dtype and absence of missing values.

        This ensures a column is properly formatted as datetime and has
        no missing values, which is critical for time-series analysis.

        Args:
            df: DataFrame to validate
            column: Name of the datetime column to validate

        Raises:
            DataValidationError: If:
                - Column is not datetime type
                - Column contains any missing/null values

        Example:
            >>> # Validate date columns
            >>> analyzer.validate_dates(df, 'order_date')
            >>> analyzer.validate_dates(df, 'ship_date')

            >>> # Convert then validate
            >>> df['date'] = pd.to_datetime(df['date_string'])
            >>> analyzer.validate_dates(df, 'date')
        """
        if column not in df.columns:
            raise DataValidationError(
                f"Column '{column}' does not exist in DataFrame"
            )

        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            raise DataValidationError(
                f"Column '{column}' is not datetime type (dtype: {df[column].dtype})"
            )

        if df[column].isna().any():
            missing_count = df[column].isna().sum()
            raise DataValidationError(
                f"Missing values found in date column '{column}' "
                f"({missing_count} missing)"
            )

        self.logger.debug(
            f"Validated column '{column}': datetime type with no missing values"
        )

    # =====================================================
    # TYPE CONVERSION
    # =====================================================

    def convert_numeric(
        self,
        series: Series,
        integer: bool = False,
    ) -> Series:
        """
        Coerce mixed-format numeric strings into numeric dtype.

        This method handles common numeric formatting issues like currency
        symbols, commas, and other non-numeric characters, converting them
        to clean numeric values.

        Args:
            series: Series to convert (typically with string dtype)
            integer: If True, convert to nullable integer (Int64).
                    If False, convert to float. Default is False.

        Returns:
            Series: Converted series with numeric dtype. Invalid values
                   become NaN.

        Example:
            >>> # Clean currency strings
            >>> prices = pd.Series(['$1,234.56', '$5,678.90', 'N/A'])
            >>> clean = analyzer.convert_numeric(prices)
            >>> print(clean)
            0    1234.56
            1    5678.90
            2        NaN

            >>> # Convert to integers
            >>> quantities = pd.Series(['1,000', '2,500', '3,750'])
            >>> clean = analyzer.convert_numeric(quantities, integer=True)
            >>> print(clean.dtype)
            Int64
        """
        try:
            # Remove all non-numeric characters except decimal point
            cleaned = series.astype(str).str.replace(
                r"[^0-9.-]", "", regex=True)

            if integer:
                return pd.to_numeric(cleaned, errors="coerce").astype("Int64")
            return pd.to_numeric(cleaned, errors="coerce")
        except Exception as e:
            raise DataTypeError(
                f"Failed to convert series to numeric: {str(e)}"
            ) from e

    def convert_datetime(
        self,
        series: Series,
    ) -> Series:
        """
        Convert a series to datetime with flexible parsing.

        This method handles various datetime formats automatically,
        converting string dates to proper datetime objects.

        Args:
            series: Series to convert (typically string dtype)

        Returns:
            Series: Converted series with datetime64 dtype. Invalid dates
                   become NaT (Not a Time).

        Example:
            >>> dates = pd.Series(['2024-01-15', '15/01/2024', 'Jan 15, 2024'])
            >>> clean = analyzer.convert_datetime(dates)
            >>> print(clean.dtype)
            datetime64[ns]

            >>> # Use in pipeline
            >>> df['order_date'] = analyzer.convert_datetime(df['order_date'])
            >>> analyzer.validate_dates(df, 'order_date')
        """
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except Exception as e:
            raise DataTypeError(
                f"Failed to convert series to datetime: {str(e)}"
            ) from e

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize_column_names(self, df: DataFrame) -> DataFrame:
        """
        Standardize column names to lowercase and trimmed format.

        This creates consistent column naming for easier programmatic access
        and avoids issues with case sensitivity and whitespace.

        Args:
            df: DataFrame to normalize (not modified)

        Returns:
            DataFrame: New DataFrame with normalized column names

        Example:
            >>> # Before: ['Customer ID', 'First Name', '  Email  ']
            >>> df_clean = analyzer.normalize_column_names(df)
            >>> print(df_clean.columns.tolist())
            ['customer id', 'first name', 'email']

            >>> # Use at start of pipeline
            >>> df = analyzer.normalize_column_names(df)
            >>> # Now can access columns consistently
            >>> df['customer id']  # No worries about case or spaces
        """
        try:
            df = df.copy()
            df.columns = df.columns.str.lower().str.strip()
            self.logger.debug(
                "Normalized column names to lowercase and trimmed")
            return df
        except Exception as e:
            raise DataEngineError(
                f"Failed to normalize column names: {str(e)}"
            ) from e

    def normalize_text_values(
        self,
        series: Series,
        method: str = "lower",
    ) -> Series:
        """
        Normalize text casing and whitespace in a series.

        This standardizes text values for consistent matching and comparison,
        trimming whitespace and applying case normalization.

        Args:
            series: Series with text values to normalize
            method: Normalization method. Options:
                - 'lower': Convert to lowercase (default)
                - 'upper': Convert to UPPERCASE
                - 'title': Convert to Title Case

        Returns:
            Series: New series with normalized text values

        Raises:
            DataEngineError: If method is not one of the valid options

        Example:
            >>> # Normalize names to title case
            >>> df['name'] = analyzer.normalize_text_values(df['name'], method='title')

            >>> # Normalize categories to lowercase
            >>> df['category'] = analyzer.normalize_text_values(df['category'], method='lower')

            >>> # Before: ['  Apple  ', 'BANANA', 'Cherry']
            >>> normalized = analyzer.normalize_text_values(series, method='lower')
            >>> print(normalized.tolist())
            ['apple', 'banana', 'cherry']
        """
        try:
            s = series.astype(str).str.strip()

            if method == "lower":
                result = s.str.lower()
            elif method == "upper":
                result = s.str.upper()
            elif method == "title":
                result = s.str.title()
            else:
                raise DataEngineError(
                    f"Invalid normalization method: '{method}'. "
                    f"Valid options are: 'lower', 'upper', 'title'"
                )

            self.logger.debug(
                f"Normalized text values using method '{method}'")
            return result
        except DataEngineError:
            raise
        except Exception as e:
            raise DataEngineError(
                f"Failed to normalize text values: {str(e)}"
            ) from e

    # =====================================================
    # MISSING DATA
    # =====================================================

    def missing_summary(
        self,
        df: DataFrame,
        column: str,
    ) -> MissingStats:
        """
        Return total, missing count, and missing percentage for a column.

        This method also updates the cumulative_missing counter for tracking
        total missing values encountered across multiple operations.

        Args:
            df: DataFrame to analyze
            column: Column name to check for missing values

        Returns:
            Tuple[int, int, float]: (total_rows, missing_count, missing_percent)

        Raises:
            DataValidationError: If column doesn't exist

        Example:
            >>> total, missing, percent = analyzer.missing_summary(df, 'salary')
            >>> print(f"Salary: {missing}/{total} missing ({percent:.1f}%)")
            Salary: 45/1000 missing (4.5%)

            >>> # Check multiple columns
            >>> for col in ['age', 'salary', 'bonus']:
            ...     total, missing, pct = analyzer.missing_summary(df, col)
            ...     if pct > 5:
            ...         print(f"Warning: {col} has {pct:.1f}% missing")
        """
        if column not in df.columns:
            raise DataValidationError(
                f"Column '{column}' not found in DataFrame"
            )

        try:
            total = len(df)
            missing = int(df[column].isna().sum())
            percent = (missing / total) * 100 if total > 0 else 0.0

            self.cumulative_missing += missing

            self.logger.debug(
                f"Column '{column}': {missing}/{total} missing ({percent:.1f}%)"
            )

            return total, missing, percent
        except DataValidationError:
            raise
        except Exception as e:
            raise DataEngineError(
                f"Failed to generate missing summary for column '{column}': {str(e)}"
            ) from e

    def drop_rows_with_missing(
        self,
        df: DataFrame,
        columns: Sequence[str],
    ) -> DataFrame:
        """
        Drop rows missing values in specified columns.

        This method removes any row that has missing values in ANY of the
        specified columns. Updates the dropped_row_count counter.

        Args:
            df: DataFrame to clean (not modified)
            columns: List of column names to check for missing values

        Returns:
            DataFrame: New DataFrame with rows containing missing values removed

        Raises:
            DataValidationError: If any specified columns don't exist

        Example:
            >>> # Drop rows missing critical columns
            >>> df_clean = analyzer.drop_rows_with_missing(
            ...     df,
            ...     ['customer_id', 'order_date', 'amount']
            ... )
            >>> print(f"Dropped {analyzer.dropped_row_count} rows")

            >>> # Before: 1000 rows
            >>> df_clean = analyzer.drop_rows_with_missing(df, ['email', 'phone'])
            >>> # After: 950 rows (50 had missing email or phone)
        """
        # Validate columns exist
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"Columns not found in DataFrame: {missing_cols}"
            )

        try:
            mask = df[columns].isna().any(axis=1)
            rows_to_drop = int(mask.sum())
            self.dropped_row_count += rows_to_drop

            self.logger.debug(
                f"Dropping {rows_to_drop} rows with missing values in {columns}"
            )

            return df.loc[~mask].copy()
        except DataValidationError:
            raise
        except Exception as e:
            raise DataEngineError(
                f"Failed to drop rows with missing values: {str(e)}"
            ) from e

    def fill_missing_forward(self, series: Series) -> Series:
        """
        Forward-fill missing values in a series.

        This propagates the last valid observation forward to fill gaps,
        useful for time-series data where values tend to persist.

        Args:
            series: Series with missing values to fill

        Returns:
            Series: New series with missing values forward-filled

        Example:
            >>> # Before: [1, NaN, NaN, 4, NaN, 6]
            >>> filled = analyzer.fill_missing_forward(series)
            >>> # After:  [1, 1,   1,   4, 4,   6]

            >>> # Fill missing prices with last known price
            >>> df['price'] = analyzer.fill_missing_forward(df['price'])
        """
        try:
            result = series.ffill()
            filled_count = series.isna().sum() - result.isna().sum()
            self.logger.debug(f"Forward-filled {filled_count} missing values")
            return result
        except Exception as e:
            raise DataEngineError(
                f"Failed to forward-fill missing values: {str(e)}"
            ) from e

    def fill_missing_backward(self, series: Series) -> Series:
        """
        Backward-fill missing values in a series.

        This propagates the next valid observation backward to fill gaps,
        useful when future values are more relevant than past ones.

        Args:
            series: Series with missing values to fill

        Returns:
            Series: New series with missing values backward-filled

        Example:
            >>> # Before: [1, NaN, NaN, 4, NaN, 6]
            >>> filled = analyzer.fill_missing_backward(series)
            >>> # After:  [1, 4,   4,   4, 6,   6]

            >>> # Fill missing categories with next category
            >>> df['category'] = analyzer.fill_missing_backward(df['category'])
        """
        try:
            result = series.bfill()
            filled_count = series.isna().sum() - result.isna().sum()
            self.logger.debug(f"Backward-filled {filled_count} missing values")
            return result
        except Exception as e:
            raise DataEngineError(
                f"Failed to backward-fill missing values: {str(e)}"
            ) from e

    # =====================================================
    # AGGREGATION
    # =====================================================

    def aggregate(
        self,
        df: DataFrame,
        value_col: str,
        group_cols: Union[str, List[str]],
        op: AggOp = "sum",
    ) -> Union[Series, DataFrame]:
        """
        Aggregate values by group using one or more operations.

        This provides a clean interface to pandas groupby aggregations,
        supporting single or multiple aggregation operations.

        Args:
            df: DataFrame to aggregate
            value_col: Column name containing values to aggregate
            group_cols: Column name(s) to group by (str or list of str)
            op: Aggregation operation(s). Options:
                - Single operation (str): 'sum', 'mean', 'median', 'min', 'max',
                  'count', 'std', 'var', etc.
                - Multiple operations (list): ['sum', 'mean', 'count']
                Default is 'sum'.

        Returns:
            Series: If single operation, returns Series with grouped results
            DataFrame: If multiple operations, returns DataFrame with one column
                      per operation

        Raises:
            DataValidationError: If columns don't exist
            DataEngineError: If aggregation operation is invalid

        Example:
            >>> # Simple aggregation
            >>> total_sales = analyzer.aggregate(
            ...     df,
            ...     value_col='amount',
            ...     group_cols='customer_id',
            ...     op='sum'
            ... )

            >>> # Multiple aggregations
            >>> stats = analyzer.aggregate(
            ...     df,
            ...     value_col='amount',
            ...     group_cols='product_category',
            ...     op=['sum', 'mean', 'count']
            ... )
            >>> print(stats.columns)
            ['sum', 'mean', 'count']

            >>> # Group by multiple columns
            >>> monthly_sales = analyzer.aggregate(
            ...     df,
            ...     value_col='revenue',
            ...     group_cols=['year', 'month'],
            ...     op='sum'
            ... )
        """
        # Validate value column exists
        if value_col not in df.columns:
            raise DataValidationError(
                f"Value column '{value_col}' not found in DataFrame"
            )

        # Validate group columns exist
        group_list = [group_cols] if isinstance(
            group_cols, str) else group_cols
        missing_cols = [c for c in group_list if c not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"Group columns not found in DataFrame: {missing_cols}"
            )

        try:
            gb = df.groupby(group_cols, observed=True)[value_col]

            if isinstance(op, (list, tuple)):
                # Multiple operations
                ops_lower = [o.lower() for o in op]
                result = gb.agg(ops_lower)
                self.logger.debug(
                    f"Aggregated '{value_col}' by {group_cols} using {ops_lower}"
                )
                return result
            else:
                # Single operation
                op_lower = op.lower()
                if not hasattr(gb, op_lower):
                    raise DataEngineError(
                        f"Invalid aggregation operation: '{op}'"
                    )
                result = getattr(gb, op_lower)()
                self.logger.debug(
                    f"Aggregated '{value_col}' by {group_cols} using '{op_lower}'"
                )
                return result
        except (DataValidationError, DataEngineError):
            raise
        except Exception as e:
            raise DataEngineError(
                f"Failed to aggregate data: {str(e)}"
            ) from e

    # =====================================================
    # JOINS
    # =====================================================

    def merge(
        self,
        df1: DataFrame,
        df2: DataFrame,
        merge_col: Union[str, List[str]],
        how: str = "left",
        validate: str = "1:1"
    ) -> DataFrame:
        """
        Merge two DataFrames with validation.

        This is a wrapper around pandas merge with built-in validation
        to catch common merging errors.

        Args:
            df1: Left DataFrame
            df2: Right DataFrame
            merge_col: Column name(s) to merge on (must exist in both DataFrames)
            how: Type of merge. Options:
                - 'left': Keep all rows from df1
                - 'right': Keep all rows from df2
                - 'inner': Keep only matching rows
                - 'outer': Keep all rows from both
                Default is 'left'.
            validate: Merge validation. Options:
                - '1:1': One-to-one (no duplicates in either DataFrame)
                - '1:m': One-to-many (duplicates allowed in df2)
                - 'm:1': Many-to-one (duplicates allowed in df1)
                - 'm:m': Many-to-many (duplicates allowed in both)
                Default is '1:1'.

        Returns:
            DataFrame: Merged DataFrame

        Raises:
            DataValidationError: If merge columns don't exist or validation fails

        Example:
            >>> # Left join on customer_id
            >>> merged = analyzer.merge(
            ...     orders_df,
            ...     customers_df,
            ...     merge_col='customer_id',
            ...     how='left',
            ...     validate='m:1'  # Many orders per customer
            ... )

            >>> # Inner join on multiple columns
            >>> merged = analyzer.merge(
            ...     df1,
            ...     df2,
            ...     merge_col=['year', 'month'],
            ...     how='inner',
            ...     validate='1:1'
            ... )
        """
        # Validate merge columns exist in both DataFrames
        merge_cols = [merge_col] if isinstance(merge_col, str) else merge_col

        missing_df1 = [c for c in merge_cols if c not in df1.columns]
        missing_df2 = [c for c in merge_cols if c not in df2.columns]

        if missing_df1:
            raise DataValidationError(
                f"Merge columns not found in first DataFrame: {missing_df1}"
            )
        if missing_df2:
            raise DataValidationError(
                f"Merge columns not found in second DataFrame: {missing_df2}"
            )

        try:
            result = pd.merge(
                df1, df2,
                on=merge_col,
                how=how,
                validate=validate
            )

            self.logger.debug(
                f"Merged DataFrames on {merge_col} (how='{how}', validate='{validate}'). "
                f"Result: {len(result)} rows"
            )

            return result
        except DataValidationError:
            raise
        except Exception as e:
            raise DataEngineError(
                f"Failed to merge DataFrames: {str(e)}"
            ) from e


# =========================
# DATA LOADER
# =========================

class DataLoader:
    """
    Lightweight data ingestion utility for loading tabular data formats.

    This class focuses strictly on I/O operations - loading data from files
    into pandas DataFrames. It does not perform validation or transformation.

    Supports multiple file formats:
    - CSV (single, multiple, or chunked streaming)
    - Excel (single file)
    - Parquet (single or multiple files)

    Attributes:
        file_paths (List[str]): List of file paths to load
        logger (Logger): Logger instance for debug/info messages
        file_handler (FileHandler): Handler for file operations and validation

    Example:
        >>> from haashi_pkg.utility.utils import Logger, FileHandler
        >>> import logging
        >>> 
        >>> logger = Logger(level=logging.INFO)
        >>> file_handler = FileHandler(logger=logger)
        >>> 
        >>> # Single file
        >>> loader = DataLoader("data.csv", logger=logger, file_handler=file_handler)
        >>> df = loader.load_csv_single()
        >>> 
        >>> # Multiple files
        >>> loader = DataLoader("jan.csv", "feb.csv", "mar.csv", logger=logger, file_handler=file_handler)
        >>> dfs = loader.load_csv_many()
        >>> 
        >>> # Large file in chunks
        >>> loader = DataLoader("huge_file.csv", logger=logger, file_handler=file_handler)
        >>> for chunk in loader.load_csv_chunk(chunk_size=10000):
        ...     process(chunk)
    """

    def __init__(
        self,
        *file_paths: str,
        logger: Optional[Logger] = None,
        file_handler: Optional[FileHandler] = None
    ) -> None:
        """
        Initialize loader with one or more file paths.

        Args:
            *file_paths: Variable number of file path strings
            logger: Optional Logger instance. If None, creates default logger
            file_handler: Optional FileHandler instance. If None, creates default
        """
        self.file_paths: List[str] = list(file_paths)
        self.logger = logger or Logger(level=logging.DEBUG)
        self.file_handler = file_handler or FileHandler(logger=self.logger)

    # ==========================
    # VALIDATE PATHS
    # ==========================

    def _validate_path(self, p: str) -> str:
        """
        Validate a single file path.

        Args:
            p: File path to validate

        Returns:
            str: Validated absolute path

        Raises:
            FileLoadError: If file doesn't exist or isn't readable
        """
        try:
            path = Path(p)
            return str(self.file_handler.ensure_readable_file(path))
        except Exception as e:
            raise FileLoadError(
                f"Invalid file path: {str(e)}",
                path=p
            ) from e

    def _validate_paths(self, paths: List[str]) -> List[str]:
        """
        Validate multiple file paths.

        Args:
            paths: List of file paths to validate

        Returns:
            List[str]: List of validated absolute paths

        Raises:
            FileLoadError: If any file doesn't exist or isn't readable
        """
        return [self._validate_path(p) for p in paths]

    # ==========================
    # LOAD CSV
    # ==========================

    def load_csv_single(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> DataFrame:
        """
        Load a single CSV file into a DataFrame.

        This method automatically detects the delimiter (comma, tab, semicolon, etc.)
        and loads the data.

        Args:
            skip_rows: Number of rows to skip at the start of the file.
                      Useful for files with metadata headers. Default is 0.
            header_row: Row number (0-indexed) containing column names.
                       Default is 0 (first row after skipped rows).

        Returns:
            DataFrame: Loaded data

        Raises:
            FileLoadError: If file cannot be loaded or no file path provided

        Example:
            >>> # Simple load
            >>> loader = DataLoader("data.csv", logger=logger, file_handler=file_handler)
            >>> df = loader.load_csv_single()

            >>> # Skip metadata rows
            >>> loader = DataLoader("data_with_header.csv", logger=logger, file_handler=file_handler)
            >>> df = loader.load_csv_single(skip_rows=3, header_row=0)

            >>> # No header row
            >>> loader = DataLoader("data_no_header.csv", logger=logger, file_handler=file_handler)
            >>> df = loader.load_csv_single(header_row=None)
        """
        if not self.file_paths:
            raise FileLoadError("No file path provided to DataLoader")

        try:
            path = self._validate_path(self.file_paths[0])

            self.logger.debug(f"Loading CSV file: {path}")

            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
            )

            self.logger.info(
                f"Loaded CSV: {path} ({len(df)} rows, {len(df.columns)} columns)"
            )

            return df
        except FileLoadError:
            raise
        except Exception as e:
            raise FileLoadError(
                f"Failed to load CSV file: {str(e)}",
                path=self.file_paths[0] if self.file_paths else None
            ) from e

    def load_csv_many(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
    ) -> List[DataFrame]:
        """
        Load multiple CSV files into a list of DataFrames.

        This is useful for loading related data files that have the same structure,
        such as monthly data files or regional reports.

        Args:
            skip_rows: Number of rows to skip at the start of each file. Default is 0.
            header_row: Row number containing column names in each file. Default is 0.

        Returns:
            List[DataFrame]: List of DataFrames, one per file

        Raises:
            FileLoadError: If no files provided or any file cannot be loaded

        Example:
            >>> # Load quarterly files
            >>> loader = DataLoader(
            ...     "q1_2024.csv",
            ...     "q2_2024.csv",
            ...     "q3_2024.csv",
            ...     "q4_2024.csv",
            ...     logger=logger,
            ...     file_handler=file_handler
            ... )
            >>> dfs = loader.load_csv_many()
            >>> 
            >>> # Combine all data
            >>> combined = pd.concat(dfs, ignore_index=True)

            >>> # Process each file separately
            >>> for i, df in enumerate(dfs):
            ...     print(f"File {i+1}: {len(df)} rows")
        """
        if not self.file_paths:
            raise FileLoadError("No file paths provided to DataLoader")

        try:
            paths = self._validate_paths(self.file_paths)

            self.logger.debug(f"Loading {len(paths)} CSV files")

            dfs = []
            for path in paths:
                df = pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    skiprows=skip_rows,
                    header=header_row,
                )
                dfs.append(df)
                self.logger.debug(
                    f"Loaded: {path} ({len(df)} rows, {len(df.columns)} columns)"
                )

            self.logger.info(
                f"Loaded {len(dfs)} CSV files (total: "
                f"{sum(len(df) for df in dfs)} rows)"
            )

            return dfs
        except FileLoadError:
            raise
        except Exception as e:
            raise FileLoadError(
                f"Failed to load CSV files: {str(e)}"
            ) from e

    def load_csv_chunk(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
        chunk_size: int = 1000,
    ) -> Iterable[DataFrame]:
        """
        Stream a large CSV file in iterable chunks.

        This is memory-efficient for processing files too large to fit in RAM.
        Returns an iterator that yields DataFrames of the specified chunk size.

        Args:
            skip_rows: Number of rows to skip at start. Default is 0.
            header_row: Row number containing column names. Default is 0.
            chunk_size: Number of rows per chunk. Default is 1000.

        Returns:
            Iterable[DataFrame]: Iterator yielding DataFrame chunks

        Raises:
            FileLoadError: If file cannot be loaded or no file path provided

        Example:
            >>> # Process large file in chunks
            >>> loader = DataLoader("huge_file.csv", logger=logger, file_handler=file_handler)
            >>> for chunk in loader.load_csv_chunk(chunk_size=10000):
            ...     # Process chunk
            ...     processed = analyze(chunk)
            ...     save(processed)

            >>> # Calculate statistics without loading full file
            >>> total_sum = 0
            >>> row_count = 0
            >>> for chunk in loader.load_csv_chunk(chunk_size=5000):
            ...     total_sum += chunk['value'].sum()
            ...     row_count += len(chunk)
            >>> average = total_sum / row_count
        """
        if not self.file_paths:
            raise FileLoadError("No file path provided to DataLoader")

        try:
            path = self._validate_path(self.file_paths[0])

            self.logger.debug(
                f"Loading CSV in chunks: {path} (chunk_size={chunk_size})"
            )

            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                skiprows=skip_rows,
                header=header_row,
                chunksize=chunk_size,
            )
        except FileLoadError:
            raise
        except Exception as e:
            raise FileLoadError(
                f"Failed to load CSV in chunks: {str(e)}",
                path=self.file_paths[0] if self.file_paths else None
            ) from e

    # ==========================
    # LOAD EXCEL
    # ==========================

    def load_excel_single(
        self,
        skip_rows: int = 0,
        header_row: int = 0,
        sheet_name: Union[str, int] = 0,
    ) -> DataFrame:
        """
        Load a single Excel file into a DataFrame.

        Uses openpyxl engine for .xlsx files.

        Args:
            skip_rows: Number of rows to skip at start. Default is 0.
            header_row: Row number containing column names. Default is 0.
            sheet_name: Sheet to load. Can be sheet name (str) or index (int).
                       Default is 0 (first sheet).

        Returns:
            DataFrame: Loaded data

        Raises:
            FileLoadError: If file cannot be loaded or no file path provided

        Example:
            >>> # Load first sheet
            >>> loader = DataLoader("report.xlsx", logger=logger, file_handler=file_handler)
            >>> df = loader.load_excel_single()

            >>> # Load specific sheet by name
            >>> df = loader.load_excel_single(sheet_name='Sales Data')

            >>> # Load second sheet, skip header rows
            >>> df = loader.load_excel_single(sheet_name=1, skip_rows=2)
        """
        if not self.file_paths:
            raise FileLoadError("No file path provided to DataLoader")

        try:
            path = self._validate_path(self.file_paths[0])

            self.logger.debug(
                f"Loading Excel file: {path} (sheet={sheet_name})")

            df = pd.read_excel(
                path,
                engine="openpyxl",
                skiprows=skip_rows,
                header=header_row,
                sheet_name=sheet_name,
            )

            self.logger.info(
                f"Loaded Excel: {path} ({len(df)} rows, {len(df.columns)} columns)"
            )

            return df
        except FileLoadError:
            raise
        except Exception as e:
            raise FileLoadError(
                f"Failed to load Excel file: {str(e)}",
                path=self.file_paths[0] if self.file_paths else None
            ) from e

    # ==========================
    # LOAD PARQUET
    # ==========================

    def load_parquet_single(self) -> DataFrame:
        """
        Load a single Parquet file into a DataFrame.

        Parquet is a columnar storage format that's efficient for large datasets.
        Uses PyArrow engine for best performance.

        Returns:
            DataFrame: Loaded data

        Raises:
            FileLoadError: If file cannot be loaded or no file path provided

        Example:
            >>> loader = DataLoader("data.parquet", logger=logger, file_handler=file_handler)
            >>> df = loader.load_parquet_single()

            >>> # Parquet preserves data types and is much faster than CSV
            >>> loader = DataLoader("large_dataset.parquet", logger=logger, file_handler=file_handler)
            >>> df = loader.load_parquet_single()
        """
        if not self.file_paths:
            raise FileLoadError("No file path provided to DataLoader")

        try:
            path = self._validate_path(self.file_paths[0])

            self.logger.debug(f"Loading Parquet file: {path}")

            df = pd.read_parquet(path, engine="pyarrow")

            self.logger.info(
                f"Loaded Parquet: {path} ({len(df)} rows, {len(df.columns)} columns)"
            )

            return df
        except FileLoadError:
            raise
        except Exception as e:
            raise FileLoadError(
                f"Failed to load Parquet file: {str(e)}",
                path=self.file_paths[0] if self.file_paths else None
            ) from e

    def load_parquet_many(self) -> List[DataFrame]:
        """
        Load multiple Parquet files into a list of DataFrames.

        Useful for loading partitioned Parquet datasets stored across multiple files.

        Returns:
            List[DataFrame]: List of DataFrames, one per file

        Raises:
            FileLoadError: If no files provided or any file cannot be loaded

        Example:
            >>> # Load partitioned data
            >>> loader = DataLoader(
            ...     "data/year=2023/data.parquet",
            ...     "data/year=2024/data.parquet",
            ...     logger=logger,
            ...     file_handler=file_handler
            ... )
            >>> dfs = loader.load_parquet_many()
            >>> combined = pd.concat(dfs, ignore_index=True)

            >>> # Load monthly partitions
            >>> import glob
            >>> files = glob.glob("data/month=*/part-*.parquet")
            >>> loader = DataLoader(*files, logger=logger, file_handler=file_handler)
            >>> dfs = loader.load_parquet_many()
        """
        if not self.file_paths:
            raise FileLoadError("No file paths provided to DataLoader")

        try:
            paths = self._validate_paths(self.file_paths)

            self.logger.debug(f"Loading {len(paths)} Parquet files")

            dfs = []
            for path in paths:
                df = pd.read_parquet(path, engine="pyarrow")
                dfs.append(df)
                self.logger.debug(
                    f"Loaded: {path} ({len(df)} rows, {len(df.columns)} columns)"
                )

            self.logger.info(
                f"Loaded {len(dfs)} Parquet files (total: "
                f"{sum(len(df) for df in dfs)} rows)"
            )

            return dfs
        except FileLoadError:
            raise
        except Exception as e:
            raise FileLoadError(
                f"Failed to load Parquet files: {str(e)}"
            ) from e


# =========================
# DATA SAVER
# =========================

class DataSaver:
    """
    Save pandas DataFrames to disk in multiple formats.

    This class provides a clean interface for persisting DataFrames to various
    file formats with automatic path validation and format enforcement.

    Supports formats:
    - CSV (comma-separated values)
    - Parquet (uncompressed or gzip-compressed)

    Attributes:
        save_path (Optional[str]): Default save path for all operations
        logger (Logger): Logger instance for debug/info messages
        file_handler (FileHandler): Handler for file operations and validation

    Example:
        >>> from haashi_pkg.utility.utils import Logger, FileHandler
        >>> import logging
        >>> 
        >>> logger = Logger(level=logging.INFO)
        >>> file_handler = FileHandler(logger=logger)
        >>> 
        >>> # With default save path
        >>> saver = DataSaver(save_path="output.csv", logger=logger, file_handler=file_handler)
        >>> saver.save_csv(df)  # Uses default path
        >>> 
        >>> # Override default path
        >>> saver.save_csv(df, path="different.csv")
        >>> 
        >>> # Save as compressed Parquet
        >>> saver = DataSaver(logger=logger, file_handler=file_handler)
        >>> saver.save_parquet_compressed(df, path="data.parquet")
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        logger: Optional[Logger] = None,
        file_handler: Optional[FileHandler] = None
    ) -> None:
        """
        Initialize DataSaver with an optional default save path.

        Args:
            save_path: Optional default path for save operations
            logger: Optional Logger instance. If None, creates default logger
            file_handler: Optional FileHandler instance. If None, creates default
        """
        self.save_path = save_path
        self.logger = logger or Logger(level=logging.DEBUG)
        self.file_handler = file_handler or FileHandler(logger=self.logger)

    # ========================
    # VALIDATE SAVE PATH
    # ========================

    def validate_save_path(
        self,
        path: Optional[str],
        file_type: str
    ) -> str:
        """
        Validate save path and enforce file extension.

        Args:
            path: Path to validate (uses default if None)
            file_type: Required file extension (e.g., '.csv', '.parquet')

        Returns:
            str: Validated absolute path

        Raises:
            FileSaveError: If no path provided or extension doesn't match
        """
        path = path or self.save_path

        if not path:
            raise FileSaveError("No save path provided")

        if not path.endswith(file_type):
            raise FileSaveError(
                f"Save path must end with '{file_type}', got '{path}'",
                path=path
            )

        try:
            return str(self.file_handler.ensure_writable_path(path))
        except Exception as e:
            raise FileSaveError(
                f"Path validation failed: {str(e)}",
                path=path
            ) from e

    # ========================
    # SAVE TO CSV
    # ========================

    def save_csv(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        """
        Save a DataFrame as a CSV file.

        Saves without the index column by default for cleaner output.

        Args:
            df: DataFrame to save
            path: Output file path. Uses default if None. Must end with '.csv'.

        Raises:
            FileSaveError: If path is invalid or save operation fails

        Example:
            >>> saver = DataSaver(logger=logger, file_handler=file_handler)
            >>> saver.save_csv(df, path="output.csv")

            >>> # With default path
            >>> saver = DataSaver(save_path="default.csv", logger=logger, file_handler=file_handler)
            >>> saver.save_csv(df)  # Saves to 'default.csv'

            >>> # Save with index
            >>> path = saver.validate_save_path("data.csv", ".csv")
            >>> df.to_csv(path, index=True)
        """
        try:
            validated_path = self.validate_save_path(path, ".csv")

            self.logger.debug(f"Saving CSV to: {validated_path}")

            df.to_csv(validated_path, index=False)

            self.logger.info(f"Saved CSV: {validated_path} ({len(df)} rows)")
        except FileSaveError:
            raise
        except Exception as e:
            raise FileSaveError(
                f"Failed to save CSV: {str(e)}",
                path=path or self.save_path
            ) from e

    # ========================
    # SAVE TO PARQUET
    # ========================

    def save_parquet_default(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        """
        Save a DataFrame as a Parquet file (uncompressed).

        Parquet is a columnar format that's much more efficient than CSV
        for large datasets and preserves data types perfectly.

        Args:
            df: DataFrame to save
            path: Output file path. Uses default if None. Must end with '.parquet'.

        Raises:
            FileSaveError: If path is invalid or save operation fails

        Example:
            >>> saver = DataSaver(logger=logger, file_handler=file_handler)
            >>> saver.save_parquet_default(df, path="output.parquet")

            >>> # Parquet preserves types
            >>> df['date'] = pd.to_datetime(df['date'])
            >>> saver.save_parquet_default(df, path="typed_data.parquet")
            >>> # When loaded, 'date' will still be datetime type
        """
        try:
            validated_path = self.validate_save_path(path, ".parquet")

            self.logger.debug(f"Saving Parquet to: {validated_path}")

            df.to_parquet(validated_path, index=False)

            self.logger.info(
                f"Saved Parquet: {validated_path} ({len(df)} rows)"
            )
        except FileSaveError:
            raise
        except Exception as e:
            raise FileSaveError(
                f"Failed to save Parquet: {str(e)}",
                path=path or self.save_path
            ) from e

    def save_parquet_compressed(
        self,
        df: DataFrame,
        path: Optional[str] = None,
    ) -> None:
        """
        Save a compressed Parquet file using Gzip.

        This creates smaller files than uncompressed Parquet at the cost of
        slightly slower read/write speeds. Good for archival or when disk
        space is limited.

        Args:
            df: DataFrame to save
            path: Output file path. Uses default if None. Must end with '.parquet'.

        Raises:
            FileSaveError: If path is invalid or save operation fails

        Example:
            >>> saver = DataSaver(logger=logger, file_handler=file_handler)
            >>> saver.save_parquet_compressed(df, path="compressed_data.parquet")

            >>> # Compare file sizes
            >>> saver.save_parquet_default(df, "uncompressed.parquet")
            >>> saver.save_parquet_compressed(df, "compressed.parquet")
            >>> # compressed.parquet will be much smaller
        """
        try:
            validated_path = self.validate_save_path(path, ".parquet")

            self.logger.debug(
                f"Saving compressed Parquet to: {validated_path}"
            )

            df.to_parquet(
                validated_path,
                engine="pyarrow",
                compression="gzip",
                index=False
            )

            self.logger.info(
                f"Saved compressed Parquet: {validated_path} ({len(df)} rows)"
            )
        except FileSaveError:
            raise
        except Exception as e:
            raise FileSaveError(
                f"Failed to save compressed Parquet: {str(e)}",
                path=path or self.save_path
            ) from e


# =========================
# DataFrame Factory
# =========================

class DataFrameFactory:
    """
    Factory for creating pandas DataFrames and Series
    Provides convenience methods without requiring direct pandas import
    """

    @staticmethod
    def create_dataframe(
        data: DataLike,
        columns: ColumnLike = None,
        index: IndexLike = None
    ) -> pd.DataFrame:
        """
        Create pandas DataFrame from various data structures

        Args:
            data: Dictionary of lists, list of dicts, or list of lists
            columns: Optional column names (for list of lists)
            index: Optional index

        Returns:
            pandas DataFrame

        Examples:
            >>> # From dictionary of lists
            >>> df = DataFrameFactory.create_dataframe({
            ...     'name': ['Alice', 'Bob'],
            ...     'age': [25, 30]
            ... })

            >>> # From list of dictionaries
            >>> df = DataFrameFactory.create_dataframe([
            ...     {'name': 'Alice', 'age': 25},
            ...     {'name': 'Bob', 'age': 30}
            ... ])

            >>> # From list of lists
            >>> df = DataFrameFactory.create_dataframe(
            ...     [['Alice', 25], ['Bob', 30]],
            ...     columns=['name', 'age']
            ... )
        """
        return pd.DataFrame(data, columns=columns, index=index)

    @staticmethod
    def create_series(
        data: Union[List[Any], Dict[Any, Any]],
        index: IndexLike = None,
        name: Optional[str] = None
    ) -> pd.Series:
        """
        Create pandas Series from list or dictionary

        Args:
            data: List or dictionary of values
            index: Optional index
            name: Optional series name

        Returns:
            pandas Series

        Examples:
            >>> # From list
            >>> series = DataFrameFactory.create_series([1, 2, 3, 4, 5])

            >>> # From dictionary
            >>> series = DataFrameFactory.create_series({'a': 1, 'b': 2})

            >>> # With name
            >>> series = DataFrameFactory.create_series(
            ...     [1, 2, 3],
            ...     name='values'
            ... )
        """
        return pd.Series(data, index=index, name=name)

    @staticmethod
    def from_records(
        records: List[Dict[Any, Any]],
        columns: ColumnLike = None
    ) -> pd.DataFrame:
        """
        Create DataFrame from list of dictionaries (records)

        Args:
            records: List of dictionaries
            columns: Optional column ordering

        Returns:
            pandas DataFrame

        Example:
            >>> records = [
            ...     {'name': 'Alice', 'age': 25, 'city': 'Lagos'},
            ...     {'name': 'Bob', 'age': 30, 'city': 'Abuja'}
            ... ]
            >>> df = DataFrameFactory.from_records(records)
        """
        df = pd.DataFrame.from_records(records)
        if columns:
            df = pd.DataFrame(df[columns])
        return df

    @staticmethod
    def empty_dataframe(columns: List[str]) -> pd.DataFrame:
        """
        Create empty DataFrame with specified columns

        Args:
            columns: Column names

        Returns:
            Empty pandas DataFrame

        Example:
            >>> df = DataFrameFactory.empty_dataframe(['name', 'age', 'city'])
        """
        return pd.DataFrame(columns=columns)
