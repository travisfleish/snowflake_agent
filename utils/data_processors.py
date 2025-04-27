"""
Data processors for Snowflake query results.
Provides functions to clean, normalize, format, and transform data from Snowflake queries.
"""

import re
import json
import decimal
import datetime
from typing import Dict, List, Any, Union, Optional, Tuple

import pandas as pd
import numpy as np


class SnowflakeDataProcessor:
    """
    Data processing utilities for Snowflake query results.
    Handles conversion, cleaning, and formatting of query results.
    """

    @staticmethod
    def cursor_to_dict(cursor) -> List[Dict[str, Any]]:
        """
        Convert a Snowflake cursor result to a list of dictionaries.

        Args:
            cursor: Snowflake cursor object with executed query

        Returns:
            List[Dict[str, Any]]: List of row dictionaries with column names as keys
        """
        columns = [col[0] for col in cursor.description]
        results = []

        for row in cursor:
            results.append(dict(zip(columns, row)))

        return results

    @staticmethod
    def cursor_to_dataframe(cursor) -> pd.DataFrame:
        """
        Convert a Snowflake cursor result to a pandas DataFrame.

        Args:
            cursor: Snowflake cursor object with executed query

        Returns:
            pd.DataFrame: DataFrame containing the query results
        """
        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()

        # Convert to pandas DataFrame
        df = pd.DataFrame(results, columns=columns)

        return df

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize column names in a DataFrame.

        Args:
            df: DataFrame with columns to clean

        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # Clean column names
        df.columns = [
            col.lower()  # Convert to lowercase
            .strip()  # Remove leading/trailing whitespace
            .replace(' ', '_')  # Replace spaces with underscores
            .replace('-', '_')  # Replace hyphens with underscores
            .replace('/', '_')  # Replace slashes with underscores
            .replace('\\', '_')  # Replace backslashes with underscores
            .replace('.', '_')  # Replace periods with underscores
            .replace('(', '')  # Remove left parentheses
            .replace(')', '')  # Remove right parentheses
            .replace('?', '')  # Remove question marks
            .replace('!', '')  # Remove exclamation points
            .replace(':', '')  # Remove colons
            .replace(';', '')  # Remove semicolons
            .replace(',', '')  # Remove commas
            .replace('#', 'num')  # Replace hash with 'num'
            .replace('$', 'usd')  # Replace dollar sign with 'usd'
            .replace('%', 'pct')  # Replace percent sign with 'pct'
            for col in df.columns
        ]

        # Handle duplicates by adding suffix
        if len(df.columns) != len(set(df.columns)):
            seen = {}
            new_columns = []

            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_columns.append(col)

            df.columns = new_columns

        return df

    @staticmethod
    def convert_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame types to appropriate Python types.

        Args:
            df: DataFrame with columns to convert

        Returns:
            pd.DataFrame: DataFrame with converted types
        """
        df = df.copy()

        # Detect and convert date columns
        date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
        datetime_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})')

        for col in df.columns:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue

            # Get a non-null sample value
            sample = df[col].dropna().iloc[0]

            if isinstance(sample, str):
                # Try to convert string columns to datetime if they match patterns
                if datetime_pattern.match(str(sample)):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
                elif date_pattern.match(str(sample)):
                    try:
                        df[col] = pd.to_datetime(df[col]).dt.date
                    except:
                        pass

        return df

    @staticmethod
    def handle_null_values(df: pd.DataFrame, strategy: str = 'none') -> pd.DataFrame:
        """
        Handle null values in DataFrame based on specified strategy.

        Args:
            df: DataFrame with null values to handle
            strategy: Strategy for handling nulls ('none', 'drop_rows', 'drop_columns',
                     'fill_mean', 'fill_median', 'fill_mode', 'fill_zero', or 'fill_empty_string')

        Returns:
            pd.DataFrame: DataFrame with null values handled
        """
        df = df.copy()

        if strategy == 'none':
            return df
        elif strategy == 'drop_rows':
            return df.dropna()
        elif strategy == 'drop_columns':
            return df.dropna(axis=1)
        elif strategy == 'fill_mean':
            # Fill numeric columns with mean, non-numeric with empty string
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna('')
        elif strategy == 'fill_median':
            # Fill numeric columns with median, non-numeric with empty string
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna('')
        elif strategy == 'fill_mode':
            # Fill with mode (most common value)
            for col in df.columns:
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value[0])
        elif strategy == 'fill_zero':
            # Fill numeric columns with 0, non-numeric with empty string
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('')
        elif strategy == 'fill_empty_string':
            # Fill all nulls with empty string
            df = df.fillna('')

        return df

    @staticmethod
    def format_numbers(df: pd.DataFrame,
                       decimal_places: Optional[Dict[str, int]] = None,
                       thousands_separator: bool = True,
                       percentage_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Format numeric columns with specified decimal places and separators.

        Args:
            df: DataFrame with numeric columns to format
            decimal_places: Dictionary mapping column names to number of decimal places
            thousands_separator: Whether to include thousands separators
            percentage_cols: List of column names to format as percentages

        Returns:
            pd.DataFrame: DataFrame with formatted numeric columns
        """
        df = df.copy()

        # Default decimal places
        if decimal_places is None:
            decimal_places = {}

        # Default percentage columns
        if percentage_cols is None:
            percentage_cols = []

        # Format specified columns
        for col in df.columns:
            # Only format numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Get decimal places for this column (default to 2)
            places = decimal_places.get(col, 2)

            # Handle percentage columns
            if col in percentage_cols:
                # Format as percentage with specified decimal places
                df[col] = df[col].apply(
                    lambda x: f"{x:.{places}f}%" if pd.notna(x) else ""
                )
            else:
                # Format as number with specified decimal places and separators
                if thousands_separator:
                    df[col] = df[col].apply(
                        lambda x: f"{x:,.{places}f}" if pd.notna(x) else ""
                    )
                else:
                    df[col] = df[col].apply(
                        lambda x: f"{x:.{places}f}" if pd.notna(x) else ""
                    )

        return df

    @staticmethod
    def format_dates(df: pd.DataFrame,
                     date_format: str = '%Y-%m-%d',
                     datetime_format: str = '%Y-%m-%d %H:%M:%S',
                     date_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Format date and datetime columns with specified formats.

        Args:
            df: DataFrame with date columns to format
            date_format: Format string for date columns
            datetime_format: Format string for datetime columns
            date_cols: Optional list of column names to format as dates

        Returns:
            pd.DataFrame: DataFrame with formatted date columns
        """
        df = df.copy()

        # If specific columns are provided, only format those
        if date_cols:
            columns_to_check = [col for col in date_cols if col in df.columns]
        else:
            columns_to_check = df.columns

        for col in columns_to_check:
            # Check if column contains datetime objects
            if pd.api.types.is_datetime64_dtype(df[col]):
                # Check if time components are all zeros (just dates)
                if (df[col].dt.time == datetime.time(0, 0, 0)).all():
                    df[col] = df[col].dt.strftime(date_format)
                else:
                    df[col] = df[col].dt.strftime(datetime_format)
            # Check if column contains date objects
            elif df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    temp = pd.to_datetime(df[col])
                    # Check if conversion worked and if time components are all zeros
                    if (temp.dt.time == datetime.time(0, 0, 0)).all():
                        df[col] = temp.dt.strftime(date_format)
                    else:
                        df[col] = temp.dt.strftime(datetime_format)
                except:
                    pass

        return df

    @staticmethod
    def normalize_values(df: pd.DataFrame, method: str = 'minmax',
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize numeric columns using specified method.

        Args:
            df: DataFrame with columns to normalize
            method: Normalization method ('minmax', 'zscore', or 'robust')
            columns: List of column names to normalize (if None, normalize all numeric columns)

        Returns:
            pd.DataFrame: DataFrame with normalized columns
        """
        df = df.copy()

        # Determine which columns to normalize
        if columns is None:
            # Find all numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Filter out non-existent columns
            columns = [col for col in columns if col in df.columns]
            # Filter out non-numeric columns
            columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]

        # Apply normalization to each column
        for col in columns:
            if method == 'minmax':
                # Min-Max normalization (scale to 0-1 range)
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:  # Avoid division by zero
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:  # Avoid division by zero
                    df[col] = (df[col] - mean) / std
            elif method == 'robust':
                # Robust scaling (using median and IQR)
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:  # Avoid division by zero
                    df[col] = (df[col] - median) / iqr

        return df

    @staticmethod
    def to_json(df: pd.DataFrame, orient: str = 'records') -> str:
        """
        Convert DataFrame to JSON string with custom handlers for special types.

        Args:
            df: DataFrame to convert to JSON
            orient: JSON orientation ('records', 'split', 'index', 'columns', or 'values')

        Returns:
            str: JSON string representation of the DataFrame
        """

        # Define custom JSON encoder to handle special data types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                elif isinstance(obj, decimal.Decimal):
                    return float(obj)
                elif pd.isna(obj):
                    return None
                return super().default(obj)

        # Convert to JSON with custom encoder
        return json.dumps(
            json.loads(df.to_json(orient=orient, date_format='iso')),
            cls=CustomJSONEncoder,
            indent=2
        )

    @staticmethod
    def to_csv(df: pd.DataFrame, index: bool = False) -> str:
        """
        Convert DataFrame to CSV string with proper handling of special types.

        Args:
            df: DataFrame to convert to CSV
            index: Whether to include index column

        Returns:
            str: CSV string representation of the DataFrame
        """
        return df.to_csv(index=index)

    @staticmethod
    def to_html_table(df: pd.DataFrame, index: bool = False,
                      classes: str = 'table table-striped',
                      max_rows: Optional[int] = None) -> str:
        """
        Convert DataFrame to HTML table with styling.

        Args:
            df: DataFrame to convert to HTML table
            index: Whether to include index column
            classes: CSS classes to apply to table
            max_rows: Maximum number of rows to include (None for all rows)

        Returns:
            str: HTML string representing the DataFrame as a table
        """
        # Limit rows if specified
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)

        # Convert to HTML
        return df.to_html(index=index, classes=classes, border=0)

    @staticmethod
    def detect_outliers(df: pd.DataFrame,
                        method: str = 'iqr',
                        columns: Optional[List[str]] = None,
                        threshold: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers in numeric columns using specified method.

        Args:
            df: DataFrame to check for outliers
            method: Detection method ('iqr' or 'zscore')
            columns: List of column names to check (if None, check all numeric columns)
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)

        Returns:
            Dict[str, List[int]]: Dictionary mapping column names to lists of outlier indices
        """
        # Determine which columns to check
        if columns is None:
            # Find all numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Filter out non-existent columns
            columns = [col for col in columns if col in df.columns]
            # Filter out non-numeric columns
            columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]

        outliers = {}

        # Check each column for outliers
        for col in columns:
            if method == 'iqr':
                # IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)

                # Find outlier indices
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices = df.index[outlier_mask].tolist()

                if outlier_indices:
                    outliers[col] = outlier_indices

            elif method == 'zscore':
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()

                if std > 0:  # Avoid division by zero
                    # Calculate z-scores
                    z_scores = (df[col] - mean) / std

                    # Find outlier indices
                    outlier_mask = abs(z_scores) > threshold
                    outlier_indices = df.index[outlier_mask].tolist()

                    if outlier_indices:
                        outliers[col] = outlier_indices

        return outliers

    @staticmethod
    def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of DataFrame contents.

        Args:
            df: DataFrame to summarize

        Returns:
            Dict[str, Any]: Dictionary containing summary information
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': {col: int(df[col].isna().sum()) for col in df.columns},
            'missing_percentage': {col: float(df[col].isna().mean() * 100) for col in df.columns},
            'numeric_summary': {},
            'categorical_summary': {},
            'date_summary': {}
        }

        # Summarize numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            summary['numeric_summary'] = {
                col: {
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None
                }
                for col in numeric_cols
            }

        # Summarize categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            summary['categorical_summary'] = {
                col: {
                    'unique_count': int(df[col].nunique()),
                    'unique_values': df[col].unique().tolist()[:10],  # Top 10 unique values
                    'top': df[col].value_counts().index[0] if not df[col].isna().all()
                                                              and len(df[col].value_counts()) > 0
                    else None,
                    'top_count': int(df[col].value_counts().iloc[0]) if not df[col].isna().all()
                                                                        and len(df[col].value_counts()) > 0
                    else 0
                }
                for col in categorical_cols
            }

        # Summarize date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            summary['date_summary'] = {
                col: {
                    'min': df[col].min().isoformat() if not df[col].isna().all() else None,
                    'max': df[col].max().isoformat() if not df[col].isna().all() else None,
                    'range_days': int((df[col].max() - df[col].min()).days)
                    if not df[col].isna().all() else None
                }
                for col in date_cols
            }

        return summary


# Helper functions for direct usage
def clean_snowflake_results(cursor) -> pd.DataFrame:
    """
    Process Snowflake cursor results into a clean, usable DataFrame.

    Args:
        cursor: Snowflake cursor object with executed query

    Returns:
        pd.DataFrame: Clean DataFrame with processed data
    """
    # Convert cursor to DataFrame
    df = SnowflakeDataProcessor.cursor_to_dataframe(cursor)

    # Clean and normalize
    df = SnowflakeDataProcessor.clean_column_names(df)
    df = SnowflakeDataProcessor.convert_types(df)

    return df


def format_for_presentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame for presentation or reporting.

    Args:
        df: DataFrame to format

    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    # Handle nulls
    df = SnowflakeDataProcessor.handle_null_values(df, strategy='fill_empty_string')

    # Format dates and numbers
    df = SnowflakeDataProcessor.format_dates(df)
    df = SnowflakeDataProcessor.format_numbers(df, thousands_separator=True)

    return df


def df_to_formatted_output(df: pd.DataFrame, output_format: str = 'html') -> str:
    """
    Convert DataFrame to formatted output string.

    Args:
        df: DataFrame to convert
        output_format: Output format ('html', 'json', 'csv')

    Returns:
        str: Formatted output string
    """
    if output_format.lower() == 'html':
        return SnowflakeDataProcessor.to_html_table(df)
    elif output_format.lower() == 'json':
        return SnowflakeDataProcessor.to_json(df)
    elif output_format.lower() == 'csv':
        return SnowflakeDataProcessor.to_csv(df)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


# Export for direct imports
__all__ = [
    'SnowflakeDataProcessor',
    'clean_snowflake_results',
    'format_for_presentation',
    'df_to_formatted_output'
]