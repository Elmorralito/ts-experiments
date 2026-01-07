# Pydantic BaseModel wrappers for forecast models with hyperparameter grid search

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerWrapper:
    """Wrapper for sklearn MinMaxScaler to handle multiple columns"""

    def __init__(self):
        self.scalers = {}
        self.fitted_columns = []  # Track which columns have been fitted

    def fit(self, data: pd.DataFrame, columns: List[str]):
        """Fit scalers for specified columns"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if not isinstance(columns, list) or len(columns) == 0:
            raise ValueError("columns must be a non-empty list")

        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if data[col].isna().all():
                raise ValueError(f"Column '{col}' contains only NaN values")

            # Fit scaler on non-null values
            non_null_data = data[col].dropna().values.reshape(-1, 1)
            if len(non_null_data) == 0:
                raise ValueError(f"Column '{col}' has no valid values to fit")

            self.scalers[col] = MinMaxScaler().fit(non_null_data)
            if col not in self.fitted_columns:
                self.fitted_columns.append(col)

    def fit_transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit scalers and transform specified columns"""
        self.fit(data, columns)
        return self.transform(data, columns)

    def transform(
        self, data: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Transform specified columns (or all fitted columns if None)"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        data = data.copy()  # Work with a copy to avoid modifying original

        # If columns not specified, use all fitted columns
        if columns is None:
            columns = list(self.scalers.keys())

        for col in columns:
            if col not in self.scalers:
                raise ValueError(
                    f"Scaler for column '{col}' has not been fitted. Call fit() first."
                )
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

            # Transform returns 2D array, flatten to 1D for pandas assignment
            transformed = self.scalers[col].transform(data[col].values.reshape(-1, 1))
            data[col] = transformed.ravel()

        return data

    def inverse_transform(
        self, data: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Inverse transform specified columns (or all fitted columns if None)

        Args:
            data: DataFrame containing scaled values to inverse transform
            columns: List of column names to inverse transform. If None, uses all fitted columns.

        Returns:
            DataFrame with inverse transformed values
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        data_ = data.copy()  # Work with a copy to avoid modifying original

        # If columns not specified, use all fitted columns that exist in data
        if columns is None:
            columns = [col for col in self.fitted_columns if col in data.columns]
        else:
            # Filter to only columns that exist in both data and scalers
            columns = [
                col for col in columns if col in data.columns and col in self.scalers
            ]

        if len(columns) == 0:
            raise ValueError(
                "No valid columns found to inverse transform. Ensure columns exist in both data and fitted scalers."
            )

        for col in columns:
            if col not in self.scalers:
                raise ValueError(
                    f"Scaler for column '{col}' has not been fitted. Call fit() first."
                )
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

            # Inverse transform returns 2D array, flatten to 1D for pandas assignment
            # Handle NaN values by keeping them as NaN
            col_values = data[col].values.reshape(-1, 1)
            nan_mask = pd.isna(data[col]).values

            # Inverse transform non-null values
            if not nan_mask.all():
                inverse_transformed = self.scalers[col].inverse_transform(col_values)
                data_[col] = inverse_transformed.ravel()
                # Restore NaN values
                data_.loc[nan_mask, col] = np.nan
            else:
                # All values are NaN, keep as NaN
                data_[col] = np.nan

        return data_

    def get_scaler(self, column: str) -> Optional[MinMaxScaler]:
        """Get the scaler for a specific column"""
        return self.scalers.get(column)

    def is_fitted(self, column: str) -> bool:
        """Check if a scaler has been fitted for a column"""
        return column in self.scalers

    def get_fitted_columns(self) -> List[str]:
        """Get list of all columns that have been fitted"""
        return self.fitted_columns.copy()
