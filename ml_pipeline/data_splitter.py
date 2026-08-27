"""Train / validation / test splitting strategies."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ml_pipeline.config import DataSplitConfig


class DataSplitter:
    """Split data into train, validation, and test sets.

    ``test_size`` and ``val_size`` are both fractions of the *full* dataset.
    Example: test_size=0.15, val_size=0.15 → ~70% train / 15% val / 15% test.
    """

    def __init__(self, config: DataSplitConfig):
        self.config = config

    def _relative_val_size(self) -> float:
        """Convert full-dataset val_size to a fraction of the train+val pool."""
        test_size = self.config.test_size
        val_size = self.config.val_size
        remaining = 1.0 - test_size
        if remaining <= 0:
            raise ValueError("test_size must be < 1.0")
        relative = val_size / remaining
        if not 0.0 < relative < 1.0:
            raise ValueError(
                f"val_size={val_size} with test_size={test_size} leaves no room for train"
            )
        return relative

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        target = self.config.target_col
        method = self.config.method
        test_size = self.config.test_size
        val_size = self.config.val_size
        relative_val = self._relative_val_size()

        if method == "random":
            train_val, test = train_test_split(
                df, test_size=test_size, random_state=42
            )
            train, val = train_test_split(
                train_val, test_size=relative_val, random_state=42
            )
        elif method == "stratified":
            train_val, test = train_test_split(
                df,
                test_size=test_size,
                stratify=df[target],
                random_state=42,
            )
            train, val = train_test_split(
                train_val,
                test_size=relative_val,
                stratify=train_val[target],
                random_state=42,
            )
        elif method == "time_based":
            split_var = self.config.split_var
            if not split_var:
                raise ValueError("split_var is required for time_based splitting")
            sorted_df = df.sort_values(split_var).reset_index(drop=True)
            n = len(sorted_df)
            test_start = int(n * (1 - test_size))
            val_start = int(n * (1 - test_size - val_size))
            train = sorted_df.iloc[:val_start]
            val = sorted_df.iloc[val_start:test_start]
            test = sorted_df.iloc[test_start:]
        elif method == "group_based":
            split_var = self.config.split_var
            if not split_var:
                raise ValueError("split_var is required for group_based splitting")
            groups = df[split_var]
            gss_test = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=42
            )
            train_val_idx, test_idx = next(gss_test.split(df, groups=groups))
            train_val_df = df.iloc[train_val_idx]
            test = df.iloc[test_idx]
            gss_val = GroupShuffleSplit(
                n_splits=1, test_size=relative_val, random_state=42
            )
            train_idx, val_idx = next(
                gss_val.split(train_val_df, groups=train_val_df[split_var])
            )
            train = train_val_df.iloc[train_idx]
            val = train_val_df.iloc[val_idx]
        else:
            raise ValueError(f"Unknown split method: {method}")

        return train, val, test

    def extract_xy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        target = self.config.target_col
        X = df.drop(columns=[target])
        y = df[target].values
        return X, y
