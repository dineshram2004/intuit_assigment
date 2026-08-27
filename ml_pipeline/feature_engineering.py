"""Time-window aggregation feature engineering."""

from __future__ import annotations

from typing import List

import pandas as pd

from ml_pipeline.config import FeatureEngineeringConfig

WINDOW_MAP = {
    "15m": "15min",
    "1h": "1h",
    "24h": "24h",
    "7d": "7D",
    "30d": "30D",
    "90d": "90D",
}

METRIC_FNS = {
    "count": lambda roll: roll.count(),
    "sum": lambda roll: roll.sum(),
    "mean": lambda roll: roll.mean(),
    "min": lambda roll: roll.min(),
    "max": lambda roll: roll.max(),
    "std": lambda roll: roll.std(),
}


def engineered_feature_names(config: FeatureEngineeringConfig) -> List[str]:
    """Return column names that FeatureEngineer.transform will create."""
    names: List[str] = []
    for agg in config.aggregations:
        for col in agg.columns:
            for metric in agg.metrics:
                names.append(f"{col}_{metric}_{agg.window}")
    return names


class FeatureEngineer:
    """Build rolling / windowed aggregation features per entity."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        entity = self.config.entity_key
        ts_col = self.config.timestamp_col
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values([entity, ts_col]).reset_index(drop=True)

        for agg in self.config.aggregations:
            window = WINDOW_MAP.get(agg.window, agg.window)
            for col in agg.columns:
                for metric in agg.metrics:
                    feat_name = f"{col}_{metric}_{agg.window}"
                    if metric not in METRIC_FNS:
                        raise ValueError(f"Unknown aggregation metric: {metric}")

                    result = pd.Series(index=df.index, dtype=float)
                    for _, group in df.groupby(entity, sort=False):
                        indexed = group.set_index(ts_col)
                        roll = indexed[col].rolling(window, min_periods=1)
                        values = METRIC_FNS[metric](roll)
                        # Keep original row positions (do not concat on timestamp index).
                        result.loc[group.index] = values.to_numpy()

                    # Rolling std is NaN for single-point windows; treat as 0.
                    if metric == "std":
                        result = result.fillna(0.0)
                    df[feat_name] = result

        return df
