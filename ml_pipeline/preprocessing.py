"""Build sklearn preprocessing pipelines from config."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from ml_pipeline.config import ColumnGroupConfig, PreprocessingConfig

SCALER_MAP = {
    "standard_scaler": StandardScaler,
    "minmax_scaler": MinMaxScaler,
    "robust_scaler": RobustScaler,
}


class TextTfidf(BaseEstimator, TransformerMixin):
    """Impute missing text then apply TfidfVectorizer (single column)."""

    def __init__(self, fill_value: str = "", tfidf_params: dict | None = None):
        self.fill_value = fill_value
        self.tfidf_params = tfidf_params

    def _to_series(self, X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 2:
            arr = arr[:, 0]
        return pd.Series(arr)

    def fit(self, X, y=None):
        texts = self._to_series(X).fillna(self.fill_value).astype(str)
        params = dict(self.tfidf_params or {})
        if params.get("stop_words") is None:
            params.pop("stop_words", None)
        # YAML loads [1, 2] as a list; TfidfVectorizer requires a tuple.
        if "ngram_range" in params and isinstance(params["ngram_range"], list):
            params["ngram_range"] = tuple(params["ngram_range"])
        self.vectorizer_ = TfidfVectorizer(**params)
        self.vectorizer_.fit(texts)
        return self

    def transform(self, X):
        texts = self._to_series(X).fillna(self.fill_value).astype(str)
        return self.vectorizer_.transform(texts)

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer_.get_feature_names_out()


def _build_imputer(group: ColumnGroupConfig) -> SimpleImputer:
    strategy_map = {
        "mean": "mean",
        "median": "median",
        "most_frequent": "most_frequent",
        "constant": "constant",
    }
    strategy = strategy_map.get(group.imputer, group.imputer)
    fill_value = group.impute_value if strategy == "constant" else None
    return SimpleImputer(strategy=strategy, fill_value=fill_value)


def _build_transformer(group: ColumnGroupConfig) -> Any:
    transformer_type = group.transformer
    params = dict(group.params)

    if transformer_type == "tfidf":
        fill_value = "" if group.impute_value is None else str(group.impute_value)
        return TextTfidf(fill_value=fill_value, tfidf_params=params)

    if transformer_type == "onehot":
        params.setdefault("handle_unknown", "ignore")
        return OneHotEncoder(**params)

    if transformer_type == "ordinal":
        params.setdefault("handle_unknown", "use_encoded_value")
        params.setdefault("unknown_value", -1)
        return OrdinalEncoder(**params)

    if transformer_type in SCALER_MAP:
        return SCALER_MAP[transformer_type](**params)

    raise ValueError(f"Unknown transformer: {transformer_type}")


def _build_group_pipeline(group: ColumnGroupConfig) -> Pipeline:
    if group.transformer == "tfidf":
        return Pipeline(steps=[("tfidf", _build_transformer(group))])

    steps: List[tuple] = [
        ("imputer", _build_imputer(group)),
        ("transformer", _build_transformer(group)),
    ]
    return Pipeline(steps)


def _expand_tfidf_groups(groups: List[ColumnGroupConfig]) -> List[ColumnGroupConfig]:
    """TfidfVectorizer expects one text column; expand multi-column groups."""
    expanded: List[ColumnGroupConfig] = []
    for group in groups:
        if group.transformer == "tfidf" and len(group.columns) > 1:
            for col in group.columns:
                expanded.append(
                    ColumnGroupConfig(
                        group_name=f"{group.group_name}_{col}",
                        columns=[col],
                        imputer=group.imputer,
                        transformer=group.transformer,
                        impute_value=(
                            "" if group.impute_value is None else group.impute_value
                        ),
                        params=dict(group.params),
                    )
                )
        else:
            expanded.append(group)
    return expanded


def build_preprocessor(config: PreprocessingConfig) -> ColumnTransformer:
    """Create a ColumnTransformer from preprocessing config column groups."""
    transformers = []
    for group in _expand_tfidf_groups(config.column_groups):
        transformers.append(
            (group.group_name, _build_group_pipeline(group), group.columns)
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")
