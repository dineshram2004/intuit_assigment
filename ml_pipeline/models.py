"""Model factory for classification and regression algorithms."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from ml_pipeline.config import ModelConfig

CLASSIFICATION_MODELS = {
    "random_forest": RandomForestClassifier,
    "log_reg": LogisticRegression,
}

REGRESSION_MODELS = {
    "random_forest_reg": RandomForestRegressor,
    "ridge": Ridge,
}


def _import_lightgbm_classifier():
    from lightgbm import LGBMClassifier

    return LGBMClassifier


def _import_lightgbm_regressor():
    from lightgbm import LGBMRegressor

    return LGBMRegressor


def build_model(config: ModelConfig, task: str) -> Any:
    """Instantiate a sklearn-compatible estimator from config."""
    algorithm = config.algorithm
    params = dict(config.params)

    if task == "classification":
        if algorithm == "lightgbm":
            return _import_lightgbm_classifier()(**params)
        if algorithm in CLASSIFICATION_MODELS:
            return CLASSIFICATION_MODELS[algorithm](**params)
        raise ValueError(f"Unknown classification algorithm: {algorithm}")

    if task == "regression":
        if algorithm == "lightgbm_reg":
            return _import_lightgbm_regressor()(**params)
        if algorithm in REGRESSION_MODELS:
            return REGRESSION_MODELS[algorithm](**params)
        raise ValueError(f"Unknown regression algorithm: {algorithm}")

    raise ValueError(f"Unknown task: {task}")
