"""Save and load pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


def save_pipeline(pipeline: Any, path: str | Path) -> Path:
    """Serialize the full sklearn pipeline to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: str | Path) -> Any:
    """Load a serialized sklearn pipeline."""
    return joblib.load(path)


def save_predictions(
    predictions: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save inference results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path


def save_artifacts(
    pipeline: Any,
    predictions: pd.DataFrame,
    metrics: Dict[str, float],
    config_path: str | Path,
    model_path: str | Path,
    predictions_path: str | Path,
    metadata_path: str | Path = "./outputs/run_metadata.pkl",
) -> Dict[str, Path]:
    """Save model, predictions, and run metadata."""
    saved: Dict[str, Path] = {}
    saved["model"] = save_pipeline(pipeline, model_path)
    saved["predictions"] = save_predictions(predictions, predictions_path)

    meta_path = Path(metadata_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metrics": metrics,
            "config_path": str(config_path),
            "model_path": str(saved["model"]),
            "predictions_path": str(saved["predictions"]),
        },
        meta_path,
    )
    saved["metadata"] = meta_path
    return saved
