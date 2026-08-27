"""End-to-end ML pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

from ml_pipeline.config import ColumnGroupConfig, PipelineConfig, load_config
from ml_pipeline.data_splitter import DataSplitter
from ml_pipeline.evaluation import Evaluator
from ml_pipeline.feature_engineering import FeatureEngineer, engineered_feature_names
from ml_pipeline.models import build_model
from ml_pipeline.preprocessing import build_preprocessor
from ml_pipeline.serialization import save_artifacts


class MLPipeline:
    """Orchestrate data split, preprocessing, training, evaluation, and saving."""

    def __init__(self, config: PipelineConfig | str | Path):
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
            self.config_path = Path(config)
        else:
            self.config = config
            self.config_path = None

        self.splitter = DataSplitter(self.config.data_split)
        self.evaluator = Evaluator(
            task=self.config.experiment.task,
            optimize_metric=self.config.evaluation.optimize_metric,
            track_metrics=self.config.evaluation.track_metrics,
        )
        self.pipeline: Optional[SklearnPipeline] = None
        self.metrics: Dict[str, float] = {}

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.feature_engineering is None:
            return df
        engineer = FeatureEngineer(self.config.feature_engineering)
        return engineer.transform(df)

    def _include_engineered_columns(self, df: pd.DataFrame) -> None:
        """Auto-add FE columns to preprocessing if the YAML omitted them."""
        if self.config.feature_engineering is None:
            return
        fe_cols = engineered_feature_names(self.config.feature_engineering)
        existing = {
            col
            for group in self.config.preprocessing.column_groups
            for col in group.columns
        }
        missing = [c for c in fe_cols if c in df.columns and c not in existing]
        if not missing:
            return
        self.config.preprocessing.column_groups.append(
            ColumnGroupConfig(
                group_name="engineered_features",
                columns=missing,
                imputer="median",
                transformer="standard_scaler",
            )
        )

    def _build_sklearn_pipeline(self) -> SklearnPipeline:
        preprocessor = build_preprocessor(self.config.preprocessing)
        model = build_model(self.config.model, self.config.experiment.task)
        return SklearnPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def _get_probabilities(
        self, pipeline: SklearnPipeline, X: pd.DataFrame
    ) -> Optional[np.ndarray]:
        model = pipeline.named_steps["model"]
        if not hasattr(model, "predict_proba"):
            return None
        proba = pipeline.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def _get_feature_importances(
        self, pipeline: SklearnPipeline
    ) -> Optional[np.ndarray]:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        return None

    def _get_feature_names(self, pipeline: SklearnPipeline) -> List[str]:
        preprocessor = pipeline.named_steps["preprocessor"]
        try:
            return list(preprocessor.get_feature_names_out())
        except AttributeError:
            return []

    def fit(
        self,
        df: pd.DataFrame,
        evaluate_on: str = "test",
    ) -> Dict[str, float]:
        """Run full pipeline: split, preprocess, fit, evaluate, save artifacts."""
        df = self._apply_feature_engineering(df)
        self._include_engineered_columns(df)
        train, val, test = self.splitter.split(df)

        X_train, y_train = self.splitter.extract_xy(train)
        X_val, y_val = self.splitter.extract_xy(val)
        X_test, y_test = self.splitter.extract_xy(test)

        self.pipeline = self._build_sklearn_pipeline()
        self.pipeline.fit(X_train, y_train)

        eval_sets: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
        X_eval, y_eval = eval_sets.get(evaluate_on, eval_sets["test"])

        y_pred = self.pipeline.predict(X_eval)
        y_prob = self._get_probabilities(self.pipeline, X_eval)
        self.metrics = self.evaluator.compute_metrics(y_eval, y_pred, y_prob)

        artifacts = self.config.evaluation.artifacts
        plot_names = artifacts.get("plots", [])
        if plot_names:
            self.evaluator.generate_all_plots(
                plot_names,
                y_eval,
                y_pred,
                y_prob,
                self._get_feature_names(self.pipeline),
                self._get_feature_importances(self.pipeline),
            )

        predictions_df = pd.DataFrame(
            {
                "y_true": y_eval,
                "y_pred": y_pred,
            }
        )
        if y_prob is not None:
            if y_prob.ndim == 1:
                predictions_df["y_prob"] = y_prob
            else:
                for i in range(y_prob.shape[1]):
                    predictions_df[f"y_prob_{i}"] = y_prob[:, i]

        model_path = artifacts.get("save_model_path", "./outputs/model_pipeline.pkl")
        predictions_path = artifacts.get(
            "save_predictions_path", "./outputs/inference_results.csv"
        )
        save_artifacts(
            self.pipeline,
            predictions_df,
            self.metrics,
            self.config_path or "inline_config",
            model_path,
            predictions_path,
        )

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        df = self._apply_feature_engineering(df)
        target = self.config.data_split.target_col
        X = df.drop(columns=[target], errors="ignore")
        return self.pipeline.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        df = self._apply_feature_engineering(df)
        target = self.config.data_split.target_col
        X = df.drop(columns=[target], errors="ignore")
        return self._get_probabilities(self.pipeline, X)
