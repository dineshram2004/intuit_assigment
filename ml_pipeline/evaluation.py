"""Metrics, evaluation, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

def _multiclass_kw(y: np.ndarray) -> dict:
    n_classes = len(np.unique(y))
    if n_classes > 2:
        return {"average": "weighted"}
    return {}


def _roc_auc(y: np.ndarray, p: np.ndarray, prob: Optional[np.ndarray] = None) -> float:
    scores = prob if prob is not None else p
    n_classes = len(np.unique(y))
    if n_classes > 2:
        return roc_auc_score(y, scores, multi_class="ovr", average="weighted")
    return roc_auc_score(y, scores)


CLASSIFICATION_METRICS = {
    "f1": lambda y, p, prob=None: f1_score(y, p, zero_division=0, **_multiclass_kw(y)),
    "roc_auc": _roc_auc,
    "precision": lambda y, p, prob=None: precision_score(
        y, p, zero_division=0, **_multiclass_kw(y)
    ),
    "recall": lambda y, p, prob=None: recall_score(
        y, p, zero_division=0, **_multiclass_kw(y)
    ),
    "accuracy": lambda y, p, prob=None: accuracy_score(y, p),
}

REGRESSION_METRICS = {
    "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
    "mae": lambda y, p: mean_absolute_error(y, p),
    "r2": lambda y, p: r2_score(y, p),
    "mape": lambda y, p: np.mean(np.abs((y - p) / np.where(y == 0, 1, y))) * 100,
}


class Evaluator:
    """Compute metrics and generate evaluation plots."""

    def __init__(self, task: str, optimize_metric: str, track_metrics: List[str]):
        self.task = task
        self.optimize_metric = optimize_metric
        self.track_metrics = track_metrics
        self.metric_fns = (
            CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
        )

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        all_metrics = set(self.track_metrics) | {self.optimize_metric}

        for name in all_metrics:
            if name not in self.metric_fns:
                continue
            if self.task == "classification":
                metrics[name] = self.metric_fns[name](y_true, y_pred, y_prob)
            else:
                metrics[name] = self.metric_fns[name](y_true, y_pred)

        return metrics

    def plot(
        self,
        plot_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None,
        output_dir: Path = Path("./outputs/plots"),
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_name == "roc_curve" and y_prob is not None:
            y_prob_arr = np.asarray(y_prob)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            if y_prob_arr.ndim == 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob_arr)
                ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
            else:
                classes = np.unique(y_true)
                for i, cls in enumerate(classes):
                    if i >= y_prob_arr.shape[1]:
                        break
                    fpr, tpr, _ = roc_curve((y_true == cls).astype(int), y_prob_arr[:, i])
                    ax.plot(fpr, tpr, label=f"class {cls} (AUC={auc(fpr, tpr):.3f})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()

        elif plot_name == "precision_recall_curve" and y_prob is not None:
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 1:
                prec, rec, _ = precision_recall_curve(y_true, y_prob_arr)
                ax.plot(rec, prec)
            else:
                classes = np.unique(y_true)
                for i, cls in enumerate(classes):
                    if i >= y_prob_arr.shape[1]:
                        break
                    prec, rec, _ = precision_recall_curve(
                        (y_true == cls).astype(int), y_prob_arr[:, i]
                    )
                    ax.plot(rec, prec, label=f"class {cls}")
                ax.legend()
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")

        elif plot_name == "confusion_matrix":
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        elif plot_name == "feature_importance" and feature_importances is not None:
            indices = np.argsort(feature_importances)[::-1][:20]
            names = (
                [feature_names[i] for i in indices]
                if feature_names
                else [str(i) for i in indices]
            )
            ax.barh(names, feature_importances[indices])
            ax.set_title("Feature Importance (top 20)")
            ax.invert_yaxis()

        elif plot_name == "residuals_plot":
            residuals = y_true - y_pred
            ax.scatter(y_pred, residuals, alpha=0.5)
            ax.axhline(0, color="k", linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals Plot")

        else:
            plt.close(fig)
            raise ValueError(f"Cannot generate plot: {plot_name}")

        path = output_dir / f"{plot_name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def generate_all_plots(
        self,
        plot_names: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None,
        output_dir: Path = Path("./outputs/plots"),
    ) -> List[Path]:
        paths = []
        for name in plot_names:
            try:
                path = self.plot(
                    name,
                    y_true,
                    y_pred,
                    y_prob,
                    feature_names,
                    feature_importances,
                    output_dir,
                )
                paths.append(path)
            except (ValueError, TypeError):
                continue
        return paths
