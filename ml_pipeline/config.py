"""Load and validate YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    task: str


@dataclass
class DataSplitConfig:
    method: str
    target_col: str
    split_var: Optional[str] = None
    test_size: float = 0.15
    val_size: float = 0.15


@dataclass
class AggregationConfig:
    window: str
    metrics: List[str]
    columns: List[str]


@dataclass
class FeatureEngineeringConfig:
    entity_key: str
    timestamp_col: str
    aggregations: List[AggregationConfig] = field(default_factory=list)


@dataclass
class ColumnGroupConfig:
    group_name: str
    columns: List[str]
    imputer: str
    transformer: str
    impute_value: Optional[Any] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    default_imputation: Dict[str, str]
    column_groups: List[ColumnGroupConfig]


@dataclass
class ModelConfig:
    algorithm: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    optimize_metric: str
    track_metrics: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig
    data_split: DataSplitConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    feature_engineering: Optional[FeatureEngineeringConfig] = None


def _parse_aggregation(raw: Dict[str, Any]) -> AggregationConfig:
    return AggregationConfig(
        window=raw["window"],
        metrics=raw["metrics"],
        columns=raw["columns"],
    )


def _parse_column_group(raw: Dict[str, Any]) -> ColumnGroupConfig:
    return ColumnGroupConfig(
        group_name=raw["group_name"],
        columns=raw["columns"],
        imputer=raw.get("imputer", "median"),
        transformer=raw["transformer"],
        impute_value=raw.get("impute_value"),
        params=raw.get("params", {}),
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    feature_engineering = None
    if "feature_engineering" in raw:
        fe_raw = raw["feature_engineering"]
        feature_engineering = FeatureEngineeringConfig(
            entity_key=fe_raw["entity_key"],
            timestamp_col=fe_raw["timestamp_col"],
            aggregations=[_parse_aggregation(a) for a in fe_raw.get("aggregations", [])],
        )

    defaults = raw["preprocessing"].get("default_imputation", {})
    column_groups_raw = []
    for g in raw["preprocessing"]["column_groups"]:
        group = dict(g)
        if "imputer" not in group:
            transformer = group.get("transformer", "")
            if transformer == "tfidf":
                group["imputer"] = "constant"
                group.setdefault("impute_value", defaults.get("text", ""))
            elif transformer in ("onehot", "ordinal"):
                group["imputer"] = defaults.get("categorical", "most_frequent")
            else:
                group["imputer"] = defaults.get("numeric", "median")
        column_groups_raw.append(group)

    return PipelineConfig(
        experiment=ExperimentConfig(task=raw["experiment"]["task"]),
        data_split=DataSplitConfig(**raw["data_split"]),
        preprocessing=PreprocessingConfig(
            default_imputation=defaults,
            column_groups=[_parse_column_group(g) for g in column_groups_raw],
        ),
        model=ModelConfig(
            algorithm=raw["model"]["algorithm"],
            params=raw["model"].get("params", {}),
        ),
        evaluation=EvaluationConfig(
            optimize_metric=raw["evaluation"]["optimize_metric"],
            track_metrics=raw["evaluation"].get("track_metrics", []),
            artifacts=raw["evaluation"].get("artifacts", {}),
        ),
        feature_engineering=feature_engineering,
    )
