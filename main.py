"""CLI entry point for running experiments from YAML config."""

import argparse
from pathlib import Path

import pandas as pd

from ml_pipeline import MLPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML pipeline from YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sample-2.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input CSV/Parquet data file",
    )
    parser.add_argument(
        "--evaluate-on",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to use for evaluation and plots",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    pipeline = MLPipeline(args.config)
    metrics = pipeline.fit(df, evaluate_on=args.evaluate_on)

    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
