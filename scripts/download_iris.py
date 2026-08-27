"""Download Iris dataset and save as CSV."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

OUTPUT = Path("data/iris.csv")


def main() -> None:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(df.head())


if __name__ == "__main__":
    main()
