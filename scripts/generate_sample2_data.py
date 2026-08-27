"""Generate synthetic CSV matching configs/sample-2.yaml schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path("data/sample-2.csv")
RANDOM_STATE = 42

PRODUCT_TITLES = [
    "wireless earbuds",
    "laptop stand",
    "coffee maker",
    "yoga mat",
    "desk lamp",
    "running shoes",
    "bluetooth speaker",
    "phone case",
    "water bottle",
    "backpack",
]

REVIEW_SNIPPETS = [
    "great product fast shipping",
    "not worth the price disappointed",
    "excellent quality would buy again",
    "arrived damaged poor packaging",
    "works as expected average experience",
    "amazing value highly recommend",
    "stopped working after a week",
    "perfect fit very comfortable",
]

COUNTRIES = ["US", "UK", "CA", "DE", "FR", "IN", "AU"]
TIERS = ["free", "basic", "premium", "enterprise"]


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n_accounts = 80
    rows_per_account = rng.integers(25, 45, size=n_accounts)
    n_rows = int(rows_per_account.sum())

    account_ids = np.repeat(np.arange(n_accounts), rows_per_account)
    base_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 60, size=n_accounts), unit="D"
    )
    account_starts = np.repeat(base_dates.values, rows_per_account)
    offsets = rng.integers(0, 24 * 60 * 60, size=n_rows)
    date_col = pd.to_datetime(account_starts) + pd.to_timedelta(offsets, unit="s")

    transaction_amount = rng.lognormal(mean=3.5, sigma=0.8, size=n_rows).round(2)
    age = rng.integers(18, 70, size=n_rows)
    account_age_days = rng.integers(30, 2000, size=n_rows)
    annual_income = rng.lognormal(mean=10.8, sigma=0.35, size=n_rows).round(2)
    country = rng.choice(COUNTRIES, size=n_rows)
    subscription_tier = rng.choice(TIERS, size=n_rows, p=[0.35, 0.3, 0.25, 0.1])
    product_title = rng.choice(PRODUCT_TITLES, size=n_rows)
    review_body = rng.choice(REVIEW_SNIPPETS, size=n_rows)

    score = (
        0.35 * (transaction_amount > np.median(transaction_amount))
        + 0.2 * (subscription_tier == "premium")
        + 0.15 * (subscription_tier == "enterprise")
        + 0.1 * (annual_income > np.median(annual_income))
        + 0.1 * (account_age_days > 365)
        + rng.normal(0, 0.25, size=n_rows)
    )
    target = (score > np.median(score)).astype(int)

    df = pd.DataFrame(
        {
            "account_id": account_ids,
            "date_col": date_col,
            "target": target,
            "transaction_amount": transaction_amount,
            "product_title": product_title,
            "review_body": review_body,
            "age": age,
            "account_age_days": account_age_days,
            "annual_income": annual_income,
            "country": country,
            "subscription_tier": subscription_tier,
        }
    )
    df = df.sort_values(["account_id", "date_col"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(df.head())
    print(f"Target distribution:\n{df['target'].value_counts()}")


if __name__ == "__main__":
    main()
