"""Shared, leakage-aware feature construction for churn modeling and rescoring."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
SNAPSHOT_DATE = pd.Timestamp("2026-03-31")

NUMERIC_FEATURES = [
    "tenure_months",
    "age",
    "add_on_count",
    "monthly_recharge_inr",
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
    "support_calls_90d",
    "n_success_6m",
    "n_failed_6m",
    "recency_days",
]
CATEGORICAL_FEATURES = ["city", "plan_id", "payment_mode", "auto_renew"]


def build_model_table(
    subscribers: pd.DataFrame | None = None,
    recharges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the same cutoff-safe feature table from files or supplied rows."""
    loaded_full_table = subscribers is None
    if loaded_full_table:
        subscribers = pd.read_csv(CLEAN_DIR / "subscribers.csv")
    else:
        subscribers = subscribers.copy()
    if recharges is None:
        recharges = pd.read_csv(CLEAN_DIR / "recharges.csv")
    else:
        recharges = recharges.copy()

    recharges["recharge_date"] = pd.to_datetime(recharges["recharge_date"])
    observed_recharges = recharges.loc[
        recharges["recharge_date"] <= SNAPSHOT_DATE
    ].copy()

    success = observed_recharges.loc[observed_recharges["status"] == "success"]
    failed = observed_recharges.loc[observed_recharges["status"] == "failed"]
    success_counts = success.groupby("subscriber_id").size().to_dict()
    failed_counts = failed.groupby("subscriber_id").size().to_dict()
    last_success = (
        success.groupby("subscriber_id")["recharge_date"].max().to_dict()
    )

    subscribers["n_success_6m"] = (
        subscribers["subscriber_id"].map(success_counts).fillna(0).astype(int)
    )
    subscribers["n_failed_6m"] = (
        subscribers["subscriber_id"].map(failed_counts).fillna(0).astype(int)
    )
    last_dates = pd.to_datetime(subscribers["subscriber_id"].map(last_success))
    subscribers["recency_days"] = (SNAPSHOT_DATE - last_dates).dt.days

    if loaded_full_table and len(subscribers) != 2_400:
        raise AssertionError("Expected 2400 rows in the modelling table.")
    if len(subscribers) == 2_400:
        if subscribers["recency_days"].isna().sum() != 76:
            raise AssertionError("Expected 76 missing recency_days values.")
        if subscribers["churned"].value_counts().to_dict() != {0: 1_886, 1: 514}:
            raise AssertionError("The Part A churn class balance changed unexpectedly.")
    return subscribers
