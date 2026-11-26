from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
TABLE_NAMES = ("subscribers", "recharges")


def describe_table(name):
    frame = pd.read_csv(RAW_DIR / f"{name}.csv")
    missing = frame.isna().sum()
    audit = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": missing,
            "missing_percent": (missing / len(frame) * 100).round(2),
        }
    )
    print(f"\n{name}.csv shape: {frame.shape}")
    print(audit.to_string())
    return frame


def main():
    subscribers = describe_table("subscribers")
    recharges = describe_table("recharges")
    existing_ids = set(subscribers["subscriber_id"])

    checks = {
        "subscribers rows": len(subscribers),
        "subscriber exact duplicate rows": int(subscribers.duplicated().sum()),
        "distinct subscriber IDs": int(subscribers["subscriber_id"].nunique()),
        "blank ages": int(subscribers["age"].isna().sum()),
        "ages outside 10-100": int(
            (subscribers["age"].notna() & ~subscribers["age"].between(10, 100)).sum()
        ),
        "blank payment modes": int(subscribers["payment_mode"].isna().sum()),
        "distinct raw city spellings": int(subscribers["city"].nunique()),
        "distinct raw auto-renew spellings": int(subscribers["auto_renew"].nunique()),
        "blank next-cycle recharge values": int(
            subscribers["next_cycle_recharge_inr"].isna().sum()
        ),
        "recharge rows": len(recharges),
        "recharge exact duplicate rows": int(recharges.duplicated().sum()),
        "distinct recharge IDs": int(recharges["recharge_id"].nunique()),
        "non-numeric recharge amounts": int(
            pd.to_numeric(recharges["amount_inr"], errors="coerce").isna().sum()
        ),
        "distinct raw recharge status spellings": int(recharges["status"].nunique()),
        "recharges with unknown subscriber IDs": int(
            (~recharges["subscriber_id"].isin(existing_ids)).sum()
        ),
    }

    print("\nRaw-data checkpoint counts:")
    for label, value in checks.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
