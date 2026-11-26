from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
DATABASE_FILE = PROJECT_DIR / "data" / "tataplay.db"
QUERIES_FILE = PROJECT_DIR / "sql" / "queries.sql"

EXPECTED_CITY_COUNTS = {
    "Mumbai": 524,
    "Delhi": 436,
    "Chennai": 391,
    "Bengaluru": 386,
    "Hyderabad": 262,
    "Pune": 245,
    "Kolkata": 156,
}
EXPECTED_REVENUE_BY_PLAN = {
    "Premium": (1_471_199, 860.35, 339),
    "Standard": (1_323_622, 492.60, 559),
    "Standard HD": (1_283_471, 645.61, 407),
    "Premium Plus": (1_262_398, 1_075.30, 228),
    "Basic": (782_554, 326.61, 493),
    "Basic Lite": (334_329, 224.83, 298),
}
EXPECTED_RANK_ONE_COUNTS = {
    "Bengaluru": 10,
    "Chennai": 10,
    "Delhi": 5,
    "Hyderabad": 5,
    "Kolkata": 2,
    "Mumbai": 6,
    "Pune": 4,
}


def clean_subscribers():
    subscribers = pd.read_csv(RAW_DIR / "subscribers.csv").drop_duplicates().copy()

    city_names = {
        "chennai": "Chennai",
        "madras": "Chennai",
        "mumbai": "Mumbai",
        "bombay": "Mumbai",
        "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru",
        "delhi": "Delhi",
        "new delhi": "Delhi",
        "hyderabad": "Hyderabad",
        "hyd": "Hyderabad",
        "pune": "Pune",
        "kolkata": "Kolkata",
        "calcutta": "Kolkata",
    }
    subscribers["city"] = (
        subscribers["city"].str.strip().str.casefold().map(city_names)
    )
    if subscribers["city"].isna().any():
        raise ValueError("An unrecognized city was found in subscribers.csv.")

    auto_renew_values = {
        "yes": "Yes",
        "y": "Yes",
        "no": "No",
        "n": "No",
    }
    subscribers["auto_renew"] = (
        subscribers["auto_renew"].str.strip().str.casefold().map(auto_renew_values)
    )
    if subscribers["auto_renew"].isna().any():
        raise ValueError("An unrecognized auto_renew value was found.")

    subscribers["age"] = pd.to_numeric(subscribers["age"], errors="coerce")
    invalid_age = subscribers["age"].notna() & ~subscribers["age"].between(10, 100)
    subscribers.loc[invalid_age, "age"] = pd.NA

    if len(subscribers) != 2_400:
        raise AssertionError(f"Expected 2400 clean subscribers; got {len(subscribers)}.")
    if subscribers["age"].isna().sum() != 104:
        raise AssertionError("Expected 104 missing ages after cleaning.")
    if subscribers["payment_mode"].isna().sum() != 114:
        raise AssertionError("Expected 114 missing payment modes after cleaning.")
    if subscribers["churned"].sum() != 514:
        raise AssertionError("Expected 514 churned subscribers.")
    if subscribers["city"].value_counts().to_dict() != EXPECTED_CITY_COUNTS:
        raise AssertionError("Clean city counts did not match the assignment reference.")
    if subscribers["auto_renew"].value_counts().to_dict() != {"No": 1_355, "Yes": 1_045}:
        raise AssertionError("Clean auto_renew counts did not match the reference.")

    return subscribers


def clean_recharges(subscribers):
    recharges = pd.read_csv(RAW_DIR / "recharges.csv").drop_duplicates().copy()
    existing_ids = set(subscribers["subscriber_id"])
    recharges = recharges[recharges["subscriber_id"].isin(existing_ids)].copy()

    recharges["status"] = recharges["status"].str.strip().str.casefold()
    if not set(recharges["status"].dropna()).issubset({"success", "failed", "pending"}):
        raise ValueError("An unrecognized recharge status was found.")

    amounts = pd.to_numeric(recharges["amount_inr"], errors="coerce")
    invalid_amount = amounts.isna()
    monthly_recharge = subscribers.set_index("subscriber_id")["monthly_recharge_inr"]
    amounts.loc[invalid_amount] = recharges.loc[invalid_amount, "subscriber_id"].map(
        monthly_recharge
    )
    if amounts.isna().any():
        raise ValueError("A recharge amount could not be parsed or rebuilt.")
    recharges["amount_inr"] = amounts
    recharges["recharge_date"] = pd.to_datetime(
        recharges["recharge_date"], format="%Y-%m-%d", errors="raise"
    )

    if len(recharges) != 12_985:
        raise AssertionError(f"Expected 12985 clean recharges; got {len(recharges)}.")
    if int(invalid_amount.sum()) != 364:
        raise AssertionError("Expected 364 non-numeric recharge amounts to be rebuilt.")
    if recharges["status"].value_counts().to_dict() != {
        "success": 11_442,
        "failed": 1_422,
        "pending": 121,
    }:
        raise AssertionError("Clean recharge status counts did not match the reference.")
    successful_total = recharges.loc[
        recharges["status"] == "success", "amount_inr"
    ].sum()
    if int(successful_total) != 6_457_573:
        raise AssertionError("Successful recharge total did not match the reference.")

    return recharges, int(invalid_amount.sum())


def execute_queries():
    query_text = QUERIES_FILE.read_text(encoding="utf-8")
    queries = [statement.strip() for statement in query_text.split(";") if statement.strip()]
    if len(queries) != 4:
        raise ValueError(f"Expected four SQL queries; found {len(queries)}.")

    with sqlite3.connect(DATABASE_FILE) as connection:
        results = [pd.read_sql_query(query, connection) for query in queries]

    revenue, never_recharged, rankings, running_total = results
    actual_revenue = {
        row.plan_name: (
            int(row.total_revenue_inr),
            round(float(row.average_recharge_inr), 2),
            int(row.distinct_subscribers),
        )
        for row in revenue.itertuples(index=False)
    }
    if actual_revenue != EXPECTED_REVENUE_BY_PLAN:
        raise AssertionError(f"Q1 revenue reference mismatch: {actual_revenue}")

    never_counts = never_recharged.iloc[0].to_dict()
    if never_counts != {"never_successful_count": 76, "zero_tenure_count": 68}:
        raise AssertionError(f"Q2 reference mismatch: {never_counts}")

    rank_one_counts = (
        rankings.loc[rankings["spend_rank"] == 1]
        .groupby("city")
        .size()
        .to_dict()
    )
    if rank_one_counts != EXPECTED_RANK_ONE_COUNTS:
        raise AssertionError(f"Q3 rank-one reference mismatch: {rank_one_counts}")

    kolkata_top_four = rankings.loc[rankings["city"] == "Kolkata"].head(4)
    expected_kolkata = [
        ("TP00074", 6_876, 1, 1, 1),
        ("TP00917", 6_876, 1, 1, 2),
        ("TP00143", 6_582, 3, 2, 3),
        ("TP01355", 6_582, 3, 2, 4),
    ]
    actual_kolkata = list(
        kolkata_top_four[
            [
                "subscriber_id",
                "total_successful_recharge_inr",
                "spend_rank",
                "dense_spend_rank",
                "spend_row_number",
            ]
        ].itertuples(index=False, name=None)
    )
    if actual_kolkata != expected_kolkata:
        raise AssertionError(f"Q3 Kolkata reference mismatch: {actual_kolkata}")

    if len(running_total) != 5 or int(running_total.iloc[-1]["running_total_inr"]) != 2_490:
        raise AssertionError("Q4 running total did not match the reference.")

    return revenue, never_recharged, rankings, running_total


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    subscribers = clean_subscribers()
    recharges, rebuilt_amount_count = clean_recharges(subscribers)
    plans = pd.read_csv(RAW_DIR / "plans.csv")

    subscribers.to_csv(CLEAN_DIR / "subscribers.csv", index=False)
    recharges.to_csv(CLEAN_DIR / "recharges.csv", index=False, date_format="%Y-%m-%d")
    plans.to_csv(CLEAN_DIR / "plans.csv", index=False)

    with sqlite3.connect(DATABASE_FILE) as connection:
        subscribers.to_sql("subscribers", connection, if_exists="replace", index=False)
        recharges.to_sql("recharges", connection, if_exists="replace", index=False)
        plans.to_sql("plans", connection, if_exists="replace", index=False)

    print("Clean data checkpoints:")
    print(f"Subscribers: {len(subscribers)}")
    print(f"City counts: {subscribers['city'].value_counts().to_dict()}")
    print(f"Auto-renew counts: {subscribers['auto_renew'].value_counts().to_dict()}")
    print(f"Missing ages: {subscribers['age'].isna().sum()}")
    print(f"Missing payment modes: {subscribers['payment_mode'].isna().sum()}")
    print(f"Churned subscribers: {subscribers['churned'].sum()} / {len(subscribers)}")
    print(f"Recharges: {len(recharges)}")
    print(f"Amounts rebuilt: {rebuilt_amount_count}")
    print(f"Recharge status counts: {recharges['status'].value_counts().to_dict()}")
    print(
        "Successful recharge total INR: "
        f"{int(recharges.loc[recharges['status'] == 'success', 'amount_inr'].sum())}"
    )

    revenue, never_recharged, rankings, running_total = execute_queries()
    pandas_revenue = (
        recharges.loc[recharges["status"] == "success"]
        .merge(subscribers[["subscriber_id", "plan_id"]], on="subscriber_id")
        .merge(plans[["plan_id", "plan_name"]], on="plan_id")
        .groupby("plan_name")["amount_inr"]
        .sum()
    )
    if int(pandas_revenue["Premium"]) != int(
        revenue.loc[revenue["plan_name"] == "Premium", "total_revenue_inr"].iloc[0]
    ):
        raise AssertionError("Pandas groupby did not confirm the Premium SQL result.")

    print("\nQ1 - successful recharge revenue by plan:")
    print(revenue.to_string(index=False))
    print("\nQ2 - subscribers with no successful recharge:")
    print(never_recharged.to_string(index=False))
    print("\nQ3 - rank-one subscribers per city:")
    print(
        rankings.loc[rankings["spend_rank"] == 1]
        .groupby("city")
        .size()
        .rename("rank_one_subscribers")
        .to_string()
    )
    print("\nQ3 - Kolkata top four:")
    print(
        rankings.loc[rankings["city"] == "Kolkata"]
        .head(4)
        .to_string(index=False)
    )
    print("\nQ4 - TP00010 running total:")
    print(running_total.to_string(index=False))
    print(f"\nSQLite database: {DATABASE_FILE}")
    print("All Part A reference checkpoints passed.")


if __name__ == "__main__":
    main()
