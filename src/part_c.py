from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SEGMENT_FEATURES = [
    "tenure_months",
    "hours_watched_per_week",
    "monthly_recharge_inr",
]
CORRELATION_FEATURES = [
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
]
SPEND_FEATURES = [
    "monthly_recharge_inr",
    "tenure_months",
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
    "support_calls_90d",
]
SPEND_TARGET = "next_cycle_recharge_inr"
ALPHAS = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 10, 30, 100]


def describe_clusters(subscribers):
    features = subscribers[SEGMENT_FEATURES]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scores = []
    for cluster_count in range(2, 9):
        model = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(scaled_features)
        scores.append(
            {
                "k": cluster_count,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled_features, labels),
            }
        )

    chosen_k = 4
    model = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    labels = model.fit_predict(scaled_features)
    segmented = subscribers[["subscriber_id", "churned"]].copy()
    segmented["segment_id"] = labels

    profile = (
        segmented.groupby("segment_id")
        .agg(
            size=("subscriber_id", "size"),
            mean_tenure_months=("segment_id", lambda ids: subscribers.loc[ids.index, "tenure_months"].mean()),
            mean_hours_per_week=("segment_id", lambda ids: subscribers.loc[ids.index, "hours_watched_per_week"].mean()),
            mean_monthly_recharge_inr=("segment_id", lambda ids: subscribers.loc[ids.index, "monthly_recharge_inr"].mean()),
            churn_rate=("churned", "mean"),
        )
        .reset_index()
    )
    at_risk_id = int(profile.loc[profile["churn_rate"].idxmax(), "segment_id"])
    premium_id = int(
        profile.loc[profile["mean_monthly_recharge_inr"].idxmax(), "segment_id"]
    )
    if at_risk_id == premium_id:
        raise AssertionError("At-Risk and Premium segments should be distinct.")

    names = {at_risk_id: "At-Risk", premium_id: "Premium"}
    remaining = [
        int(segment_id)
        for segment_id in profile["segment_id"]
        if int(segment_id) not in names
    ]
    remaining_by_tenure = (
        profile.loc[profile["segment_id"].isin(remaining)]
        .sort_values("mean_tenure_months", ascending=False)["segment_id"]
        .astype(int)
        .tolist()
    )
    remaining_names = ["Established Viewers", "Value Seekers"]
    for segment_id, name in zip(remaining_by_tenure, remaining_names):
        names[segment_id] = name

    actions = {
        "At-Risk": "Offer a proactive retention contact and check for recharge or support issues.",
        "Premium": "Provide priority support and recommend relevant add-ons.",
        "Established Viewers": "Recognize tenure with a relevant loyalty benefit.",
        "Value Seekers": "Offer a suitable lower-cost plan or simple recharge support.",
    }
    profile["segment_name"] = profile["segment_id"].map(names)
    profile["recommended_action"] = profile["segment_name"].map(actions)
    segmented["segment_name"] = segmented["segment_id"].map(names)

    new_subscriber = pd.DataFrame(
        [[4, 15, 520]],
        columns=SEGMENT_FEATURES,
    )
    new_scaled = (new_subscriber.to_numpy() - scaler.mean_) / scaler.scale_
    distances = np.linalg.norm(model.cluster_centers_ - new_scaled, axis=1)
    manual_segment_id = int(np.argmin(distances))
    predicted_segment_id = int(model.predict(new_scaled)[0])
    if manual_segment_id != predicted_segment_id:
        raise AssertionError("Manual nearest-centroid assignment disagrees with KMeans.")

    if len(segmented) != 2_400 or segmented["segment_name"].isna().any():
        raise AssertionError("Every clean subscriber must receive one named segment.")
    if int(profile.loc[profile["segment_name"] == "At-Risk", "size"].iloc[0]) != 784:
        raise AssertionError("The chosen K=4 At-Risk segment size missed its reference.")

    segment_output = segmented[["subscriber_id", "segment_id", "segment_name"]]
    return (
        pd.DataFrame(scores),
        scaler,
        model,
        profile,
        segment_output,
        distances,
        manual_segment_id,
    )


def fit_spend_models(subscribers):
    stayers = subscribers.loc[subscribers["churned"] == 0].copy()
    if stayers[SPEND_TARGET].isna().any():
        raise AssertionError("A stayer is missing its next-cycle recharge target.")

    X_train, X_test, y_train, y_test = train_test_split(
        stayers[SPEND_FEATURES],
        stayers[SPEND_TARGET],
        test_size=0.2,
        random_state=42,
    )
    if (len(X_train), len(X_test)) != (1_508, 378):
        raise AssertionError("Unexpected spend-model train/test split sizes.")

    linear_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    linear_cv_scores = cross_val_score(
        linear_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
    )
    linear_pipeline.fit(X_train, y_train)

    ridge_search = GridSearchCV(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge()),
            ]
        ),
        param_grid={"regressor__alpha": ALPHAS},
        cv=5,
        scoring="r2",
        n_jobs=1,
        refit=True,
    )
    ridge_search.fit(X_train, y_train)

    lasso_search = GridSearchCV(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Lasso(max_iter=10_000)),
            ]
        ),
        param_grid={"regressor__alpha": ALPHAS},
        cv=5,
        scoring="r2",
        n_jobs=1,
        refit=True,
    )
    lasso_search.fit(X_train, y_train)

    model_results = []
    fitted_models = {
        "LinearRegression": (linear_pipeline, float(linear_cv_scores.mean()), None),
        "Ridge": (ridge_search.best_estimator_, ridge_search.best_score_, ridge_search.best_params_["regressor__alpha"]),
        "Lasso": (lasso_search.best_estimator_, lasso_search.best_score_, lasso_search.best_params_["regressor__alpha"]),
    }
    for name, (pipeline, cv_r2, alpha) in fitted_models.items():
        predictions = pipeline.predict(X_test)
        model_results.append(
            {
                "model": name,
                "best_alpha": alpha,
                "cv_r2": float(cv_r2),
                "test_r2": float(r2_score(y_test, predictions)),
                "test_mae": float(mean_absolute_error(y_test, predictions)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            }
        )

    lasso_model = lasso_search.best_estimator_.named_steps["regressor"]
    zero_coefficients = [
        feature
        for feature, coefficient in zip(SPEND_FEATURES, lasso_model.coef_)
        if coefficient == 0
    ]

    stability = {"LinearRegression": [], "Ridge(alpha=30)": []}
    for seed in range(5):
        sample_indices = X_train.sample(frac=0.8, random_state=seed).index
        sample_x = X_train.loc[sample_indices]
        sample_y = y_train.loc[sample_indices]
        for name, model in [
            ("LinearRegression", LinearRegression()),
            ("Ridge(alpha=30)", Ridge(alpha=30)),
        ]:
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", model),
                ]
            )
            pipeline.fit(sample_x, sample_y)
            feature_index = SPEND_FEATURES.index("hours_watched_per_week")
            coefficient = pipeline.named_steps["regressor"].coef_[feature_index]
            stability[name].append(float(coefficient))
    stability_std = {
        name: float(np.std(coefficients, ddof=0))
        for name, coefficients in stability.items()
    }

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        ridge_search.best_estimator_,
        pd.DataFrame(model_results),
        zero_coefficients,
        stability_std,
        subscribers[CORRELATION_FEATURES].corr(),
    )


def build_retention_priority(subscribers, ridge_model, segment_output):
    test_scores = pd.read_csv(OUTPUT_DIR / "test_scores.csv")
    test_subscribers = subscribers.merge(
        test_scores[["subscriber_id", "churn_probability", "churned"]],
        on="subscriber_id",
        how="inner",
        suffixes=("", "_test"),
        validate="one_to_one",
    )
    if len(test_subscribers) != 480:
        raise AssertionError("Expected 480 Part B test subscribers.")

    test_subscribers["predicted_next_recharge_inr"] = ridge_model.predict(
        test_subscribers[SPEND_FEATURES]
    )
    test_subscribers["revenue_at_risk_inr"] = (
        test_subscribers["churn_probability"]
        * test_subscribers["predicted_next_recharge_inr"]
    )
    priority = test_subscribers.merge(
        segment_output[["subscriber_id", "segment_name"]],
        on="subscriber_id",
        how="left",
        validate="one_to_one",
    )
    priority = (
        priority.sort_values(
            "revenue_at_risk_inr",
            ascending=False,
            kind="mergesort",
        )
        .head(15)
    )
    priority = priority[
        [
            "subscriber_id",
            "churn_probability",
            "predicted_next_recharge_inr",
            "revenue_at_risk_inr",
            "segment_name",
            "churned_test",
        ]
    ].rename(columns={"churned_test": "churned"})
    priority.to_csv(OUTPUT_DIR / "retention_priority.csv", index=False)
    churned_count = int(priority["churned"].sum())
    if len(priority) != 15 or churned_count < 9:
        raise AssertionError(
            f"Priority list acceptance failed: 15 rows, {churned_count} churned."
        )
    return priority


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subscribers = pd.read_csv(CLEAN_DIR / "subscribers.csv")
    (
        cluster_scores,
        segment_scaler,
        kmeans,
        segment_profile,
        segment_output,
        new_subscriber_distances,
        assigned_segment_id,
    ) = describe_clusters(subscribers)
    segment_output.to_csv(OUTPUT_DIR / "segments.csv", index=False)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        best_ridge,
        regression_results,
        zero_coefficients,
        stability_std,
        correlation_matrix,
    ) = fit_spend_models(subscribers)
    retention_priority = build_retention_priority(
        subscribers,
        best_ridge,
        segment_output,
    )

    print("K-means inertia / silhouette:")
    print(cluster_scores.to_string(index=False, float_format="%.3f"))
    print("\nChosen K=4 segment profiles:")
    print(segment_profile.to_string(index=False, float_format="%.2f"))
    print("\nNew subscriber (4 months, 15 hours/week, INR 520):")
    print(f"Distances by cluster ID: {np.round(new_subscriber_distances, 3).tolist()}")
    new_subscriber_scaled = (
        np.array([[4, 15, 520]]) - segment_scaler.mean_
    ) / segment_scaler.scale_
    predicted_segment_id = int(kmeans.predict(new_subscriber_scaled)[0])
    print(
        f"Nearest cluster: {assigned_segment_id} "
        f"({segment_profile.loc[segment_profile['segment_id'] == assigned_segment_id, 'segment_name'].iloc[0]})"
    )
    print(
        "Hand calculation matches kmeans.predict: "
        f"{assigned_segment_id == predicted_segment_id}"
    )
    print(f"Segments saved: {OUTPUT_DIR / 'segments.csv'} ({len(segment_output)} rows)")

    print("\nCorrelation matrix:")
    print(correlation_matrix.round(3).to_string())
    print("\nSpend-model split: train/test =", len(X_train), "/", len(X_test))
    print("Spend model results:")
    print(regression_results.to_string(index=False, float_format="%.3f"))
    print(f"Features with exactly zero Lasso coefficient: {zero_coefficients}")
    print(f"Hours-watched coefficient population standard deviations: {stability_std}")

    total_priority_risk = float(retention_priority["revenue_at_risk_inr"].sum())
    actual_churners = int(retention_priority["churned"].sum())
    print("\nTop-15 revenue-at-risk subscribers:")
    print(f"Sum of estimated revenue at risk: INR {total_priority_risk:,.2f}")
    print(f"Actually churned among the top 15: {actual_churners}")
    print(f"Priority list saved: {OUTPUT_DIR / 'retention_priority.csv'}")
    print("Part C pipeline completed.")


if __name__ == "__main__":
    main()
