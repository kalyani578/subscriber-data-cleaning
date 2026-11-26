from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__:
    from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_model_table
else:
    from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_model_table


PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
OUTPUT_DIR = PROJECT_DIR / "outputs"
LEAKAGE_NUMERIC_FEATURES = NUMERIC_FEATURES + ["next_cycle_recharge_inr"]
LEAKAGE_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ["closure_request_logged"]
COST_MISSED_CHURN = 1_200
COST_UNNECESSARY_OFFER = 300


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def make_pipeline(model, numeric_features=NUMERIC_FEATURES,
                  categorical_features=CATEGORICAL_FEATURES):
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(numeric_features, categorical_features),
            ),
            ("model", model),
        ]
    )


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def threshold_cost(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    cost = COST_MISSED_CHURN * fn + COST_UNNECESSARY_OFFER * fp
    return {
        "threshold": float(threshold),
        "cost_inr": int(cost),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    table = build_model_table()
    y = table["churned"].astype(int)
    indices = np.arange(len(table))
    train_validation_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=0.25,
        stratify=y.iloc[train_validation_indices],
        random_state=42,
    )

    train = table.iloc[train_indices]
    validation = table.iloc[validation_indices]
    test = table.iloc[test_indices]
    y_train = train["churned"].astype(int)
    y_validation = validation["churned"].astype(int)
    y_test = test["churned"].astype(int)

    if (len(train), len(validation), len(test)) != (1_440, 480, 480):
        raise AssertionError("The train/validation/test split sizes are incorrect.")
    split_churners = (
        int(y_train.sum()),
        int(y_validation.sum()),
        int(y_test.sum()),
    )
    if split_churners != (308, 103, 103):
        raise AssertionError(f"Unexpected split churn counts: {split_churners}")

    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_validation = validation[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    leakage_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["next_cycle_recharge_inr", "closure_request_logged"]
    )
    leakage_model = make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42),
        LEAKAGE_NUMERIC_FEATURES,
        LEAKAGE_CATEGORICAL_FEATURES,
    )
    leakage_model.fit(train[leakage_features], y_train)
    leakage_metrics = classification_metrics(
        y_validation,
        leakage_model.predict(validation[leakage_features]),
    )

    majority_predictions = np.zeros(len(y_validation), dtype=int)
    majority_metrics = classification_metrics(y_validation, majority_predictions)

    baseline_models = {
        "LogisticRegression": LogisticRegression(max_iter=1_000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    baseline_results = {}
    for name, model in baseline_models.items():
        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)
        validation_metrics = classification_metrics(
            y_validation,
            pipeline.predict(X_validation),
        )
        training_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
        baseline_results[name] = {
            "training_accuracy": training_accuracy,
            **validation_metrics,
        }

    tree_depth_results = []
    for depth in range(1, 11):
        tree = make_pipeline(
            DecisionTreeClassifier(max_depth=depth, random_state=42)
        )
        tree.fit(X_train, y_train)
        tree_depth_results.append(
            {
                "max_depth": depth,
                "training_accuracy": accuracy_score(
                    y_train, tree.predict(X_train)
                ),
                "validation_f1": f1_score(
                    y_validation, tree.predict(X_validation), zero_division=0
                ),
            }
        )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logistic_search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1_000)),
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        scoring="f1",
        cv=folds,
        n_jobs=1,
        refit=True,
    )
    logistic_search.fit(X_train, y_train)

    forest_search = GridSearchCV(
        make_pipeline(RandomForestClassifier(random_state=42)),
        param_grid={
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__max_features": [3, 4, 5],
        },
        scoring="f1",
        cv=folds,
        n_jobs=1,
        refit=True,
    )
    forest_search.fit(X_train, y_train)

    fitted_preprocessor = forest_search.best_estimator_.named_steps["preprocessor"]
    transformed_train = fitted_preprocessor.transform(X_train)
    if hasattr(transformed_train, "data"):
        has_missing_after_transform = bool(np.isnan(transformed_train.data).any())
    else:
        has_missing_after_transform = bool(np.isnan(transformed_train).any())
    if has_missing_after_transform:
        raise AssertionError("Missing values remain after preprocessing.")

    feature_names = fitted_preprocessor.get_feature_names_out()
    numeric_pipeline = fitted_preprocessor.named_transformers_["numeric"]
    learned_median_age = float(numeric_pipeline.named_steps["imputer"].statistics_[1])
    learned_tenure_mean = float(numeric_pipeline.named_steps["scaler"].mean_[0])
    full_data_tenure_mean = float(table["tenure_months"].mean())

    validation_probabilities = {
        "random_forest": forest_search.best_estimator_.predict_proba(
            X_validation
        )[:, 1],
        "logistic_regression": logistic_search.best_estimator_.predict_proba(
            X_validation
        )[:, 1],
    }
    tuned_models = {
        "random_forest": forest_search.best_estimator_,
        "logistic_regression": logistic_search.best_estimator_,
    }
    threshold_results = {}
    thresholds = np.round(np.arange(0.10, 0.901, 0.05), 2)
    for name, probabilities in validation_probabilities.items():
        costs = [
            threshold_cost(y_validation, probabilities, threshold)
            for threshold in thresholds
        ]
        threshold_results[name] = min(costs, key=lambda result: result["cost_inr"])

    selected_name = min(
        threshold_results,
        key=lambda name: threshold_results[name]["cost_inr"],
    )
    selected_model = tuned_models[selected_name]

    test_probabilities = {
        name: model.predict_proba(X_test)[:, 1]
        for name, model in tuned_models.items()
    }
    test_results = {}
    for name, probabilities in test_probabilities.items():
        threshold = threshold_results[name]["threshold"]
        result = threshold_cost(y_test, probabilities, threshold)
        predictions = (probabilities >= threshold).astype(int)
        result.update(classification_metrics(y_test, predictions))
        test_results[name] = result

    no_action_cost = int(y_test.sum()) * COST_MISSED_CHURN
    offer_everyone_cost = (len(y_test) - int(y_test.sum())) * COST_UNNECESSARY_OFFER
    selected_test = test_results[selected_name]
    if selected_test["cost_inr"] > 0.60 * no_action_cost:
        raise AssertionError("The selected model exceeds the test cost acceptance bar.")
    if selected_test["recall"] < 0.55 or selected_test["f1"] < 0.50:
        raise AssertionError("The selected model misses a test-set acceptance bar.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "churn_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)
    reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]
    if not np.array_equal(reloaded_probabilities, test_probabilities[selected_name]):
        raise AssertionError("Reloaded model did not reproduce test probabilities.")

    test_scores = pd.DataFrame(
        {
            "subscriber_id": test["subscriber_id"].to_numpy(),
            "churn_probability": test_probabilities[selected_name],
            "churned": y_test.to_numpy(),
        }
    )
    test_scores.to_csv(OUTPUT_DIR / "test_scores.csv", index=False)

    forest = forest_search.best_estimator_.named_steps["model"]
    top_importance_indices = np.argsort(forest.feature_importances_)[::-1][:5]
    top_importances = [
        (str(feature_names[index]), float(forest.feature_importances_[index]))
        for index in top_importance_indices
    ]
    logistic_pipeline = logistic_search.best_estimator_
    logistic_feature_names = (
        logistic_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    logistic_coefficients = logistic_pipeline.named_steps["model"].coef_[0]
    coefficient_signs = {}
    for feature in ["support_calls_90d", "tenure_months", "n_failed_6m"]:
        feature_name = f"numeric__{feature}"
        feature_index = list(logistic_feature_names).index(feature_name)
        coefficient_signs[feature] = float(logistic_coefficients[feature_index])

    print("Problem framing:")
    print("Decision: prioritize subscribers for retention contact before April cancellation.")
    print("Decision owner: retention desk; offer eligible subscribers a retention action.")
    print("Target: churned (1 = cancelled in April 2026, 0 = stayed), a binary category.")
    print("Missed churn costs INR 1200; an unnecessary offer costs INR 300.")
    print(f"Class balance: {int(y.sum())} / {len(y)} ({y.mean():.2%}) churned.")
    print(f"Split sizes train/validation/test: 1440 / 480 / 480; churners: {split_churners}.")
    print("Test set used only after validation-based model and threshold selection.")

    print("\nLeakage experiment (validation):")
    print(f"With leakage accuracy/F1: {leakage_metrics['accuracy']:.3f} / {leakage_metrics['f1']:.3f}")
    print(
        "Without leakage accuracy/F1: "
        f"{baseline_results['RandomForest']['accuracy']:.3f} / "
        f"{baseline_results['RandomForest']['f1']:.3f}"
    )
    print("Leaking columns: closure_request_logged, next_cycle_recharge_inr.")

    print("\nPreprocessing fitted on training rows:")
    print(f"Output columns: {len(feature_names)}")
    print(f"Learned median age: {learned_median_age:.1f}")
    print(
        "Scaler mean tenure / full-data mean tenure: "
        f"{learned_tenure_mean:.2f} / {full_data_tenure_mean:.2f}"
    )
    print("Missing values after transform: none.")

    print("\nValidation baselines and models:")
    print(f"Always stay accuracy/F1: {majority_metrics['accuracy']:.3f} / {majority_metrics['f1']:.3f}")
    for name, metrics in baseline_results.items():
        print(
            f"{name}: train_accuracy={metrics['training_accuracy']:.3f}, "
            f"validation accuracy/precision/recall/F1="
            f"{metrics['accuracy']:.3f}/{metrics['precision']:.3f}/"
            f"{metrics['recall']:.3f}/{metrics['f1']:.3f}"
        )
    print("Decision-tree depth sweep:")
    print(pd.DataFrame(tree_depth_results).to_string(index=False, float_format="%.3f"))

    logistic_best_index = logistic_search.best_index_
    forest_best_index = forest_search.best_index_
    print("\nCross-validation:")
    print(
        f"Logistic combinations/fits: 5 / 25; best={logistic_search.best_params_}; "
        f"mean F1={logistic_search.cv_results_['mean_test_score'][logistic_best_index]:.3f}, "
        f"std={logistic_search.cv_results_['std_test_score'][logistic_best_index]:.3f}"
    )
    print(
        f"Random-forest combinations/fits: 45 / 225; best={forest_search.best_params_}; "
        f"mean F1={forest_search.cv_results_['mean_test_score'][forest_best_index]:.3f}, "
        f"std={forest_search.cv_results_['std_test_score'][forest_best_index]:.3f}"
    )

    print("\nValidation cost thresholds:")
    for name, result in threshold_results.items():
        print(f"{name}: threshold={result['threshold']:.2f}, cost=INR {result['cost_inr']}")
    print(f"Selected from validation cost: {selected_name}")

    print("\nTest results (each tuned model evaluated at its validation-selected threshold):")
    for name, result in test_results.items():
        print(
            f"{name}: threshold={result['threshold']:.2f}, "
            f"TN/FP/FN/TP={result['tn']}/{result['fp']}/{result['fn']}/{result['tp']}, "
            f"precision={result['precision']:.3f}, recall={result['recall']:.3f}, "
            f"F1={result['f1']:.3f}, cost=INR {result['cost_inr']}"
        )
    print(f"No-action test cost: INR {no_action_cost}")
    print(f"Offer-everyone test cost: INR {offer_everyone_cost}")
    print(f"Top random-forest importances: {top_importances}")
    print(f"Logistic coefficient values: {coefficient_signs}")
    print(f"Selected model saved and reload-verified: {model_path}")
    print(f"Test scores saved: {OUTPUT_DIR / 'test_scores.csv'} ({len(test_scores)} rows)")
    print("Part B pipeline completed.")


if __name__ == "__main__":
    main()
