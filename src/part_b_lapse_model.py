
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths and feature definitions
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = OUTPUT_DIR / "lapse_features.csv"
SEGMENTS_PATH = OUTPUT_DIR / "customer_segments.csv"
MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"
RETENTION_PATH = OUTPUT_DIR / "retention_list.csv"

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 2. Load data and create a stratified train/test split
# --------------------------------------------------
df = pd.read_csv(FEATURES_PATH)

assert not df[MODEL_FEATURES].isna().any().any(), (
    "Missing values found in model features."
)

X = df[MODEL_FEATURES].copy()
y = df["lapsed"].astype(int).copy()
customer_ids = df["customer_id"].astype(str).reset_index(drop=True)

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    stratify=y,
    random_state=42,
)

X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

train_ids = customer_ids.iloc[train_idx].reset_index(drop=True)
test_ids = customer_ids.iloc[test_idx].reset_index(drop=True)

print("\nDATA SPLIT")
print("----------")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Training class counts:\n{y_train.value_counts().sort_index()}")
print(f"Test class counts:\n{y_test.value_counts().sort_index()}")

assert len(X_train) == 924
assert len(X_test) == 232


# --------------------------------------------------
# 3. Baseline: always predict not lapsed
# --------------------------------------------------
baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\nBASELINE — ALWAYS PREDICT NOT LAPSED")
print("------------------------------------")
print(f"Accuracy: {accuracy_score(y_test, baseline_predictions):.4f}")
print(
    "Lapsed-class precision:",
    f"{precision_score(y_test, baseline_predictions, zero_division=0):.4f}",
)
print(
    "Lapsed-class recall:",
    f"{recall_score(y_test, baseline_predictions, zero_division=0):.4f}",
)
print(
    "Lapsed-class F1:",
    f"{f1_score(y_test, baseline_predictions, zero_division=0):.4f}",
)
print("Confusion matrix:\n", confusion_matrix(y_test, baseline_predictions))


# --------------------------------------------------
# 4. Build leakage-safe preprocessing pipelines
# --------------------------------------------------
def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# --------------------------------------------------
# 5. Tune Logistic Regression using training rows only
# --------------------------------------------------
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", make_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

logistic_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid={
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True,
)

logistic_search.fit(X_train, y_train)
best_logistic = logistic_search.best_estimator_

print("\nLOGISTIC REGRESSION GRID SEARCH")
print("--------------------------------")
print("Best parameters:", logistic_search.best_params_)
print(f"Best cross-validated F1: {logistic_search.best_score_:.4f}")


# --------------------------------------------------
# 6. Tune Random Forest using training rows only
# --------------------------------------------------
forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", make_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

forest_search = GridSearchCV(
    estimator=forest_pipeline,
    param_grid={
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True,
)

forest_search.fit(X_train, y_train)
best_forest = forest_search.best_estimator_

print("\nRANDOM FOREST GRID SEARCH")
print("-------------------------")
print("Best parameters:", forest_search.best_params_)
print(f"Best cross-validated F1: {forest_search.best_score_:.4f}")


# --------------------------------------------------
# 7. Evaluate only after both searches are complete
# --------------------------------------------------
def report_metrics(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    print(f"\n{name}")
    print("-" * len(name))
    print("Confusion matrix:\n", confusion_matrix(y_eval, predictions))
    print(
        f"Precision: {precision_score(y_eval, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_eval, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1:        {f1_score(y_eval, predictions, zero_division=0):.4f}"
    )

    return predictions


logistic_predictions = report_metrics(
    "TUNED LOGISTIC REGRESSION — TEST SET",
    best_logistic,
    X_test,
    y_test,
)

forest_predictions = report_metrics(
    "TUNED RANDOM FOREST — TEST SET",
    best_forest,
    X_test,
    y_test,
)

logistic_f1 = f1_score(
    y_test, logistic_predictions, zero_division=0
)
forest_f1 = f1_score(
    y_test, forest_predictions, zero_division=0
)

# Select the model with higher test F1, as a comparison
# requested by the assignment. This selection uses test
# results, so the final test estimate is not fully independent
# of the model-selection decision.
selected_model = (
    best_forest if forest_f1 >= logistic_f1 else best_logistic
)

print(
    "\nSelected model:",
    "Random Forest" if forest_f1 >= logistic_f1
    else "Logistic Regression",
)

if max(logistic_f1, forest_f1) < 0.55:
    print(
        "WARNING: Neither model reached the required "
        "test F1 of 0.55. Investigate before proceeding."
    )


# --------------------------------------------------
# 8. Compare training scaler mean with full-cohort mean
# --------------------------------------------------
fitted_scaler = (
    best_logistic.named_steps["preprocessor"]
    .named_transformers_["numeric"]
    .named_steps["scaler"]
)

frequency_position = NUMERIC_FEATURES.index("frequency")

print("\nSCALER CHECK")
print("------------")
print(
    "Training-set frequency mean learned by scaler:",
    round(fitted_scaler.mean_[frequency_position], 4),
)
print(
    "Full-cohort frequency mean:",
    round(df["frequency"].mean(), 4),
)
print(
    "The means can differ because the scaler is fitted only "
    "on training rows, not on the full cohort."
)


# --------------------------------------------------
# 9. Oversample lapsed training rows for Logistic Regression
# --------------------------------------------------
# Preserve every original training row and add duplicate
# lapsed rows sampled with replacement until classes match.
negative_positions = np.flatnonzero(y_train.to_numpy() == 0)
positive_positions = np.flatnonzero(y_train.to_numpy() == 1)

extra_needed = len(negative_positions) - len(positive_positions)

if extra_needed < 0:
    raise ValueError(
        "Expected lapsed customers to be the minority class."
    )

rng = np.random.RandomState(42)
extra_positions = rng.choice(
    positive_positions,
    size=extra_needed,
    replace=True,
)

balanced_positions = np.concatenate(
    [
        np.arange(len(X_train)),
        extra_positions,
    ]
)

X_balanced = X_train.iloc[balanced_positions].reset_index(drop=True)
y_balanced = y_train.iloc[balanced_positions].reset_index(drop=True)

print("\nOVERSAMPLING CHECK")
print("------------------")
print("Balanced training class counts:")
print(y_balanced.value_counts().sort_index())

balanced_logistic = clone(best_logistic)
balanced_logistic.fit(X_balanced, y_balanced)

balanced_predictions = balanced_logistic.predict(X_test)

print("\nBALANCED LOGISTIC REGRESSION — UNCHANGED TEST SET")
print("-------------------------------------------------")
print("Confusion matrix:\n", confusion_matrix(y_test, balanced_predictions))
print(
    f"Precision: {precision_score(y_test, balanced_predictions, zero_division=0):.4f}"
)
print(
    f"Recall:    {recall_score(y_test, balanced_predictions, zero_division=0):.4f}"
)
print(
    f"F1:        {f1_score(y_test, balanced_predictions, zero_division=0):.4f}"
)


# --------------------------------------------------
# 10. Interpret Random Forest feature importance
# --------------------------------------------------
forest_preprocessor = best_forest.named_steps["preprocessor"]
forest_classifier = best_forest.named_steps["classifier"]

forest_feature_names = forest_preprocessor.get_feature_names_out()

importance_table = pd.DataFrame(
    {
        "feature": forest_feature_names,
        "importance": forest_classifier.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nTOP FIVE RANDOM FOREST FEATURES")
print("--------------------------------")
print(importance_table.head(5).to_string(index=False))


# --------------------------------------------------
# 11. Interpret three Logistic Regression coefficients
# --------------------------------------------------
logistic_preprocessor = best_logistic.named_steps["preprocessor"]
logistic_classifier = best_logistic.named_steps["classifier"]

logistic_feature_names = logistic_preprocessor.get_feature_names_out()

coefficient_table = pd.DataFrame(
    {
        "feature": logistic_feature_names,
        "coefficient": logistic_classifier.coef_[0],
    }
)

print("\nTHREE LOGISTIC REGRESSION COEFFICIENTS")
print("--------------------------------------")

for feature in ["numeric__recency", "numeric__frequency", "numeric__monetary"]:
    row = coefficient_table.loc[
        coefficient_table["feature"] == feature
    ]
    if not row.empty:
        print(
            f"{feature}: {row['coefficient'].iloc[0]:.4f}"
        )


# --------------------------------------------------
# 12. Save, reload, and verify the selected model
# --------------------------------------------------
joblib.dump(selected_model, MODEL_PATH)
reloaded_model = joblib.load(MODEL_PATH)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

assert np.array_equal(
    original_predictions, reloaded_predictions
), "Reloaded model predictions differ."

print("\nMODEL PERSISTENCE CHECK")
print("-----------------------")
print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
print("Reloaded predictions exactly match original predictions.")


# --------------------------------------------------
# 13. Rank test-set Premium customers for retention
# --------------------------------------------------
segments = pd.read_csv(SEGMENTS_PATH)

segments["customer_id"] = segments["customer_id"].astype(str)

premium_ids = set(
    segments.loc[
        segments["is_premium"].astype(str).str.lower().isin(
            ["true", "1"]
        ),
        "customer_id",
    ]
)

test_results = pd.DataFrame(
    {
        "customer_id": test_ids,
        "actually_lapsed": y_test.to_numpy(),
        "lapse_probability": selected_model.predict_proba(X_test)[
            :, list(selected_model.named_steps["classifier"].classes_).index(1)
        ],
    }
)

premium_test = test_results.loc[
    test_results["customer_id"].isin(premium_ids)
].copy()

retention_list = (
    premium_test.sort_values(
        "lapse_probability", ascending=False
    )
    .head(10)
    [["customer_id", "lapse_probability", "actually_lapsed"]]
)

retention_list.to_csv(RETENTION_PATH, index=False)

print("\nRETENTION LIST")
print("--------------")
print(retention_list.to_string(index=False))
print(f"Premium customers in test set: {len(premium_test)}")
print(
    "Truly lapsed customers in retention list:",
    int(retention_list["actually_lapsed"].sum()),
)
print(f"Saved: {RETENTION_PATH.relative_to(ROOT)}")

print("\nPART B LAPSE MODEL PIPELINE FINISHED")
