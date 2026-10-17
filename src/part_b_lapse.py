
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths and prediction dates
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")

ORDERS_PATH = DATA_DIR / "orders.csv"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"


# --------------------------------------------------
# 2. Load cleaned tables
# --------------------------------------------------
orders = pd.read_csv(ORDERS_PATH)
customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["status"] = (
    orders["status"].astype(str).str.strip().str.lower()
)

for column in ["order_amount", "discount_pct", "delivery_minutes"]:
    orders[column] = pd.to_numeric(
        orders[column], errors="coerce"
    )

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

customers["city"] = (
    customers["city"].astype("string").str.strip().str.title()
)
customers["gender"] = (
    customers["gender"].astype("string").str.strip().str.lower()
)

# Keep only the fields needed for the category join.
products = products[["product_id", "category"]].copy()


# --------------------------------------------------
# 3. Define the cohort and lapse label
# --------------------------------------------------
completed = orders.loc[orders["status"] == "completed"].copy()
cancelled = orders.loc[orders["status"] == "cancelled"].copy()

# Only completed orders strictly before the cutoff
# are eligible for model features.
completed_before = completed.loc[
    completed["order_ts"] < CUTOFF
].copy()

# A customer enters the cohort only if they had at
# least one completed order before the cutoff.
cohort_ids = completed_before["customer_id"].unique()

# Lapse label uses activity in September only.
september_completed = completed.loc[
    (completed["order_ts"] >= CUTOFF)
    & (completed["order_ts"] < LABEL_END)
]

september_active_ids = set(
    september_completed["customer_id"].unique()
)

cohort = pd.DataFrame({"customer_id": cohort_ids})
cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_active_ids)
).astype(int)


# --------------------------------------------------
# 4. Engineer behaviour features using pre-cutoff data
# --------------------------------------------------
last_order = (
    completed_before.groupby("customer_id")["order_ts"]
    .max()
    .rename("last_order_ts")
)

recency = (
    (CUTOFF - last_order).dt.days.rename("recency")
)

frequency = (
    completed_before.groupby("customer_id")["order_id"]
    .nunique()
    .rename("frequency")
)

monetary = (
    completed_before.groupby("customer_id")["order_amount"]
    .sum(min_count=1)
    .rename("monetary")
)

avg_discount = (
    completed_before.groupby("customer_id")["discount_pct"]
    .mean()
    .rename("avg_discount_pct")
)

avg_delivery = (
    completed_before.groupby("customer_id")["delivery_minutes"]
    .mean()
    .rename("avg_delivery_minutes")
)

# Cancellation rate uses orders with a final status
# before the cutoff: completed or cancelled.
final_before = orders.loc[
    (orders["order_ts"] < CUTOFF)
    & orders["status"].isin(["completed", "cancelled"])
].copy()

final_counts = final_before.groupby("customer_id").size()
cancel_counts = cancelled.loc[
    cancelled["order_ts"] < CUTOFF
].groupby("customer_id").size()

cancellation_rate = (
    cancel_counts.div(final_counts)
    .fillna(0)
    .rename("cancellation_rate")
)

# Most frequently purchased category among completed
# pre-cutoff orders. Ties are resolved alphabetically.
category_orders = completed_before.merge(
    products, on="product_id", how="left"
)

favourite_category = (
    category_orders.groupby("customer_id")["category"]
    .agg(
        lambda values: (
            values.dropna().value_counts()
            .sort_index()
            .sort_values(ascending=False)
            .index[0]
            if not values.dropna().empty
            else "Unknown"
        )
    )
    .rename("favourite_category")
)

behaviour = pd.concat(
    [
        recency,
        frequency,
        monetary,
        avg_discount,
        avg_delivery,
        cancellation_rate,
        favourite_category,
    ],
    axis=1,
)


# --------------------------------------------------
# 5. Add clean customer attributes and tenure
# --------------------------------------------------
customer_features = customers[
    ["customer_id", "city", "signup_date", "age", "gender"]
].copy()

customer_features["age"] = pd.to_numeric(
    customer_features["age"], errors="coerce"
)

customer_features["tenure_days"] = (
    CUTOFF - customer_features["signup_date"]
).dt.days

customer_features = customer_features.drop(
    columns=["signup_date"]
)

features = (
    cohort
    .merge(behaviour, on="customer_id", how="left")
    .merge(customer_features, on="customer_id", how="left")
)

# Ensure every model feature has a value.
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "avg_delivery_minutes",
    "cancellation_rate",
    "tenure_days",
    "age",
]

for column in numeric_features:
    features[column] = pd.to_numeric(
        features[column], errors="coerce"
    ).fillna(0)

categorical_features = [
    "city",
    "gender",
    "favourite_category",
]

for column in categorical_features:
    features[column] = (
        features[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

# Keep a stable, documented column order.
output_columns = [
    "customer_id",
    "lapsed",
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days",
]

features = features[output_columns]

# Save the engineered dataset for inspection and reuse.
output_path = OUTPUT_DIR / "lapse_features.csv"
features.to_csv(output_path, index=False)


# --------------------------------------------------
# 6. Validate assignment checkpoints
# --------------------------------------------------
print("\nLAPSE COHORT CHECKPOINTS")
print("------------------------")
print(f"Cohort customers: {len(features)}")
print(f"Lapsed customers: {int(features['lapsed'].sum())}")
print(
    f"Lapse rate: {features['lapsed'].mean() * 100:.2f}%"
)
print(f"Not lapsed: {int((features['lapsed'] == 0).sum())}")
print(f"Feature rows with missing values: {int(features.drop(columns=['customer_id', 'lapsed']).isna().any(axis=1).sum())}")
print(f"Saved: {output_path.relative_to(ROOT)}")

assert len(features) == 1156, "Unexpected cohort size"
assert int(features["lapsed"].sum()) == 443, "Unexpected lapse count"
assert not features.drop(
    columns=["customer_id", "lapsed"]
).isna().any().any(), "Missing model feature values detected"

print("\nLAPSE FEATURE CHECKPOINTS PASSED")
