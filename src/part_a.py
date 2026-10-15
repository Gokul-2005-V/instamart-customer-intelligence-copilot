"""Part A: audit and clean the synthetic Instamart data, then build SQLite."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
DB_PATH = ROOT / "data" / "instamart.db"


def audit_frame(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    return {
        "shape": list(df.shape),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing": {
            str(col): {"count": int(missing[col]), "percent": round(float(missing[col] / len(df) * 100), 3)}
            for col in df.columns
        },
    }


def audit_raw(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> dict:
    amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
    age_numeric = pd.to_numeric(customers["age"], errors="coerce")
    gender = customers["gender"].fillna("").astype(str).str.strip()
    age_outside = age_numeric.notna() & ~age_numeric.between(10, 100)
    return {
        "tables": {
            "customers": audit_frame(customers),
            "products": audit_frame(products),
            "orders": audit_frame(orders),
        },
        "figures": {
            "raw_order_rows": int(len(orders)),
            "exact_duplicate_order_rows": int(orders.duplicated().sum()),
            "pending_order_rows": int(orders["status"].astype(str).str.lower().eq("pending").sum()),
            "distinct_order_ids": int(orders["order_id"].nunique()),
            "non_numeric_order_amounts": int(amount_numeric.isna().sum()),
            "negative_delivery_minutes": int(pd.to_numeric(orders["delivery_minutes"], errors="coerce").lt(0).sum()),
            "blank_delivery_minutes": int(orders["delivery_minutes"].isna().sum() + orders["delivery_minutes"].astype(str).str.strip().eq("").sum()),
            "blank_customer_ages": int(customers["age"].isna().sum() + customers["age"].astype(str).str.strip().eq("").sum()),
            "customer_ages_outside_10_100": int(age_outside.sum()),
            "distinct_raw_city_spellings": int(customers["city"].nunique(dropna=True)),
            "distinct_nonblank_raw_gender_spellings": int(gender[gender.ne("")].nunique()),
            "blank_customer_genders": int(gender.eq("").sum()),
        },
    }


def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["city"] = df["city"].astype("string").str.strip().str.title()
    # Standardise all observed spelling/case variants into the three allowed values.
    gender_map = {
        "m": "male", "male": "male", "man": "male", "men": "male",
        "f": "female", "female": "female", "woman": "female", "women": "female",
        "unknown": "unknown", "unspecified": "unknown", "na": "unknown", "n/a": "unknown",
        "none": "unknown", "": "unknown",
    }
    gender_key = df["gender"].fillna("").astype(str).str.strip().str.lower()
    df["gender"] = gender_key.map(gender_map).fillna("unknown")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].between(10, 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df.groupby("city")["age"].transform("median"))
    df["age"] = df["age"].round().astype("int64")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    return df


def clean_orders(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")
    before = len(df)
    df = df.drop_duplicates().copy()
    df.attrs["exact_duplicates_removed"] = before - len(df)
    # Keep the latest record for each order_id. Stable sorting makes tie handling deterministic.
    df = df.sort_values(["order_id", "order_ts"], kind="mergesort")
    df = df.drop_duplicates(subset=["order_id"], keep="last").copy()
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
    product_price = products.set_index("product_id")["price_inr"]
    prices = df["product_id"].map(product_price)
    fill_amount = df["order_amount"].isna()
    df.loc[fill_amount, "order_amount"] = (
        prices[fill_amount] * df.loc[fill_amount, "quantity"] *
        (1 - df.loc[fill_amount, "discount_pct"] / 100)
    ).round(2)
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    completed = df["status"].astype(str).str.lower().eq("completed")
    category_by_product = products.set_index("product_id")["category"]
    df["product_category"] = df["product_id"].map(category_by_product)
    # Median is computed only from valid delivery observations on completed orders.
    completed_medians = df.loc[completed].groupby("product_category")["delivery_minutes"].transform("median")
    median_by_category = (
        df.loc[completed, ["product_category", "delivery_minutes"]]
        .groupby("product_category")["delivery_minutes"].median()
    )
    missing_completed = completed & df["delivery_minutes"].isna()
    df.loc[missing_completed, "delivery_minutes"] = df.loc[missing_completed, "product_category"].map(median_by_category)
    df = df.drop(columns=["product_category"])
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["order_amount"] = df["order_amount"].round(2)
    # A final table should contain only the required persisted fields and no pending records.
    df = df.loc[df["status"].isin(["completed", "cancelled"])].copy()
    df = df.sort_values(["order_ts", "order_id"], kind="mergesort").reset_index(drop=True)
    return df


def write_outputs(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for name, frame in [("customers", customers), ("products", products), ("orders", orders)]:
        export = frame.copy()
        for col in export.columns:
            if pd.api.types.is_datetime64_any_dtype(export[col]):
                export[col] = export[col].dt.strftime("%Y-%m-%d %H:%M:%S" if col == "order_ts" else "%Y-%m-%d")
        export.to_csv(CLEAN / f"{name}.csv", index=False)
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        for name, frame in [("customers", customers), ("products", products), ("orders", orders)]:
            export = frame.copy()
            for col in export.columns:
                if pd.api.types.is_datetime64_any_dtype(export[col]):
                    export[col] = export[col].dt.strftime("%Y-%m-%d %H:%M:%S" if col == "order_ts" else "%Y-%m-%d")
            export.to_sql(name, conn, index=False, if_exists="replace")


def validate(customers: pd.DataFrame, orders: pd.DataFrame) -> None:
    completed = orders[orders["status"].eq("completed")]
    cancelled = orders[orders["status"].eq("cancelled")]
    checks = {
        "clean customer count": len(customers) == 1200,
        "six standardised cities": customers["city"].nunique() == 6,
        "no missing ages": customers["age"].isna().sum() == 0,
        "gender categories constrained": set(customers["gender"].unique()) <= {"male", "female", "unknown"},
        "unique order IDs": orders["order_id"].nunique() == 12309,
        "clean order row count": len(orders) == 12309,
        "completed order count": len(completed) == 11110,
        "cancelled order count": len(cancelled) == 1199,
        "no pending orders": not orders["status"].eq("pending").any(),
        "no missing order amounts": orders["order_amount"].isna().sum() == 0,
        "completed delivery times present": completed["delivery_minutes"].isna().sum() == 0,
        "cancelled delivery times missing": cancelled["delivery_minutes"].isna().sum() == 1199,
        "completed revenue checkpoint": round(float(completed["order_amount"].sum()), 2) == 2998668.80,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError("Checkpoint failures: " + ", ".join(failed))
    print("ALL CHECKPOINTS PASSED")


def main() -> None:
    customers_raw = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders_raw = pd.read_csv(RAW / "orders.csv")
    audit = audit_raw(customers_raw, products, orders_raw)
    (OUTPUTS / "part_a_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    customers = clean_customers(customers_raw)
    orders = clean_orders(orders_raw, products)
    write_outputs(customers, products, orders)
    validate(customers, orders)
    orders_after_exact_dedup = orders_raw.drop_duplicates().copy()
    orders_after_exact_dedup["order_ts"] = pd.to_datetime(orders_after_exact_dedup["order_ts"], errors="coerce")
    orders_after_latest = orders_after_exact_dedup.sort_values(["order_id", "order_ts"], kind="mergesort").drop_duplicates("order_id", keep="last")
    latest_amounts = pd.to_numeric(orders_after_latest["order_amount"], errors="coerce")
    latest_delivery = pd.to_numeric(orders_after_latest["delivery_minutes"], errors="coerce")
    latest_status = orders_after_latest["status"].astype(str).str.lower()
    completed_needing_delivery = latest_status.eq("completed") & (latest_delivery.isna() | latest_delivery.lt(0))
    age_raw = pd.to_numeric(customers_raw["age"], errors="coerce")
    age_imputations = int(age_raw.isna().sum() + (age_raw.notna() & ~age_raw.between(10, 100)).sum())
    summary = {
        "customers": {"rows": len(customers), "cities": customers["city"].value_counts().to_dict(), "genders": customers["gender"].value_counts().to_dict(), "missing_ages": int(customers["age"].isna().sum())},
        "orders": {"rows": len(orders), "statuses": orders["status"].value_counts().to_dict(), "missing_amounts": int(orders["order_amount"].isna().sum()), "missing_delivery_completed": int(orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum()), "missing_delivery_cancelled": int(orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum()), "completed_revenue_inr": round(float(orders.loc[orders["status"].eq("completed"), "order_amount"].sum()), 2)},
        "cleaning_changes": {
            "city_spellings_standardised_from": int(customers_raw["city"].nunique(dropna=True)),
            "city_spellings_after_standardisation": int(customers["city"].nunique()),
            "age_values_imputed": age_imputations,
            "blank_genders_mapped_to_unknown": int(customers_raw["gender"].isna().sum() + customers_raw["gender"].astype(str).str.strip().eq("").sum()),
            "exact_duplicate_order_rows_removed": int(len(orders_raw) - len(orders_raw.drop_duplicates())),
            "older_order_id_records_removed": int(len(orders_after_exact_dedup) - len(orders_after_latest)),
            "order_amount_values_imputed_after_deduplication": int(latest_amounts.isna().sum()),
            "completed_delivery_values_imputed_after_deduplication": int(completed_needing_delivery.sum()),
            "negative_delivery_values_in_raw_data": int(pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce").lt(0).sum()),
        },
    }
    (OUTPUTS / "part_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
