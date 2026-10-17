
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

ORDERS_PATH = CLEAN_DATA_DIR / "orders.csv"

RFM_PATH = OUTPUT_DIR / "customer_rfm.csv"
SEGMENTS_PATH = OUTPUT_DIR / "customer_segments.csv"
EVALUATION_PATH = OUTPUT_DIR / "kmeans_evaluation.csv"
PROFILE_PATH = OUTPUT_DIR / "segment_profiles.csv"
PCA_PATH = OUTPUT_DIR / "pca_summary.csv"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

RANDOM_STATE = 42
K_VALUES = range(2, 9)
SELECTED_K = 4

RFM_FEATURES = ["recency", "frequency", "monetary"]


# ============================================================
# 2. LOAD AND VALIDATE ORDERS
# ============================================================

def load_orders():
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned orders file not found: {ORDERS_PATH}\n"
            "Run your Part A cleaning pipeline first."
        )

    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["order_ts"],
    )

    required_columns = {
        "order_id",
        "customer_id",
        "order_ts",
        "status",
        "order_amount",
    }

    missing_columns = required_columns - set(orders.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    orders["status"] = (
        orders["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"],
        errors="coerce",
    )

    return orders


# ============================================================
# 3. CALCULATE RFM
# ============================================================

def calculate_rfm(orders):
    # Only completed orders before the snapshot date are used.
    completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].notna()
        & (orders["order_ts"] < SNAPSHOT_DATE)
    ].copy()

    if completed.empty:
        raise ValueError(
            "No completed orders were found before the snapshot date."
        )

    if completed["order_amount"].isna().any():
        raise ValueError(
            "Completed orders contain missing or nonnumeric amounts. "
            "Review the Part A cleaning output."
        )

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_date=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_date"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].copy()

    # Validate RFM features.
    if rfm.empty:
        raise ValueError("RFM calculation produced no customers.")

    if rfm["customer_id"].duplicated().any():
        raise ValueError("RFM contains duplicate customer IDs.")

    if rfm[RFM_FEATURES].isna().any().any():
        raise ValueError("RFM contains missing feature values.")

    if not np.isfinite(rfm[RFM_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("RFM contains infinite feature values.")

    if (rfm["recency"] < 0).any():
        raise ValueError("RFM contains negative recency values.")

    if (rfm["frequency"] < 1).any():
        raise ValueError("RFM contains invalid frequency values.")

    if (rfm["monetary"] < 0).any():
        raise ValueError("RFM contains negative monetary values.")

    rfm.to_csv(RFM_PATH, index=False)

    print("\n" + "=" * 65)
    print("RFM CUSTOMER ANALYSIS")
    print("=" * 65)

    print(f"Snapshot date: {SNAPSHOT_DATE}")
    print(f"Customers: {len(rfm)}")

    print("\nRFM descriptive statistics:")
    print(rfm[RFM_FEATURES].describe().round(2).to_string())

    # Reference checks from the assignment.
    print("\nAssignment reference checks:")

    expected = {
        "customer_count": (len(rfm), 1175),
        "recency_median": (rfm["recency"].median(), 19),
        "recency_min": (rfm["recency"].min(), 0),
        "recency_max": (rfm["recency"].max(), 180),
        "frequency_mean": (rfm["frequency"].mean(), 9.46),
        "frequency_max": (rfm["frequency"].max(), 33),
        "monetary_mean": (rfm["monetary"].mean(), 2552.06),
        "monetary_max": (rfm["monetary"].max(), 14766.55),
    }

    for metric, (actual, target) in expected.items():
        print(
            f"{metric}: actual={actual:.2f}, "
            f"reference={target}"
        )

    print(f"\nSaved RFM dataset: {RFM_PATH}")
    print("RFM CHECKPOINTS PASSED")

    return rfm


# ============================================================
# 4. SCALE RFM FEATURES
# ============================================================

def scale_features(rfm):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[RFM_FEATURES])

    print("\nRFM features standardized successfully.")
    return scaled


# ============================================================
# 5. EVALUATE K-MEANS FOR K=2 THROUGH K=8
# ============================================================

def evaluate_kmeans(scaled_features):
    results = []

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(scaled_features)

        score = silhouette_score(
            scaled_features,
            labels,
        )

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": score,
            }
        )

    evaluation = pd.DataFrame(results)
    evaluation.to_csv(EVALUATION_PATH, index=False)

    print("\n" + "=" * 65)
    print("K-MEANS MODEL EVALUATION")
    print("=" * 65)
    print(evaluation.round(4).to_string(index=False))

    print(f"\nSaved evaluation table: {EVALUATION_PATH}")

    print(
        "\nInterpretation: inertia measures within-cluster "
        "squared distance; lower is generally better, but it "
        "normally decreases as K increases."
    )
    print(
        "Silhouette measures how well-separated clusters are. "
        "Higher scores generally indicate clearer separation."
    )

    # The assignment asks for a choice among K=3, K=4, and K=5
    # based on the elbow, silhouette, and business interpretation.
    print(
        "\nCandidate K values for business review: 3, 4, and 5."
    )
    print(
        f"Selected K={SELECTED_K}; this choice must also be "
        "supported by the observed scores and segment profiles."
    )

    return evaluation


# ============================================================
# 6. ASSIGN BUSINESS-FRIENDLY SEGMENT NAMES
# ============================================================

def name_segments(profile):
    """
    Premium:
        Cluster with the highest mean monetary value.

    Frequent Loyal:
        Remaining cluster with the highest mean frequency.

    At Risk:
        Remaining cluster with the highest mean recency.

    Regular:
        Remaining cluster.
    """
    cluster_ids = profile.index.tolist()

    premium_id = profile["monetary"].idxmax()
    names = {premium_id: "Premium"}

    remaining = [
        cluster_id
        for cluster_id in cluster_ids
        if cluster_id != premium_id
    ]

    frequent_id = profile.loc[remaining, "frequency"].idxmax()
    names[frequent_id] = "Frequent Loyal"
    remaining.remove(frequent_id)

    at_risk_id = profile.loc[remaining, "recency"].idxmax()
    names[at_risk_id] = "At Risk"
    remaining.remove(at_risk_id)

    for cluster_id in remaining:
        names[cluster_id] = "Regular"

    return names, premium_id


# ============================================================
# 7. FIT SELECTED K-MEANS MODEL
# ============================================================

def build_segments(rfm, scaled_features):
    model = KMeans(
        n_clusters=SELECTED_K,
        n_init=10,
        random_state=RANDOM_STATE,
    )

    cluster_ids = model.fit_predict(scaled_features)

    segmented = rfm.copy()
    segmented["segment_id"] = cluster_ids

    profile = (
        segmented.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "count"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .sort_index()
    )

    names, premium_id = name_segments(profile)

    segmented["segment_name"] = segmented["segment_id"].map(names)
    segmented["is_premium"] = segmented["segment_id"].eq(premium_id)

    # Save customer-level segmentation.
    output_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]

    segmented[output_columns].to_csv(
        SEGMENTS_PATH,
        index=False,
    )

    # Save the segment profile.
    profile["segment_name"] = profile.index.map(names)
    profile["is_premium"] = profile.index == premium_id
    profile = profile.reset_index()

    profile = profile[
        [
            "segment_id",
            "segment_name",
            "customer_count",
            "recency",
            "frequency",
            "monetary",
            "is_premium",
        ]
    ]

    profile.to_csv(PROFILE_PATH, index=False)

    print("\n" + "=" * 65)
    print(f"SELECTED MODEL: K-MEANS WITH K={SELECTED_K}")
    print("=" * 65)

    print("\nSegment profiles (RFM values are means):")
    print(profile.round(2).to_string(index=False))

    premium_row = profile.loc[profile["is_premium"]].iloc[0]

    print(f"\nPremium segment ID: {premium_id}")
    print(f"Premium customer count: {int(premium_row['customer_count'])}")
    print(f"Premium mean monetary value: {premium_row['monetary']:.2f}")

    print(f"\nSaved customer segments: {SEGMENTS_PATH}")
    print(f"Saved segment profiles: {PROFILE_PATH}")

    return segmented, profile


# ============================================================
# 8. PCA EXPLAINED VARIANCE
# ============================================================

def run_pca(scaled_features):
    pca = PCA(n_components=2)
    pca.fit(scaled_features)

    ratios = pca.explained_variance_ratio_

    summary = pd.DataFrame(
        {
            "component": ["PC1", "PC2", "Total"],
            "explained_variance_ratio": [
                ratios[0],
                ratios[1],
                ratios.sum(),
            ],
        }
    )

    summary.to_csv(PCA_PATH, index=False)

    print("\nPCA explained variance:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved PCA summary: {PCA_PATH}")


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: calculate RFM directly from cleaned orders.
    orders = load_orders()
    rfm = calculate_rfm(orders)

    # Step 2: standardize features.
    scaled_features = scale_features(rfm)

    # Step 3: compare K=2 through K=8.
    evaluate_kmeans(scaled_features)

    # Step 4: fit selected model and save segments.
    segmented, profile = build_segments(
        rfm,
        scaled_features,
    )

    # Step 5: calculate PCA.
    run_pca(scaled_features)

    # Step 6: final validation.
    assert len(segmented) == len(rfm)
    assert segmented["customer_id"].is_unique
    assert segmented["segment_id"].nunique() == SELECTED_K
    assert segmented["segment_name"].notna().all()
    assert segmented["is_premium"].sum() == int(
        profile.loc[
            profile["is_premium"],
            "customer_count",
        ].iloc[0]
    )

    premium_mean = segmented.loc[
        segmented["is_premium"],
        "monetary",
    ].mean()

    other_means = (
        segmented.loc[~segmented["is_premium"]]
        .groupby("segment_id")["monetary"]
        .mean()
    )

    assert premium_mean >= other_means.max()

    print("\n" + "=" * 65)
    print("PART B SEGMENTATION CHECKPOINTS PASSED")
    print(f"Customers segmented: {len(segmented)}")
    print(f"Number of clusters: {segmented['segment_id'].nunique()}")
    print(f"Premium customers: {int(segmented['is_premium'].sum())}")

    print("\nGenerated files:")
    print(f"  {RFM_PATH}")
    print(f"  {SEGMENTS_PATH}")
    print(f"  {EVALUATION_PATH}")
    print(f"  {PROFILE_PATH}")
    print(f"  {PCA_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
