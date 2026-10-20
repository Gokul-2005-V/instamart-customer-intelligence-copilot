# Instamart Customer Intelligence & Support Copilot

## Overview

A reproducible, offline capstone built on fictional synthetic data. It combines data preparation and exploratory data analysis, customer segmentation, lapse prediction, privacy-aware support ticket triage, policy retrieval, and a guarded refund agent.

All monetary amounts are in INR. The supplied `generate_data.py` and `mock_llm.py` are retained unchanged. The mock LLM requires no API key and makes no external network calls.

## Reproduce

Run the following commands from the repository root with Python 3.11 or newer. On Windows, use `py` in place of `python` if necessary.

### 1. Set up the environment

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 2. Generate the synthetic data

```powershell
py generate_data.py
```

The generator's final line should be:

`generated: customers=1200 products=40 orders_raw=13084 tickets=164`

### 3. Run Part A — Data audit, cleaning, SQL and EDA

```powershell
py src/part_a.py
py src/run_sql.py
py src/part_a_eda.py
```

### 4. Run Part B — Segmentation and lapse prediction

```powershell
py src/part_b_segments.py
py src/part_b_lapse.py
py src/part_b_lapse_model.py
```

### 5. Run Part C — Ticket triage

```powershell
py src/part_c_triage.py
```

### 6. Run automated tests

```powershell
py -m unittest discover -s tests -v
```

The current automated test suite contains two Part A artifact tests. Parts B and C also contain script-level checks, but do not yet have dedicated unit-test coverage.

## Part A — Data Audit, Cleaning, SQL and EDA

### Raw-data audit

The generated source data is fictional. Audit results are saved in `outputs/part_a_audit.json`.

| Audit item | Result |
|---|---:|
| Raw orders | 13,084 rows |
| Exact duplicate order rows | 119 |
| Rows with pending status | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank customer genders | 30 |

The audit JSON records the shape, data type, missing-value count, and missing-value percentage for every column in each raw table. Ticket text is loaded with `keep_default_na=False` when empty strings must be preserved.

### Cleaning rules and validation

- **Customers:** Standardise 24 raw city spellings to six using whitespace trimming and title case. Map gender variants to `male`, `female`, or `unknown`; the 30 blank values become `unknown`. Treat six ages outside 10–100 and 61 blank ages as missing, then impute 67 ages using the city median with `groupby(...).transform('median')`. Parse `signup_date` as a datetime.
- **Orders:** Parse `order_ts`; remove 119 exact duplicate rows; retain the latest timestamp for each `order_id`, removing 656 older records and leaving one row per order. Convert `order_amount` to numeric and impute 112 missing or unparseable amounts after de-duplication using `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. Convert negative delivery durations to missing values. Impute 774 missing or negative delivery durations for completed orders using the product-category median among completed orders. Leave all 1,199 cancelled-order delivery durations missing and remove pending records from the clean table.
- **Products:** Retain the generated product catalogue. Product IDs and prices are used when repairing order amounts.
- **Persistence:** Write cleaned CSV files to `data/clean/` and SQLite tables named `customers`, `products`, and `orders` to `data/instamart.db`. The SQLite `order_ts` column is stored as a timestamp string.

### Cleaned-data checkpoints

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| City counts | Pune 227; Delhi 210; Bengaluru 197; Chennai 191; Hyderabad 191; Mumbai 184 |
| Gender counts | Female 610; male 560; unknown 30 |
| Missing customer ages | 0 |
| Cleaned age range | 18–60 |
| Clean orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending rows | 0 |
| Missing order amounts | 0 |
| Missing delivery times among completed orders | 0 |
| Missing delivery times among cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

`src/part_a.py` validates 13 reference checkpoints. The script's successful run prints `ALL CHECKPOINTS PASSED`.

### SQL analysis

The seven queries are stored in `sql/queries.sql`. `src/run_sql.py` executes them using `pandas.read_sql_query`. Results are saved to `outputs/part_a_sql_results.json`.

| Query | Result |
|---|---|
| Q1 — WHERE then HAVING | 30 customers |
| Q2 — City/category revenue join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 — Products above average catalogue price | 11 products |
| Q4 — Customers with no completed order | 25 customers |
| Q5 — Spend above average | 550 customers; average completed spend per customer with completed orders is INR 2,552.06 |
| Q6 — Ranking window functions | City leaders match the reference. Eight rows have `RANK()` different from `ROW_NUMBER()` because ties share a rank but row numbers remain unique. |
| Q7 — Cumulative daily revenue | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

### pandas cross-check and EDA

`src/part_a_eda.py` saves results to `outputs/part_a_eda.json`. The pivot-table grand total is INR 2,998,668.80, matching SQL.

| City | Completed revenue | Mean order amount | Completed orders |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

Completed order amounts have a mean of INR 269.91, median of INR 135.00, skewness of 4.61, Q1 of INR 64.80, Q3 of INR 297.00, and IQR of INR 232.20. The 1.5 × IQR fences are INR -283.50 and INR 645.30. There are no values below the lower fence and 897 above the upper fence.

The largest absolute correlation is between `order_amount` and `discount_pct`, at approximately -0.095. This is a small correlation and does not establish causation.

## Part B — Customer Segmentation and Lapse Prediction

### RFM analysis and customer segmentation

The RFM snapshot date is 2025-10-01. Only completed orders before the snapshot are used to calculate recency, frequency, and monetary value.

The resulting RFM table contains 1,175 customers.

| RFM metric | Result |
|---|---:|
| Median recency | 19 days |
| Minimum / maximum recency | 0 / 180 days |
| Mean frequency | 9.46 orders |
| Maximum frequency | 33 orders |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

K-means was evaluated for K = 2 through K = 8 using standardised RFM features, `n_init=10`, and `random_state=42`.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.7606 | 0.4650 |
| 3 | 1268.2346 | 0.4452 |
| 4 | 743.0711 | 0.5255 |
| 5 | 617.3666 | 0.4805 |
| 6 | 503.1706 | 0.4819 |
| 7 | 408.4605 | 0.4163 |
| 8 | 366.5035 | 0.4123 |

K = 4 was selected because it achieved the highest silhouette score among the evaluated cluster counts.

### Customer segment profiles

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary value |
|---|---:|---:|---:|---:|
| Regular | 436 | 28.86 | 3.77 | INR 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | INR 7,129.70 |
| Frequent Loyal | 419 | 8.77 | 19.40 | INR 3,410.02 |
| At Risk | 182 | 109.04 | 2.20 | INR 685.99 |

The Premium segment is the cluster with the highest mean monetary value. The PCA summary reports explained variance ratios of 0.6343 and 0.2180 for the first two principal components, totalling 0.8523.

Segment outputs include:

- `outputs/customer_rfm.csv`
- `outputs/customer_segments.csv`
- `outputs/segment_profiles.csv`
- `outputs/kmeans_evaluation.csv`
- `outputs/pca_summary.csv`

### Lapse prediction

The lapse cutoff is 2025-09-01. The target is based on whether a customer placed a completed order during September 1–30, 2025. Model features are derived from data before the cutoff.

| Metric | Result |
|---|---:|
| Customer cohort | 1,156 |
| Lapsed customers | 443 (38.32%) |
| Non-lapsed customers | 713 |
| Missing feature values | 0 |
| Training rows | 924 |
| Test rows | 232 |

A majority-class baseline achieved accuracy of 0.6164 and positive-class F1 of 0.

Logistic Regression and Random Forest were tuned using `GridSearchCV` with F1 as the scoring metric.

| Model | Best parameters | Cross-validation F1 |
|---|---|---:|
| Logistic Regression | `classifier__C=0.1` | 0.6392 |
| Random Forest | `classifier__max_depth=3`, `classifier__min_samples_leaf=5` | 0.6748 |

### Test-set evaluation

| Metric | Tuned Logistic Regression | Tuned Random Forest |
|---|---:|---:|
| Confusion matrix | TN 106, FP 37, FN 17, TP 72 | TN 112, FP 31, FN 23, TP 66 |
| Precision | 0.6606 | 0.6804 |
| Recall | 0.8090 | 0.7416 |
| F1 score | 0.7273 | 0.7097 |

Logistic Regression was selected based on the higher observed test-set F1 score. Because the test set was used to compare the final models, this comparison is not an unbiased final performance estimate; a separate validation set would be preferable for model selection.

### Training-only oversampling

Oversampling was applied only to the training set, balancing its classes from 570 non-lapsed and 354 lapsed customers to 570 examples per class.

| Metric | Tuned Logistic Regression | Oversampled Logistic Regression |
|---|---:|---:|
| True negatives | 106 | 86 |
| False positives | 37 | 57 |
| False negatives | 17 | 8 |
| True positives | 72 | 81 |
| Precision | 0.6606 | 0.5870 |
| Recall | 0.8090 | 0.9101 |
| F1 score | 0.7273 | 0.7137 |

Oversampling increased recall and reduced missed lapses from 17 to 8, at the cost of more false positives and lower precision.

Random Forest feature importances rank frequency (0.352171), monetary value (0.288692), recency (0.114974), cancellation rate (0.053057), and average discount (0.040657) as the five most important features in that model.

The illustrative retention list is saved to `outputs/retention_list.csv`. It contains the top 10 Premium test customers, of whom two were truly lapsed. Because the Premium flag is derived from the full-window RFM segmentation, this list is illustrative and should not be treated as a leakage-free model evaluation.

Other Part B artifacts include `outputs/lapse_features.csv` and `outputs/lapse_model.joblib`. The saved model was reloaded, and its predictions matched the original model's predictions.

## Part C — Privacy-Aware LLM Ticket Triage

### Objective and guardrails

`src/part_c_triage.py` uses the supplied offline mock LLM to classify support tickets and route them to appropriate teams.

- **PII redaction:** Email addresses, phone numbers, and order IDs are redacted before the ticket text enters the model prompt.
- **Input checks:** Empty or whitespace-only text is marked `skipped_empty`. Tickets whose estimated input length exceeds 400 tokens are marked `flagged_long`. Neither is sent to the model.
- **Prompt comparison:** Zero-shot and few-shot prompts are evaluated independently.
- **Retry handling:** Transient errors are retried using jittered backoff. The observed runs recovered all 19 transient failures.
- **Output validation:** Valid JSON category outputs are marked `classified_json`. Recognised categories extracted from non-JSON responses are marked `classified_keyword`. Invalid outputs are marked `flagged_invalid_output`.
- **Token budget:** A separate zero-shot run demonstrates stopping new model calls when the running token total reaches the configured pre-call budget.
- **Business routing:** Tickets are routed to human review, the security team, the Premium desk, or the general support queue.

### PII redaction results

| Redaction type | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

The script asserts that the targeted PII patterns no longer occur in redacted ticket text and in the model prompts.

### Classification results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets in dataset | 164 | 164 |
| Classified tickets | 147 | 147 |
| Accuracy on classified tickets | 85.71% | 100.00% |
| Total model tokens | 12,988 | 22,660 |
| Tokens per classified ticket | 88.35 | 154.15 |

Each comparison run produced 135 `classified_json`, 12 `classified_keyword`, and 9 `flagged_invalid_output` statuses. Four empty tickets were skipped and four long tickets were blocked.

Few-shot prompting improved observed accuracy by 14.29 percentage points, but used 9,672 additional tokens, approximately 74.5% more than zero-shot prompting. The extra examples increase prompt-token consumption. In a production system, this trade-off should be assessed against accuracy, latency, and cost.

These results are based on the supplied synthetic tickets and mock LLM. They do not establish that a real LLM will achieve 100% accuracy.

### Token-budget demonstration

The zero-shot budget demonstration used a 6,000-token pre-call budget. It classified 68 tickets and marked 84 tickets `budget_exhausted`.

The recorded total was 6,004 tokens. The guard checks the running total before a call, so a successful call can push usage beyond the configured threshold. It demonstrates pre-call budget checking rather than a strict hard cap on total token consumption.

### Final routing rules

The final few-shot run uses a token budget of 100,000. Routing follows these rules:

1. `skipped_empty`, `flagged_long`, `flagged_invalid_output`, and `budget_exhausted` tickets go to `human_review` with `NORMAL` priority.
2. Validly classified `fraud` tickets go to `security_team` with `HIGH` priority.
3. Other validly classified tickets for Premium customers go to `premium_desk` with `HIGH` priority.
4. Remaining validly classified tickets go to `support_queue` with `NORMAL` priority.
5. Customers without a matching segment record are treated as non-Premium.

The final CSV has the required columns: `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`. Unavailable category and segment values are left blank.

### Part C outputs

- `outputs/part_c_metrics.json`
- `outputs/part_c_redaction_report.json`
- `outputs/part_c_zero_shot_confusion.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

The Part C script completed its built-in checkpoints, including checks for expected status counts, redaction counts, budget exhaustion, and 164 final routing rows.

## Part D — Policy Retrieval and Guarded Refund Agent

Part D is not yet implemented.

Planned work:

- Build TF-IDF policy retrieval over the supplied policy documents.
- Generate policy-grounded answers using retrieved evidence.
- Implement a tool-using refund agent with explicit eligibility checks and guardrails.
- Add budget, time, and iteration limits with safe fallbacks.
- Implement checkpoint and resume behaviour.
- Test agent decisions, policy grounding, failure handling, and reproducibility.

## Limitations and Next Steps

- All datasets are synthetic, and the mock LLM is offline test infrastructure rather than a production language model.
- No real customer data, external model API, or API key is used.
- Part A has two passing automated artifact tests. Parts B and C have script-level checks but need dedicated automated unit tests.
- Lapse-model results depend on the specified synthetic dataset and temporal cutoffs. Model selection based on the test set may overstate generalisation performance.
- The Part C token-budget guard checks usage before each call and may allow the final successful call to exceed the configured budget.
- The Premium retention list is illustrative because its segment flag is derived from the full-window RFM data.
- Part D remains to be implemented.

Next steps are to add dedicated tests for Parts B and C, finish Part D, run the complete pipeline from a clean environment, and verify the final repository artifacts before release.
