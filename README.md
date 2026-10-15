# Instamart Customer Intelligence & Support Copilot

## Overview

A reproducible, offline capstone built on fictional synthetic data. It combines a trusted data foundation, customer segmentation and lapse prediction, privacy-aware support ticket triage, policy retrieval, and a guarded refund agent. All monetary amounts are INR. The supplied `generate_data.py` and `mock_llm.py` are retained verbatim; the mock LLM makes no network calls and requires no API key.

## Reproduce

Run these commands from the repository root with Python 3.11 or newer. On Windows, use `.venv\\Scripts\\activate`; on macOS/Linux, use `source .venv/bin/activate`.

1. `python -m venv .venv`
2. Activate the virtual environment using the command for your operating system.
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python src/part_a.py`
7. `python src/run_sql.py`
8. `python src/part_a_eda.py`

The generator's final line must be `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Part A validation prints `ALL CHECKPOINTS PASSED`; SQL and EDA scripts print their own pass messages.

## Part A

### Raw-data audit

The generated source data is fictional. The audit results are saved in `outputs/part_a_audit.json`.

| Audit item | Result |
|---|---:|
| Raw orders | 13,084 rows |
| Exact duplicate order rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank customer genders | 30 |

The JSON audit also records the shape, dtype, and missing count/percentage for every column in the three raw tables.

### Cleaning rules and validation

- **Customers:** standardise 24 raw city spellings to six by trimming whitespace and applying title case; map gender variants to `male`, `female`, or `unknown` (30 blank values become `unknown`); treat six ages outside 10–100 plus 61 blank ages as missing and impute 67 ages using the city median with `groupby(...).transform('median')`; parse `signup_date` as a datetime.
- **Orders:** parse `order_ts`; remove 119 exact duplicate rows; retain the latest timestamp for each `order_id` (656 older records removed, leaving one row per order); coerce `order_amount` to numeric and impute 112 missing/unparseable amounts after de-duplication using `round(price_inr × quantity × (1 − discount_pct / 100), 2)`; turn negative delivery durations into missing values (106 negative durations in raw data); impute 774 missing/negative delivery durations for completed orders using the median for the product category among completed orders; leave all 1,199 cancelled-order delivery durations missing; remove pending records from the clean table.
- **Products:** retain the generated product catalogue, whose product IDs and prices are used to repair order amounts.
- **Persistence:** write clean CSVs to `data/clean/` and tables named `customers`, `products`, and `orders` to `data/instamart.db`. The SQLite `order_ts` column is stored as a timestamp string.

Cleaned checkpoints:

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| City counts | Pune 227; Delhi 210; Bengaluru 197; Chennai 191; Hyderabad 191; Mumbai 184 |
| Gender counts | female 610; male 560; unknown 30 |
| Missing customer ages | 0; all ages are between 18 and 60 |
| Clean orders | 12,309 |
| Completed / cancelled | 11,110 / 1,199 |
| Pending rows | 0 |
| Missing order amounts | 0 |
| Missing delivery times among completed orders | 0 |
| Missing delivery times among cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

`src/part_a.py` validates 13 reference checkpoints and prints `ALL CHECKPOINTS PASSED` if all hold.

### SQL analysis

The seven queries are stored in `sql/queries.sql`; `src/run_sql.py` executes them using `pandas.read_sql_query`. Results are saved to `outputs/part_a_sql_results.json`.

| Query | Result |
|---|---|
| Q1 — WHERE then HAVING | 30 customers |
| Q2 — city/category revenue join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 — products above average catalogue price | 11 products |
| Q4 — customers with no completed order | 25 customers |
| Q5 — spend above average | 550 customers; average completed spend per customer with completed orders is INR 2,552.06 |
| Q6 — ranking window functions | City leaders match the reference; 8 rows have `RANK()` different from `ROW_NUMBER()` because tied spend values share a rank but row numbers remain unique. |
| Q7 — cumulative daily revenue | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

### pandas cross-check and EDA

`src/part_a_eda.py` writes its results to `outputs/part_a_eda.json`. The pivot-table grand total is INR 2,998,668.80, matching SQL.

| City | Completed revenue | Mean order amount | Completed orders |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

Completed order amounts have mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, and IQR INR 232.20. The 1.5×IQR fences are INR -283.50 and INR 645.30; there are 0 values below the lower fence and 897 above the upper fence. The largest absolute correlation is between `order_amount` and `discount_pct` (approximately -0.095). This is a small correlation and does not establish causation.

## Part B

Not yet implemented. Next: build the RFM table, compare K-means cluster counts using silhouette score, name the customer segments, and train/evaluate the lapse model without leakage using the specified cutoff.

## Part C

Not yet implemented. This part will use the offline mock model, redact personal information, validate structured JSON output, retry transient errors, enforce a token budget, and route tickets using Part B segments.

## Part D

Not yet implemented. This part will add TF-IDF policy retrieval, policy-grounded answers, a tool-using refund agent, budget/time/iteration fallbacks, and checkpoint/resume.

## Limitations and Next Steps

All data and language-model behaviour are synthetic. The offline mock is deterministic test infrastructure, not a production LLM. Part A is implemented and its data, SQL, and EDA checkpoints pass locally. Next: implement and test Parts B, C, and D in dependency order. No real customer data, external API, or API key is used.
