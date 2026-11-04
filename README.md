
# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

The Instamart Customer Intelligence & Support Copilot is an end-to-end, offline AI/ML capstone project built using synthetic e-commerce data.

The project combines data engineering, exploratory data analysis, machine learning, customer intelligence, LLM-based support ticket classification, policy retrieval, and a guarded refund agent.

It demonstrates how data science and LLM systems can support everyday e-commerce operations, including:

- Understanding customer purchasing behaviour.
- Identifying valuable and at-risk customers.
- Predicting customer lapse.
- Classifying customer support tickets.
- Protecting personally identifiable information (PII).
- Retrieving relevant refund policies.
- Supporting refund decisions through a controlled agent workflow.

**Project characteristics**

- Dataset: Synthetic Instamart-style e-commerce data.
- Currency: Indian Rupees (INR).
- Execution: Offline.
- LLM: Supplied mock LLM.
- External API keys: Not required.
- External network calls: Not required by the mock LLM.

The supplied `generate_data.py` and `mock_llm.py` files are retained unchanged.

---

## 2. Project Architecture

The project is divided into four major parts.

| Part | Component | Purpose |
|---|---|---|
| A | Data Preparation and EDA | Audit, clean, analyse and query data |
| B | Customer Intelligence | RFM segmentation and lapse prediction |
| C | LLM Ticket Triage | Privacy-aware support ticket classification |
| D | RAG and Refund Agent | Policy retrieval and guarded agent execution |

### Overall workflow

```text
Synthetic Data
      |
      v
Data Audit and Cleaning
      |
      v
Exploratory Data Analysis
      |
      v
SQLite Database and SQL Analysis
      |
      v
Customer Segmentation
      |
      v
Customer Lapse Prediction
      |
      v
Privacy-Aware LLM Ticket Triage
      |
      v
Policy Retrieval (RAG)
      |
      v
Guarded Refund Agent
      |
      v
Structured Outputs and Checkpoints
```

---

## 3. Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Data visualisation | Matplotlib |
| Database | SQLite |
| SQL analysis | SQLite |
| Machine learning | Scikit-learn |
| Clustering | K-Means |
| Classification | Logistic Regression, Random Forest |
| Dimensionality reduction | PCA |
| LLM integration | Supplied mock LLM |
| Retrieval | Policy document chunking and retrieval |
| Model persistence | Joblib |
| Testing | Python unittest |
| Version control | Git and GitHub |

---

## 4. Project Setup

### Requirements

- Python 3.11 or later.
- Git.
- VS Code or another Python editor.
- Windows, macOS or Linux.

### Step 1: Clone the repository

```powershell
git clone https://github.com/Gokul-2005-V/instamart-customer-intelligence-copilot.git
cd instamart-customer-intelligence-copilot
```

### Step 2: Create a virtual environment

```powershell
py -m venv .venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

---

## 5. Reproduce the Project

Run the following commands from the repository root.

### Generate synthetic data

```powershell
py generate_data.py
```

Expected generator summary:

```text
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```

### Part A — Data Preparation, SQL and EDA

```powershell
py src/part_a.py
py src/run_sql.py
py src/part_a_eda.py
```

### Part B — Customer Segmentation and Lapse Prediction

```powershell
py src/part_b_segments.py
py src/part_b_lapse.py
py src/part_b_lapse_model.py
```

### Part C — LLM Support Ticket Triage

```powershell
py src/part_c_triage.py
```

### Part D — Policy Retrieval and Refund Agent

```powershell
py src/part_d_rag.py
py src/part_d_agent.py
```

### Run automated tests

```powershell
py -m unittest discover -s tests -v
```

The current automated unittest suite contains two Part A artifact tests. Parts B, C and D also have script-level checkpoints; broader dedicated unit-test coverage remains future work.

---

## 6. Part A — Data Preparation and Exploratory Data Analysis

### Objective

Prepare reliable customer, product and order datasets for downstream analytics and machine learning.

### Raw data audit

The generated dataset contains:

| Dataset | Raw records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 13,084 |
| Support tickets | 164 |

Important raw order-data quality findings:

| Data quality issue | Count |
|---|---:|
| Exact duplicate order rows | 119 |
| Pending orders | 656 |
| Distinct order IDs | 12,309 |
| Nonnumeric order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |

Customer-data quality findings:

- 61 blank ages.
- 6 ages outside the valid 10–100 range.
- 24 distinct raw city spellings.
- 10 nonblank raw gender spellings.
- 30 blank gender values.

### Data cleaning

The cleaning workflow includes:

1. Standardising city names.
2. Standardising gender values.
3. Imputing missing and invalid customer ages using city-level medians.
4. Parsing order timestamps.
5. Removing exact duplicate order rows.
6. Keeping the latest record for each order ID.
7. Imputing missing or unparseable order amounts using product price, quantity and discount.
8. Converting negative delivery durations to missing values.
9. Imputing missing delivery durations for completed orders using product-category medians.
10. Excluding pending orders from the cleaned order dataset.

Cancelled orders retain missing delivery durations.

### Cleaned dataset results

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| Cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders | 0 |
| Completed-order revenue | INR 2,998,668.80 |

The cleaned customer dataset contains six standardised cities and three gender categories: Female, male and unknown.

### SQL analysis

SQLite is used to store and analyse the cleaned datasets.

The SQL analysis covers:

- Customer purchase behaviour.
- Revenue analysis.
- Product performance.
- Customer spending.
- Aggregations and filtering.
- Ranking and window functions.
- Daily cumulative revenue.

The final cumulative completed-order revenue is INR 2,998,668.80.

### EDA findings

Completed order amount statistics:

| Metric | Value |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Q1 | INR 64.80 |
| Q3 | INR 297.00 |
| IQR | INR 232.20 |
| Skewness | 4.61 |

The distribution is positively skewed, with 897 upper outliers using the IQR method.

The correlation between order amount and discount percentage is approximately -0.095.

### Part A outputs

- `data/clean/`
- `data/instamart.db`
- EDA reports and analysis outputs.

Checkpoint results:

```text
ALL CHECKPOINTS PASSED
ALL EDA CHECKPOINTS PASSED
```

---

## 7. Part B — Customer Intelligence and Machine Learning

### Objective

Understand customer purchasing behaviour, identify customer segments and predict customer lapse.

### A. RFM Analysis

RFM represents:

- Recency: How recently a customer purchased.
- Frequency: How frequently a customer purchased.
- Monetary: How much a customer spent.

The analysis uses completed orders and an observation snapshot of:

`2025-10-01`

RFM results:

| Metric | Result |
|---|---:|
| Customers analysed | 1,175 |
| Mean recency | 33.56 days |
| Median recency | 19 days |
| Mean frequency | 9.46 |
| Maximum frequency | 33 |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

### B. K-Means Customer Segmentation

K-Means clustering was evaluated for K values from 2 to 8.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.7606 | 0.4650 |
| 3 | 1268.2346 | 0.4452 |
| 4 | 743.0711 | 0.5255 |
| 5 | 617.3666 | 0.4805 |
| 6 | 503.1706 | 0.4819 |
| 7 | 408.4606 | 0.4163 |
| 8 | 366.5035 | 0.4123 |

K = 4 was selected based on the highest silhouette score.

### Customer segments

| Segment | Customers | Mean recency | Mean frequency | Mean monetary value |
|---|---:|---:|---:|---:|
| Regular | 436 | 28.86 | 3.77 | INR 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | INR 7,129.70 |
| Frequent Loyal | 419 | 8.77 | 19.40 | INR 3,410.02 |
| At Risk | 182 | 109.04 | 2.20 | INR 685.99 |

PCA explained variance:

- First component: 63.43%.
- Second component: 21.80%.
- Combined: 85.23%.

### C. Customer Lapse Prediction

The lapse prediction workflow identifies customers who did not place a completed order during September 2025.

Cohort results:

| Metric | Result |
|---|---:|
| Customers in cohort | 1,156 |
| Lapsed customers | 443 |
| Non-lapsed customers | 713 |
| Lapse rate | 38.32% |
| Missing feature rows | 0 |

The train/test split contains 924 training records and 232 test records.

Models evaluated:

- Logistic Regression.
- Random Forest.

Best cross-validation results:

| Model | Best parameters | CV F1 |
|---|---|---:|
| Logistic Regression | C = 0.1 | 0.6392 |
| Random Forest | max_depth = 3, min_samples_leaf = 5 | 0.6748 |

The baseline accuracy was 0.6164.

The test-set results were also evaluated, and Logistic Regression was selected based on its observed test F1 score.

**Evaluation note:** Selecting a model using test-set performance can introduce optimistic selection bias. A separate validation set should be used for unbiased model selection in a production-quality workflow.

### Part B outputs

- `outputs/customer_rfm.csv`
- `outputs/customer_segments.csv`
- `outputs/kmeans_evaluation.csv`
- `outputs/segment_profiles.csv`
- `outputs/pca_summary.csv`
- `outputs/lapse_features.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

---

## 8. Part C — Privacy-Aware LLM Support Ticket Triage

### Objective

Build an offline LLM-based ticket classification workflow that protects customer information and produces structured support decisions.

### Main capabilities

1. PII detection and redaction.
2. Email, phone number and order ID redaction.
3. Empty-ticket handling.
4. Long-ticket detection.
5. Zero-shot classification.
6. Few-shot classification.
7. Structured JSON output validation.
8. Retry handling for transient failures.
9. Keyword-based fallback.
10. Token budget management.
11. Customer segment-based ticket routing.
12. Human-review routing for invalid or unsupported cases.

### PII redaction results

| PII type | Redactions |
|---|---:|
| Email | 55 |
| Phone | 91 |
| Order IDs | 132 |
| Total | 278 |

### Classification results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets classified | 147 | 147 |
| Accuracy | 85.71% | 100% |
| Tokens used | 12,988 | 22,660 |
| Tokens per classified ticket | 88.35 | 154.15 |

Few-shot classification improved observed accuracy by 14.29 percentage points but used 74.5% more tokens.

The evaluation uses synthetic tickets and a mock LLM; these results do not establish production-level generalisation.

### Error handling

The workflow includes:

- Transient failure retries.
- Invalid JSON handling.
- Keyword fallback.
- Empty-ticket skipping.
- Long-ticket blocking.
- Token budget checks.
- Human-review routing.

A budget demonstration used a 6,000-token budget and classified 68 tickets before the budget was exhausted.

### Ticket routing

The routing workflow includes:

- `human_review`
- `security_team`
- `premium_desk`
- `support_queue`

Fraud-related tickets are routed to the security team. Invalid, empty, long or budget-exhausted cases are sent for human review.

### Part C outputs

- `outputs/part_c_metrics.json`
- `outputs/part_c_redaction_report.json`
- `outputs/part_c_zero_shot_confusion.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

---

## 9. Part D — Policy Retrieval and Guarded Refund Agent

### Objective

Build an offline policy retrieval system and a controlled agent workflow for refund-related support requests.

### A. Retrieval-Augmented Generation (RAG)

The retrieval component:

1. Loads refund policy evidence.
2. Splits policy content into chunks.
3. Evaluates overlapping and non-overlapping chunk strategies.
4. Retrieves relevant policy evidence.
5. Uses the mock LLM to produce grounded responses.
6. Handles questions when sufficient evidence is unavailable.

The RAG evaluation confirmed:

- 15 chunks with overlap.
- 13 chunks without overlap.
- 12/12 expected documents or facts retrieved with overlap.
- 10/12 expected facts retrieved without overlap.
- 2/2 unanswerable questions handled correctly.

### B. Refund Agent

The refund agent uses a controlled tool-based workflow.

Its five tools are:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
5. `escalate_to_human`

The agent follows an iterative tool-execution process and records intermediate state.

### Refund policy rules

- Only completed orders are eligible for refund consideration.
- Delivery time greater than 45 minutes qualifies for a full order-amount refund.
- Delivery time of 45 minutes or less qualifies for zero refund.
- Refunds greater than INR 1,000 require human escalation.
- Invalid or ineligible orders are not refunded.
- Refund execution is guarded by order eligibility and calculated refund values.
- Iteration, token and time limits prevent uncontrolled agent execution.
- Unsupported or unsafe cases use fallback or human handoff.

The refund workflow is simulated. It does not make real payments.

### Agent scenario results

| Scenario | Status | Refund amount |
|---|---|---:|
| S1 | REFUND | INR 128.25 |
| S2 | REFUND | INR 135.00 |
| S3 | ESCALATE | INR 1,999.00 |
| S4 | DENIED | INR 0.00 |
| S5 | DENIED | — |
| S6 | NOT_FOUND | — |
| S7 | FALLBACK_MAX_ITERATIONS | — |

### Safety demonstrations

The agent was tested with:

- Token-limit fallback.
- Timeout fallback.
- Maximum-iteration fallback.
- Resume after checkpoint.
- Prevention of repeated completed tool calls.

The resume demonstration completed a refund of INR 128.25.

The recorded tool sequence across both runs was:

```text
get_order
search_policy
calculate_refund
issue_refund
```

The agent printed:

```text
ALL PART D REFUND AGENT CHECKPOINTS PASSED.
```

### Part D outputs

RAG outputs:

- `outputs/policy_chunks_overlap_10.csv`
- `outputs/policy_chunks_0.csv`
- `outputs/part_d_rag_metrics.json`

Agent outputs:

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `outputs/part_d_agent_resume_demo.json`
- `outputs/part_d_agent_safety_demos.json`
- `state/`

---

## 10. Testing and Validation

The project includes:

- Data cleaning checkpoints.
- EDA checkpoints.
- SQL execution checks.
- Machine learning evaluation.
- Ticket triage evaluation.
- RAG retrieval evaluation.
- Refund agent scenario checks.
- Agent safety demonstrations.
- Python unittest tests.

Run the automated test suite:

```powershell
py -m unittest discover -s tests -v
```

The currently available automated unittest suite contains two tests:

- `test_cleaned_checkpoints`
- `test_sqlite_tables_exist`

Both passed during project verification.

The Part A, Part B, Part C and Part D scripts also contain their own validation or checkpoint workflows.

---

## 11. Project Limitations

1. **Synthetic data:** The project uses generated data rather than real Instamart customer records.
2. **Mock LLM:** LLM outputs are generated using the supplied mock implementation, not a live production model.
3. **Offline execution:** No external API integration or live production service is included.
4. **Model generalisation:** Machine learning results require validation on independent real-world data.
5. **Test-set selection:** Selecting a model using test-set performance can introduce selection bias.
6. **Token budget:** The Part C pre-call token budget check is not a strict hard cap on total tokens consumed.
7. **Customer segments:** The retention list is illustrative and uses customer segment information calculated over the RFM observation window.
8. **Refund execution:** Refund operations are simulated and do not connect to a real payment gateway.
9. **Test coverage:** The automated unittest suite currently covers two Part A artifact checks. More dedicated tests are needed for Parts B, C and D.

---

## 12. Business Impact

This project demonstrates how an e-commerce business can use data and AI systems to improve operational decision-making.

| Business area | Potential impact |
|---|---|
| Customer analytics | Understand customer behaviour and purchasing patterns |
| Customer retention | Identify customers at risk of lapsing |
| Marketing | Support targeted customer engagement |
| Support operations | Classify and route customer tickets |
| Data privacy | Redact sensitive information before LLM processing |
| Customer support | Retrieve relevant refund policy evidence |
| Refund operations | Apply controlled refund rules and escalation |
| Operational safety | Use limits, checkpoints and fallback handling |

The system is an educational prototype demonstrating these capabilities, not a production deployment.

---

## 13. Repository Structure

```text
instamart-customer-intelligence-copilot/
|
|-- data/
|   |-- clean/
|   |-- raw/
|   |-- instamart.db
|
|-- outputs/
|   |-- customer_rfm.csv
|   |-- customer_segments.csv
|   |-- kmeans_evaluation.csv
|   |-- segment_profiles.csv
|   |-- pca_summary.csv
|   |-- lapse_features.csv
|   |-- lapse_model.joblib
|   |-- retention_list.csv
|   |-- triage_results.csv
|   |-- triage_budget_demo.csv
|   |-- part_d_rag_metrics.json
|   |-- agent_runs.jsonl
|   |-- refund_ledger.jsonl
|
|-- src/
|   |-- part_a.py
|   |-- run_sql.py
|   |-- part_a_eda.py
|   |-- part_b_segments.py
|   |-- part_b_lapse.py
|   |-- part_b_lapse_model.py
|   |-- part_c_triage.py
|   |-- part_d_rag.py
|   |-- part_d_agent.py
|
|-- state/
|-- tests/
|-- generate_data.py
|-- mock_llm.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

The repository may contain additional generated artifacts and supporting files beyond those listed above.

---

## 14. Final Summary

The Instamart Customer Intelligence & Support Copilot combines four important areas:

- Data preparation and exploratory analysis.
- Customer segmentation and predictive machine learning.
- Privacy-aware LLM-based support ticket triage.
- Retrieval-augmented generation and a guarded refund agent.

The project demonstrates an end-to-end offline workflow, from synthetic data generation to customer intelligence and controlled AI-assisted support operations.

**The goal is to demonstrate how data science, machine learning and LLM systems can work together to solve practical e-commerce business problems.**
