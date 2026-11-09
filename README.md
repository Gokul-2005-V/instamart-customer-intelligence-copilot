Instamart Customer Intelligence & Support Copilot
# Overview
The Instamart Customer Intelligence & Support Copilot is an end-to-end, offline AI/ML capstone project built using synthetic e-commerce data.
The project combines data engineering, exploratory data analysis, machine learning, customer intelligence, LLM-based support ticket classification, policy retrieval, and a guarded refund agent.
It demonstrates how data science and LLM systems can support everyday e-commerce operations, including:
Understanding customer purchasing behaviour.
Identifying valuable and at-risk customers.
Predicting customer lapse.
Classifying customer support tickets.
Protecting personally identifiable information (PII).
Retrieving relevant refund policies.
Supporting refund decisions through a controlled agent workflow.
Project characteristics
Dataset: Synthetic Instamart-style e-commerce data.
Currency: Indian Rupees (INR).
Execution: Offline.
LLM: Supplied mock LLM.
External API keys: Not required.
External network calls: Not required by the mock LLM.
The supplied `generate_data.py` and `mock_llm.py` files are retained unchanged.
---
2. Project Architecture
The project is divided into four major parts.
Part	Component	Purpose
A	Data Preparation and EDA	Audit, clean, analyse and query data
B	Customer Intelligence	RFM segmentation and lapse prediction
C	LLM Ticket Triage	Privacy-aware support ticket classification
D	RAG and Refund Agent	Policy retrieval and guarded agent execution
Overall workflow
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
3. Technology Stack
Area	Technology
Programming	Python
Data processing	Pandas, NumPy
Data visualisation	Matplotlib
Database	SQLite
SQL analysis	SQLite
Machine learning	Scikit-learn
Clustering	K-Means
Classification	Logistic Regression, Random Forest
Dimensionality reduction	PCA
LLM integration	Supplied mock LLM
Retrieval	Policy document chunking and retrieval
Model persistence	Joblib
Testing	Python unittest
Version control	Git and GitHub
---
4. Project Setup
Requirements
Python 3.11 or later.
Git.
VS Code or another Python editor.
Windows, macOS or Linux.
Step 1: Clone the repository
```powershell
git clone https://github.com/Gokul-2005-V/instamart-customer-intelligence-copilot.git
cd instamart-customer-intelligence-copilot
```
Step 2: Create a virtual environment
```powershell
py -m venv .venv
```
Step 3: Activate the environment
Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```
Step 4: Install dependencies
```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```
---
# Reproduce
Run the following commands from the repository root.
Generate synthetic data
```powershell
py generate_data.py
```
Expected generator summary:
```text
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```
Part A — Data Preparation, SQL and EDA
```powershell
py src/part_a.py
py src/run_sql.py
py src/part_a_eda.py
```
Part B — Customer Segmentation and Lapse Prediction
```powershell
py src/part_b_segments.py
py src/part_b_lapse.py
py src/part_b_lapse_model.py
```
Part C — LLM Support Ticket Triage
```powershell
py src/part_c_triage.py
```
Part D — Policy Retrieval and Refund Agent
```powershell
py src/part_d_rag.py
py src/part_d_agent.py
```
Run automated tests
```powershell
py -m unittest discover -s tests -v
```
The current automated unittest suite contains two Part A artifact tests. Parts B, C and D also have script-level checkpoints; broader dedicated unit-test coverage remains future work.
---
# Part A
Objective
Prepare reliable customer, product and order datasets for downstream analytics and machine learning.
Raw data audit
The generated dataset contains:
Dataset	Raw records
Customers	1,200
Products	40
Orders	13,084
Support tickets	164
Important raw order-data quality findings:
Data quality issue	Count
Exact duplicate order rows	119
Pending orders	656
Distinct order IDs	12,309
Nonnumeric order amounts	120
Negative delivery durations	106
Blank delivery durations	2,541
Customer-data quality findings:
61 blank ages.
6 ages outside the valid 10–100 range.
24 distinct raw city spellings.
10 nonblank raw gender spellings.
30 blank gender values.
Data cleaning
The cleaning workflow includes:
Standardising city names.
Standardising gender values.
Imputing missing and invalid customer ages using city-level medians.
Parsing order timestamps.
Removing exact duplicate order rows.
Keeping the latest record for each order ID.
Imputing missing or unparseable order amounts using product price, quantity and discount.
Converting negative delivery durations to missing values.
Imputing missing delivery durations for completed orders using product-category medians.
Excluding pending orders from the cleaned order dataset.
Cancelled orders retain missing delivery durations.
Cleaned dataset results
Metric	Result
Customers	1,200
Cleaned orders	12,309
Completed orders	11,110
Cancelled orders	1,199
Pending orders	0
Completed-order revenue	INR 2,998,668.80
The cleaned customer dataset contains six standardised cities and three gender categories: Female, male and unknown.
SQL analysis
SQLite is used to store and analyse the cleaned datasets.
The SQL analysis covers:
Customer purchase behaviour.
Revenue analysis.
Product performance.
Customer spending.
Aggregations and filtering.
Ranking and window functions.
Daily cumulative revenue.
The final cumulative completed-order revenue is INR 2,998,668.80.
The seven SQL questions cover:
Q1: 30 customers after `WHERE` and `HAVING` filters.
Q2: 30 city/category revenue rows, totalling INR 2,998,668.80; the largest cell is Bengaluru / Electronics at INR 233,909.45.
Q3: 11 products priced above the average catalogue price.
Q4: 25 customers with no completed order.
Q5: 550 customers above the average customer spend of INR 2,552.06.
Q6: top spenders by city — Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10). Eight rows have `RANK()` different from `ROW_NUMBER()` because ties share a rank, while `ROW_NUMBER()` assigns each row a unique sequential number.
Q7: 183 daily revenue rows, ending at INR 2,998,668.80.
The pandas city-level cross-check reports completed revenue of INR 523,643.40 for Bengaluru (mean INR 276.47; 1,894 orders) and INR 479,058.50 for Mumbai (mean INR 297.00; 1,613 orders).
EDA findings
Completed order amount statistics:
Metric	Value
Mean	INR 269.91
Median	INR 135.00
Q1	INR 64.80
Q3	INR 297.00
IQR	INR 232.20
Skewness	4.61
The distribution is positively skewed, with 897 upper outliers using the IQR method.
The correlation between order amount and discount percentage is approximately -0.095, the largest absolute correlation among the specified pairs. The order amount/quantity correlation is approximately +0.087. These are weak associations and do not establish causation.
The IQR fences are approximately INR -283.50 and INR 645.30. There are 897 observations above the upper fence and none below the lower fence.
EDA interpretation (under 200 words): The data describes completed customer orders across six cities and multiple product categories. Order amounts are strongly right-skewed: the mean (INR 269.91) is substantially higher than the median (INR 135.00), and skewness is about 4.61. The middle 50% of order amounts lie between INR 64.80 and INR 297.00. The IQR method identifies 897 upper outliers and no lower outliers; these may represent unusually large baskets rather than data errors, so they should be investigated rather than automatically deleted. Among the specified numeric variables, order amount and discount percentage have the largest absolute correlation (about -0.095), followed by order amount and quantity (about +0.087). Both relationships are weak, and correlation does not establish causation.
Part A outputs
`data/clean/`
`data/instamart.db`
EDA reports and analysis outputs.
Checkpoint results:
```text
ALL CHECKPOINTS PASSED
ALL EDA CHECKPOINTS PASSED
```
---
# Part B
Objective
Understand customer purchasing behaviour, identify customer segments and predict customer lapse.
A. RFM Analysis
RFM represents:
Recency: How recently a customer purchased.
Frequency: How frequently a customer purchased.
Monetary: How much a customer spent.
The analysis uses completed orders and an observation snapshot of:
`2025-10-01`
RFM results:
Metric	Result
Customers analysed	1,175
Mean recency	33.56 days
Median recency	19 days
Mean frequency	9.46
Maximum frequency	33
Mean monetary value	INR 2,552.06
Maximum monetary value	INR 14,766.55
B. K-Means Customer Segmentation
K-Means clustering was evaluated for K values from 2 to 8.
K	Inertia	Silhouette score
2	1820.7606	0.4650
3	1268.2346	0.4452
4	743.0711	0.5255
5	617.3666	0.4805
6	503.1706	0.4819
7	408.4606	0.4163
8	366.5035	0.4123
K = 4 was selected based on the highest silhouette score.
Customer segments
Segment	Customers	Mean recency	Mean frequency	Mean monetary value
Regular	436	28.86	3.77	INR 1,057.62
Premium	138	24.12	6.77	INR 7,129.70
Frequent Loyal	419	8.77	19.40	INR 3,410.02
At Risk	182	109.04	2.20	INR 685.99
PCA explained variance:
First component: 63.43%.
Second component: 21.80%.
Combined: 85.23%.
C. Customer Lapse Prediction
The lapse prediction workflow identifies customers who did not place a completed order during September 2025.
Cohort results:
Metric	Result
Customers in cohort	1,156
Lapsed customers	443
Non-lapsed customers	713
Lapse rate	38.32%
Missing feature rows	0
The train/test split contains 924 training records and 232 test records.
Models evaluated:
Logistic Regression.
Random Forest.
Best cross-validation results:
Model	Best parameters	CV F1
Logistic Regression	C = 0.1	0.6392
Random Forest	max_depth = 3, min_samples_leaf = 5	0.6748
The baseline accuracy was 0.6164.
The held-out test results were:
Model	TN	FP	FN	TP	Precision (lapsed)	Recall (lapsed)	F1 (lapsed)
Logistic Regression	106	37	17	72	0.6606	0.8090	0.7273
Random Forest	112	31	23	66	0.6804	0.7416	0.7097
Random Forest was selected using the higher cross-validation F1 score (0.6748 versus 0.6392). The held-out test set was reserved for final evaluation. Logistic Regression achieved test F1 of 0.7273, while Random Forest achieved 0.7097. Both results are reported for comparison; test performance was not used to select the model. A false negative means missing a customer who is about to lapse; a false positive means spending an incentive on a customer who may have remained active. The preferred trade-off depends on campaign cost and the value of retaining each customer.
Class-balance experiment: The training set was oversampled only, from 570 not-lapsed / 354 lapsed to 570 / 570. On the unchanged test set, the oversampled Logistic Regression produced TN=86, FP=57, FN=8, TP=81, precision=0.5870, recall=0.9101 and F1=0.7137. Recall increased substantially, while precision decreased: the model catches more lapsing customers but also targets more customers who are not actually lapsed.
Random Forest feature importance (top five):
Feature	Importance
Frequency	0.352171
Monetary	0.288692
Recency	0.114974
Cancellation rate	0.053057
Average discount	0.040657
Frequency and monetary value are the strongest recorded predictors in this fitted forest.
Model-selection protocol: Cross-validation F1 was used to select the Random Forest model. The held-out test set was used only for final evaluation.

Preprocessing evidence:
- Training-only scaler frequency mean: 8.1158.
- Full-cohort frequency mean: 8.0285.

Logistic Regression coefficients:
- Recency: -0.0285.
- Frequency: -1.2811.
- Monetary: +0.0295.
Part B outputs
`outputs/customer_rfm.csv`
`outputs/customer_segments.csv`
`outputs/kmeans_evaluation.csv`
`outputs/segment_profiles.csv`
`outputs/pca_summary.csv`
`outputs/lapse_features.csv`
`outputs/lapse_model.joblib`
`outputs/retention_list.csv`
---
# Part C
Objective
Build an offline LLM-based ticket classification workflow that protects customer information and produces structured support decisions.
Main capabilities
PII detection and redaction.
Email, phone number and order ID redaction.
Empty-ticket handling.
Long-ticket detection.
Zero-shot classification.
Few-shot classification.
Structured JSON output validation.
Retry handling for transient failures.
Keyword-based fallback.
Token budget management.
Customer segment-based ticket routing.
Human-review routing for invalid or unsupported cases.
PII redaction results
PII type	Redactions
Email	55
Phone	91
Order IDs	132
Total	278
Classification results
Metric	Zero-shot	Few-shot
Tickets classified	147	147
Accuracy	85.71%	100%
Tokens used	12,988	22,660
Tokens per classified ticket	88.35	154.15
Few-shot classification improved observed accuracy by 14.29 percentage points but used 74.5% more tokens. In the zero-shot run, 126 of 147 classified tickets were correct. The recorded errors were 11 fraud tickets and 10 technical tickets predicted as billing. The full per-category confusion matrix should be taken from `outputs/part_c_zero_shot_confusion.csv` rather than reconstructed from these aggregate figures.
The evaluation uses synthetic tickets and a mock LLM; these results do not establish production-level generalisation.
Error handling
The full runs processed 156 eligible tickets after the four empty and four long tickets were excluded. Each run produced 135 `classified_json`, 12 `classified_keyword` and 9 `flagged_invalid_output` statuses, for 147 classified tickets. The retry experiment encountered 19 transient failures and recovered all of them.
The workflow includes:
Transient failure retries.
Invalid JSON handling.
Keyword fallback.
Empty-ticket skipping.
Long-ticket blocking.
Token budget checks.
Human-review routing.
A budget demonstration used a 6,000-token budget and classified 68 tickets before the remaining 84 tickets were stopped as `budget_exhausted`. The budget check is performed before a call using accumulated usage from completed calls; consequently, a single response can take the total slightly beyond the configured threshold (the recorded demo total was 6,004 tokens).
Ticket routing
The routing workflow includes:
`human_review`
`security_team`
`premium_desk`
`support_queue`
Fraud-related tickets are routed to the security team. Invalid, empty, long or budget-exhausted cases are sent for human review.
Part C outputs
`outputs/part_c_metrics.json`
`outputs/part_c_redaction_report.json`
`outputs/part_c_zero_shot_confusion.csv`
`outputs/triage_budget_demo.csv`
`outputs/triage_results.csv`
---
# Part D
Objective
Build an offline policy retrieval system and a controlled agent workflow for refund-related support requests.
A. Retrieval-Augmented Generation (RAG)
The retrieval component:
Loads refund policy evidence.
Splits policy content into chunks.
Evaluates overlapping and non-overlapping chunk strategies.
Retrieves relevant policy evidence.
Uses the mock LLM to produce grounded responses.
Handles questions when sufficient evidence is unavailable.
The recorded RAG evaluation summary reports:
15 chunks with 10-word overlap.
13 chunks with zero overlap.
12/12 expected documents/facts retrieved with overlap.
10/12 expected facts retrieved without overlap.
2/2 unanswerable questions handled with the fallback response.
The zero-overlap result demonstrates why chunk overlap matters: information near a chunk boundary can be separated from the context needed to answer a question. The exact failed question and separate expected-document versus expected-fact counts are not included in the recorded summary.
B. Refund Agent
The refund agent uses a controlled tool-based workflow.
Its five tools are:
`get_order`
`search_policy`
`calculate_refund`
`issue_refund`
`escalate_to_human`
The agent follows an iterative tool-execution process and records intermediate state.
Refund policy rules
Only completed orders are eligible for refund consideration.
Delivery time greater than 45 minutes qualifies for a full order-amount refund.
Delivery time of 45 minutes or less qualifies for zero refund.
Refunds greater than INR 1,000 require human escalation.
Invalid or ineligible orders are not refunded.
Refund execution is guarded by order eligibility and calculated refund values.
Iteration, token and time limits prevent uncontrolled agent execution.
Unsupported or unsafe cases use fallback or human handoff.
The refund workflow is simulated. It does not make real payments.
Agent scenario results
Scenario	Status	Refund amount	Model calls	Tokens used
S1	REFUND	INR 128.25	5	1,202
S2	REFUND	INR 135.00	5	1,202
S3	ESCALATE	INR 1,999.00	5	1,189
S4	DENIED	INR 0.00	4	845
S5	DENIED	—	3	517
S6	NOT_FOUND	—	2	192
S7	FALLBACK_MAX_ITERATIONS	—	6	3,360
Safety demonstrations
The agent was tested with:
Token-limit fallback.
Timeout fallback.
Maximum-iteration fallback.
Resume after checkpoint.
Prevention of repeated completed tool calls.
The resume demonstration completed a refund of INR 128.25. The run used 5 model calls and 1,202 tokens across the original run and resume. The token-limit demonstration returned `FALLBACK_TOKEN_LIMIT` (2 calls, 222 tokens); the zero-second timeout demonstration returned `FALLBACK_TIMEOUT` before making a model call (0 calls, 0 tokens).
The recorded tool sequence across both runs was:
```text
get_order
search_policy
calculate_refund
issue_refund
```
ReAct versus plan-and-execute: The implemented ReAct workflow makes decisions incrementally: it observes an order lookup or policy result, then chooses the next tool. Recorded runs used 5 model calls for S1, 3 for S5 and 2 for S6, so this design has a per-step token and latency cost. A plan-and-execute design could ask the model for a complete tool plan once and execute that plan, potentially reducing model calls and tokens for predictable cases. However, an unexpected observation such as an order not being found or an order being cancelled can invalidate a precomputed plan; ReAct can adapt after each observation. For this refund workflow, ReAct is preferable because refund eligibility and escalation depend on actual order and policy observations, while iteration, token and time limits constrain execution.
The agent printed:
```text
ALL PART D REFUND AGENT CHECKPOINTS PASSED.
```
Part D outputs
RAG outputs:
`outputs/policy_chunks_overlap_10.csv`
`outputs/policy_chunks_overlap_0.csv`
`outputs/part_d_rag_metrics.json`
Agent outputs:
`outputs/agent_runs.jsonl`
`outputs/refund_ledger.jsonl`
`outputs/part_d_agent_resume_demo.json`
`outputs/part_d_agent_safety_demos.json`
`state/`
---
10. Testing and Validation
The project includes:
Data cleaning checkpoints.
EDA checkpoints.
SQL execution checks.
Machine learning evaluation.
Ticket triage evaluation.
RAG retrieval evaluation.
Refund agent scenario checks.
Agent safety demonstrations.
Python unittest tests.
Run the automated test suite:
```powershell
py -m unittest discover -s tests -v
```
The currently available automated unittest suite contains two tests:
`test_cleaned_checkpoints`
`test_sqlite_tables_exist`
Both passed during project verification.
The Part A, Part B, Part C and Part D scripts also contain their own validation or checkpoint workflows.
---
# Limitations and Next Steps
Synthetic data: The project uses generated data rather than real Instamart customer records.
Mock LLM: LLM outputs are generated using the supplied mock implementation, not a live production model.
Offline execution: No external API integration or live production service is included.
Model generalisation: Machine learning results require validation on independent real-world data.
Model selection: The model is selected using cross-validation. Further evaluation on independent real-world data is required to assess generalisation.
Token budget: The Part C pre-call token budget check is not a strict hard cap on total tokens consumed.
Customer segments: The retention list is illustrative and uses customer segment information calculated over the RFM observation window.
Refund execution: Refund operations are simulated and do not connect to a real payment gateway.
Test coverage: The automated unittest suite currently covers two Part A artifact checks. More dedicated tests are needed for Parts B, C and D.
RAG evaluation detail: The recorded summary gives aggregate overlap results, but does not identify the exact failed zero-overlap question or separately list expected-document and expected-fact hit counts. The aggregate values are reported without inventing those details.

Next Steps:
- Validate model performance using independent, real-world data.
- Expand automated tests for Parts B, C and D.
- Improve token-budget enforcement and RAG evaluation.
- Integrate real services only with appropriate authorization and safety controls.
Ticket privacy/retry implementation evidence: Redaction totals and recovered retry counts are reported. The README does not reproduce the exact regex assertion output or individual retry log waits; those are implementation-level checks in the source/run logs.
---
12. Business Impact
This project demonstrates how an e-commerce business can use data and AI systems to improve operational decision-making.
Business area	Potential impact
Customer analytics	Understand customer behaviour and purchasing patterns
Customer retention	Identify customers at risk of lapsing
Marketing	Support targeted customer engagement
Support operations	Classify and route customer tickets
Data privacy	Redact sensitive information before LLM processing
Customer support	Retrieve relevant refund policy evidence
Refund operations	Apply controlled refund rules and escalation
Operational safety	Use limits, checkpoints and fallback handling
The system is an educational prototype demonstrating these capabilities, not a production deployment.
---
13. Repository Structure
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
14. Final Summary
The Instamart Customer Intelligence & Support Copilot combines four important areas:
Data preparation and exploratory analysis.
Customer segmentation and predictive machine learning.
Privacy-aware LLM-based support ticket triage.
Retrieval-augmented generation and a guarded refund agent.
The project demonstrates an end-to-end offline workflow, from synthetic data generation to customer intelligence and controlled AI-assisted support operations.
The goal is to demonstrate how data science, machine learning and LLM systems can work together to solve practical e-commerce business problems.