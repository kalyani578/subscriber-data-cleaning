# Tata Play Subscriber Retention & Support Copilot

## Overview

This capstone uses synthetic, fictional data and builds one subscriber-retention copilot in four connected parts. The deliverable is one public GitHub repository on the `main` branch.

- **Part A — Data Foundation:** audit, clean, store, and query the source data.
- **Part B — Churn Prediction:** predict April 2026 cancellations without leakage.
- **Part C — Segments and Revenue at Risk:** group subscribers and estimate spend and risk.
- **Part D — Support Copilot:** classify tickets, extract ticket fields, answer policy questions, and write retention briefs.

All amounts are in Indian Rupees (INR). The supplied `generate_data.py` and `mock_llm.py` files are used unmodified.

## Reproduce

Run these commands from the repository root in Windows PowerShell with Python 3.12 installed:

1. `py -3.12 -m venv .venv`
2. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `.\.venv\Scripts\python.exe generate_data.py`
4. `.\.venv\Scripts\python.exe src\audit_raw.py`
5. `.\.venv\Scripts\python.exe src\part_a.py`
6. `.\.venv\Scripts\python.exe src\part_b.py`
7. `.\.venv\Scripts\python.exe src\part_c.py`
8. `.\.venv\Scripts\python.exe part_d_client_demo.py`
9. `.\.venv\Scripts\python.exe part_d_classifier.py`
10. `.\.venv\Scripts\python.exe part_d_extraction.py`
11. `.\.venv\Scripts\python.exe policy_rag.py`
12. `.\.venv\Scripts\python.exe copilot.py`

The generator's final line is:

```text
generated: subscribers=2415 recharges_raw=13055 tickets=150 policy_files=5
```

## Part A — Data Foundation: Audit, Clean, Store, Query

### Raw-data audit

The audit script reports the shape, pandas data type, missing count, and missing percentage for every raw column.

`subscribers.csv`: 2,415 rows × 17 columns; 15 exact duplicate rows; 2,400 distinct subscriber IDs.

| Column | Type | Missing | Column | Type | Missing |
|---|---:|---:|---|---:|---:|
| subscriber_id | str | 0 (0.00%) | city | str | 0 (0.00%) |
| plan_id | str | 0 (0.00%) | tenure_months | int64 | 0 (0.00%) |
| age | float64 | 99 (4.10%) | payment_mode | str | 116 (4.80%) |
| auto_renew | str | 0 (0.00%) | add_on_count | int64 | 0 (0.00%) |
| monthly_recharge_inr | int64 | 0 (0.00%) | hours_watched_per_week | float64 | 0 (0.00%) |
| hd_hours_per_week | float64 | 0 (0.00%) | sports_hours_per_week | float64 | 0 (0.00%) |
| channels_watched | int64 | 0 (0.00%) | support_calls_90d | int64 | 0 (0.00%) |
| closure_request_logged | str | 0 (0.00%) | next_cycle_recharge_inr | float64 | 516 (21.37%) |
| churned | int64 | 0 (0.00%) |  |  |  |

Other raw subscriber audit figures: 6 ages outside 10–100, 22 raw city spellings, and 6 raw auto-renew spellings.

`recharges.csv`: 13,055 rows × 6 columns; 60 exact duplicate rows; 12,995 distinct recharge IDs; 10 rows reference unknown subscriber IDs; 366 amounts are not plain numbers; 6 raw status spellings.

| Column | Type | Missing |
|---|---:|---:|
| recharge_id | str | 0 (0.00%) |
| subscriber_id | str | 0 (0.00%) |
| recharge_date | str | 0 (0.00%) |
| amount_inr | str | 0 (0.00%) |
| status | str | 0 (0.00%) |
| channel | str | 0 (0.00%) |

### Cleaning rules and results

- Subscribers: removed 15 exact duplicates, leaving 2,400 rows. Trimmed and case-normalized city names, then mapped aliases to the seven standard cities. Mapped auto-renew values to `Yes` or `No`. Set the six ages outside 10–100 to missing. No missing values were filled in Part A.
- Clean city counts: Mumbai 524, Delhi 436, Chennai 391, Bengaluru 386, Hyderabad 262, Pune 245, Kolkata 156.
- Clean auto-renew counts: No 1,355; Yes 1,045. Missing age: 104. Missing payment mode: 114. Churned: 514 of 2,400 (21.42%).
- Recharges: removed 60 exact duplicates and 10 rows with unknown subscriber IDs, leaving 12,985 rows. Lower-cased status, parsed recharge dates, and rebuilt the 364 non-numeric amounts from each subscriber's `monthly_recharge_inr`.
- Clean recharge status counts: success 11,442; failed 1,422; pending 121. Total successful recharge amount: INR 6,457,573.
- Plans were copied unchanged. Clean CSVs are in `data/clean/`; the SQLite database is `data/tataplay.db` with `subscribers`, `recharges`, and `plans` tables.

### SQL results

All results use successful recharges unless stated otherwise. The four queries are in `sql/queries.sql` and are executed by `src/part_a.py`.

**Q1 — successful recharge revenue by plan**

| Plan | Revenue (INR) | Average recharge (INR) | Distinct subscribers |
|---|---:|---:|---:|
| Premium | 1,471,199 | 860.35 | 339 |
| Standard | 1,323,622 | 492.60 | 559 |
| Standard HD | 1,283,471 | 645.61 | 407 |
| Premium Plus | 1,262,398 | 1,075.30 | 228 |
| Basic | 782,554 | 326.61 | 493 |
| Basic Lite | 334,329 | 224.83 | 298 |

A pandas groupby independently confirms Premium revenue of INR 1,471,199.

**Q2 — subscribers who never successfully recharged**

There are 76 such subscribers; 68 have tenure of zero months. An INNER JOIN would discard subscribers without a matching successful-recharge row, hiding them from the result.

**Q3 — rank within each city**

| City | Subscribers with RANK() = 1 |
|---|---:|
| Bengaluru | 10 |
| Chennai | 10 |
| Delhi | 5 |
| Hyderabad | 5 |
| Kolkata | 2 |
| Mumbai | 6 |
| Pune | 4 |

Kolkata's first four rows:

| Subscriber | Total (INR) | RANK | DENSE_RANK | ROW_NUMBER |
|---|---:|---:|---:|---:|
| TP00074 | 6,876 | 1 | 1 | 1 |
| TP00917 | 6,876 | 1 | 1 | 2 |
| TP00143 | 6,582 | 3 | 2 | 3 |
| TP01355 | 6,582 | 3 | 2 | 4 |

Use `ROW_NUMBER()` when exactly one top spender per city is needed: the subscriber-ID tie-breaker selects one row, while `RANK()` and `DENSE_RANK()` include tied top spenders.

**Q4 — TP00010 running total**

| Recharge ID | Date | Amount (INR) | Running total (INR) |
|---|---|---:|---:|
| R000053 | 2025-10-15 | 498 | 498 |
| R000055 | 2025-12-15 | 498 | 996 |
| R000056 | 2026-01-15 | 498 | 1,494 |
| R000057 | 2026-02-15 | 498 | 1,992 |
| R000058 | 2026-03-13 | 498 | 2,490 |

## Part B — Churn Prediction You Can Trust

### Frame the decision and create the modelling table

| Question | Answer |
|---|---|
| What decision does the prediction drive? | Which subscribers should be prioritized for retention contact before the April 2026 cancellation window. |
| Who acts, and what changes? | The retention desk contacts flagged subscribers and considers an eligible retention action. |
| Exact target | `churned` |
| Target type | Binary category: 1 cancelled in April 2026; 0 stayed. |
| Cost of a wrong prediction | Missing a subscriber who leaves costs INR 1,200; an unnecessary offer costs INR 300. |

The class balance is 514 churners out of 2,400 subscribers (21.42%). The model table uses the 12 numeric features `tenure_months`, `age`, `add_on_count`, `monthly_recharge_inr`, `hours_watched_per_week`, `hd_hours_per_week`, `sports_hours_per_week`, `channels_watched`, `support_calls_90d`, `n_success_6m`, `n_failed_6m`, and `recency_days`; and the four categorical features `city`, `plan_id`, `payment_mode`, and `auto_renew`. Recharge counts cover the six-month observation window through 2026-03-31. Subscribers with no recharge rows have zero success/failure counts; `recency_days` remains missing for 76 subscribers.

The split is made in two steps: first hold out the test set, then split the remaining rows into training and validation. Training has 1,440 rows / 308 churners, validation has 480 / 103, and test has 480 / 103. Training fits preprocessing and models; validation compares models and selects thresholds; the test set is evaluated only after those choices.

### Leakage and preprocessing

`closure_request_logged` and `next_cycle_recharge_inr` are not available at prediction time, so neither is used as a model feature. On validation, deliberately including both gives accuracy 0.983 and churn F1 0.961; excluding them gives 0.823 and 0.514 with the default random forest. The leaky score is not shippable: the closure request is recorded while a subscriber is leaving, and next-cycle recharge is only known after the prediction window.

One preprocessing structure is used inside each model pipeline: median imputation then standardization for numeric features, and most-frequent imputation then one-hot encoding with unknown categories ignored for categorical features. Each pipeline is fit on training rows only. The transformed training data has 31 columns and no missing values. The learned median age is 38.0; the scaler's training-only mean tenure is 22.65 months versus 22.46 across all data. They differ because the scaler must learn from training rows alone, not from validation or test rows.

### Baselines, overfitting, and cross-validation

| Validation model | Train accuracy | Validation accuracy | Churn precision | Churn recall | Churn F1 |
|---|---:|---:|---:|---:|---:|
| Always predict stays | - | 0.785 | 0.000 | 0.000 | 0.000 |
| Logistic regression, `max_iter=1000` | 0.833 | 0.833 | 0.646 | 0.495 | 0.560 |
| Unrestricted decision tree | 1.000 | 0.750 | 0.429 | 0.495 | 0.459 |
| Default random forest | 1.000 | 0.823 | 0.625 | 0.437 | 0.514 |

| Decision-tree max depth | Training accuracy | Validation churn F1 |
|---:|---:|---:|
| 1 | 0.815 | 0.434 |
| 2 | 0.832 | 0.413 |
| 3 | 0.839 | 0.408 |
| 4 | 0.844 | 0.422 |
| 5 | 0.863 | 0.529 |
| 6 | 0.883 | 0.545 |
| 7 | 0.908 | 0.488 |
| 8 | 0.924 | 0.489 |
| 9 | 0.944 | 0.468 |
| 10 | 0.957 | 0.440 |

The widening training/validation gap shows that a deeper tree increasingly memorizes training examples. Greedy splits choose locally useful patterns, including noise, so unrestricted depth overfits.

Five-fold stratified cross-validation runs on training rows only (`shuffle=True`, `random_state=42`):

| Model | Grid combinations | Fits | Best parameters | Mean CV F1 | CV F1 standard deviation |
|---|---:|---:|---|---:|---:|
| Logistic regression | 5 | 25 | `C=100` | 0.507 | 0.035 |
| Random forest | 45 | 225 | `max_depth=8`, `max_features=5`, `n_estimators=100` | 0.439 | 0.061 |

Logistic regression is more consistent across folds because its F1 standard deviation is lower. Lower fold-to-fold variation makes its expected performance less dependent on which training examples happen to land in a fold.

### Cost-based threshold and final test evaluation

The threshold search uses validation probabilities only, testing every threshold from 0.10 to 0.90 in 0.05 increments. The random forest's lowest validation cost is INR 53,100 at threshold 0.30; logistic regression's is INR 56,400 at threshold 0.20. The random forest is selected using this validation cost, before test evaluation.

| Model | Threshold | TN | FP | FN | TP | Precision | Recall | F1 | Test cost (INR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random forest (selected) | 0.30 | 310 | 67 | 29 | 74 | 0.525 | 0.718 | 0.607 | 54,900 |
| Logistic regression | 0.20 | 275 | 102 | 17 | 86 | 0.457 | 0.835 | 0.591 | 51,000 |

The no-action test cost is INR 123,600; offering everyone costs INR 113,100. The selected random forest costs 44.4% of the no-action policy and meets the acceptance bars: recall 0.718, F1 0.607, and cost below INR 74,160. It is selected because it had the lower validation cost, even though logistic regression has the lower test cost here. Logistic regression is easier to explain and has higher test recall, while the random forest better minimizes validation cost and its leading risk signals can still be described to the retention desk.

### Explanation and saved outputs

| Top random-forest feature | Importance |
|---|---:|
| `support_calls_90d` | 0.196 |
| `recency_days` | 0.091 |
| `tenure_months` | 0.082 |
| `n_success_6m` | 0.079 |
| `hd_hours_per_week` | 0.067 |

Logistic-regression coefficient readings: `support_calls_90d` is positive (+1.075), so more support calls are associated with higher churn risk; `tenure_months` is negative (-0.511), so longer tenure is associated with lower risk; `n_failed_6m` is positive (+0.211), so more failed recharges are associated with higher risk.

The selected pipeline is saved to `outputs/churn_model.joblib`, reloaded, and verified to reproduce the test probabilities exactly. `outputs/test_scores.csv` contains 480 test subscribers and the columns `subscriber_id`, `churn_probability`, and actual `churned` label. No imputer, scaler, encoder, grid search, or threshold search saw test rows: split first; fit and tune on training; select thresholds on validation; evaluate the test set last.

## Part C — Segments, Spend Model and Revenue at Risk

### K-means clustering and segment profiles

K-means uses only `tenure_months`, `hours_watched_per_week`, and `monthly_recharge_inr`, standardized with `StandardScaler`. The churn label is used only to describe the resulting segments.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 3,728.219 | 0.420 |
| 3 | 2,812.958 | 0.346 |
| 4 | 2,193.933 | 0.340 |
| 5 | 1,781.092 | 0.343 |
| 6 | 1,502.476 | 0.344 |
| 7 | 1,292.830 | 0.364 |
| 8 | 1,191.417 | 0.336 |

K=4 is chosen: the inertia curve begins to flatten around four groups, and four profiles support distinct business actions. Inertia decreases as K grows because more centroids can only reduce each point's distance to its nearest centroid. K=2 has the best silhouette here, but that score alone favors separation rather than useful business detail; neither metric alone determines the final K.

| Segment ID | Business name | Subscribers | Mean tenure (months) | Mean hours/week | Mean recharge (INR) | Churn rate | Recommended action |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | Value Seekers | 593 | 23.58 | 7.08 | 301.33 | 22% | Offer a suitable lower-cost plan or simple recharge support. |
| 1 | Established Viewers | 543 | 34.16 | 22.24 | 559.73 | 13% | Recognize tenure with a relevant loyalty benefit. |
| 2 | Premium | 480 | 34.20 | 33.64 | 942.32 | 11% | Provide priority support and recommend relevant add-ons. |
| 3 | At-Risk | 784 | 6.32 | 14.85 | 513.51 | 33% | Offer a proactive retention contact and check for recharge or support issues. |

Cluster IDs are arbitrary labels, not an ordered ranking. The highest-churn segment is named At-Risk; the highest-recharge segment is Premium.

For a new subscriber (tenure 4 months, 15 hours/week, INR 520 monthly recharge), the standardized distances to cluster IDs 0, 1, 2, and 3 are 1.708, 2.094, 3.117, and 0.153. The hand-computed nearest centroid is segment 3 (At-Risk), matching `kmeans.predict`. `outputs/segments.csv` contains 2,400 rows with `subscriber_id`, `segment_id`, and `segment_name`.

### Collinearity and next-cycle spend model

| Feature | hours_watched_per_week | hd_hours_per_week | sports_hours_per_week | channels_watched |
|---|---:|---:|---:|---:|
| hours_watched_per_week | 1.000 | 0.979 | 0.947 | 0.938 |
| hd_hours_per_week | 0.979 | 1.000 | 0.927 | 0.921 |
| sports_hours_per_week | 0.947 | 0.927 | 1.000 | 0.891 |
| channels_watched | 0.938 | 0.921 | 0.891 | 1.000 |

Pairs with absolute correlation above 0.9 are hours-HD (0.979), hours-sports (0.947), hours-channels (0.938), HD-sports (0.927), and HD-channels (0.921). Sports-channels (0.891) is below the cutoff.

The spend models use only the 1,886 subscribers who stayed (`churned=0`), with a 1,508/378 train/test split. Each pipeline standardizes the seven specified predictors; Ridge and Lasso alpha values are selected using five-fold cross-validation on training rows.

| Model | Best alpha | Mean CV R2 | Test R2 | Test MAE (INR) | Test RMSE (INR) |
|---|---:|---:|---:|---:|---:|
| Linear regression | N/A | 0.994 | 0.993 | 20.157 | 24.701 |
| Ridge | 0.01 | 0.994 | 0.993 | 20.156 | 24.701 |
| Lasso | 0.1 | 0.994 | 0.993 | 20.113 | 24.649 |

Lasso sets `tenure_months`, `hours_watched_per_week`, and `channels_watched` coefficients exactly to zero. Highly correlated features contain overlapping information; Lasso can retain one representative and shrink other redundant coefficients to zero, which does not mean those features have no relationship with spend.

For five 80% training-row subsamples (seeds 0–4), the population standard deviation of the standardized `hours_watched_per_week` coefficient is 1.945 for LinearRegression and 0.896 for Ridge(alpha=30). Ridge's shrinkage makes the coefficient more stable, but the variation also cautions against over-interpreting individual coefficients when predictors are correlated.

### Revenue at risk

The best Ridge model predicts next-cycle recharge for the 480 Part B test subscribers, including subscribers who left. Revenue at risk is `churn_probability × predicted_next_recharge_inr`. The top 15 sum to INR 8,716.53; 12 of these subscribers actually churned. `outputs/retention_priority.csv` contains the required six columns: `subscriber_id`, `churn_probability`, `predicted_next_recharge_inr`, `revenue_at_risk_inr`, `segment_name`, and `churned`.

This priority list uses test subscribers whose outcomes are already known, so it illustrates the method and is not a forward-looking evaluation.

## Part D — Support Copilot

### Reliable LLM client

`llm_client.py` uses the supplied offline mock by default. It retries only HTTP 429 responses, up to three times, with exponential delays; other HTTP errors are raised immediately. A response whose `finish_reason` is `length` produces a truncation warning. Optional live mode is selected with `LLM_MODE=live` and reads its key from `LLM_API_KEY`; the key is never stored in the repository.

Run the four offline checks with `.\.venv\Scripts\python.exe part_d_client_demo.py`:

```text
missing Authorization: 401
wrong URL: 404
max_tokens=3: finish_reason=length; returned text='I am'
conversation memory: Your name is Asha.
30 successful requests; 3 rate limits recovered; 33 total mock calls
```

Retention briefs are still in progress.

### Ticket classification

The classifier uses `ChatPromptTemplate | call_prompt | StrOutputParser()` and sends prompts to the offline mock. It drops the four `...` tickets, runs the remaining 146 tickets with a reset mock counter for each prompt, and recovers 18 rate limits per run.

| Prompt | Valid labels | Correct | Accuracy | Recovered 429s |
|---|---:|---:|---:|---:|
| Weak: `TICKET CLASSIFIER. Classify the ticket.` | 0/146 | 0/146 | 0.00% | 18 |
| Strong zero-shot | 146/146 | 123/146 | 84.25% | 18 |
| Strong few-shot with 13 examples | 146/146 | 142/146 | 97.26% | 18 |

`outputs/ticket_classification.csv` contains the 146 readable tickets with `ticket_id`, `true_category`, `zero_shot`, and `few_shot`.

The four remaining few-shot errors are:

| Ticket | True label | Predicted label | Competing intents in the text |
|---|---|---|---|
| T0041 | `plan_change` | `billing` | Says the last recharge succeeded, then asks to move to Premium. |
| T0094 | `cancellation` | `signal` | Requests cancellation because the picture keeps freezing. |
| T0110 | `signal` | `cancellation` | Explicitly says they do not want cancellation, but reports no signal. |
| T0136 | `signal` | `cancellation` | Explicitly says they do not want cancellation, but reports no signal. |

All four mention two intents, so one-label classification can confuse the incidental issue with the customer's requested action. Add primary and secondary intent fields, and explicitly treat negated requests such as “do not want to cancel” as not an active cancellation. Add contrastive examples for tickets mixing plan changes, billing history, service faults, and cancellation requests, or send ambiguous cases for human review.

Chain-of-thought prompting can be worth the extra tokens when a case requires several reasoning steps or resolving conflicting policy rules and the explanation can be audited. It is not needed for this small fixed-label task: the few-shot prompt reaches 97.26% accuracy, and the mock's labels are driven by explicit intent words rather than generated reasoning.

### Validated ticket extraction

`part_d_extraction.py` uses a Pydantic `TicketRecord` and `PydanticOutputParser`. Urgency `"3"` is converted to integer `3`; urgency `9` fails on field `urgency` with `less_than_equal`, and category `"refund"` fails on field `category` with `literal_error`. The chain sends the exact ticket text as the human message, then retries a parsing or validation failure up to two times by appending the bad assistant reply and a `VALIDATION ERROR:` user message.

| Result across 150 tickets | Count |
|---|---:|
| Valid on first attempt | 118 |
| Valid after one retry | 28 |
| `manual_review` after retries | 4 |
| Rate-limited responses recovered | 23 |

The 32 first-attempt failures comprise 16 invalid-JSON replies (12 wrapped in chatty prose and 4 unreadable `...` tickets) and 16 schema errors (6 urgency values of 7 and 10 out-of-list categories). The 146 valid records include 31 with `wants_refund=true` and 75 with a null `subscriber_id`. The four `manual_review` tickets are T0018, T0059, T0104, and T0141; their JSONL `record` is null after three total attempts.

In production, a `manual_review` ticket should be placed in a human-agent queue with its original text and failure history, and no refund or account action should be triggered from an unvalidated extraction. `outputs/ticket_extractions.jsonl` has one line per input ticket with `ticket_id`, `status`, `attempts`, and the validated record (or null).

### Policy retrieval: chunking

The first retrieval step splits each policy into chunks of 400 characters with 60 characters of overlap. Chunks begin every 340 characters, and the final start position is included only when `start + overlap < len(text)`.

| Policy file | Chunks |
|---|---:|
| `cancellation_policy.txt` | 2 |
| `plan_change_policy.txt` | 3 |
| `recharge_and_refund_policy.txt` | 3 |
| `retention_offers_policy.txt` | 3 |
| `technical_support_policy.txt` | 2 |
| **Total** | **13** |

`policy_rag.py` implements `chunk_text(text, size=400, overlap=60)` and checks these exact counts.

### Policy retrieval, grounding, and calibration

The in-memory Chroma collection uses cosine distance (`metadata={"hnsw:space": "cosine"}`); reported similarity is `1 - distance`. Embeddings use `SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")` with sentence-transformers 6.3.0 and PyTorch 2.15.0+cpu. Retrieval returns the top 3 chunks with their source, text, and similarity.

| Evaluation | Result |
|---|---:|
| Correct policy file in top 3, answerable questions | 12/12 |
| Correct policy file at top 1, answerable questions | 12/12 |
| Answerable top-1 similarity | 0.334–0.637 (mean 0.514) |
| Keyword baseline top-1 accuracy | 9/12 |

Unanswerable-question top-1 similarities, in evaluation-file order, are 0.069, -0.047, 0.295, and 0.304. The closest to an answerable question is “Is international roaming available on the mobile app?” (0.304), because “app” also appears in the technical-support policy even though roaming is not covered.

Embedding search retrieves paraphrases such as “top-up” for “recharge” and “cheaper” for “downgrade,” even when the exact policy words are absent. Keyword overlap is distracted by terms shared across topics, so it can miss conceptually matching files despite their meaning being close.

The rejection threshold is **0.32**, between the lowest answerable similarity (0.334) and the highest unanswerable similarity (0.304). The separation is only 0.030 (about 0.014 above the threshold for the closest answerable and 0.016 below it for the closest unanswerable), so a different embedding build could change a borderline result. The grounded prompt uses only retrieved context; all 12 answerable responses contain the expected fact, and all 4 unanswerable questions receive exactly `I don't have enough information to answer that.` Without grounding, all 16 questions receive the same invented answer: `As of my last update, this is usually processed within 3 working days and the fee is typically INR 500.`

`outputs/rag_eval.csv` stores the 16 questions and the columns `question`, `expected_file`, `top1_source`, `top1_similarity`, `hit_at_3`, `grounded_answer`, and `ungrounded_answer`.

With smaller 200-character chunks and 30-character overlap, the collection has 25 chunks; expected facts appear in the top 3 chunks for 11/12 answerable questions, and 8/12 grounded answers contain the expected fact (versus 12/12 with 400/60 chunks). Smaller chunks sometimes split a rule from its timing or exception, so retrieval may return only a fragment. With less surrounding context, the mock may not find the full fact in the retrieved sentences.

### Retention Copilot

`src/features.py` is shared by Part B and `copilot.py`; it builds the recharge-window features through 2026-03-31. For each of the 15 priority subscribers, the copilot reads tenure and failed-recharge count from SQLite, re-scores with `outputs/churn_model.joblib`, and confirms the saved and recomputed churn probabilities agree to four decimal places. It also reads the Part C segment, retrieves three policy chunks, and validates the JSON response and its consistency with the facts and policy.

All **15/15 briefs** were valid and consistent, and every brief cites `retention_offers_policy.txt`. This run produced **14 High**, **1 Medium**, and **0 Low** risk bands. TP01015 has probability 0.4021 on this build, so it is High at the 0.40 cutoff; the assignment reference notes this subscriber can move between bands on another library build.

| Rule, in priority order | Recommended action | Count |
|---|---|---:|
| High risk and at least 3 failed recharges in six months | Auto-renew setup call and 7-day grace extension | 4 |
| High risk, fewer than 3 failed recharges, and tenure at least 12 months | 15% loyalty discount on the next three recharges | 4 |
| High risk, fewer than 3 failed recharges, and tenure under 12 months | Free add-on pack for 2 months | 6 |
| Medium or Low risk | No retention offer applies; keep monitoring the account | 1 |

`outputs/retention_briefs.jsonl` contains one line per subscriber with the facts used and the validated brief. Every subscriber's band and recommended action are checked against the probability thresholds and policy rule; the selected policy source must also exist in `data/policies/`.

## Limitations and Next Steps

All four parts are implemented. The main limitations are:

- The subscriber, recharge, ticket, and policy-evaluation data are synthetic, so the results do not establish performance on real Tata Play customers.
- Churn features use one fixed snapshot ending 2026-03-31; changing the prediction window or using later data could change both model scores and retention rules.
- The mock LLM is deterministic and offline; it cannot represent the language quality, latency, or failure patterns of a production model.
- The Part C priority list is drawn from the Part B test set, whose outcomes are already known; it demonstrates the workflow, not a forward-looking campaign evaluation.
- The RAG rejection window is only about 0.03 similarity points wide, so a different embedding build could change a borderline answer or risk an incorrect refusal.

Next steps would be to validate the workflows on approved real-world data, evaluate the classifier and retrieval thresholds with support agents, and run a prospective retention pilot before connecting a live model or taking account actions.
