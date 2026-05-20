# 🏦 NexBank Analytics — End-to-End Banking Data Project

> Built a production-grade data pipeline and fraud analytics system on a simulated banking dataset — from raw SQL extraction through ETL, data quality validation, exploratory analysis, and statistical hypothesis testing.

---

| | |
|---|---|
| **Client** | NexBank *(simulated)* |
| **Role** | Data Analyst · Risk Analytics |
| **Stack** | Python · PostgreSQL · Supabase · pandas · SciPy · matplotlib · seaborn · SQLAlchemy |
| **Scale** | 50,102 transactions · 5 tables · 40 columns |
| **Type** | Batch ETL · Star Schema · Cloud → Local · EDA · Statistical Testing |
| **Status** | Complete |

---

## Why This Project Exists

Most portfolio projects use pre-cleaned CSV files and skip the hard parts.

This one doesn't.

This project replicates what a junior data analyst actually does in a bank's risk team — write the SQL to extract from a live cloud database, build the pipeline that cleans and validates it, debug it when it fails, and then answer the business questions the Chief Risk Officer is actually asking.

Every bug in this repo is a real bug that showed up in a real pipeline log and had to be diagnosed and fixed.

---

## What I Built

### 1. Star Schema SQL — 5-table extraction from Supabase

Wrote a production SQL query joining `transactions` (fact) to `accounts`, `customers`, `loans`, and `fraud_alerts` across a normalised banking schema.

Key decisions:
- Used `DISTINCT ON (customer_id)` subquery for loans to prevent row fan-out — a silent data duplication bug that inflates row counts without throwing an error
- `fraud_alerts` modelled as a LEFT JOIN enrichment — 94% null is correct, not a data quality failure
- Diagnosed and fixed a `banking.branches` relation error by tracing `branch` to a denormalised column on `accounts`

```sql
FROM      banking.transactions  t
JOIN      banking.accounts      a   ON a.account_id     = t.account_id
JOIN      banking.customers     c   ON c.customer_id    = t.customer_id
LEFT JOIN (
    SELECT DISTINCT ON (customer_id)
        customer_id, loan_id, loan_type, principal,
        outstanding_balance, monthly_payment, risk_grade, status
    FROM  banking.loans
    WHERE status IN ('Current', 'Delinquent')
    ORDER BY customer_id, disbursed_date DESC
) l                               ON l.customer_id     = t.customer_id
LEFT JOIN banking.fraud_alerts  fa  ON fa.transaction_id = t.transaction_id
```

---

### 2. ETL Pipeline — Extract · Validate · Transform · Load &nbsp;·&nbsp; [View on GitHub](https://github.com/samuadesina/banking-etl-pipeline)

```
Supabase (Postgres)
    └── SQL extraction
            └── raw-data.csv  (50,102 rows × 40 cols)
                    └── DataValidator  (5 checks · CRITICAL / WARNING classification)
                            └── DataTransformer
                                    └── processed-data.csv  ✓
```

**`DataValidator`** — custom rules engine, read-only by design:

| Check | Purpose |
|---|---|
| `check_not_empty` | Catches zero-row extracts before anything downstream runs |
| `check_nulls` | Flags columns above null threshold; whitelists LEFT JOIN sparse columns |
| `check_duplicates` | Catches fan-out duplicates and double-loads |
| `check_numeric_ranges` | Catches impossible negatives; whitelists overdraft and delta columns |
| `compute_stats` | Produces full dataset profile for every run |

**Real debugging — from actual pipeline logs:**

| Error | Root cause | Fix |
|---|---|---|
| Loan columns 100% null | `status = 'active'` filter — actual values are `'Current'` / `'Delinquent'` | Queried `DISTINCT status`, updated subquery filter |
| Fraud alert columns flagged CRITICAL | LEFT JOIN sparse columns treated as data errors | Added nullable column whitelist to `check_nulls` |
| `relation "banking.branches" does not exist` | Schema had no branches table | Pulled `branch` from `accounts.branch` directly |

---

### 3. Exploratory Data Analysis — 4 Business Questions

The EDA layer answers what the Chief Risk Officer actually wants to know.

**Q1 — Fraud Patterns**
Where does fraud concentrate — by merchant category, channel, and time period?
Used `groupby` aggregations with fraud rate benchmarked against the dataset average. Monthly trend line to detect seasonal spikes.

**Q2 — Customer Segment Risk Profiling**
Built a `SegmentProfiler` class computing per-segment: mean amount, median amount, total spend, fraud count, and fraud rate — sorted by fraud rate descending so the highest-risk segment is always row 0.

**Q3 — Chi-Square Independence Test**
Formally tested whether fraud rates are independent of merchant category using `scipy.stats.chi2_contingency()`.

```
H₀: fraud is independent of merchant_category
H₁: fraud concentrates in specific categories
Decision: p < 0.05 → reject H₀
```

Visualised as observed vs expected count heatmaps — so the deviation from H₀ is visible, not just a number.

**Q4 — Anomaly Detection (Two Types)**

| Type | Method | What it finds |
|---|---|---|
| `HIGH_VALUE` | IQR + Z-score consensus | Statistically unusual transaction amounts |
| `STEALTH_FRAUD` | `is_fraud=True` but `\|z\| ≤ 2` | Fraud cases that look completely normal — bypass any amount-threshold rule |

The stealth fraud detection is the hard part. A rules-based system misses these entirely.

---

## Technical Depth

**Object-Oriented Design**

Every component is a class with a single responsibility:

```
DataValidator    → inspects only, never modifies
DataTransformer  → modifies only, never inspects
ETLPipeline      → orchestrates only, never touches data directly
EDAEngine        → analyses only, read-only
SegmentProfiler  → profiles segments, single output format
```

Method chaining throughout: `engine.load().profile().group_analysis().correlation().report()`

**Configuration over hardcoding**

Null thresholds, duplicate thresholds, file paths, logging format — all in `config.py`. One file change switches the environment.

**Whitelist pattern for exceptions**

LEFT JOIN sparse columns and overdraft-valid negatives are explicitly whitelisted with documented reasoning. Every exception is a deliberate architectural decision, not a global suppression.

**Defensive copying**

Every class stores `df.copy()` on initialisation. The caller's DataFrame is never mutated by inspection or analysis logic.

---

## Project Structure

```
banking-etl/
├── config.py                        # Shared settings: thresholds, logger, paths
├── data/
│   ├── raw/raw-data.csv             # Extracted from Supabase
│   └── processed/processed-data.csv # Clean pipeline output
├── reports/
│   ├── analysis_report.txt          # EDA text report
│   ├── anomalies.csv                # HIGH_VALUE + STEALTH_FRAUD transactions
│   └── segment_profile.csv          # Per-segment risk statistics
├── notebooks/
│   └── banking_eda.ipynb            # Interactive analysis notebook
└── src/
    ├── etl/
    │   ├── etl_pipeline.py          # Pipeline orchestrator
    │   ├── validator.py             # DataValidator
    │   └── transformer.py           # DataTransformer
    └── eda_engine.py                # EDAEngine + SegmentProfiler
```

---

## Running It

```bash
pip install -r requirements.txt

export SUPABASE_URL=your_url
export SUPABASE_KEY=your_key

# ETL pipeline
python src/etl/etl_pipeline.py

# EDA notebook
jupyter notebook notebooks/banking_eda.ipynb
```

---

## What I'd Add Next

**Incremental loads** — checkpoint the last `transaction_date` so each run only extracts new rows, not the full 50K.

**Great Expectations** — replace the custom validator with a declarative expectation suite that auto-generates an HTML quality report per run.

**Airflow DAG** — wrap extraction and ETL as tasks with retry logic, SLA monitoring, and email alerts on CRITICAL failures.

**dbt models** — move SQL transformations out of pandas and into version-controlled, tested, documented dbt models.



---

## Skills This Project Demonstrates

`SQL` · `Star Schema Design` · `ETL Pipeline Engineering` · `Data Quality Validation` · `Python OOP` · `pandas` · `Exploratory Data Analysis` · `Statistical Hypothesis Testing` · `Fraud Analytics` · `Anomaly Detection` · `Data Visualisation` · `Debugging from Production Logs` · `Supabase` · `PostgreSQL` · `SciPy`

## 👨🏾‍💻 Author

**Samuel Adesina**  
Data Analyst | Python · SQL · Data Engineering  
[GitHub](https://github.com/samueladesina) · ([LinkedIn](https://www.linkedin.com/in/samuadesina/))
