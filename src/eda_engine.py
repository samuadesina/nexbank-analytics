# ================================================================
# src/eda_engine.py
# ================================================================
# CONTEXT:
#   We have processed-data.csv — clean, typed, enriched by Module 05.
#   Now we need to UNDERSTAND what is in it.
#
# THE BUSINESS QUESTION:
#   The Chief Risk Officer at NexBank wants to know:
#     - Where does fraud concentrate — which channels and merchant categories?
#     - How do transaction amounts differ across customer segments?
#     - Are fraud rates statistically independent of merchant category?
#     - Which transactions are anomalous by amount — and which fraud cases
#       are hardest to detect because they look completely normal?
#
# THE ANALOGY:
#   Imagine you just received a report from every branch in the bank.
#   Before presenting to the CRO, you need to read it, find the patterns,
#   and summarise the key findings.
#   EDAEngine reads the data report, finds the patterns, and summarises them.
#
# WHY A CLASS AND NOT JUST FUNCTIONS?
#   Because we need to run multiple types of analysis and keep ALL results.
#   A class stores everything in self.results so any other module can access:
#     engine.results["group_analysis"]  → group stats
#     engine.results["correlation"]     → correlation pairs
#     engine.results["chi_square"]      → independence test
#   Functions would run and throw away results. The class remembers.
#
# DESIGN PRINCIPLE: READ-ONLY
#   EDAEngine never modifies the DataFrame. It only reads and summarises.
#   (Same as DataValidator in Module 05 — analysts inspect, they do not edit.)
# ================================================================

# ── IMPORTS ───────────────────────────────────────────────────────
import sys        # sys: for manipulating Python's module search path
import pathlib    # pathlib: cross-platform file paths

# Walk up from this file's directory until we find config.py
# This makes the import work whether the file is run from any directory
_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd                        # pandas: the core Python data library
import numpy as np                         # numpy: numerical operations
from scipy.stats import chi2_contingency   # chi-square independence test

# Import our settings from config.py
from config import (
    INDUSTRY,              # which industry schema
    DATA_PATH,             # where processed-data.csv lives
    REPORTS_DIR,           # where to save the report
    TOP_N_GROUPS,          # how many top groups to show (8)
    CORRELATION_THRESHOLD, # minimum r to include (0.3)
    logger                 # shared logger
)


# ================================================================
# CLASS 1: EDAEngine
# ================================================================

class EDAEngine:
    """
    Runs Exploratory Data Analysis on the processed banking dataset.

    WHAT IS EDA?
    ─────────────
    EDA (Exploratory Data Analysis) is the process of examining a dataset
    to discover patterns, relationships, and anomalies before building models.
    It was formalised by statistician John Tukey in 1977 and is now standard
    practice at every data-driven company.

    Every data scientist and analyst runs EDA as their FIRST step after
    receiving clean data. It answers the question: "What is in here?"

    WHAT THIS CLASS DOES:
    ──────────────────────
    Six methods, each answering a different business question:
      1. load()             → How many rows/columns? What types?
      2. profile()          → What are the distributions and completeness?
      3. group_analysis()   → How do key metrics vary by category?
      4. correlation()      → Which numeric variables move together?
      5. time_trends()      → How do metrics change over time?
      6. chi_square_test()  → Are fraud rates independent of merchant category?

    METHOD CHAIN PATTERN:
    ──────────────────────
    engine.load().profile().group_analysis().correlation().time_trends().report()

    Each method returns self so they can be chained like this.
    This is the same pattern we used in Module 05 ETL.

    Attributes
    ──────────
    df         pd.DataFrame  the loaded processed data
    results    dict          all analysis outputs (keyed by analysis name)
    num_cols   list[str]     numeric column names (set by load())
    cat_cols   list[str]     categorical column names (set by load())
    _status    str           lifecycle state
    """

    def __init__(self):
        """
        Initialise the EDA engine.

        We do NOT load data here — that is load()'s job.
        This separation allows:
          - Object creation without any I/O
          - Testing without needing a real CSV file
          - Clear lifecycle: ready → loaded → analysed → reported
        """
        self.df       = None    # will hold the DataFrame after load()
        self.results  = {}      # will hold all analysis outputs
        self.num_cols = []      # numeric columns (identified in load())
        self.cat_cols = []      # categorical columns (identified in load())
        self._status  = "ready"

        logger.info(f"EDAEngine initialised — industry: {INDUSTRY}")

    def load(self) -> "EDAEngine":
        """
        Load processed-data.csv and identify column types.

        WHY DO WE REMOVE METADATA COLUMNS?
        ─────────────────────────────────────
        Module 05 added three columns starting with _:
          _industry, _processed_at, _pipeline_version
        These describe the pipeline run — NOT the business data.
        Including them in groupby or correlation analysis would add noise.
        We exclude them for analysis but keep the full DataFrame for saving.

        WHY select_dtypes?
        ─────────────────
        select_dtypes(include=["number"]) returns a subset of the DataFrame
        containing ONLY numeric columns (int64, float64).
        select_dtypes(include=["object"]) returns only text/categorical columns.
        This is how pandas separates column types automatically.

        Returns self for method chaining.
        """
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"processed-data.csv not found at: {DATA_PATH}\n"
                "Run Module 05 first:\n"
                "  python run.py\n"
                "Then copy processed-data.csv to data/"
            )

        logger.info(f"[EDA] Loading: {DATA_PATH.name}")

        # pd.read_csv() loads a CSV file from disk into a pandas DataFrame
        # low_memory=False reads the entire file before inferring column types
        self.df = pd.read_csv(DATA_PATH, low_memory=False)

        logger.info(f"[EDA] Loaded {len(self.df):,} rows × {self.df.shape[1]} columns")

        # Remove pipeline metadata columns (start with _) from analysis
        analysis_df = self.df.drop(
            columns=[c for c in self.df.columns if c.startswith("_")],
            errors="ignore"
        )

        # Identify column types using pandas type detection
        self.num_cols = analysis_df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = analysis_df.select_dtypes(include=["object"]).columns.tolist()

        self._status = "loaded"

        logger.info(
            f"[EDA] Column types: "
            f"{len(self.num_cols)} numeric, "
            f"{len(self.cat_cols)} categorical"
        )

        return self

    def profile(self) -> "EDAEngine":
        """
        Compute a complete statistical profile of the dataset.

        WHY PROFILE FIRST?
        ────────────────────
        Before asking "which merchant category has the highest fraud rate?"
        you need to know: "Do we have merchant_category for all transactions,
        or is 5% missing?" The profile gives you confidence in — or warnings
        about — the data before you draw any conclusions.

        WHAT pd.DataFrame.describe() DOES:
        ─────────────────────────────────
        For each numeric column, it computes:
          count  → how many non-null values
          mean   → arithmetic average
          std    → standard deviation (how spread out values are)
          min    → smallest value
          25%    → 25th percentile (first quartile)
          50%    → median (middle value)
          75%    → 75th percentile (third quartile)
          max    → largest value

        WHY IS MEDIAN (50%) OFTEN MORE USEFUL THAN MEAN?
        ──────────────────────────────────────────────────
        Transaction amounts are right-skewed — a few very large transactions
        drag the mean upward. The median gives the genuine middle value.

        Returns self.
        """
        logger.info("[EDA] Computing dataset profile...")

        profile = {
            "rows":             len(self.df),
            "columns":          len(self.df.columns),
            "numeric_cols":     len(self.num_cols),
            "categorical_cols": len(self.cat_cols),

            # Grand total of null cells across the entire DataFrame
            "total_nulls":      int(self.df.isna().sum().sum()),

            # Null as a percentage of all cells
            "null_pct":         round(
                                    self.df.isna().sum().sum() / self.df.size * 100, 2
                                ),

            # Memory consumed by this DataFrame in megabytes
            "memory_mb":        round(
                                    self.df.memory_usage(deep=True).sum() / 1024**2, 2
                                ),

            # Rows that are exact copies of another row
            "duplicates":       int(self.df.duplicated().sum()),
        }

        # Descriptive statistics for numeric columns
        if self.num_cols:
            desc = self.df[self.num_cols].describe().round(3)
            profile["descriptive_stats"] = desc.to_dict()

        # Value counts for categorical columns (top values and their frequencies)
        cat_profiles = {}
        for col in self.cat_cols[:8]:
            vc = self.df[col].value_counts()
            cat_profiles[col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_5":        vc.head(5).to_dict(),
                "null_count":   int(self.df[col].isna().sum()),
            }
        profile["categorical_profiles"] = cat_profiles

        self.results["profile"] = profile

        logger.info(
            f"[EDA] Profile complete — "
            f"{profile['rows']:,} rows | "
            f"{profile['null_pct']}% nulls"
        )

        return self

    def group_analysis(self) -> "EDAEngine":
        """
        Group numeric metrics by categorical columns and compute aggregates.

        WHY THIS IS THE MOST IMPORTANT EDA STEP FOR BANKING:
        ──────────────────────────────────────────────────────
        "What is the average transaction amount?" is a weak question.
        "What is the average transaction amount per merchant category
         and channel?" is a strong question that drives risk decisions.

        For NexBank the CRO cares about:
          - Average amount by channel (mobile vs branch vs ATM)
          - Fraud rate by merchant_category
          - Transaction count by customer segment

        HOW pd.DataFrame.groupby() WORKS:
        ───────────────────────────────────
        df.groupby("channel")["amount"].agg(["mean", "median", "std"])
          → splits the DataFrame into one group per unique channel
          → takes the amount column from each group
          → computes mean, median, std for each group
          → returns a new DataFrame with one row per channel

        Returns self.
        """
        logger.info("[EDA] Running group analysis...")

        group_results = {}

        for cat in self.cat_cols[:2]:
            col_results = {}
            for num in self.num_cols[:3]:

                agg = (
                    self.df.groupby(cat)[num]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .round(2)
                    .reset_index()
                )

                # rank(ascending=False) gives rank 1 to the highest mean
                agg["rank"] = agg["mean"].rank(ascending=False).astype(int)

                col_results[num] = (
                    agg.sort_values("rank")
                       .head(TOP_N_GROUPS)
                       .to_dict(orient="records")
                )

            group_results[cat] = col_results

        # ── Banking-specific: fraud rate by channel and merchant_category ──
        fraud_col = "is_fraud"
        if fraud_col in self.df.columns:
            fraud_groups = {}
            for cat in ["channel", "merchant_category", "segment"]:
                if cat not in self.df.columns:
                    continue
                fraud_rate = (
                    self.df.groupby(cat)[fraud_col]
                    .agg(
                        fraud_count="sum",
                        total_transactions="count",
                    )
                    .reset_index()
                )
                fraud_rate["fraud_rate_pct"] = (
                    fraud_rate["fraud_count"] / fraud_rate["total_transactions"] * 100
                ).round(2)
                fraud_rate = fraud_rate.sort_values("fraud_rate_pct", ascending=False)
                fraud_groups[cat] = fraud_rate.to_dict(orient="records")

            group_results["fraud_by_group"] = fraud_groups

        self.results["group_analysis"] = group_results

        logger.info(f"[EDA] Group analysis: {len(group_results)} grouping variables")

        return self

    def correlation(self) -> "EDAEngine":
        """
        Compute Pearson correlation between all numeric column pairs.

        WHAT IS PEARSON CORRELATION?
        ─────────────────────────────
        The Pearson correlation coefficient (r) measures the LINEAR relationship
        between two numeric variables.

        r ranges from -1 to +1:
          +1.0 → perfect positive relationship
           0.0 → no linear relationship
          -1.0 → perfect negative relationship

        BUSINESS INTERPRETATION FOR BANKING:
        ──────────────────────────────────────
        We are especially interested in:
          - correlation between amount and is_fraud
            (do fraudsters transact at unusual amounts?)
          - correlation between credit_score and fraud rate
            (does credit quality predict fraud exposure?)
          - correlation between balance_after and amount
            (are large transactions depleting accounts?)

        |r| > 0.7 → STRONG
        |r| > 0.5 → MODERATE
        |r| > 0.3 → WEAK
        |r| < 0.3 → NEGLIGIBLE (excluded from report)

        Returns self.
        """
        if len(self.num_cols) < 2:
            logger.warning("[EDA] Not enough numeric columns for correlation")
            return self

        logger.info("[EDA] Computing correlation matrix...")

        corr_matrix = self.df[self.num_cols].corr(numeric_only=True).round(3)

        corr_pairs = []
        for i, col_a in enumerate(self.num_cols):
            for j, col_b in enumerate(self.num_cols):
                if i >= j:
                    continue

                val = corr_matrix.loc[col_a, col_b]

                if abs(val) < CORRELATION_THRESHOLD:
                    continue

                if abs(val) > 0.7:
                    strength = "STRONG"
                elif abs(val) > 0.5:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"

                corr_pairs.append({
                    "col_a":       col_a,
                    "col_b":       col_b,
                    "correlation": float(val),
                    "strength":    strength,
                    "direction":   "positive" if val > 0 else "negative",
                })

        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        self.results["correlation"] = {
            "matrix":       corr_matrix.to_dict(),
            "strong_pairs": corr_pairs[:10],
        }

        logger.info(
            f"[EDA] Correlation: {len(corr_pairs)} meaningful pairs "
            f"(threshold |r| > {CORRELATION_THRESHOLD})"
        )

        return self

    def time_trends(self) -> "EDAEngine":
        """
        Detect time/period columns and compute month-over-month trends.

        WHY TIME TRENDS MATTER FOR BANKING:
        ─────────────────────────────────────
        A single fraud rate is a snapshot. A monthly fraud rate trend is a movie.
        "Fraud rate is 6%" — is that growing or shrinking?
        Are certain months higher risk (e.g. December holiday fraud spikes)?

        METRICS WE COMPUTE FOR EACH TIME PERIOD:
        ──────────────────────────────────────────
          sum          → total transaction value for this period
          mean         → average transaction amount for this period
          mom_change   → mean minus previous period mean (absolute change)
          mom_pct      → percentage change from previous period
          rolling_3m   → 3-period rolling average (smooths short-term noise)

        Returns self.
        """
        time_keywords = ["month", "period", "quarter", "year", "date"]
        time_cols = [
            c for c in self.df.columns
            if any(kw in c.lower() for kw in time_keywords)
            and "date" not in c.lower()
        ]

        if not time_cols:
            # Try to extract month from transaction_date if it exists
            if "transaction_date" in self.df.columns:
                self.df["_month"] = pd.to_datetime(
                    self.df["transaction_date"], errors="coerce"
                ).dt.to_period("M").astype(str)
                time_cols = ["_month"]
            else:
                logger.info("[EDA] No time column found — skipping time trends")
                return self

        time_col = time_cols[0]

        if self.df[time_col].nunique() > 200:
            logger.info(f"[EDA] {time_col} has too many unique values — skipping")
            return self

        logger.info(f"[EDA] Computing time trends on '{time_col}'...")

        trend_results = {}
        for num_col in self.num_cols[:2]:

            monthly = (
                self.df.groupby(time_col)[num_col]
                .agg(["count", "sum", "mean"])
                .round(2)
                .reset_index()
                .sort_values(time_col)
            )

            monthly["mom_change"] = monthly["mean"].diff()
            monthly["mom_pct"]    = monthly["mean"].pct_change().mul(100).round(1)
            monthly["rolling_3m"] = monthly["mean"].rolling(3).mean().round(2)

            trend_results[num_col] = monthly.to_dict(orient="records")

        # ── Banking-specific: fraud count per month ──────────────────
        if "is_fraud" in self.df.columns:
            fraud_trend = (
                self.df.groupby(time_col)["is_fraud"]
                .agg(fraud_count="sum", total="count")
                .reset_index()
                .sort_values(time_col)
            )
            fraud_trend["fraud_rate_pct"] = (
                fraud_trend["fraud_count"] / fraud_trend["total"] * 100
            ).round(2)
            trend_results["fraud_rate"] = fraud_trend.to_dict(orient="records")

        self.results["time_trends"] = trend_results
        logger.info(f"[EDA] Time trends: {len(trend_results)} metrics analysed")

        return self

    def chi_square_test(self, col_a: str = "is_fraud",
                         col_b: str = "merchant_category") -> "EDAEngine":
        """
        Test whether two categorical variables are statistically independent.
        Uses scipy.stats.chi2_contingency().

        WHAT IS THE CHI-SQUARE TEST?
        ─────────────────────────────
        It answers: "Is the pattern I see in this crosstab real, or just noise?"

        H₀ (null hypothesis):      col_a and col_b are INDEPENDENT
                                   (fraud rate is the same across all merchant categories)
        H₁ (alternative):          col_a and col_b are NOT independent
                                   (fraud clusters in specific merchant categories)

        DECISION RULE:
          p < 0.05  → REJECT H₀  — the relationship is statistically significant
          p ≥ 0.05  → FAIL TO REJECT H₀ — the pattern may be noise

        WHY p=0.05?
        ─────────────
        By convention, we accept a 5% chance of being wrong when we reject H₀.
        In a banking fraud context you might use p=0.01 (stricter) to reduce
        false positives before escalating to the risk committee.

        Args:
            col_a   first variable (default: 'is_fraud')
            col_b   second variable (default: 'merchant_category')

        Returns self.
        """
        # Validate columns exist
        for col in [col_a, col_b]:
            if col not in self.df.columns:
                logger.warning(f"[EDA] Chi-square skipped — column '{col}' not found")
                return self

        logger.info(f"[EDA] Chi-square test: '{col_a}' vs '{col_b}'")

        # pd.crosstab() builds a contingency table:
        # rows = unique values of col_a, columns = unique values of col_b
        # each cell = count of rows where that combination occurs
        contingency = pd.crosstab(self.df[col_a], self.df[col_b])

        # chi2_contingency() returns:
        #   chi2   → the test statistic (larger = stronger evidence against H₀)
        #   p      → probability of seeing this result if H₀ were true
        #   dof    → degrees of freedom ((rows-1) × (cols-1))
        #   expected → the expected counts if H₀ were true
        chi2, p, dof, expected = chi2_contingency(contingency)

        result = {
            "col_a":            col_a,
            "col_b":            col_b,
            "chi2_statistic":   round(float(chi2), 4),
            "p_value":          round(float(p), 6),
            "degrees_of_freedom": int(dof),
            "significant":      bool(p < 0.05),
            "conclusion": (
                f"REJECT H₀ — '{col_a}' and '{col_b}' are NOT independent "
                f"(p={p:.4f}). Fraud concentrates in specific {col_b} groups. "
                f"The relationship is statistically significant."
                if p < 0.05 else
                f"FAIL TO REJECT H₀ — no significant relationship between "
                f"'{col_a}' and '{col_b}' detected (p={p:.4f}). "
                f"The groupby pattern from Q1 may be noise."
            ),
            "contingency_table": contingency.to_dict(),
        }

        self.results["chi_square"] = result

        logger.info(
            f"[EDA] Chi-square: χ²={chi2:.2f} | p={p:.4f} | "
            f"significant={p < 0.05}"
        )

        return self

    def report(self, save: bool = True) -> None:
        """
        Print and optionally save the structured analysis report.

        This is the final step — it turns numbers into language the CRO can read.
        A good EDA report:
          - States findings as sentences, not just tables
          - Ranks items (highest to lowest) so the most important comes first
          - Flags anomalies and exceptions explicitly
          - Recommends next steps

        Args:
            save   if True, saves the report to reports/analysis_report.txt
        """
        lines = []

        lines += [
            "═" * 65,
            f"  NEXBANK EDA REPORT  |  INDUSTRY: {INDUSTRY.upper()}",
            "═" * 65,
        ]

        # ── Dataset profile ────────────────────────────────────────────
        if "profile" in self.results:
            p = self.results["profile"]
            lines += [
                "",
                "  DATASET OVERVIEW",
                f"    Records:              {p['rows']:,}",
                f"    Columns:              {p['columns']}",
                f"    Numeric columns:      {p['numeric_cols']}",
                f"    Categorical columns:  {p['categorical_cols']}",
                f"    Missing values:       {p['total_nulls']:,} ({p['null_pct']}%)",
                f"    Duplicate rows:       {p['duplicates']:,}",
                f"    Memory usage:         {p['memory_mb']} MB",
            ]

            if "descriptive_stats" in p:
                lines += ["", "  DESCRIPTIVE STATISTICS"]
                lines.append(
                    f"    {'Metric':<30} {'Mean':>12} {'Median':>12} {'Std Dev':>10}"
                )
                lines.append("    " + "-" * 65)
                for col in list(p["descriptive_stats"].get("mean", {}).keys())[:6]:
                    mean = p["descriptive_stats"].get("mean",  {}).get(col)
                    med  = p["descriptive_stats"].get("50%",   {}).get(col)
                    std  = p["descriptive_stats"].get("std",   {}).get(col)
                    m_s  = f"{mean:>12,.2f}" if isinstance(mean, float) else f"{mean:>12}"
                    md_s = f"{med:>12,.2f}"  if isinstance(med,  float) else f"{med:>12}"
                    st_s = f"{std:>10,.2f}"  if isinstance(std,  float) else f"{std:>10}"
                    lines.append(f"    {col:<30} {m_s} {md_s} {st_s}")

        # ── Fraud by group ─────────────────────────────────────────────
        if "group_analysis" in self.results:
            fraud_groups = self.results["group_analysis"].get("fraud_by_group", {})
            if fraud_groups:
                lines += ["", "  FRAUD RATES BY GROUP"]
                for group_col, rows in fraud_groups.items():
                    lines += [
                        "",
                        f"    {group_col.upper()}",
                        f"    {'Group':<28} {'Fraud Count':>12} {'Total':>10} {'Fraud %':>9}",
                        "    " + "-" * 62,
                    ]
                    for row in rows[:TOP_N_GROUPS]:
                        g  = str(row.get(group_col, ""))[:27]
                        fc = row.get("fraud_count", 0)
                        tt = row.get("total_transactions", 0)
                        fr = row.get("fraud_rate_pct", 0.0)
                        lines.append(f"    {g:<28} {fc:>12,} {tt:>10,} {fr:>8.2f}%")

        # ── Correlation ────────────────────────────────────────────────
        if "correlation" in self.results:
            pairs = self.results["correlation"]["strong_pairs"]
            if pairs:
                lines += [
                    "",
                    f"  TOP CORRELATIONS (|r| > {CORRELATION_THRESHOLD})",
                    f"    {'Column A':<28} {'Column B':<28} {'r':>8}  Strength",
                    "    " + "-" * 70,
                ]
                for pair in pairs[:8]:
                    lines.append(
                        f"    {pair['col_a']:<28} {pair['col_b']:<28} "
                        f"{pair['correlation']:>8.3f}  {pair['strength']} "
                        f"{pair['direction']}"
                    )

        # ── Chi-square result ──────────────────────────────────────────
        if "chi_square" in self.results:
            cs = self.results["chi_square"]
            lines += [
                "",
                "  CHI-SQUARE INDEPENDENCE TEST",
                f"    Variables:   {cs['col_a']} vs {cs['col_b']}",
                f"    χ² statistic: {cs['chi2_statistic']}",
                f"    p-value:      {cs['p_value']}",
                f"    Significant:  {'YES — reject H₀' if cs['significant'] else 'NO — fail to reject H₀'}",
                f"    Conclusion:   {cs['conclusion']}",
            ]

        # ── Time trends ────────────────────────────────────────────────
        if "time_trends" in self.results:
            lines += ["", "  TIME TRENDS (last 6 periods)"]
            for metric, rows in self.results["time_trends"].items():
                if not rows:
                    continue
                period_key = list(rows[0].keys())[0]
                lines += [
                    "",
                    f"    {metric.upper()}",
                    f"    {'Period':<15} {'Mean':>12} {'MoM %':>9} {'Rolling 3':>12}",
                    "    " + "-" * 52,
                ]
                for row in rows[-6:]:
                    period  = str(row.get(period_key, ""))
                    mean_v  = row.get("mean", 0) or row.get("fraud_rate_pct", 0)
                    mom_p   = row.get("mom_pct")  or 0.0
                    rolling = row.get("rolling_3m") or 0.0
                    lines.append(
                        f"    {period:<15} {mean_v:>12,.2f} {mom_p:>8.1f}% "
                        f"{rolling:>12,.2f}"
                    )

        # ── Next steps ─────────────────────────────────────────────────
        lines += [
            "",
            "  NEXT STEPS:",
            "    → Use chi-square findings to prioritise fraud controls by category",
            "    → Pass segment profiles to risk committee for targeted interventions",
            "    → Use correlation findings for ML feature selection (Module 09)",
            "    → Use profile as drift detection baseline (Module 14)",
            "",
            "═" * 65,
        ]

        report_text = "\n".join(lines)
        print(report_text)

        if save:
            report_path = REPORTS_DIR / "analysis_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            logger.info(f"[EDA] Report saved: {report_path}")

    def __str__(self) -> str:
        """Human-readable summary — shown by print(engine)."""
        rows = len(self.df) if self.df is not None else 0
        return (
            f"EDAEngine("
            f"industry={INDUSTRY!r}, "
            f"rows={rows:,}, "
            f"analyses={list(self.results.keys())})"
        )

    def __repr__(self) -> str:
        """Developer representation — shown in debugger."""
        return (
            f"EDAEngine("
            f"industry={INDUSTRY!r}, "
            f"status={self._status!r})"
        )


# ================================================================
# CLASS 2: SegmentProfiler
# ================================================================

class SegmentProfiler:
    """
    Computes per-segment statistics for banking customer analysis.

    BUSINESS CONTEXT:
    ──────────────────
    NexBank serves four customer segments:
      Retail    → standard personal accounts
      Premium   → high-net-worth individuals
      Business  → SME and corporate accounts
      Student   → low-balance, low-limit accounts

    The CRO wants to know:
      - Which segment transacts at the highest average amount?
      - Which segment has the highest fraud exposure?
      - Are Premium customers generating disproportionate fraud losses?

    DESIGN PRINCIPLE: SINGLE RESPONSIBILITY
    ─────────────────────────────────────────
    SegmentProfiler does ONE thing: compute statistics grouped by segment.
    It does not load data, validate data, or produce charts.
    The notebook calls it, reads self.results, and decides what to do next.

    Attributes
    ──────────
    df        pd.DataFrame   the processed dataset (copy — never mutated)
    results   list[dict]     one dict per segment, sorted by fraud_rate_pct desc
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df   the processed DataFrame (from EDAEngine.df or read directly)
        """
        # df.copy() protects the caller's DataFrame from any accidental mutation
        self.df      = df.copy()
        self.results = []

        logger.info(
            f"SegmentProfiler initialised — "
            f"{len(self.df):,} rows"
        )

    def profile(self,
                segment_col: str = "segment",
                amount_col:  str = "amount",
                fraud_col:   str = "is_fraud") -> "SegmentProfiler":
        """
        Compute statistics per customer segment.

        METRICS COMPUTED PER SEGMENT:
        ──────────────────────────────
          transaction_count   how many transactions this segment made
          mean_amount         average transaction value
          median_amount       middle transaction value (robust to outliers)
          total_spend         sum of all transactions — total exposure
          std_amount          spread of transaction values
          fraud_count         absolute number of fraud cases
          fraud_rate_pct      fraud cases as % of total transactions

        WHY SORT BY fraud_rate_pct DESCENDING?
        ────────────────────────────────────────
        The CRO's first question is always "who is most at risk?"
        Sorting by fraud rate puts the highest-risk segment at the top
        of every table and chart without needing extra code in the notebook.

        Args:
            segment_col   column containing segment labels (default: 'segment')
            amount_col    column containing transaction amounts (default: 'amount')
            fraud_col     boolean fraud flag column (default: 'is_fraud')

        Returns self for optional chaining.
        """
        # Validate required columns exist before running
        for col in [segment_col, amount_col, fraud_col]:
            if col not in self.df.columns:
                logger.warning(
                    f"[SEGMENT] Column '{col}' not found — "
                    f"skipping profile"
                )
                return self

        logger.info(f"[SEGMENT] Profiling by '{segment_col}'...")

        profiles = []

        # df.groupby(segment_col) splits the DataFrame into one group per segment
        # We iterate over (segment_name, group_dataframe) pairs
        for segment, group in self.df.groupby(segment_col):

            # group[fraud_col].sum()  counts True values (True = 1 in pandas)
            # group[fraud_col].mean() gives the fraud rate as a decimal (0.06 = 6%)
            profiles.append({
                "segment":            segment,
                "transaction_count":  len(group),
                "mean_amount":        round(float(group[amount_col].mean()),   2),
                "median_amount":      round(float(group[amount_col].median()), 2),
                "total_spend":        round(float(group[amount_col].sum()),    2),
                "std_amount":         round(float(group[amount_col].std()),    2),
                "fraud_count":        int(group[fraud_col].sum()),
                "fraud_rate_pct":     round(float(group[fraud_col].mean()) * 100, 2),
            })

        # Sort: highest fraud rate first — most at-risk segment is always row 0
        self.results = sorted(
            profiles,
            key=lambda x: x["fraud_rate_pct"],
            reverse=True
        )

        logger.info(
            f"[SEGMENT] Profiled {len(self.results)} segments | "
            f"highest fraud rate: "
            f"{self.results[0]['segment']} "
            f"({self.results[0]['fraud_rate_pct']}%)"
            if self.results else "[SEGMENT] No segments found"
        )

        return self

    def save(self, path=None) -> "SegmentProfiler":
        """
        Save segment profiles to CSV.

        Args:
            path   file path to save to. Defaults to reports/segment_profile.csv

        Returns self for chaining.
        """
        if not self.results:
            logger.warning("[SEGMENT] No results to save — run profile() first")
            return self

        save_path = path or (REPORTS_DIR / "segment_profile.csv")
        pd.DataFrame(self.results).to_csv(save_path, index=False)
        logger.info(f"[SEGMENT] Saved: {save_path}")

        return self

    def summary(self) -> str:
        """
        Return a plain-English summary of segment findings.
        Useful for pasting into the analysis_report.txt or notebook markdown.
        """
        if not self.results:
            return "No segment profile computed yet. Run profile() first."

        top    = self.results[0]
        bottom = self.results[-1]

        return (
            f"Segment Analysis ({len(self.results)} segments)\n"
            f"  Highest fraud rate: {top['segment']} "
            f"({top['fraud_rate_pct']}% | "
            f"{top['fraud_count']:,} cases)\n"
            f"  Lowest  fraud rate: {bottom['segment']} "
            f"({bottom['fraud_rate_pct']}% | "
            f"{bottom['fraud_count']:,} cases)\n"
            f"  Highest avg amount: "
            f"{max(self.results, key=lambda x: x['mean_amount'])['segment']} "
            f"(£{max(self.results, key=lambda x: x['mean_amount'])['mean_amount']:,.2f})"
        )

    def __str__(self) -> str:
        return (
            f"SegmentProfiler("
            f"segments={len(self.results)}, "
            f"profiled={'yes' if self.results else 'no'})"
        )

    def __repr__(self) -> str:
        return f"SegmentProfiler(rows={len(self.df):,}, segments={len(self.results)})"


# ================================================================
# QUICK SELF-TEST
# ================================================================
# This block only runs when you execute this file DIRECTLY:
#   python src/eda_engine.py
# It does NOT run when another file imports EDAEngine or SegmentProfiler.
# ================================================================

if __name__ == "__main__":
    print("Running EDAEngine + SegmentProfiler self-test...")
    print("=" * 55)

    # Minimal synthetic banking DataFrame
    test_df = pd.DataFrame({
        "transaction_id":    range(1, 101),
        "amount":            np.random.uniform(10, 5000, 100).round(2),
        "balance_after":     np.random.uniform(0, 20000, 100).round(2),
        "is_fraud":          np.random.choice([True, False], 100, p=[0.06, 0.94]),
        "channel":           np.random.choice(["mobile", "online", "branch", "atm"], 100),
        "merchant_category": np.random.choice(["retail", "food", "travel", "gaming"], 100),
        "segment":           np.random.choice(["Retail", "Premium", "Business", "Student"], 100),
        "transaction_date":  pd.date_range("2025-01-01", periods=100, freq="D").astype(str),
    })

    # Test SegmentProfiler
    profiler = SegmentProfiler(test_df)
    profiler.profile()
    print(profiler.summary())
    print()

    print("Self-test complete.")