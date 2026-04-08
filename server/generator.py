"""
generator.py — Deterministic dataset generation for all DataGym tasks.

Each generator function produces (dirty_df, ground_truth_df) pairs.
The seed is applied globally before generation so results are fully
reproducible for the same (task_id, seed) pair.

detect_issues() produces human-readable issue descriptions that are
injected into the initial observation's issues_detected field, giving
the agent concrete signal about what needs repairing — without revealing
the exact action sequence required.
"""

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Task generators
# ──────────────────────────────────────────────────────────────────────────────

def generate_task1_easy(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Single-table repair: nulls, mixed types, duplicate rows.

    Ground truth: 200-row employee table (employee_id, department, salary,
                  performance_score).
    Corruptions:
      • 15 % nulls in 'department'
      • 10 % of 'salary' values become "<value> USD" strings
      • 15 duplicate rows injected
    Optimal sequence (≤ 4 steps):
      1. fill_nulls department (mode)
      2. normalize_values salary (strip " USD")
      3. cast_type salary → int
      4. deduplicate
    """
    gt_df = pd.DataFrame({
        "employee_id":       range(1000, 1200),
        "department":        np.random.choice(
                                 ["Engineering", "Sales", "HR", "Marketing"], 200
                             ),
        "salary":            np.random.randint(50_000, 150_000, 200),
        "performance_score": np.random.uniform(1.0, 5.0, 200).round(1),
    })

    dirty = gt_df.copy()

    # Corrupt 1: 15 % nulls in department
    dirty.loc[dirty.sample(frac=0.15, random_state=seed).index, "department"] = np.nan

    # Corrupt 2: mixed salary types
    dirty["salary"] = dirty["salary"].astype(object)
    str_idx = dirty.sample(frac=0.10, random_state=seed + 1).index
    dirty.loc[str_idx, "salary"] = (
        dirty.loc[str_idx, "salary"].astype(str) + " USD"
    )

    # Corrupt 3: 15 duplicate rows
    dupes = dirty.sample(n=15, random_state=seed + 2)
    dirty = pd.concat([dirty, dupes]).reset_index(drop=True)

    return dirty, gt_df


def generate_task2_medium(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-column repair with sequencing constraints.

    Ground truth: 300-row transaction table.
    Corruptions (must be fixed in order):
      • Mixed date formats (DD/MM/YYYY, MM-DD-YY, Unix timestamp)
      • 20 % of 'amount' prefixed with '$'
      • Inconsistent category casing + trailing spaces
      • Outlier injection (amount = 999999.99, 5 rows)
      • 8 % nulls across three columns
    Optimal sequence (≤ 8 steps):
      1. normalize_values transaction_date (unify formats or cast_type)
      2. normalize_values amount (strip '$')
      3. cast_type amount → float
      4. normalize_values category (strip trailing spaces)
      5. normalize_values category (standardise casing)
      6. fill_nulls <columns>
      7. normalize_values amount (replace outliers)
      8. deduplicate (safety)
    """
    base_dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(300)]
    gt_df = pd.DataFrame({
        "transaction_id":   range(1, 301),
        "transaction_date": [d.strftime("%Y-%m-%d") for d in base_dates],
        "amount":           np.random.uniform(10.0, 500.0, 300).round(2),
        "category":         np.random.choice(["Electronics", "Clothing", "Food"], 300),
    })

    dirty = gt_df.copy().astype(object)

    # Corrupt 1: mixed date formats
    rng = random.Random(seed)

    def _mess_date(d_str):
        d = datetime.strptime(d_str, "%Y-%m-%d")
        choice = rng.choice([1, 2, 3])
        if choice == 1:
            return d.strftime("%d/%m/%Y")
        if choice == 2:
            return d.strftime("%m-%d-%y")
        return str(int(d.timestamp()))

    dirty["transaction_date"] = dirty["transaction_date"].apply(_mess_date)

    # Corrupt 2: currency symbols in amount
    curr_idx = dirty.sample(frac=0.20, random_state=seed).index
    dirty.loc[curr_idx, "amount"] = (
        "$" + dirty.loc[curr_idx, "amount"].astype(str)
    )

    # Corrupt 3: inconsistent casing + trailing spaces in category
    def _mangle_cat(x):
        x = str(x)
        x = x.lower() if rng.random() > 0.5 else (x.upper() if rng.random() > 0.5 else x)
        if rng.random() > 0.7:
            x = x + "   "
        return x

    dirty["category"] = dirty["category"].apply(_mangle_cat)

    # Corrupt 4: outlier injection (5 rows)
    dirty.loc[dirty.sample(n=5, random_state=seed + 3).index, "amount"] = 999_999.99

    # Corrupt 5: 8 % nulls across three columns
    for col in ["transaction_date", "amount", "category"]:
        dirty.loc[dirty.sample(frac=0.08, random_state=seed + 4).index, col] = np.nan

    return dirty, gt_df


def generate_task3_hard(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Schema drift reconstruction.

    Ground truth: 400-row product catalogue with 6 clean columns.
    Corruptions (structural, requiring new action types):
      • 'first_name' + 'last_name' merged into 'full_name'
      • 'price_usd' scaled ×1.3 and renamed 'cost'
      • 'tag_1' and 'tag_2' JSONified into 'tags'
      • Two spurious ETL columns injected
      • Column order shuffled
    Required repair sequence (≤ 12 steps):
      1. drop_column etl_timestamp
      2. drop_column _internal_row_uuid
      3. split_column full_name → [first_name, last_name]
      4. arithmetic_transform cost / 1.3 → price_usd
      5. parse_json_column tags key=t1 → tag_1
      6. parse_json_column tags key=t2 → tag_2
      7. drop_column tags       (now extracted)
      8. submit
    """
    rng_np = np.random.default_rng(seed)

    gt_df = pd.DataFrame({
        "product_id": range(5000, 5400),
        "first_name": rng_np.choice(["Alice", "Bob", "Charlie", "Diana"], 400),
        "last_name":  rng_np.choice(["Smith", "Jones", "Brown", "Taylor"], 400),
        "price_usd":  rng_np.uniform(20.0, 100.0, 400).round(2),
        "tag_1":      rng_np.choice(["sale", "new", "clearance"], 400),
        "tag_2":      rng_np.choice(["summer", "winter", "fall"], 400),
    })

    dirty = gt_df.copy().astype(object)

    # Corrupt 1: merge name columns
    dirty["full_name"] = dirty["first_name"] + " " + dirty["last_name"]
    dirty.drop(columns=["first_name", "last_name"], inplace=True)

    # Corrupt 2: scale price and rename
    dirty["cost"] = (dirty["price_usd"].astype(float) * 1.3).round(2)
    dirty.drop(columns=["price_usd"], inplace=True)

    # Corrupt 3: JSONify tags
    dirty["tags"] = dirty.apply(
        lambda row: json.dumps({"t1": row["tag_1"], "t2": row["tag_2"]}), axis=1
    )
    dirty.drop(columns=["tag_1", "tag_2"], inplace=True)

    # Corrupt 4: inject spurious ETL columns
    dirty["etl_timestamp"]      = "2026-03-30T11:22:31Z"
    dirty["_internal_row_uuid"] = [f"uuid-{i}" for i in range(400)]

    # Corrupt 5: shuffle column order (remove positional hints)
    cols = list(dirty.columns)
    random.Random(seed).shuffle(cols)
    dirty = dirty[cols]

    return dirty, gt_df


# ──────────────────────────────────────────────────────────────────────────────
# Issue detection — provides agent signal without revealing the solution
# ──────────────────────────────────────────────────────────────────────────────

def detect_issues(df: pd.DataFrame, task_id: str) -> list[str]:
    """
    Inspect a DataFrame and return a list of human-readable issue strings.

    These appear in observations[issues_detected] from the very first step,
    giving the agent concrete diagnostic information rather than a blank slate.
    """
    issues = []

    # ── Universal checks ──────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            pct = cnt / len(df) * 100
            issues.append(f"Column '{col}' has {cnt} null values ({pct:.0f}%).")

    dupe_count = df.duplicated().sum()
    if dupe_count > 0:
        issues.append(
            f"Dataset contains {dupe_count} duplicate rows. "
            f"Fix: deduplicate (no column needed)."
        )

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str)
        numeric_like = pd.to_numeric(sample, errors="coerce").notna().mean()
        if 0.5 < numeric_like < 0.95:
            issues.append(
                f"Column '{col}' appears numeric but contains non-numeric strings "
                f"(~{(1 - numeric_like)*100:.0f}% non-parseable)."
            )

    # ── Task-specific checks ──────────────────────────────────────────────────
    if task_id == "task2_medium":
        if "transaction_date" in df.columns:
            # Only warn if the column is still string/object — once cast to
            # datetime the format issue is resolved and the alert must clear.
            if not pd.api.types.is_datetime64_any_dtype(df["transaction_date"]):
                sample_str = df["transaction_date"].dropna().astype(str)
                has_mixed = (
                    sample_str.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").any()
                    or sample_str.str.match(r"^\d{1,2}-\d{1,2}-\d{2}$").any()
                    or sample_str.str.match(r"^\d{9,10}$").any()
                )
                if has_mixed:
                    issues.append(
                        "Column 'transaction_date' has inconsistent date formats "
                        "(detected DD/MM/YYYY, MM-DD-YY, and Unix timestamp variants). "
                        "Fix: cast_type datetime — handles all formats automatically."
                    )
        if "amount" in df.columns:
            sample = df["amount"].dropna().astype(str)
            has_dollar = sample.str.startswith("$").mean()
            if has_dollar > 0.05:
                issues.append(
                    f"Column 'amount' has currency prefix '$' in "
                    f"~{has_dollar*100:.0f}% of rows — normalize before casting."
                )
            numeric_vals = pd.to_numeric(
                sample.str.replace("$", "", regex=False), errors="coerce"
            )
            has_outlier = numeric_vals > 10_000
            n_outliers  = int(has_outlier.sum())
            if n_outliers > 0:
                median_val = numeric_vals[~has_outlier].median()
                issues.append(
                    f"Column 'amount' contains {n_outliers} extreme outliers (> 10,000). "
                    f"Replace with median (~{median_val:.2f}) using normalize_values "
                    f"mapping {{\"999999.99\": \"{median_val:.2f}\"}} — "
                    f"do NOT use fill_nulls mean (outliers skew the mean)."
                )
        if "category" in df.columns:
            sample = df["category"].dropna().astype(str).str.strip()
            trailing_raw = df["category"].dropna().astype(str)
            trailing = trailing_raw.str.endswith("   ").mean()
            if trailing > 0.05:
                issues.append(
                    "Column 'category' has trailing whitespace. "
                    "Fix: normalize_values with mapping {\"   \": \"\"} "
                    "(three literal spaces → empty string). "
                    "Do NOT use \\s+ — it strips spaces inside words."
                )
            # Casing hint: check if any values are not title-case
            non_title = sample[sample.str.len() > 0].apply(
                lambda x: x != x.title()
            ).mean()
            if non_title > 0.05:
                issues.append(
                    "Column 'category' has inconsistent casing. "
                    "Target values are title-case: 'Electronics', 'Clothing', 'Food'. "
                    "Fix: one normalize_values call with ALL variants: "
                    "{\"electronics\": \"Electronics\", "
                    "\"ELECTRONICS\": \"Electronics\", "
                    "\"clothing\": \"Clothing\", "
                    "\"CLOTHING\": \"Clothing\", "
                    "\"food\": \"Food\", "
                    "\"FOOD\": \"Food\"}."
                )

    if task_id == "task3_hard":
        gt_expected = {
            "product_id", "first_name", "last_name",
            "price_usd", "tag_1", "tag_2",
        }
        current_cols  = set(df.columns)
        missing_cols  = gt_expected - current_cols
        spurious_cols = current_cols - gt_expected

        if spurious_cols:
            issues.append(
                f"Spurious columns detected (not in target schema): "
                f"{sorted(spurious_cols)}. Use drop_column to remove them."
            )
        if missing_cols:
            issues.append(
                f"Target schema columns not yet present: {sorted(missing_cols)}. "
                f"These may need to be reconstructed via split_column, "
                f"parse_json_column, or arithmetic_transform."
            )
        if "full_name" in df.columns:
            issues.append(
                "Column 'full_name' detected. Target schema expects separate "
                "'first_name' and 'last_name' columns — use split_column."
            )
        if "cost" in df.columns:
            issues.append(
                "Column 'cost' detected. Target schema expects 'price_usd' — "
                "values appear scaled by ~1.3x. Use arithmetic_transform (/ 1.3) "
                "then the column will be matched automatically."
            )
        if "tags" in df.columns:
            issues.append(
                "Column 'tags' contains JSON-encoded tag data. Use "
                "parse_json_column with key='t1' → 'tag_1' and key='t2' → 'tag_2', "
                "then drop_column 'tags'."
            )

    return issues


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────

def load_task_data(task_id: str, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route to the correct generator with a fixed global RNG seed."""
    np.random.seed(seed)
    random.seed(seed)

    if task_id == "task1_easy":
        return generate_task1_easy(seed=seed)
    if task_id == "task2_medium":
        return generate_task2_medium(seed=seed)
    if task_id == "task3_hard":
        return generate_task3_hard(seed=seed)
    raise ValueError(
        f"Unknown task_id '{task_id}'. "
        "Valid values: 'task1_easy', 'task2_medium', 'task3_hard'."
    )