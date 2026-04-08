"""
grader.py — Deterministic scoring for all DataGym tasks.

Design goals (panel requirements):
  • Score range 0.0–1.0, partial progress rewarded at every step.
  • Extra rows (duplicates) lower precision → dedup gives positive reward.
  • Missing rows (from drop_nulls) lower recall → large negative reward.
  • DateTime columns compared as dates, not strings, so cast_type(datetime)
    gives real signal.
  • NaN in current vs real value in GT = wrong (not "neutral").
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def _compare_col(s_c: pd.Series, s_g: pd.Series):
    """
    Return a boolean array: True where current cell matches GT cell.
    Rules:
      - GT null  → uncountable target, skip (treated as match to avoid penalising
                   columns the agent cannot fix; marked True so they don't hurt)
      - curr null vs GT value → mismatch
      - Both datetime-parseable  → compare date portion only
      - Both numeric             → np.isclose (no NaN==NaN credit)
      - Otherwise                → strip-normalised string compare
    """
    gt_null  = s_g.isna().values

    # ── Datetime branch ───────────────────────────────────────────────────────
    sg_dt = pd.to_datetime(s_g, errors="coerce", utc=False)
    sc_dt = pd.to_datetime(s_c, errors="coerce", utc=False)
    sg_date_frac = sg_dt.notna().mean()
    sc_date_frac = sc_dt.notna().mean()

    if sg_date_frac > 0.7 and sc_date_frac > 0.7:
        # Normalise to date-only (drop time component) before comparing.
        # .copy() is required — .values on a comparison result is read-only.
        sg_norm  = sg_dt.dt.normalize()
        sc_norm  = sc_dt.dt.normalize()
        matches  = (sc_norm == sg_norm).values.copy()
        matches[sc_dt.isna().values & ~gt_null] = False
        matches[gt_null] = True
        return matches

    # ── Numeric branch ────────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(s_g) and pd.api.types.is_numeric_dtype(s_c):
        with np.errstate(invalid="ignore"):
            matches = np.isclose(
                pd.to_numeric(s_c, errors="coerce").values,
                pd.to_numeric(s_g, errors="coerce").values,
                equal_nan=False,
            ).copy()
        matches[gt_null] = True
        return matches

    # ── String branch ─────────────────────────────────────────────────────────
    sc_str  = s_c.astype(str).str.strip()
    sg_str  = s_g.astype(str).str.strip()
    matches = (sc_str == sg_str).values.copy()
    matches[gt_null] = True
    return matches


def _cell_f1(current_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """
    Cell-level F1 with row-count-sensitive precision/recall.

    precision denominator = n_current_rows × n_common_cols
      → extra duplicate rows lower precision → dedup gives positive reward
    recall denominator    = n_gt_rows × n_common_cols
      → dropped rows lower recall → drop_nulls gives large negative reward

    Only cells corresponding to overlapping rows (min of both lengths) are
    compared; extra rows in current contribute 0 correct cells.
    """
    common_cols = [c for c in gt_df.columns if c in current_df.columns]
    if not common_cols:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    n_curr    = len(current_df)
    n_gt      = len(gt_df)
    n_compare = min(n_curr, n_gt)

    curr_s = current_df.iloc[:n_compare][common_cols].reset_index(drop=True)
    gt_s   = gt_df.iloc[:n_compare][common_cols].reset_index(drop=True)

    correct = 0
    for col in common_cols:
        matches  = _compare_col(curr_s[col], gt_s[col])
        correct += int(matches.sum())

    total_curr = n_curr * len(common_cols)
    total_gt   = n_gt   * len(common_cols)

    precision = correct / total_curr if total_curr > 0 else 0.0
    recall    = correct / total_gt   if total_gt   > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"f1": float(f1), "precision": float(precision), "recall": float(recall)}


def _fuzzy_match_columns(source_cols, target_cols, threshold=0.8):
    used = set()
    mapping = {}
    for src in source_cols:
        best_score, best_tgt = 0.0, None
        for tgt in target_cols:
            if tgt in used:
                continue
            score = fuzz.ratio(src, tgt) / 100.0
            if score > best_score:
                best_score, best_tgt = score, tgt
        if best_score >= threshold and best_tgt:
            mapping[src] = best_tgt
            used.add(best_tgt)
    return mapping


def _score_task3(current_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    curr_cols      = list(current_df.columns)
    gt_cols        = list(gt_df.columns)
    match_map      = _fuzzy_match_columns(curr_cols, gt_cols, threshold=0.8)
    gt_recovered   = set(match_map.values())
    column_recall  = len(gt_recovered) / len(gt_cols) if gt_cols else 0.0
    n_spurious     = sum(1 for c in curr_cols if c not in match_map)
    spurious_pen   = min(0.25, 0.05 * n_spurious)

    if match_map:
        renamed  = current_df.rename(columns=match_map)[list(gt_recovered)]
        gt_sub   = gt_df[list(gt_recovered)]
        cell_m   = _cell_f1(renamed, gt_sub)
        cell_f1v = cell_m["f1"]
        cell_prec= cell_m["precision"]
    else:
        cell_f1v = cell_prec = 0.0

    combined = max(0.0, min(1.0,
        0.45 * column_recall + 0.45 * cell_f1v - spurious_pen
    ))
    return {"f1": float(combined), "precision": float(cell_prec),
            "recall": float(column_recall)}


def calculate_similarity(current_df, ground_truth, task_id):
    try:
        if task_id == "task3_hard":
            return _score_task3(current_df, ground_truth)
        return _cell_f1(current_df, ground_truth)
    except Exception:
        import traceback; traceback.print_exc()
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}