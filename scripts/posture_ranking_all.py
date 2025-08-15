from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# ========================= CONFIG =========================
# Path to the one-file summary you already generated (Sheet "Summary")
SUMMARY_XLSX = Path(r"C:\Users\hcisg\Documents\Annotation\humanAnnotatorLive_all_one_hour\all_participants_posture_metrics.xlsx")

# Posture labels we care about (fine-grained).
# We’ll only rank these; group aggregates like "(GROUP)" are ignored.
EXPECTED_LABELS = [
    "UPRIGHT", "UPRIGHT LEFT", "UPRIGHT RIGHT",
    "FORWARD", "FORWARD LEFT", "FORWARD RIGHT",
    "BACK", "BACK LEFT", "BACK RIGHT",
]
# ==========================================================


def _extract_label_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return two lists of column names in df:
      - time_min_cols: columns like "<LABEL> total_time_mins"
      - episode_cols : columns like "<LABEL> episode_count"
    Only for fine-grained labels (no '(GROUP)' columns).
    """
    time_min_cols = []
    episode_cols = []
    for c in df.columns:
        if "(GROUP)" in c:
            continue  # skip aggregates
        if c.endswith(" total_time_mins"):
            # keep only those whose <LABEL> is in EXPECTED_LABELS
            label = c.replace(" total_time_mins", "")
            if label in EXPECTED_LABELS:
                time_min_cols.append(c)
        elif c.endswith(" episode_count"):
            label = c.replace(" episode_count", "")
            if label in EXPECTED_LABELS:
                episode_cols.append(c)
    # maintain a stable order following EXPECTED_LABELS
    time_min_cols = sorted(time_min_cols, key=lambda x: EXPECTED_LABELS.index(x.replace(" total_time_mins", "")))
    episode_cols  = sorted(episode_cols,  key=lambda x: EXPECTED_LABELS.index(x.replace(" episode_count", "")))
    return time_min_cols, episode_cols


def _rank_series(values: Dict[str, float]) -> List[Tuple[int, str, float]]:
    """
    Given a dict {label: value}, return a ranking list:
      [(rank, label, value), ...] sorted by value desc, ties by label asc.
    Rank starts at 1.
    """
    items = list(values.items())
    # sort: value desc, label asc for stable tie-break
    items.sort(key=lambda kv: (-float(kv[1]), kv[0]))
    ranked = [(i + 1, lab, float(val)) for i, (lab, val) in enumerate(items)]
    return ranked


def build_dominance_tables(df_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From the one-row-per-participant 'Summary' sheet, compute:
      - top1 table (dominant posture by time and by episode count)
      - full rankings table (by time and by episode count)
    """
    # Identify relevant columns
    time_min_cols, episode_cols = _extract_label_columns(df_summary)

    # Sanity
    if not time_min_cols or not episode_cols:
        raise ValueError("Could not find per-label total_time_mins and/or episode_count columns in Summary sheet.")

    # Convenience: ID columns if present
    pid_col = "participant_id" if "participant_id" in df_summary.columns else None
    file_col = "source_file" if "source_file" in df_summary.columns else None

    top_rows = []
    ranks_rows = []

    for _, row in df_summary.iterrows():
        # Build dicts: label -> minutes / episodes
        time_vals = {}
        for c in time_min_cols:
            lab = c.replace(" total_time_mins", "")
            time_vals[lab] = float(row.get(c, 0.0) or 0.0)

        ep_vals = {}
        for c in episode_cols:
            lab = c.replace(" episode_count", "")
            ep_vals[lab] = float(row.get(c, 0.0) or 0.0)

        # Rankings
        time_rank = _rank_series(time_vals)
        ep_rank   = _rank_series(ep_vals)

        # Top-1 values
        top_time_rank, top_time_label, top_time_value = time_rank[0] if time_rank else (None, None, None)
        top_ep_rank,   top_ep_label,   top_ep_value   = ep_rank[0] if ep_rank else (None, None, None)

        top_rec = {
            "participant_id": row.get(pid_col) if pid_col else None,
            "source_file": row.get(file_col) if file_col else None,
            "top_time_label": top_time_label,
            "top_time_minutes": top_time_value,
            "top_repetition_label": top_ep_label,
            "top_repetition_count": top_ep_value,
        }
        top_rows.append(top_rec)

        # Full ranking rows (time)
        for rnk, lab, val in time_rank:
            ranks_rows.append({
                "participant_id": row.get(pid_col) if pid_col else None,
                "source_file": row.get(file_col) if file_col else None,
                "metric": "time_minutes",
                "rank": rnk,
                "label": lab,
                "value": val,
            })
        # Full ranking rows (episodes)
        for rnk, lab, val in ep_rank:
            ranks_rows.append({
                "participant_id": row.get(pid_col) if pid_col else None,
                "source_file": row.get(file_col) if file_col else None,
                "metric": "episode_count",
                "rank": rnk,
                "label": lab,
                "value": val,
            })

    top_df   = pd.DataFrame(top_rows)
    ranks_df = pd.DataFrame(ranks_rows)

    # Sort nicely
    sort_keys = ["participant_id", "source_file"]
    sort_keys = [k for k in sort_keys if k in top_df.columns]
    if sort_keys:
        top_df = top_df.sort_values(sort_keys).reset_index(drop=True)
        ranks_df = ranks_df.sort_values(sort_keys + ["metric", "rank"]).reset_index(drop=True)

    return top_df, ranks_df


def main():
    if not SUMMARY_XLSX.exists():
        raise FileNotFoundError(f"Cannot find summary workbook: {SUMMARY_XLSX}")

    # Read existing Summary sheet
    summary_df = pd.read_excel(SUMMARY_XLSX, sheet_name="Summary")

    # Build dominance tables
    top_df, ranks_df = build_dominance_tables(summary_df)

    # Write back to the SAME workbook as extra sheets
    # Requires pandas >= 1.4 for mode='a' and if_sheet_exists
    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        # Do not touch the existing Summary sheet; only add/replace these:
        top_df.to_excel(writer, index=False, sheet_name="Dominance (top1)")
        ranks_df.to_excel(writer, index=False, sheet_name="Dominance (rankings)")

    print(f"Updated workbook with dominance sheets:\n  - {SUMMARY_XLSX}")


if __name__ == "__main__":
    main()
