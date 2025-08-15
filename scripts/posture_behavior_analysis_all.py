from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ========================= CONFIG =========================
BASE_DIR = Path(r"C:\Users\hcisg\Documents\Annotation\humanAnnotatorLive_all_one_hour")
PARTICIPANT_IDS = list(range(1, 9))  # 1..8
INPUT_PATTERNS = [
    "Participant_{pid}_phase_2_annotations.xlsx",
    "Participant_{pid}_phase_2_annotations.csv",  # fallback if someone saved as CSV
]
SHEET_NAME = "annotations"
OUTPUT_XLSX = BASE_DIR / "all_participants_posture_metrics.xlsx"
# ===========================================================

# Good posture = exact "UPRIGHT"
GOOD_LABEL = "UPRIGHT"

# Always show these (even if absent)
EXPECTED_LABELS = [
    "UPRIGHT", "UPRIGHT LEFT", "UPRIGHT RIGHT",
    "FORWARD", "FORWARD LEFT", "FORWARD RIGHT",
    "BACK", "BACK LEFT", "BACK RIGHT",
]

# ===== Major groups (as you requested) =====
# - 'upright_neutral' = UPRIGHT only
# - 'upright_lean'    = UPRIGHT LEFT / UPRIGHT RIGHT
# - forward/back as usual
MAJOR_GROUPS = {
    "UPRIGHT": "upright_neutral",
    "UPRIGHT LEFT": "upright_lean",
    "UPRIGHT RIGHT": "upright_lean",

    "FORWARD": "forward",
    "FORWARD LEFT": "forward",
    "FORWARD RIGHT": "forward",

    "BACK": "back",
    "BACK LEFT": "back",
    "BACK RIGHT": "back",
}

# ---------- Helpers ----------
def _to_timedelta(s: pd.Series) -> pd.Series:
    return pd.to_timedelta(s.astype(str))

def _add_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - 't': timestamp as Timedelta
      - 'dt': duration to next row (fills last delta with median; guards non-positive)
    """
    out = df.copy()
    if "t" not in out.columns:
        if "timestamp" not in out.columns:
            raise ValueError("Input must contain 'timestamp' column.")
        out["t"] = _to_timedelta(out["timestamp"])

    deltas = out["t"].shift(-1) - out["t"]
    med = deltas.dropna()
    med = med[med > pd.Timedelta(0)]
    fallback = med.median() if not med.empty else pd.Timedelta(seconds=0)
    deltas = deltas.fillna(fallback)
    deltas = deltas.mask(deltas <= pd.Timedelta(0), fallback)
    out["dt"] = deltas
    return out

def _runs(series: pd.Series, durations: pd.Series) -> List[Dict]:
    """
    Collapse consecutive identical values into runs with summed durations.
    Each run: {'value': v, 'duration': Timedelta, 'count': n}
    """
    values = series.values
    dts = durations.values
    if len(values) == 0:
        return []
    runs = []
    cur = values[0]
    tot = pd.Timedelta(0)
    cnt = 0
    for v, dt in zip(values, dts):
        if v == cur:
            tot += dt
            cnt += 1
        else:
            runs.append({"value": cur, "duration": tot, "count": cnt})
            cur = v
            tot = dt
            cnt = 1
    runs.append({"value": cur, "duration": tot, "count": cnt})
    return runs

# ---------- Metric engines (minutes-focused) ----------
def _time_in_good_bad(df: pd.DataFrame) -> Tuple[pd.Timedelta, pd.Timedelta, float, float]:
    d = _add_durations(df)
    good_mask = d["label"] == GOOD_LABEL
    good_time = d.loc[good_mask, "dt"].sum()
    bad_time  = d.loc[~good_mask, "dt"].sum()
    total     = d["dt"].sum()
    good_ratio = float(good_time / total) if total.total_seconds() > 0 else np.nan
    bad_ratio  = float(bad_time  / total) if total.total_seconds() > 0 else np.nan
    return good_time, bad_time, good_ratio, bad_ratio

def _avg_time_until_any_change(df: pd.DataFrame) -> Tuple[pd.Timedelta, float]:
    d = _add_durations(df)
    runs = _runs(d["label"], d["dt"])
    if not runs:
        return pd.Timedelta(0), np.nan
    durations = [r["duration"] for r in runs]
    avg = sum(durations, pd.Timedelta(0)) / len(durations)
    return avg, avg.total_seconds() / 60.0  # minutes

def _avg_time_until_major_change(df: pd.DataFrame) -> Tuple[pd.Timedelta, float]:
    d = _add_durations(df)
    groups = d["label"].map(MAJOR_GROUPS).fillna("other")
    runs = _runs(groups, d["dt"])
    if not runs:
        return pd.Timedelta(0), np.nan
    durations = [r["duration"] for r in runs]
    avg = sum(durations, pd.Timedelta(0)) / len(durations)
    return avg, avg.total_seconds() / 60.0  # minutes

def _posture_type_breakdown(df: pd.DataFrame) -> Dict[str, float]:
    """
    Flat dict with:
      - Per-label totals (HH:MM:SS + minutes) and episode_count (consecutive runs)
      - Per-group totals (HH:MM:SS + minutes) and episode_count
    Always includes EXPECTED_LABELS and groups even if absent (zeros).
    """
    d = _add_durations(df)

    # total time in seconds per label
    tot_sec_by_label = d.groupby("label")["dt"].sum().apply(lambda x: x.total_seconds()).to_dict()

    # episode counts per label (consecutive runs)
    runs = _runs(d["label"], d["dt"])
    ep_count_by_label: Dict[str, int] = {}
    for r in runs:
        lab = r["value"]
        ep_count_by_label[lab] = ep_count_by_label.get(lab, 0) + 1

    # ensure all expected labels appear, preserve order
    labels_present = set(tot_sec_by_label.keys()) | set(ep_count_by_label.keys())
    all_labels_ordered = [lab for lab in EXPECTED_LABELS] + sorted(labels_present - set(EXPECTED_LABELS))

    flat: Dict[str, float] = {}
    for lab in all_labels_ordered:
        total_sec = float(tot_sec_by_label.get(lab, 0.0))
        total_hms = str(pd.Timedelta(seconds=total_sec))
        total_mins = round(total_sec / 60.0, 4)
        ep_cnt = int(ep_count_by_label.get(lab, 0))
        flat[f"{lab} total_time_hms"] = total_hms
        flat[f"{lab} total_time_mins"] = total_mins
        flat[f"{lab} episode_count"] = ep_cnt

    # ===== Groups =====
    group_map = MAJOR_GROUPS.copy()
    expected_groups = {"upright_neutral", "upright_lean", "forward", "back", "other"}
    group_labels = sorted(set(group_map.values()) | {"other"} | expected_groups)

    group_sec = {g: 0.0 for g in group_labels}
    group_count = {g: 0 for g in group_labels}

    for lab, sec in tot_sec_by_label.items():
        group = group_map.get(lab, "other")
        group_sec[group] += sec
    for r in runs:
        group = group_map.get(r["value"], "other")
        group_count[group] += 1

    for g in group_labels:
        total_hms = str(pd.Timedelta(seconds=group_sec[g]))
        total_mins = round(group_sec[g] / 60.0, 4)
        flat[f"{g} (GROUP) total_time_hms"] = total_hms
        flat[f"{g} (GROUP) total_time_mins"] = total_mins
        flat[f"{g} (GROUP) episode_count"] = group_count[g]

    return flat

def compute_summary_row(df: pd.DataFrame) -> Dict[str, object]:
    """
    Build one summary row (minutes-focused) for a single participant/session.
    """
    # 1) Good/Bad time + ratios
    good_td, bad_td, good_ratio, bad_ratio = _time_in_good_bad(df)
    good_pct = None if np.isnan(good_ratio) else round(good_ratio * 100, 2)
    bad_pct  = None if np.isnan(bad_ratio)  else round(bad_ratio  * 100, 2)

    # 2) Avg times
    any_avg_td, any_avg_min = _avg_time_until_any_change(df)
    major_avg_td, major_avg_min = _avg_time_until_major_change(df)

    # 3) Per-posture/per-group
    per_label_and_group = _posture_type_breakdown(df)

    row = {
        # Global times
        "Time in good posture (HH:MM:SS)": str(good_td),
        "Time in good posture (minutes)":   round(good_td.total_seconds() / 60.0, 4),
        "Time in bad posture (HH:MM:SS)":  str(bad_td),
        "Time in bad posture (minutes)":    round(bad_td.total_seconds() / 60.0, 4),
        "Good posture time (%)": good_pct,
        "Bad posture time (%)":  bad_pct,

        # Change dynamics
        "Avg time before ANY posture label change (HH:MM:SS)": str(any_avg_td),
        "Avg time before ANY posture label change (minutes)":    None if pd.isna(any_avg_min) else round(any_avg_min, 4),

        "Avg time before MAJOR posture change (HH:MM:SS)": str(major_avg_td),
        "Avg time before MAJOR posture change (minutes)":    None if pd.isna(major_avg_min) else round(major_avg_min, 4),
    }
    row.update(per_label_and_group)
    return row

# ---------- Batch runner ----------
def _load_annotation_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name=SHEET_NAME)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def _find_input_for_pid(pid: int) -> Path | None:
    for pat in INPUT_PATTERNS:
        candidate = BASE_DIR / pat.format(pid=pid)
        if candidate.exists():
            return candidate
    return None

def main():
    rows: List[Dict[str, object]] = []
    missing: List[int] = []

    for pid in PARTICIPANT_IDS:
        in_path = _find_input_for_pid(pid)
        if in_path is None:
            print(f"[WARN] Missing file for participant {pid}. Expected one of: "
                  + ", ".join([str(BASE_DIR / p.format(pid=pid)) for p in INPUT_PATTERNS]))
            missing.append(pid)
            continue

        try:
            df = _load_annotation_file(in_path)
            if not {"timestamp", "label"}.issubset(df.columns):
                raise ValueError(f"{in_path.name} must contain 'timestamp' and 'label' columns.")

            row = compute_summary_row(df)
            row["participant_id"] = pid
            row["source_file"] = in_path.name
            rows.append(row)
            print(f"[OK] Processed participant {pid}: {in_path.name}")

        except Exception as e:
            print(f"[ERROR] Participant {pid} ({in_path.name}): {e}")

    if not rows:
        raise RuntimeError("No participants processed. Check input paths and file formats.")

    # Create DataFrame with participant rows first
    # Put participant_id, source_file as the first columns
    df_all = pd.DataFrame(rows)
    first_cols = ["participant_id", "source_file"]
    other_cols = [c for c in df_all.columns if c not in first_cols]
    df_all = df_all[first_cols + other_cols]

    # Save only ONE Excel file with ONE sheet
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_all.to_excel(writer, index=False, sheet_name="Summary")

    print(f"\nSaved: {OUTPUT_XLSX.resolve()}")
    if missing:
        print(f"Missing participants (no file found): {missing}")

if __name__ == "__main__":
    main()
