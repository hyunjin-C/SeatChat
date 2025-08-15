from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

# ========================= CONFIG =========================
INPUT_DIR = Path(r"C:\Users\hcisg\Documents\Annotation\humanAnnotatorLive_all_one_hour")
SHEET_NAME = "annotations"
PARTICIPANT_FILES = [f"Participant_{i}_phase_2_annotations.xlsx" for i in range(1, 9)]

# We will mine BOTH lengths below (exactly 3 and exactly 2)
MOTIF_LENGTHS = [3, 2]

# Filters to suppress noise; motif is kept if (count >= MIN_COUNT) OR (support >= MIN_SUPPORT_RATIO)
MIN_COUNT = 2
MIN_SUPPORT_RATIO = 0.01  # 1% of all windows of that length

# Output workbook (one file, two sheets)
OUTPUT_XLSX = INPUT_DIR / "posture_sequences_L3_and_L2.xlsx"
# ==========================================================


# --------------------- Helpers --------------------- #
def _to_timedelta(s: pd.Series) -> pd.Series:
    return pd.to_timedelta(s.astype(str))

def _add_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Add 't' (Timedelta) and 'dt' (delta to next row) with robust last/invalid handling."""
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

def collapse_runs(d: pd.DataFrame) -> List[Dict]:
    """
    Collapse consecutive identical labels into runs:
      each run = {'label','start','end','duration'}
    """
    if d.empty:
        return []
    labels = d["label"].astype(str).tolist()
    t = d["t"].tolist()
    dt = d["dt"].tolist()

    runs: List[Dict] = []
    cur_lab = labels[0]
    start = t[0]
    dur = pd.Timedelta(0)

    for lab, ti, dti in zip(labels, t, dt):
        if lab == cur_lab:
            dur += dti
        else:
            runs.append({"label": cur_lab, "start": start, "end": start + dur, "duration": dur})
            cur_lab = lab
            start = ti
            dur = dti
    runs.append({"label": cur_lab, "start": start, "end": start + dur, "duration": dur})
    return runs

def enumerate_motifs(runs: List[Dict], L: int) -> Tuple[List[Tuple[Tuple[str,...], int, int, float]], int]:
    """
    Compute all mixed (at least 2 distinct labels) contiguous L-grams over the run sequence.
    Returns:
      - occurrences: list of (motif_tuple, start_idx, end_idx, cycle_minutes)
      - total_windows: number of L-windows (for support)
    """
    if len(runs) < L:
        return [], 0
    occs = []
    for i in range(len(runs) - L + 1):
        gram = tuple(runs[j]["label"] for j in range(i, i + L))
        if len(set(gram)) < 2:  # ensure mixed
            continue
        cycle_min = (runs[i + L - 1]["end"] - runs[i]["start"]).total_seconds() / 60.0
        occs.append((gram, i, i + L - 1, cycle_min))
    total_windows = max(0, len(runs) - L + 1)
    return occs, total_windows

def weighted_interval_max_nonoverlap(intervals: List[Tuple[int,int,float]]) -> float:
    """
    Exact maximum total minutes from non-overlapping intervals using weighted interval scheduling.
    Each interval = (start_idx, end_idx, weight_minutes).
    """
    if not intervals:
        return 0.0
    intervals = sorted(intervals, key=lambda x: (x[1], x[0]))
    starts = [s for s, e, w in intervals]
    ends   = [e for s, e, w in intervals]
    weights= [w for s, e, w in intervals]
    # p(j): rightmost i < j such that intervals[i] ends before intervals[j] starts
    p = []
    for j in range(len(intervals)):
        lo, hi, idx = 0, j - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ends[mid] < starts[j]:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        p.append(idx)
    # DP
    M = [0.0] * (len(intervals) + 1)
    for j in range(1, len(intervals) + 1):
        take = weights[j-1] + (M[p[j-1] + 1] if p[j-1] >= 0 else 0.0)
        M[j] = max(M[j-1], take)
    return M[-1]

def stats_for_length(runs: List[Dict], L: int) -> Dict[Tuple[str,...], Dict[str, float]]:
    """
    Build stats for each distinct mixed motif of length L:
      - count
      - support_pct (% of L-windows)
      - avg_cycle_min (average per occurrence)
      - total_nonoverlap_min (max sum of non-overlapping occurrences; exposure)
    Applies MIN_COUNT / MIN_SUPPORT_RATIO filtering.
    """
    occs, total_windows = enumerate_motifs(runs, L)
    by_motif = defaultdict(list)
    for motif, i, j, minutes in occs:
        by_motif[motif].append((i, j, minutes))

    out: Dict[Tuple[str,...], Dict[str, float]] = {}
    for motif, ivals in by_motif.items():
        count = len(ivals)
        support = (count / total_windows * 100.0) if total_windows > 0 else 0.0
        avg_min = float(np.mean([w for _, _, w in ivals]))
        total_nonoverlap = weighted_interval_max_nonoverlap(ivals)

        if count < MIN_COUNT and support < (MIN_SUPPORT_RATIO * 100.0):
            continue

        out[motif] = dict(
            L=L,
            count=int(count),
            support_pct=round(support, 2),
            avg_cycle_min=round(avg_min, 3),
            total_nonoverlap_min=round(total_nonoverlap, 3),
        )
    return out

def pick_most_repeated(stats: Dict[Tuple[str,...], Dict[str, float]]) -> Tuple[str,int,float,int,float]:
    """
    Pick motif with max count; tie-break by longer L, larger total_nonoverlap_min, alphabetical motif.
    Returns (motif_str, count, avg_cycle_min, L, total_nonoverlap_min)
    """
    if not stats:
        return None, 0, None, None, None
    key = max(stats.keys(), key=lambda k: (stats[k]["count"], stats[k]["L"], stats[k]["total_nonoverlap_min"], k))
    s = stats[key]
    return " → ".join(key), s["count"], s["avg_cycle_min"], s["L"], s["total_nonoverlap_min"]

def pick_most_time(stats: Dict[Tuple[str,...], Dict[str, float]]) -> Tuple[str,float,float,int,int]:
    """
    Pick motif with max total_nonoverlap_min; tie-break by longer L, higher count, alphabetical motif.
    Returns (motif_str, total_nonoverlap_min, avg_cycle_min, L, count)
    """
    if not stats:
        return None, 0.0, None, None, None
    key = max(stats.keys(), key=lambda k: (stats[k]["total_nonoverlap_min"], stats[k]["L"], stats[k]["count"], k))
    s = stats[key]
    return " → ".join(key), s["total_nonoverlap_min"], s["avg_cycle_min"], s["L"], s["count"]

def _pid_from_name(filename: str) -> int | None:
    try:
        base = filename.split(".")[0]
        parts = base.split("_")
        for i, p in enumerate(parts):
            if p.lower() == "participant":
                return int(parts[i+1])
    except Exception:
        pass
    return None


# --------------------- Per-participant mining --------------------- #
def analyze_participant(path: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Returns:
      - summary row with most repeated & most time-consuming motifs for L=3 and L=2 (separately)
      - list of tidy motif rows (all motifs for L=3 and L=2)
    """
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    if not {"timestamp", "label"}.issubset(df.columns):
        raise ValueError(f"{path.name} must contain 'timestamp' and 'label' columns.")
    d = _add_durations(df)
    runs = collapse_runs(d)

    per_length_stats: Dict[int, Dict[Tuple[str,...], Dict[str,float]]] = {}
    all_motif_rows: List[Dict[str, object]] = []

    for L in MOTIF_LENGTHS:  # [3,2]
        stats = stats_for_length(runs, L)
        per_length_stats[L] = stats

        # add to tidy rows
        for motif, s in stats.items():
            all_motif_rows.append({
                "participant_id": _pid_from_name(path.name),
                "source_file": path.name,
                "L": L,
                "motif": " → ".join(motif),
                "count": s["count"],
                "support_pct": s["support_pct"],
                "avg_cycle_min": s["avg_cycle_min"],
                "total_nonoverlap_min": s["total_nonoverlap_min"],
            })

    # Build summary (per L separately)
    pid = _pid_from_name(path.name)
    row = {
        "participant_id": pid,
        "source_file": path.name,
    }

    # For L=3
    seq, cnt, avgm, Lpicked, ton = pick_most_repeated(per_length_stats.get(3, {}))
    row.update({
        "L3_most_repeated_sequence": seq,
        "L3_most_repeated_count": cnt,
        "L3_most_repeated_avg_cycle_min": avgm,
    })
    seqT, totalMin, avgmT, LpickedT, cntT = pick_most_time(per_length_stats.get(3, {}))
    row.update({
        "L3_most_time_sequence": seqT,
        "L3_most_time_total_min": totalMin,
        "L3_most_time_avg_cycle_min": avgmT,
    })

    # For L=2
    seq2, cnt2, avgm2, Lpicked2, ton2 = pick_most_repeated(per_length_stats.get(2, {}))
    row.update({
        "L2_most_repeated_sequence": seq2,
        "L2_most_repeated_count": cnt2,
        "L2_most_repeated_avg_cycle_min": avgm2,
    })
    seq2T, totalMin2, avgm2T, Lpicked2T, cnt2T = pick_most_time(per_length_stats.get(2, {}))
    row.update({
        "L2_most_time_sequence": seq2T,
        "L2_most_time_total_min": totalMin2,
        "L2_most_time_avg_cycle_min": avgm2T,
    })

    return row, all_motif_rows


# --------------------- Batch driver --------------------- #
def main():
    summary_rows: List[Dict[str, object]] = []
    tidy_rows: List[Dict[str, object]] = []

    for fname in PARTICIPANT_FILES:
        path = INPUT_DIR / fname
        if not path.exists():
            print(f"[WARN] Missing file: {path.name}")
            continue
        try:
            row, motifs = analyze_participant(path)
            summary_rows.append(row)
            tidy_rows.extend(motifs)
            print(f"[OK] {path.name}")
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")

    if not summary_rows:
        raise RuntimeError("No participants processed.")

    df_summary = pd.DataFrame(summary_rows).sort_values("participant_id").reset_index(drop=True)
    df_tidy    = pd.DataFrame(tidy_rows).sort_values(["participant_id","L","count","support_pct"], ascending=[True,True,False,False]).reset_index(drop=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Summary L2-L3")
        df_tidy.to_excel(writer, index=False, sheet_name="All motifs (tidy)")

    print(f"\nSaved -> {OUTPUT_XLSX.resolve()}")

if __name__ == "__main__":
    main()
