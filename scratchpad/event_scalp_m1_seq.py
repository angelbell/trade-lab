"""STEP C for spec card: FOMC statement gold 1min scalp -- sequential (causal) threshold.

Runs only because STEP A survived. Replaces the in-sample "top 50%/33% of confirm_move_atr,
computed from the WHOLE history" conditioning (event_scalp_cond.threshold_subset) with a
walk-forward version: at event i, the threshold is the median (or top-1/3 point = 2/3 quantile)
of confirm_move_atr among events[0:i] ONLY (strictly prior events). First m=10 events are
warmup (not traded). This is the only new logic in this file -- event selection, ATR/no-lookahead
price primitives, and the g_H outcome all come from event_scalp_m1_exec.build_scalp_table_exec
(fill=F2, slip=0.5, matching event_scalp_m1_stop.py's STEP B operating point) unchanged.

Execution: .venv/bin/python scratchpad/event_scalp_m1_seq.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.event_scalp import COST_ROUNDTRIP
from scratchpad.event_scalp_cond import threshold_subset, pctile_of_real_in_pool
from scratchpad.event_scalp_m1_exec import (
    build_scalp_table_exec, null_scalp_table_exec, profit_factor, EVENTS_CSV, GOLD_M1, DRAWS,
)

COST_BASE = COST_ROUNDTRIP["GOLD"]["base"]
W_C = 1
H = 5
FILL, SLIP = "F2", 0.5
M_WARMUP = 10


def sequential_subset(real_sorted, quantile):
    """quantile=0.50 -> median threshold; quantile=2/3 -> top-1/3 cutoff. Uses ONLY
    events[0:i] (strictly prior, expanding window) to set event i's threshold. First
    M_WARMUP events are never traded (insufficient history)."""
    cm = real_sorted["confirm_move_atr"].to_numpy()
    n = len(real_sorted)
    take = np.zeros(n, dtype=bool)
    thr_used = np.full(n, np.nan)
    for i in range(M_WARMUP, n):
        prior = cm[:i]
        thr = np.quantile(prior, quantile)
        thr_used[i] = thr
        take[i] = cm[i] >= thr
    out = real_sorted.copy()
    out["seq_thr"] = thr_used
    out["seq_take"] = take
    return out[out["seq_take"]]


def report_subset(name, sub, null_pool, cost=COST_BASE):
    gcol = f"g_{H}"
    g = sub[gcol].dropna()
    if len(g) == 0:
        print(f"  [{name}] n=0, skipped")
        return
    net = g - cost
    pctile, n_pool = pctile_of_real_in_pool(g, null_pool[gcol].dropna(), cost, B=2000)
    print(f"  [{name}] n={len(g)}  net_mean={net.mean():+.4f}  win%={(g>0).mean()*100:.1f}  "
          f"PF={profit_factor(net):.4f}  null_pctile(vs unconditioned same-clock null)={pctile:.1f} "
          f"(pool n={n_pool})")
    t = sub.dropna(subset=[gcol]).copy()
    t["year"] = t["t0"].dt.year
    netc = t[gcol] - cost
    ann = t.assign(net=netc).groupby("year").agg(n=("net", "size"), net_mean=("net", "mean"),
                                                   net_sum=("net", "sum"),
                                                   win_pct=(gcol, lambda s: (s > 0).mean() * 100))
    print(ann.round(4).to_string())


def main():
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["dt_utc", "dt_broker"])
    ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
    events = list(ev["dt_broker"].sort_values())

    df = load_mt5_csv(GOLD_M1)
    print(f"GOLD m1: {len(df)} bars, span {df.index.min()} .. {df.index.max()}")

    real_full = build_scalp_table_exec(df, events, W_C, [H], FILL, SLIP, "GOLD-seqC")
    null_full = null_scalp_table_exec(df, events, W_C, [H], FILL, SLIP, "GOLD-seqC", draws_target=DRAWS)
    real_sorted = real_full.sort_values("t0").reset_index(drop=True)
    print(f"usable real events: {len(real_sorted)} (warmup m={M_WARMUP} -> {len(real_sorted)-M_WARMUP} eligible)")

    print(f"\n{'='*100}\nSTEP C -- sequential (causal) threshold vs in-sample (whole-history) threshold, "
          f"H={H}, fill={FILL}(slip={SLIP}), cost=${COST_BASE}/oz\n{'='*100}")

    print("\n--- IN-SAMPLE (whole-history) thresholds, for comparison (same events, non-causal thr) ---")
    for frac, label in [(1.00, "frac=1.00 (all, incl. warmup events)"), (0.50, "frac=0.50 (median thr)"),
                         (0.333, "frac=0.333 (top-1/3 thr)")]:
        sub, thr = threshold_subset(real_sorted, "confirm_move_atr", frac)
        print(f"[IS {label}] thr={thr:.4f}")
        report_subset(f"IS {label}", sub, null_full)

    print("\n--- SEQUENTIAL (causal, walk-forward) thresholds ---")
    seq_median = sequential_subset(real_sorted, 0.50)
    print(f"[SEQ median] n_eligible(after warmup)={len(real_sorted)-M_WARMUP}, n_taken={len(seq_median)}")
    report_subset("SEQ median (>= expanding-window median of confirm_move_atr)", seq_median, null_full)

    seq_top13 = sequential_subset(real_sorted, 2 / 3)
    print(f"\n[SEQ top-1/3] n_eligible(after warmup)={len(real_sorted)-M_WARMUP}, n_taken={len(seq_top13)}")
    report_subset("SEQ top-1/3 (>= expanding-window 2/3-quantile of confirm_move_atr)", seq_top13, null_full)

    # Flag whether the single dominant outlier event is in both selections (transparency, not
    # part of the spec's required output but material to interpreting the result honestly).
    outlier_t0 = pd.Timestamp("2026-06-17 21:00:00", tz="UTC")
    print(f"\n[transparency] is the 2026-06-17 outlier event (g_5={real_sorted[real_sorted['t0']==outlier_t0]['g_5'].values}) "
          f"in SEQ median selection: {(seq_median['t0']==outlier_t0).any()}  "
          f"in SEQ top-1/3 selection: {(seq_top13['t0']==outlier_t0).any()}")


if __name__ == "__main__":
    main()
