"""STEP B for spec card: FOMC statement gold 1min scalp -- stop-loss / worst-loss.

Runs only because STEP A (event_scalp_m1_exec.py) survived (F1 net>0 and s=0.25 net>0 at the
reported live cell frac=0.50/H=5). Per spec: "約定規約は A で生き残った最も保守的なもの
（最低でも F1）を使う" -- this file uses F2 with slip=0.5 (the most conservative fill variant
that STILL kept net>0 at frac=0.50/H=5 in the STEP A grid: net_mean=+0.9945), which is more
conservative than the spec's stated floor of F1.

Reuses without rewriting: event_scalp_m1_exec.build_scalp_table_exec / null_scalp_table_exec /
scalp_metrics_exec / bar_after / profit_factor (this file's only new code is the MAE/stop-walk
logic, which cannot be built from the existing tools since none of them model an intrabar stop).

No lookahead / same-bar tie-break (CLAUDE.md checklist #11): the bar-by-bar stop walk starts at
the ENTRY bar itself (not entry+1), and if a bar's low/high would touch the stop level, that bar
is treated as a loss at the stop distance regardless of what the time-exit would have shown for
that same bar.

Execution: .venv/bin/python experiments/event_scalp_m1_stop.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset, pctile_of_real_in_pool
from experiments.event_scalp_m1_exec import (
    build_scalp_table_exec, null_scalp_table_exec, profit_factor, EVENTS_CSV, GOLD_M1, DRAWS,
)

COST_BASE = COST_ROUNDTRIP["GOLD"]["base"]
W_C = 1
H = 5
FRAC = 0.50           # the reported live operating point
FILL, SLIP = "F2", 0.5  # most conservative fill that still nets positive at frac=0.50,H=5
ATR_MULTS = [0.5, 1.0, 1.5, 2.0, 3.0]
FIXED_STOPS = [1.0, 2.0, 3.0, 5.0]   # $/oz
USD_JPY = 150.0


# ----------------------------------------------------------------------------
# NEW: intrabar MAE / stop walk. Not in any existing tool (event_scalp*/fomc_event_study only
# ever look at close prices via price_before -- they never touch high/low intrabar paths).
# ----------------------------------------------------------------------------
def mae_window(df, entry_time, exit_time, d, P_entry):
    """Max adverse excursion ($/oz) from P_entry over bars in [entry_time, exit_time)."""
    pos_start = df.index.searchsorted(entry_time, side="left")
    pos_end = df.index.searchsorted(exit_time, side="left")
    if pos_end <= pos_start:
        return np.nan
    win = df.iloc[pos_start:pos_end]
    if d > 0:
        adverse = P_entry - win["low"].to_numpy()
    else:
        adverse = win["high"].to_numpy() - P_entry
    adverse = np.maximum(adverse, 0.0)
    return adverse.max() if len(adverse) else 0.0


def stop_walk(df, entry_time, exit_time, d, P_entry, distance, existing_g):
    """Bar-by-bar walk from the ENTRY bar (inclusive) to exit_time (exclusive). If any bar's
    low/high touches the stop distance, exit there at EXACTLY the stop distance (same-bar
    tie-break = stop wins, i.e. we never let that bar's favorable close override the breach).
    Otherwise fall back to the already-computed time-exit gross g (existing_g)."""
    pos_start = df.index.searchsorted(entry_time, side="left")
    pos_end = df.index.searchsorted(exit_time, side="left")
    if pos_end <= pos_start:
        return existing_g, False
    win = df.iloc[pos_start:pos_end]
    if d > 0:
        breach = win["low"].to_numpy() <= (P_entry - distance)
    else:
        breach = win["high"].to_numpy() >= (P_entry + distance)
    if breach.any():
        return -distance, True
    return existing_g, False


def apply_stop_to_table(df, tbl, h, distance):
    """Vectorized-by-row application of stop_walk over a scalp table (needs t_entry, d,
    P_fill_entry, g_h columns already present, as built by build_scalp_table_exec)."""
    g_out = np.empty(len(tbl))
    stopped = np.empty(len(tbl), dtype=bool)
    t_entry_arr = tbl["t_entry"].to_numpy()
    d_arr = tbl["d"].to_numpy()
    p_arr = tbl["P_fill_entry"].to_numpy()
    g_arr = tbl[f"g_{h}"].to_numpy()
    for i in range(len(tbl)):
        t_exit = pd.Timestamp(t_entry_arr[i]) + pd.Timedelta(minutes=h)
        g_out[i], stopped[i] = stop_walk(df, pd.Timestamp(t_entry_arr[i]), t_exit,
                                          d_arr[i], p_arr[i], distance, g_arr[i])
    return g_out, stopped


def main():
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["dt_utc", "dt_broker"])
    ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
    events = list(ev["dt_broker"].sort_values())

    df = load_mt5_csv(GOLD_M1)
    print(f"GOLD m1: {len(df)} bars, span {df.index.min()} .. {df.index.max()}")
    print(f"Operating point: w_c={W_C}, H={H}, frac={FRAC}, fill={FILL}(slip={SLIP}), cost=${COST_BASE}/oz")

    real_full = build_scalp_table_exec(df, events, W_C, [H], FILL, SLIP, "GOLD-stopB")
    null_full = null_scalp_table_exec(df, events, W_C, [H], FILL, SLIP, "GOLD-stopB", draws_target=DRAWS)

    sub, thr = threshold_subset(real_full, "confirm_move_atr", FRAC)
    nsub = null_full[null_full["confirm_move_atr"] >= thr]
    print(f"real subset n={len(sub)} (thr={thr:.4f}), null-conditioned pool n={len(nsub)}")

    # -------------------------------------------------------------------
    # STEP B.1: worst loss with NO stop -- MAE distribution ($/oz, and JPY at 1$=150y)
    # -------------------------------------------------------------------
    mae_vals = []
    for _, row in sub.iterrows():
        t_exit = row["t_entry"] + pd.Timedelta(minutes=H)
        m = mae_window(df, row["t_entry"], t_exit, row["d"], row["P_fill_entry"])
        mae_vals.append(m)
    mae = pd.Series(mae_vals, index=sub.index)
    print(f"\n{'='*100}\nSTEP B.1 -- MAE distribution, NO stop (n={len(mae)}, unit=$/oz; 1oz lot=0.01 lot)\n{'='*100}")
    print(f"  median={mae.median():.4f}  std={mae.std():.4f}  max={mae.max():.4f}  p95={mae.quantile(0.95):.4f}")
    print(f"  in JPY (x{USD_JPY:.0f}):  median={mae.median()*USD_JPY:.1f}  std={mae.std()*USD_JPY:.1f}  "
          f"max={mae.max()*USD_JPY:.1f}  p95={mae.quantile(0.95)*USD_JPY:.1f}")

    # -------------------------------------------------------------------
    # STEP B.2: stop-loss variants
    # -------------------------------------------------------------------
    print(f"\n{'='*100}\nSTEP B.2 -- stop-loss variants (H={H}min time exit as fallback, "
          f"same-bar tie-break=stop, entry bar included in walk)\n{'='*100}")
    rows = []
    for kind, val in ([("ATRx", k) for k in ATR_MULTS] + [("fixed$", f) for f in FIXED_STOPS]):
        g_stop = np.empty(len(sub))
        stopped = np.empty(len(sub), dtype=bool)
        g_stop_null = np.empty(len(nsub))
        for i, (_, row) in enumerate(sub.iterrows()):
            dist = val * row["ATR"] if kind == "ATRx" else val
            t_exit = row["t_entry"] + pd.Timedelta(minutes=H)
            g_stop[i], stopped[i] = stop_walk(df, row["t_entry"], t_exit, row["d"],
                                               row["P_fill_entry"], dist, row[f"g_{H}"])
        for i, (_, row) in enumerate(nsub.iterrows()):
            dist = val * row["ATR"] if kind == "ATRx" else val
            t_exit = row["t_entry"] + pd.Timedelta(minutes=H)
            g_stop_null[i], _ = stop_walk(df, row["t_entry"], t_exit, row["d"],
                                           row["P_fill_entry"], dist, row[f"g_{H}"])
        g_stop_s = pd.Series(g_stop)
        net = g_stop_s - COST_BASE
        pctile, n_pool = pctile_of_real_in_pool(g_stop_s, pd.Series(g_stop_null), COST_BASE, B=2000)
        rows.append({
            "stop": f"{val:g}xATR" if kind == "ATRx" else f"${val:g} fixed",
            "n": len(sub), "n_stopped": int(stopped.sum()), "pct_stopped": stopped.mean() * 100,
            "net_mean": net.mean(), "win_pct": (g_stop_s > 0).mean() * 100,
            "PF": profit_factor(net), "worst_loss": net.min(), "null_pctile": pctile,
        })
    no_stop_net = sub[f"g_{H}"] - COST_BASE
    no_stop_pct, _ = pctile_of_real_in_pool(sub[f"g_{H}"], nsub[f"g_{H}"], COST_BASE, B=2000)
    rows.insert(0, {"stop": "NO STOP (time-exit only)", "n": len(sub), "n_stopped": 0, "pct_stopped": 0.0,
                     "net_mean": no_stop_net.mean(), "win_pct": (sub[f"g_{H}"] > 0).mean() * 100,
                     "PF": profit_factor(no_stop_net), "worst_loss": no_stop_net.min(),
                     "null_pctile": no_stop_pct})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
