"""STEP 0 tie-back for spec card: FOMC statement gold 1min scalp -- execution realism /
stop-loss / sequential threshold.

Reproduces the currently-reported numbers using the EXISTING machinery unchanged:
  event_scalp.build_scalp_table / null_scalp_table
  event_scalp_cond.threshold_subset / pctile_of_real_in_pool
  event_scalp.is_oos_table

w_c=1min, H=5min, cost=$0.30/oz round trip, events = experiments/fomc_stmt_2019.csv (dt_broker).
frac=1.00 (all signals) and frac=0.50 (top half by confirm_move_atr).

Expected (per spec card, to confirm before doing anything else):
  frac=1.00 -> net ~= +0.09 / null%ile ~= 95
  frac=0.50 -> net ~= +1.97 / null%ile 100, win% 66, IS +0.94 / OOS +2.92, n=29
If this does not match, STOP and report -- do not proceed to STEP A/B/C.
"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, null_scalp_table, is_oos_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset, pctile_of_real_in_pool

W_C = 1
H = 5
COST = COST_ROUNDTRIP["GOLD"]["base"]  # 0.30
DRAWS = 3000

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
print(f"total candidate events in fomc_stmt_2019.csv: {len(events)}")

df = load_mt5_csv("data/vantage_xauusd_m1.csv")
print(f"GOLD m1: {len(df)} bars, span {df.index.min()} .. {df.index.max()}")

real = build_scalp_table(df, events, W_C, [H], "GOLD-step0")
null = null_scalp_table(df, events, W_C, [H], "GOLD-step0", draws_target=DRAWS)
print(f"usable real events (w_c={W_C}): {len(real)}")

for frac in [1.00, 0.50]:
    sub, thr = threshold_subset(real, "confirm_move_atr", frac)
    nsub = null if frac >= 1.0 else null[null["confirm_move_atr"] >= thr]
    g = sub[f"g_{H}"].dropna()
    gn = nsub[f"g_{H}"].dropna()
    net = g - COST
    pct, n_pool = pctile_of_real_in_pool(g, gn, COST, B=2000)
    print(f"\n--- frac={frac:.2f} (thr={thr:.4f}) ---")
    print(f"  n={len(g)}  net_mean={net.mean():+.4f}  win%={ (g>0).mean()*100:.1f}  "
          f"null%ile={pct:.1f} (pool n={n_pool})")

    ist, span_desc = is_oos_table(sub, [H], COST)
    print(f"  IS/OOS split ({span_desc}):")
    print(ist.round(4).to_string(index=False))
