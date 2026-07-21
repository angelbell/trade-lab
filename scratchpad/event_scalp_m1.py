"""FOMC statement scalp at 1-min resolution: does a faster confirm window (1/2/3 min)
beat the 5-min confirm? Reuses event_scalp / event_scalp_cond machinery unchanged.
Data: data/vantage_xauusd_m1.csv (2019-2026, built 2026-07-19 via mt5-mcp bridge).
Events: 2019+ FOMC statements (every meeting has a 14:30 ET presser; edge lives here)."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.event_scalp import (build_scalp_table, null_scalp_table, atr14, COST_ROUNDTRIP)
from scratchpad.event_scalp_cond import threshold_subset, pctile_of_real_in_pool

COST = COST_ROUNDTRIP["GOLD"]["base"]          # $0.30/oz
COST2 = COST_ROUNDTRIP["GOLD"]["alt"][0]        # $0.60/oz
WCS = [1, 2, 3, 5]
HSET = [5, 10, 15]
FRACS = [1.00, 0.50, 0.33, 0.25]
DRAWS = 3000

ev = pd.read_csv("scratchpad/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
print(f"FOMC 2019+ statements: {len(events)}")

df = load_mt5_csv("data/vantage_xauusd_m1.csv")
print(f"m1 data: {len(df)} bars, span {df.index.min()} .. {df.index.max()}")

for wc in WCS:
    real = build_scalp_table(df, events, wc, HSET, f"wc{wc}")
    nyr = (real["t0"].max() - real["t0"].min()).days / 365.25
    null = null_scalp_table(df, events, wc, HSET, f"wc{wc}", draws_target=DRAWS)
    print(f"\n===== w_c={wc}min  (usable n={len(real)}, {len(real)/nyr:.1f}/yr, null pool {len(null)}) =====")
    print(f"{'frac':>5} {'thr_Catr':>8} {'n':>4} | " +
          " | ".join(f"H={h}: net/net60/win%/nullpct" for h in HSET))
    for frac in FRACS:
        sub, thr = threshold_subset(real, "confirm_move_atr", frac)
        nsub, _ = threshold_subset(null, "confirm_move_atr", frac) if frac < 1 else (null, -np.inf)
        cells = []
        for h in HSET:
            g = sub[f"g_{h}"].dropna()
            gn = nsub[f"g_{h}"].dropna()
            net = (g - COST).mean()
            net2 = (g - COST2).mean()
            win = (g > 0).mean() * 100
            pct, _ = pctile_of_real_in_pool(g, gn, COST) if len(gn) >= 5 else (np.nan, 0)
            cells.append(f"{net:+.2f}/{net2:+.2f}/{win:.0f}/{pct:.0f}")
        print(f"{frac:>5.2f} {thr:>8.3f} {len(sub):>4} | " + " | ".join(cells))

# IS/OOS for the headline cell: w_c that looks best x frac=0.50 x H=5/10
print("\n===== IS/OOS (frac=0.50, H=5 & H=10) by w_c =====")
for wc in WCS:
    real = build_scalp_table(df, events, wc, HSET, f"wc{wc}")
    sub, _ = threshold_subset(real, "confirm_move_atr", 0.50)
    sub = sub.sort_values("t0"); half = len(sub) // 2
    for h in [5, 10]:
        isg = sub.iloc[:half][f"g_{h}"].dropna(); oosg = sub.iloc[half:][f"g_{h}"].dropna()
        print(f"  w_c={wc} H={h}: IS net={ (isg-COST).mean():+.2f} (n{len(isg)}) | "
              f"OOS net={ (oosg-COST).mean():+.2f} (n{len(oosg)})")
