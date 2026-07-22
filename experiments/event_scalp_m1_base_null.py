"""Put gold and USDJPY on the SAME ladder: w_c 1..7 x frac {1.00,0.50,0.25}, net + null %ile.

Point of the run: the gold ledger never recorded the all-signals (frac=1.00) null percentile,
so "gold's base is alive / USDJPY's base is at the 1st percentile" was never actually compared.
If gold's base is ALSO below null, the two instruments agree and the gold shoot is a
top-quartile SELECTION, not a concentration of a live base (falsifier 1).

Machinery unchanged: event_scalp / event_scalp_cond. Costs from the same table each script used.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, null_scalp_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset, pctile_of_real_in_pool

WCS = [1, 2, 3, 4, 5, 7]
HSET = [5, 10]
FRACS = [1.00, 0.50, 0.25]
DRAWS = 3000

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())

SPECS = [
    ("GOLD  ", "data/vantage_xauusd_m1.csv", COST_ROUNDTRIP["GOLD"]["base"], 1.0, "$/oz"),
    ("USDJPY", "data/vantage_usdjpy_m1.csv", 0.009, 0.01, "pip"),
]

for name, path, cost, unit, ulab in SPECS:
    df = load_mt5_csv(path)
    print(f"\n########## {name}  ({len(df)} bars, {df.index.min().date()}..{df.index.max().date()}, "
          f"cost {cost/unit:.2f} {ulab} RT) ##########")
    print(f"{'w_c':>4} | " + " | ".join(f"frac{f:.2f} H={h}: net/null%ile/n"
                                        for f in FRACS for h in HSET))
    for wc in WCS:
        real = build_scalp_table(df, events, wc, HSET, f"{name}wc{wc}")
        null = null_scalp_table(df, events, wc, HSET, f"{name}wc{wc}", draws_target=DRAWS)
        cells = []
        for frac in FRACS:
            sub, _ = threshold_subset(real, "confirm_move_atr", frac)
            nsub, _ = threshold_subset(null, "confirm_move_atr", frac) if frac < 1 else (null, -np.inf)
            for h in HSET:
                g = sub[f"g_{h}"].dropna(); gn = nsub[f"g_{h}"].dropna()
                pct, _ = pctile_of_real_in_pool(g, gn, cost)
                cells.append(f"{(g-cost).mean()/unit:+6.2f}/{pct:3.0f}/{len(g):3d}")
        print(f"{wc:>4} | " + " | ".join(cells))
