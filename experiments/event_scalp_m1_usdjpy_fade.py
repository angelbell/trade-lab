"""USDJPY: is the statement impulse actually FADED (the mirror of gold's continuation)?

The base-null ladder showed USDJPY sitting BELOW the same-time random null at almost every
confirm window (1st/0th percentile at w_c=1/7) while gold sits at 95-100th. That is not
"no signal" -- it is a signal with the opposite sign. Measure it directly: take the trade
against the confirm move, same cost, same null (reversed), plus per-year signs.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, null_scalp_table
from experiments.event_scalp_cond import threshold_subset, pctile_of_real_in_pool

PIP = 0.01; COST = 0.9 * PIP; COST2 = 1.8 * PIP
WCS = [1, 2, 3, 4, 5, 7]; HSET = [5, 10, 15]

def pf(x):
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return np.inf if neg == 0 else pos / neg

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_utc", "dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
df = load_mt5_csv("data/vantage_usdjpy_m1.csv")

print("FADE side (trade AGAINST the confirm move), all signals, cost 0.9 pip RT")
print(f"{'w_c':>4} | " + " | ".join(f"H={h}: net/net1.8/win%/PF/null%ile" for h in HSET))
tabs = {}
for wc in WCS:
    real = build_scalp_table(df, events, wc, HSET, f"f{wc}")
    null = null_scalp_table(df, events, wc, HSET, f"f{wc}", draws_target=3000)
    tabs[wc] = real
    cells = []
    for h in HSET:
        g = -real[f"g_{h}"].dropna(); gn = -null[f"g_{h}"].dropna()
        net = g - COST
        pct, _ = pctile_of_real_in_pool(g, gn, COST)
        cells.append(f"{net.mean()/PIP:+5.2f}/{(g-COST2).mean()/PIP:+5.2f}/"
                     f"{(g > COST).mean()*100:3.0f}/{pf(net):4.2f}/{pct:3.0f}")
    print(f"{wc:>4} | " + " | ".join(cells))

print("\nPer-year net pip (FADE, H=5, all signals)")
rows = {}
for wc in WCS:
    t = tabs[wc].dropna(subset=["g_5"]).copy(); t["yr"] = t["t0"].dt.year
    rows[f"wc{wc}"] = t.groupby("yr").apply(lambda s: (-s["g_5"] - COST).mean() / PIP)
print(pd.DataFrame(rows).round(2).to_string())
