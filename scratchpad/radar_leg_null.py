"""Random-drop null for the radar subset of the kama4h BTC 15m cell: is radar's
DD-cut real selection or just n-trimming? Sample equal-size random subsets of the
kama4h trade set, compare totR/maxDD and maxDD percentiles vs the radar subset."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_leg_gate import ARGS, RT, radar_up, kama_up
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

d = load_mt5_csv("data/vantage_btcusd_m15.csv")
d15 = resample(d[d.index >= "2018-10-01"], "15min")
g_k4 = kama_up(d15, "240min")
g_r = radar_up(d15, "120min") & radar_up(d15, "240min")
t = run(d15, SimpleNamespace(**ARGS))
Rn = t["R"].values - RT / t["risk"].values
pos = d15.index.get_indexer(t["time"])
k4 = g_k4[pos]; rk = k4 & g_r[pos]
base = Rn[k4]; sub = Rn[rk]
def stats(x):
    eq = np.cumsum(x); dd = (np.maximum.accumulate(eq) - eq).max()
    return x.sum() / dd, dd
rd_real, dd_real = stats(sub)
rng = np.random.default_rng(7)
rd_n, dd_n = [], []
for _ in range(2000):
    idx = np.sort(rng.choice(len(base), size=len(sub), replace=False))
    r, dd = stats(base[idx])
    rd_n.append(r); dd_n.append(dd)
rd_n, dd_n = np.array(rd_n), np.array(dd_n)
print(f"radar&kama subset (n={len(sub)} of {len(base)}): totR/maxDD={rd_real:.2f} maxDD={dd_real:.1f}R")
print(f"random-drop null totR/maxDD: med={np.median(rd_n):.2f} sd={rd_n.std():.2f}  radar %ile={(rd_n<rd_real).mean()*100:.1f}")
print(f"random-drop null maxDD:      med={np.median(dd_n):.1f}R sd={dd_n.std():.1f}  radar %ile(lower=better)={(dd_n>dd_real).mean()*100:.1f}")
print(f"kama4h full set: totR/maxDD={stats(base)[0]:.2f} maxDD={stats(base)[1]:.1f}R totR={base.sum():.0f}")
