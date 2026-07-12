"""Paired head-to-head: candidate (kama4h&stack 20/50/10) vs incumbents (kama4h, kama4h&1d).
Same months resampled together (paired block bootstrap, L=5 months) -> distribution of the
ret/DD DIFFERENCE + P(cand > incumbent). Also per-year win count. Answers the narrow question
PBO can't: is the candidate reliably better than the gate it would replace?"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE, comps_tf, kama_up
from audit_kama_stack import stack4h
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

d = load_mt5_csv("data/vantage_btcusd_m15.csv")
d15 = resample(d[d.index >= "2018-10-01"], "15min")
t = run(d15, SimpleNamespace(**BASE, pullback_frac=0.3))
Rn = t["R"].values - 15.0 / t["risk"].values
pos = d15.index.get_indexer(t["time"])
mon = t["time"].dt.to_period("M")
k4, k1 = kama_up(d15, "240min"), kama_up(d15, "1D")
G = {"cand": (k4 & stack4h(d15, 20, 50, 10))[pos],
     "kama4h": k4[pos], "kama4h&1d": (k4 & k1)[pos]}
midx = pd.period_range(mon.min(), mon.max(), freq="M")
S = {k: pd.Series(Rn[m], index=mon[m]).groupby(level=0).sum().reindex(midx, fill_value=0.0).values
     for k, m in G.items()}

def retdd(x):
    eq = np.cumsum(x); dd = (np.maximum.accumulate(eq) - eq).max()
    return x.sum() / max(dd, 1e-9)

yr = midx.year.values
print("per-year totR (cand / kama4h / kama4h&1d) + winner vs each incumbent:")
for y in np.unique(yr):
    a, b, c = (S[k][yr == y].sum() for k in ("cand", "kama4h", "kama4h&1d"))
    print(f"  {y}: {a:+6.1f} / {b:+6.1f} / {c:+6.1f}   vs4h:{'W' if a>b else 'L'}  vsAND:{'W' if a>c else 'L'}")

rng = np.random.default_rng(7)
T = len(midx); L = 5; B = 4000
for inc in ("kama4h", "kama4h&1d"):
    diffs = []
    for _ in range(B):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T)
            idx.extend(range(s, min(s + L, T)))
        idx = np.array(idx[:T])
        diffs.append(retdd(S["cand"][idx]) - retdd(S[inc][idx]))
    diffs = np.array(diffs)
    print(f"\ncand - {inc}: ret/DD diff  med={np.median(diffs):+.2f}  sd={diffs.std():.2f}"
          f"  P(cand>{inc})={(diffs>0).mean()*100:.1f}%  CI5/95=[{np.percentile(diffs,5):+.2f},{np.percentile(diffs,95):+.2f}]")
