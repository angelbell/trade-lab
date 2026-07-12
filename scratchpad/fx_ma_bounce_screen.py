"""Screen: does FX (USDJPY, a documented mean-reverter) respect the 200MA bounce
better than trend-follower gold, and is the LENGTH a plateau (watched-level theory)?
Per verification order step 1-2: BOUNCE RATE (barrier, robust to intrabar order) +
length sensitivity + a quick limit-entry meanR (H1-OHLC approx -> read bounce rate as
primary; meanR is intrabar-approximate). BOTH sides. USDJPY H1 & 4H vs GOLD H1."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

def barrier_long(h, l, s, ref, up, dn, fwd):
    hi = ref + up; lo = ref - dn
    for j in range(s + 1, min(s + 1 + fwd, len(h))):
        if l[j] <= lo: return 0
        if h[j] >= hi: return 1
    return 0

def barrier_short(h, l, s, ref, up, dn, fwd):
    hi = ref + up; lo = ref - dn
    for j in range(s + 1, min(s + 1 + fwd, len(h))):
        if h[j] >= hi: return 0
        if l[j] <= lo: return 1
    return 0

def screen(name, d, n, cost, kind, fwd=60):
    cl = d["close"]
    ma = (cl.ewm(span=n, adjust=False).mean() if kind == "EMA" else cl.rolling(n).mean()).values
    at = ta.atr(d["high"], d["low"], d["close"], 14).values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    SK = 20
    lr_b, lr_m, sr_b, sr_m = [], [], [], []
    for s in range(n + SK, len(c) - 1):
        if np.isnan(ma[s - 1]) or np.isnan(at[s - 1]) or at[s - 1] <= 0: continue
        up_tr = ma[s - 1] > ma[s - 1 - SK]
        # LONG: rising-ish MA touched from above (low<=MA), confirmed prior bar
        if l[s] <= ma[s - 1] and c[s - 1] > ma[s - 1]:
            br = barrier_long(h, l, s, ma[s - 1], at[s - 1], at[s - 1], fwd)
            lr_b.append(br)
            e = min(o[s], ma[s - 1]); sd = at[s - 1]
            R = br * 1.0 + (1 - br) * (-1.0) - cost / sd
            lr_m.append(R)
        # SHORT: MA touched from below (high>=MA)
        if h[s] >= ma[s - 1] and c[s - 1] < ma[s - 1]:
            br = barrier_short(h, l, s, ma[s - 1], at[s - 1], at[s - 1], fwd)
            sr_b.append(br)
            e = max(o[s], ma[s - 1]); sd = at[s - 1]
            R = br * 1.0 + (1 - br) * (-1.0) - cost / sd
            sr_m.append(R)
    def fmt(b, m):
        b = np.array(b); m = np.array(m)
        return f"n={len(b):>4} bounce={b.mean()*100:>4.1f}% meanR(RR1)={m.mean():+.3f}"
    print(f"  {name:<14} {kind}{n}  LONG  {fmt(lr_b, lr_m)}")
    print(f"  {'':<14} {kind}{n}  SHORT {fmt(sr_b, sr_m)}")

gj = load_mt5_csv("data/vantage_usdjpy_h1.csv")
gj4 = gj.resample("4H").agg(AGG).dropna()
xg = load_mt5_csv("data/vantage_xauusd_h1.csv")
print("breakeven bounce-rate at RR1 = 50%.  (cost: USDJPY 1.5pip=0.015, gold $0.40)\n")
for kind in ("SMA", "EMA"):
    print(f"== USDJPY H1 (16yr) — {kind} ==")
    for n in (100, 150, 200, 250): screen("USDJPY H1", gj, n, 0.015, kind)
    print(f"\n== USDJPY 4H — {kind} ==")
    for n in (100, 150, 200, 250): screen("USDJPY 4H", gj4, n, 0.015, kind)
    print(f"\n== GOLD H1 benchmark — {kind} ==")
    for n in (100, 150, 200, 250): screen("GOLD H1", xg, n, 0.40, kind)
    print()
