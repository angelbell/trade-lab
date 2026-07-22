"""exhaustion_bounce.py -- faithful mechanization of pine/gold_5m_exhaustion_bounce.pine
Bounce-family verification ORDER (cost=0, GROSS):
  1) BASE bounce-rate (barrier) of all-signals exhaustion+confirm entries, vs random-same-time long.
  2) Selectability: do trend gate / exhaustion depth / RSI-turn SEPARATE winners from base?
  3) Excursion (realizable MFE before 1-ATR stop): median/p90/std ATR.
  4) Only then RR: PF/N/N-yr/maxDD/win/meanR/IS-OOS/per-year, vs beta null.
No-lookahead: all setup conditions on CLOSED bars; entry fill = open[s+1] after confirm bar s.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

START = "2018-11-01"   # genuine M5 density begins here

def atr(h, l, c, n=14):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values

def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().values
    rs = ru / np.maximum(rd, 1e-12)
    return 100 - 100 / (1 + rs)

def build_signals(d, whtf=240, clusterK=5, needWicks=2, wickRatio=0.5,
                  upClosePos=0.55, confirmM=6, bigBodyATR=1.5, lookbkBig=40,
                  atrLen=14, minTgtATR=2.0):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = atr(h, l, c, atrLen)
    n = len(d)
    rng = h - l
    lwr = np.where(rng > 0, (np.minimum(o, c) - l) / np.maximum(rng, 1e-12), 0.0)
    wickflag = (lwr >= wickRatio).astype(float)
    nwick = pd.Series(wickflag).rolling(clusterK).sum().values
    clusterLow = pd.Series(l).rolling(clusterK).min().values
    # freshLow: cluster low <= lowest low of whtf window ending clusterK bars ago
    lowest_whtf = pd.Series(l).rolling(whtf).min().shift(clusterK).values
    freshLow = clusterLow <= lowest_whtf
    upBar = (c > o) & (rng > 0) & ((c - l) / np.maximum(rng, 1e-12) >= upClosePos)
    isExhaust = freshLow & (nwick >= needWicks) & upBar
    # big-down-candle target: nearest prior bar with (open-close)>=bigBodyATR*atr, its high
    bigdown = ((o - c) >= bigBodyATR * a)
    # state machine
    fills = []   # (fill_idx, arm_idx, armHigh, armLow, tgt, atr_at_confirm)
    armed = False; armBar = -1; armHigh = armLow = armTgt = np.nan
    for i in range(1, n - 1):
        if armed:
            if i - armBar > confirmM:
                armed = False
            elif l[i] < armLow:
                armed = False
            elif (c[i] > armHigh) and (not np.isnan(armTgt)) and ((armTgt - c[i]) >= minTgtATR * a[i]):
                fills.append((i + 1, armBar, armHigh, armLow, armTgt, a[i]))
                armed = False
        if isExhaust[i] and not armed:
            tgt = np.nan
            for k in range(1, lookbkBig + 1):
                if i - k < 0: break
                if bigdown[i - k]:
                    tgt = h[i - k]; break
            if not np.isnan(tgt):
                armed = True; armBar = i; armHigh = h[i]; armLow = clusterLow[i]; armTgt = tgt
    return dict(o=o, h=h, l=l, c=c, a=a, fills=fills, isExhaust=isExhaust,
                clusterLow=clusterLow, times=d.index)

def barrier(sig, UP=1.0, DOWN=1.0, fwd=48):
    """bounce-rate: from fill open, does +UP*atr hit before -DOWN*atr? (long)"""
    o, h, l, a = sig["o"], sig["h"], sig["l"], sig["a"]
    n = len(o); res = []
    for (fi, arm, aH, aL, tgt, ac) in sig["fills"]:
        if fi >= n: continue
        e = o[fi]; up = e + UP * ac; dn = e - DOWN * ac
        out = 0
        for j in range(fi, min(fi + fwd, n)):
            hitdn = l[j] <= dn; hitup = h[j] >= up
            if hitdn and hitup:  # same bar: assume stop first (conservative)
                out = -1; break
            if hitdn: out = -1; break
            if hitup: out = 1; break
        res.append((sig["times"][fi], out))
    return res

def random_baseline(sig, UP=1.0, DOWN=1.0, fwd=48, nsamp=4000, seed=1):
    o, h, l, a = sig["o"], sig["h"], sig["l"], sig["a"]
    n = len(o); rng = np.random.default_rng(seed)
    idx = rng.integers(60, n - fwd - 1, size=nsamp)
    wins = 0; tot = 0
    for fi in idx:
        if not np.isfinite(a[fi]) or a[fi] <= 0: continue
        e = o[fi]; up = e + UP * a[fi]; dn = e - DOWN * a[fi]
        out = 0
        for j in range(fi, min(fi + fwd, n)):
            hitdn = l[j] <= dn; hitup = h[j] >= up
            if hitdn and hitup: out = -1; break
            if hitdn: out = -1; break
            if hitup: out = 1; break
        if out != 0: tot += 1
        if out == 1: wins += 1
    return wins / max(tot, 1), tot

def main():
    d = load_mt5_csv("data/vantage_xauusd_m5.csv")
    d = d[d.index >= START]
    print(f"# rows {len(d)}  span {d.index[0]} -> {d.index[-1]}\n")
    sig = build_signals(d)
    nfill = len(sig["fills"])
    nyr = (d.index[-1] - d.index[0]).days / 365.25
    print(f"STEP1 BASE (all-signals exhaustion+confirm entries): N={nfill}  N/yr={nfill/nyr:.1f}")
    for UP, DOWN in [(1,1),(1,1.5),(1.5,1),(2,1)]:
        b = barrier(sig, UP, DOWN)
        outs = [x[1] for x in b if x[1] != 0]
        wr = np.mean([1 if o>0 else 0 for o in outs]) if outs else 0
        rb, rt = random_baseline(sig, UP, DOWN)
        print(f"  barrier +{UP}/-{DOWN}ATR: bounce(win)%={wr*100:4.1f}  (decisive N={len(outs)})   random-same-TF={rb*100:4.1f}%  delta={100*(wr-rb):+.1f}")
    print()

if __name__ == "__main__":
    main()
