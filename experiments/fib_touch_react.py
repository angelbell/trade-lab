"""Does the LINE itself produce a bounce? First-TOUCH reaction rate at fib levels
(38.2/50/61.8) of a ZigZag impulse vs (a) same-hour beta, (b) FAKE lines = random depth
u~U[0.30,0.90] on the SAME impulses (the decisive null: is the fib RATIO special, or does
any line inside a pullback 'bounce' equally?). Race +-0.5 and +-1.0 ATR from touch close.
USDJPY 15m + GOLD 15m control. Confluence subset reported too."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

WATCH, K = 96, 96
def screen(name, d, seed=7):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c); span = (d.index[-1]-d.index[0]).days/365.25
    races = {}
    for B in (0.5, 1.0):
        t_up = np.full(n, K+1, np.int32); t_dn = np.full(n, K+1, np.int32)
        for k in range(1, K+1):
            hs, ls = np.empty(n), np.empty(n)
            hs[:n-k], ls[:n-k] = h[k:], l[k:]
            hs[n-k:], ls[n-k:] = -np.inf, np.inf
            t_up = np.where((t_up > K) & (hs >= c + B*atr), k, t_up)
            t_dn = np.where((t_dn > K) & (ls <= c - B*atr), k, t_dn)
        races[B] = (np.minimum(t_up, t_dn) <= K) & (t_up < t_dn)
    valid = ~np.isnan(atr) & (np.arange(n) < n-K)
    hours = d.index.hour.values
    beta = {B: {hh: races[B][valid & (hours == hh)].mean() for hh in range(24)} for B in races}

    sw = swings_zigzag(h, l, np.where(np.isnan(atr), np.nanmean(atr), atr), 2.0)
    lows = [(cc, pp) for cc, ii, pp, kk in sw if kk == -1]
    rng = np.random.default_rng(seed)
    def touches(levels_fn):
        evs = []
        for t in range(1, len(sw)):
            cH, iH, pH, kH = sw[t]; cL, iL, pL, kL = sw[t-1]
            if not (kH == 1 and kL == -1 and pH > pL): continue
            if cH >= n-K or np.isnan(atr[cH]) or atr[cH] <= 0: continue
            prior = [p for cc, p in lows if cc < cH][-30:]
            for z in levels_fn(pH, pL):
                conf = any(abs(p - z)/atr[cH] <= 0.5 for p in prior)
                for j in range(cH+1, min(cH+WATCH, n-K)):
                    if l[j] <= z:
                        evs.append((j, conf)); break
                    if l[j] < pL: break
        return evs
    real = touches(lambda pH, pL: [pH - f*(pH-pL) for f in (0.382, 0.5, 0.618)])
    fake = touches(lambda pH, pL: [pH - u*(pH-pL) for u in rng.uniform(0.30, 0.90, 3)])
    print(f"\n===== {name} ({span:.1f}yr) first-TOUCH reaction (long side) =====")
    for tag, evs in [("real fib", real), ("real fib∩構造線", [e for e in real if e[1]]),
                     ("FAKE lines (乱数深さ)", fake)]:
        idx = np.array([e[0] for e in evs])
        if len(idx) < 50: print(f"  {tag:<22} too few"); continue
        line = f"  {tag:<22} n={len(idx):5d}"
        for B in (0.5, 1.0):
            w = races[B][idx].mean()*100
            b = np.mean([beta[B][hh] for hh in hours[idx]])*100
            line += f"  ±{B}: {w:4.1f}% (beta {b:4.1f}, diff {w-b:+4.1f})"
        print(line)

jp = load_mt5_csv("data/vantage_usdjpy_m15.csv")
cnt = jp.groupby(jp.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 80]
screen("USDJPY 15m", jp[jp.index.date >= ok.index[0]])
gd = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
screen("GOLD 15m", gd.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna())
