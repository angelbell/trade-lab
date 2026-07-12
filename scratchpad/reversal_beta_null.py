"""BETA NULL on the high-TF long-fade: does the EXTREME trigger (BB / extMA) beat
just 'buy any dip below the mean' (anyDip)? If BB-L PF ~ anyDip-L PF, the extreme adds
nothing = it's mean-drift/beta, not a reversion edge. Report PF + N + N/yr per TF."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

def fade(o, h, l, c, ma, s, side, e, stopd, cost):
    if stopd <= 0: return None
    lvl = ma[s]
    if side == 1:
        if lvl <= e: return None
        stop = e - stopd
        for j in range(s + 1, min(s + 1 + 200, len(c))):
            if l[j] <= stop: return -1.0 - cost / stopd
            if h[j] >= lvl: return (lvl - e) / stopd - cost / stopd
        return (c[min(s + 200, len(c) - 1)] - e) / stopd - cost / stopd
    if lvl >= e: return None
    stop = e + stopd
    for j in range(s + 1, min(s + 1 + 200, len(c))):
        if h[j] >= stop: return -1.0 - cost / stopd
        if l[j] <= lvl: return (e - lvl) / stopd - cost / stopd
    return (e - c[min(s + 200, len(c) - 1)]) / stopd - cost / stopd

def pfn(v):
    v = np.array(v)
    if len(v) < 12: return "  -"
    pf = v[v > 0].sum() / abs(v[v <= 0].sum()) if (v <= 0).any() else 9.9
    return f"{pf:.2f}/{len(v)}"

def run(name, base, cost, tfs, yrs):
    print(f"\n===== {name}  PF/n  (anyDip = beta null; does BB/extMA beat it?) =====")
    print(f"  {'TF':<5}{'anyDip-L':>12}{'BB-L':>12}{'extMA-L':>12} | {'anyRally-S':>12}{'BB-S':>12}")
    for lbl, fr in tfs:
        df = base if fr is None else base.resample(fr).agg(AGG).dropna()
        if len(df) < 300: continue
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        atr = ta.atr(df["high"], df["low"], df["close"], 14).values
        ma = df["close"].rolling(20).mean().values
        sd = df["close"].rolling(20).std().values
        ub, lb = ma + 2 * sd, ma - 2 * sd
        R = {k: [] for k in ("anyDipL", "BBL", "extL", "anyRallyS", "BBS")}
        for s in range(30, len(c) - 1):
            if np.isnan(atr[s]) or atr[s] <= 0 or np.isnan(ma[s]): continue
            a = atr[s]; e = o[s + 1]
            if c[s] < ma[s]:
                r = fade(o, h, l, c, ma, s + 1, 1, e, a, cost)
                if r is not None: R["anyDipL"].append(r)
            if c[s] > ma[s]:
                r = fade(o, h, l, c, ma, s + 1, -1, e, a, cost)
                if r is not None: R["anyRallyS"].append(r)
            if c[s] < lb[s]:
                r = fade(o, h, l, c, ma, s + 1, 1, e, a, cost)
                if r is not None: R["BBL"].append(r)
            if c[s] > ub[s]:
                r = fade(o, h, l, c, ma, s + 1, -1, e, a, cost)
                if r is not None: R["BBS"].append(r)
            if c[s] <= ma[s] - 2 * a:
                r = fade(o, h, l, c, ma, s + 1, 1, e, a, cost)
                if r is not None: R["extL"].append(r)
        print(f"  {lbl:<5}{pfn(R['anyDipL']):>12}{pfn(R['BBL']):>12}{pfn(R['extL']):>12} | {pfn(R['anyRallyS']):>12}{pfn(R['BBS']):>12}")

goldm5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
jpy = load_mt5_csv("data/vantage_usdjpy_h1.csv")
run("USDJPY", jpy, 0.015, [("4h","240min"),("8h","480min"),("1d","1440min")], 16)
run("GOLD", goldm5, 0.40, [("4h","240min"),("8h","480min"),("1d","1440min")], 19)
run("BTC", btc, 15.0, [("4h","240min"),("8h","480min"),("1d","1440min")], 8)
