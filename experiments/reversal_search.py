"""SEARCH for a real REVERSAL edge (mean-reversion from an extreme), incl. the user's
confluence: RSI>=80/<=20 + price at a HTF HH/LL swing zone + a climax candle (big
bull/bear bar). Decisive lens: a real reversal edge is SYMMETRIC on a mean-reverter
(USDJPY) and FAILS as beta on trend-followers (gold/BTC, only the with-trend side wins).
Entry next bar open, RR1, 1-ATR stop, cost. Report win% + meanR per (trigger, side)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

def prep(d):
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    rsi = ta.rsi(d["close"], 14).values
    ma = d["close"].rolling(20).mean().values
    sd = d["close"].rolling(20).std().values
    ub, lb = ma + 2 * sd, ma - 2 * sd
    # HTF (4H) swing zone, causal (prior completed 4H bars), mapped to H1
    r4 = d.resample("4H").agg(AGG).dropna()
    sh = r4.high.rolling(20).max().shift(1).reindex(d.index, method="ffill").values
    sl = r4.low.rolling(20).min().shift(1).reindex(d.index, method="ffill").values
    rng = h - l
    return o, h, l, c, atr, rsi, ma, ub, lb, sh, sl, rng

def barrier(o, h, l, c, s, side, e, sd, RR, cost):
    if sd <= 0: return None
    if side == 1:
        stop = e - sd; tgt = e + RR * sd
        for j in range(s + 1, min(s + 1 + 200, len(c))):
            if l[j] <= stop: return -1.0 - cost / sd
            if h[j] >= tgt: return RR - cost / sd
        return (c[min(s + 200, len(c) - 1)] - e) / sd - cost / sd
    stop = e + sd; tgt = e - RR * sd
    for j in range(s + 1, min(s + 1 + 200, len(c))):
        if h[j] >= stop: return -1.0 - cost / sd
        if l[j] <= tgt: return RR - cost / sd
    return (e - c[min(s + 200, len(c) - 1)]) / sd - cost / sd

def fade(o, h, l, c, ma, s, side, e, stopd, cost):
    """natural fade exit: target = the mean (MA20), stop = extreme +/- stopd. meanR."""
    if stopd <= 0: return None
    if side == -1:                                   # short: revert DOWN to mean
        stop = e + stopd; lvl = ma[s]
        if lvl >= e: return None                     # mean not below -> no fade
        for j in range(s + 1, min(s + 1 + 200, len(c))):
            if h[j] >= stop: return -1.0 - cost / stopd
            if l[j] <= lvl: return (e - lvl) / stopd - cost / stopd
        return (e - c[min(s + 200, len(c) - 1)]) / stopd - cost / stopd
    else:                                            # long: revert UP to mean
        stop = e - stopd; lvl = ma[s]
        if lvl <= e: return None
        for j in range(s + 1, min(s + 1 + 200, len(c))):
            if l[j] <= stop: return -1.0 - cost / stopd
            if h[j] >= lvl: return (lvl - e) / stopd - cost / stopd
        return (c[min(s + 200, len(c) - 1)] - e) / stopd - cost / stopd

def screen(name, d, cost, RR=1.0):
    o, h, l, c, atr, rsi, ma, ub, lb, sh, sl, rng = prep(d)
    n = len(c); tol = 0.5
    triggers = {}; fades = {}
    for s in range(30, n - 1):
        if np.isnan(atr[s]) or atr[s] <= 0 or np.isnan(rsi[s]): continue
        a = atr[s]
        big_bull = c[s] > o[s] and rng[s] >= 1.5 * a
        big_bear = c[s] < o[s] and rng[s] >= 1.5 * a
        near_hi = not np.isnan(sh[s]) and abs(h[s] - sh[s]) <= tol * a
        near_lo = not np.isnan(sl[s]) and abs(l[s] - sl[s]) <= tol * a
        def add(key, side):
            e = o[s + 1]; r = barrier(o, h, l, c, s, side, e, a, RR, cost)
            if r is not None: triggers.setdefault(key, []).append(1 if r > 0 else 0)
            rf = fade(o, h, l, c, ma, s + 1, side, e, a, cost)   # mean-target fade exit
            if rf is not None: fades.setdefault(key, []).append(rf)
        # generic extension triggers (fade), both sides
        if rsi[s] >= 70: add("RSI70 S", -1)
        if rsi[s] <= 30: add("RSI30 L", 1)
        if rsi[s] >= 80: add("RSI80 S", -1)
        if rsi[s] <= 20: add("RSI20 L", 1)
        if c[s] > ub[s]: add("BB S", -1)
        if c[s] < lb[s]: add("BB L", 1)
        if c[s] >= ma[s] + 2 * a: add("extMA S", -1)
        if c[s] <= ma[s] - 2 * a: add("extMA L", 1)
        if s >= 4 and all(c[s - k] > c[s - k - 1] for k in range(4)): add("4up S", -1)
        if s >= 4 and all(c[s - k] < c[s - k - 1] for k in range(4)): add("4dn L", 1)
        # USER's confluence: RSI extreme + HTF HH/LL zone + climax bar
        if rsi[s] >= 80 and near_hi and big_bull: add("USER S", -1)
        if rsi[s] <= 20 and near_lo and big_bear: add("USER L", 1)
    print(f"\n== {name} H1  [RR1 win%]  +  [fade-to-mean meanR/win%] ==")
    for k in ("RSI70 S","RSI30 L","RSI80 S","RSI20 L","BB S","BB L","extMA S","extMA L","4up S","4dn L","USER S","USER L"):
        v = triggers.get(k, []); f = fades.get(k, [])
        if len(v) >= 10:
            fs = f"fade meanR={np.mean(f):+.3f} win={np.mean([1 if x>0 else 0 for x in f])*100:.0f}% (n{len(f)})" if len(f) >= 10 else ""
            print(f"  {k:<9} RR1 n={len(v):>4} win={np.mean(v)*100:>4.1f}%   |  {fs}")
        else:
            print(f"  {k:<9} n={len(v):>4} (too few)")

gold = load_mt5_csv("data/vantage_xauusd_h1.csv")
btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
jpy = load_mt5_csv("data/vantage_usdjpy_h1.csv")
for label, d, cost in [("USDJPY", jpy, 0.015), ("GOLD", gold, 0.40), ("BTC", btc, 15.0)]:
    screen(label, d, cost)
