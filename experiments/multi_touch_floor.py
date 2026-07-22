"""User claim: REPEATED bounces at the same floor -> visible upward flow (accumulation).
Different from the killed 'retouch reaction' -- this is about DIRECTIONAL DRIFT after
multiple defenses. Event: a ZigZag swing-low CONFIRMS whose price sits within 0.3*ATR of
k-1 prior confirmed swing-lows (lookback 20 lows). Measure from the confirm bar close:
(a) +-1 ATR race, (b) barrier-free forward drift (c[+96]-c)/ATR, both vs same-hour beta.
Rows: k=1 (lone low) / k=2 (double) / k>=3 (triple+). USDJPY 15m + GOLD 5m + GOLD 15m."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

K = 96
def screen(name, d):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c); span = (d.index[-1]-d.index[0]).days/365.25
    t_up = np.full(n, K+1, np.int32); t_dn = np.full(n, K+1, np.int32)
    for k in range(1, K+1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n-k], ls[:n-k] = h[k:], l[k:]
        hs[n-k:], ls[n-k:] = -np.inf, np.inf
        t_up = np.where((t_up > K) & (hs >= c + atr), k, t_up)
        t_dn = np.where((t_dn > K) & (ls <= c - atr), k, t_dn)
    race = (np.minimum(t_up, t_dn) <= K) & (t_up < t_dn)
    fret = np.full(n, np.nan)
    fret[:n-K] = (c[K:] - c[:n-K])
    fret = fret / atr
    valid = ~np.isnan(atr) & ~np.isnan(fret)
    hours = d.index.hour.values
    bR = {hh: race[valid & (hours == hh)].mean() for hh in range(24)}
    bD = {hh: np.nanmean(fret[valid & (hours == hh)]) for hh in range(24)}

    sw = swings_zigzag(h, l, np.where(np.isnan(atr), np.nanmean(atr), atr), 2.0)
    lows = [(cc, pp) for cc, ii, pp, kk in sw if kk == -1]
    evs = []          # (confirm_bar, k_touches)
    for i, (cc, pp) in enumerate(lows):
        if cc >= n - K or np.isnan(atr[cc]) or atr[cc] <= 0: continue
        prior = lows[max(0, i-20):i]
        kk = 1 + sum(1 for _, p2 in prior if abs(p2 - pp) <= 0.3 * atr[cc])
        evs.append((cc, kk))
    idx = np.array([e[0] for e in evs]); kt = np.array([e[1] for e in evs])
    print(f"\n===== {name} ({span:.1f}yr, confirmed swing-lows n={len(idx)}) =====")
    print(f"  {'k touches':<12}{'n':>6}{'race+1ATR':>10}{'beta':>6}{'diff':>6} | {'drift/ATR':>10}{'beta':>7}{'diff':>7}")
    for tag, m in [("k=1", kt == 1), ("k=2", kt == 2), ("k>=3", kt >= 3)]:
        if m.sum() < 40: print(f"  {tag:<12} n={m.sum()} too few"); continue
        w = race[idx[m]].mean()*100
        br = np.mean([bR[hh] for hh in hours[idx[m]]])*100
        dr = np.nanmean(fret[idx[m]])
        bd = np.mean([bD[hh] for hh in hours[idx[m]]])
        print(f"  {tag:<12}{m.sum():>6}{w:>9.1f}%{br:>6.1f}{w-br:>+6.1f} | {dr:>+10.3f}{bd:>+7.3f}{dr-bd:>+7.3f}")

jp = load_mt5_csv("data/vantage_usdjpy_m15.csv")
cnt = jp.groupby(jp.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 80]
screen("USDJPY 15m", jp[jp.index.date >= ok.index[0]])
g5 = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
screen("GOLD 5m", g5)
screen("GOLD 15m", g5.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna())
