"""Is 15m + 2h + 4h the sweet combo? 15m is instrument-dependent (helps BTC, hurts gold), so test
15m added to the robust 2h-4h band directly. fwd_ret + per-year robustness. Gold & BTC 5m."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
K = 48


def strength_tf(d1):
    c = d1["close"]; atr = ta.atr(d1["high"], d1["low"], d1["close"], 14)
    er = c.diff(20).abs() / c.diff().abs().rolling(20).sum()
    adx = ta.adx(d1["high"], d1["low"], d1["close"], 14)["ADX_14"]
    adxN = ((adx - 15) / 25).clip(0, 1)
    emaF = c.ewm(span=20, adjust=False).mean(); emaS = c.ewm(span=50, adjust=False).mean()
    stack = np.sign(c - emaF) + np.sign(emaF - emaS) + np.sign(emaS - emaS.shift(10))
    atrexpN = ((atr / atr.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    strength = 10.0 * (er.fillna(0) + adxN.fillna(0) + (stack.abs() / 3.0).fillna(0) + atrexpN.fillna(0)) / 4.0
    return strength, stack


def upge5(d5, rule):
    d = d5.resample(rule).agg(AGG).dropna(); s, st = strength_tf(d)
    return (((st > 0) & (s >= 5)).astype(float)).shift(1).reindex(d5.index, method="ffill").fillna(0).values.astype(bool)


def analyze(name, csv, start="2018-01-01"):
    d5 = load_mt5_csv(csv).loc[start:]
    if "volume" not in d5.columns: d5["volume"] = 1.0
    G = {tf: upge5(d5, r) for tf, r in [("15m", "15min"), ("1h", "60min"), ("2h", "120min"), ("4h", "240min")]}
    c = d5["close"].values; atr = ta.atr(d5["high"], d5["low"], d5["close"], 14).values; yr = d5.index.year.values
    n = len(c); fret = np.full(n, np.nan)
    for t in range(200, n - K - 1):
        if not np.isnan(atr[t]) and atr[t] > 0: fret[t] = (c[t + K] - c[t]) / atr[t]
    valid = ~np.isnan(fret); beta = np.nanmean(fret[valid])
    byr = {y: fret[valid & (yr == y)].mean() for y in np.unique(yr)}
    print(f"\n===== {name} 5m long (beta fwd_ret={beta:+.3f}) =====")
    print(f"  {'combo':>18}{'n':>9}{'fwd_ret':>9}{'vs beta':>9}{'yrs>beta':>10}")
    def A(*k):
        m = valid.copy()
        for x in k: m = m & G[x]
        return m
    for tag, m in [("4h", A("4h")), ("2h & 4h", A("2h", "4h")), ("15m & 4h", A("15m", "4h")),
                   ("15m & 2h", A("15m", "2h")), ("15m & 2h & 4h", A("15m", "2h", "4h")),
                   ("15m & 1h & 4h", A("15m", "1h", "4h"))]:
        if m.sum() < 200: print(f"  {tag:>18}{m.sum():>9}  too few"); continue
        fr = fret[m].mean(); ys = np.unique(yr[m]); beat = tot = 0
        for y in ys:
            ym = m & (yr == y)
            if ym.sum() > 30: tot += 1; beat += 1 if fret[ym].mean() > byr[y] else 0
        print(f"  {tag:>18}{m.sum():>9}{fr:>+9.3f}{fr-beta:>+9.3f}{f'{beat}/{tot}':>10}")


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv")
    analyze("BTC", "data/vantage_btcusd_m5.csv", start="2018-10-01")
