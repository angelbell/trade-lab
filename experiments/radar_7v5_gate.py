"""Did adding MACD-hist + RSI actually make the METER a better GATE (not just orthogonal)?
Build the 4H strength 4 ways (5-comp / +MACDhist / +RSI / 7-comp), gate 5m longs by
'4H up & strength in top P% (matched selectivity, so scale shift doesn't confound)', and compare
forward return + per-year robustness. Orthogonal != helpful -- acceleration may just add noise."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
K, P = 48, 0.40   # forward bars; keep top P fraction of UP bars (matched selectivity)


def strengths_4h(d5):
    d = d5.resample("240min").agg(AGG).dropna()
    c = d["close"]; atr = ta.atr(d["high"], d["low"], d["close"], 14)
    er = (c.diff(20).abs() / c.diff().abs().rolling(20).sum()).clip(0, 1)
    adxN = ((ta.adx(d["high"], d["low"], d["close"], 14)["ADX_14"] - 15) / 25).clip(0, 1)
    emaF = c.ewm(span=20, adjust=False).mean(); emaS = c.ewm(span=50, adjust=False).mean()
    stack = np.sign(c - emaF) + np.sign(emaF - emaS) + np.sign(emaS - emaS.shift(10))
    align = stack.abs() / 3.0
    slopeN = ((emaS - emaS.shift(10)).abs() / (atr * 1.5)).clip(0, 1)
    atrexpN = ((atr / atr.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    rsiN = ((ta.rsi(c, 14) - 50).abs() / 50).clip(0, 1)
    hist = ta.macd(c, 12, 26, 9)["MACDh_12_26_9"]
    macdN = (hist.abs() / (atr * 1.0)).clip(0, 1)
    base5 = er + adxN + align + slopeN + atrexpN
    V = {"5-comp": base5 / 5, "+MACDhist": (base5 + macdN) / 6,
         "+RSI": (base5 + 0.5 * rsiN) / 5.5, "7-comp": (base5 + macdN + 0.5 * rsiN) / 6.5}
    out = {}
    for k, s in V.items():
        out[k] = s.shift(1).reindex(d5.index, method="ffill").values
    up = ((stack > 0).astype(float)).shift(1).reindex(d5.index, method="ffill").fillna(0).values.astype(bool)
    return out, up


def analyze(name, csv, start="2018-01-01"):
    d5 = load_mt5_csv(csv).loc[start:]
    if "volume" not in d5.columns: d5["volume"] = 1.0
    V, up = strengths_4h(d5)
    c = d5["close"].values; atr = ta.atr(d5["high"], d5["low"], d5["close"], 14).values; yr = d5.index.year.values
    n = len(c); fret = np.full(n, np.nan)
    for t in range(200, n - K - 1):
        if not np.isnan(atr[t]) and atr[t] > 0: fret[t] = (c[t + K] - c[t]) / atr[t]
    valid = ~np.isnan(fret); beta = np.nanmean(fret[valid])
    byr = {y: fret[valid & (yr == y)].mean() for y in np.unique(yr)}
    print(f"\n===== {name} — 4H gate: UP & top {P:.0%} strength (beta fwd_ret={beta:+.3f}) =====")
    print(f"  {'variant':>12}{'n':>9}{'fwd_ret':>9}{'vs beta':>9}{'yrs>beta':>10}")
    for k, s in V.items():
        m0 = valid & up & ~np.isnan(s)
        if m0.sum() < 200: continue
        thr = np.nanpercentile(s[m0], 100 * (1 - P))       # top P% within UP bars
        m = m0 & (s >= thr)
        fr = fret[m].mean(); ys = np.unique(yr[m]); beat = tot = 0
        for y in ys:
            ym = m & (yr == y)
            if ym.sum() > 30: tot += 1; beat += 1 if fret[ym].mean() > byr[y] else 0
        print(f"  {k:>12}{m.sum():>9}{fr:>+9.3f}{fr-beta:>+9.3f}{f'{beat}/{tot}':>10}")


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv")
    analyze("BTC", "data/vantage_btcusd_m5.csv", start="2018-10-01")
