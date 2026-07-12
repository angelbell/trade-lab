"""Verify the '1h strength>=5 & up -> 5m long' gate. Disentangle a possible ARTIFACT: the 2:1
ATR-barrier meanR can differ just because high-vol periods have wider barriers (fewer stops), not
because price goes up more. So measure BOTH:
  (a) fwd_ret = (c[t+K]-c[t]) / atr[t]   -- BARRIER-FREE directional travel (the real question)
  (b) 2:1 barrier R (+cost) + stop-rate  -- the practical outcome
Per-year robustness of (a) for UP&>=5 vs UP&weak<5 (one-era check). Gold & BTC 5m."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
K, TGT, STP = 48, 2.0, 1.0


def strength_1h(d1):
    c = d1["close"]; atr = ta.atr(d1["high"], d1["low"], d1["close"], 14)
    er = c.diff(20).abs() / c.diff().abs().rolling(20).sum()
    adx = ta.adx(d1["high"], d1["low"], d1["close"], 14)["ADX_14"]
    adxN = ((adx - 15) / 25).clip(0, 1)
    emaF = c.ewm(span=20, adjust=False).mean(); emaS = c.ewm(span=50, adjust=False).mean()
    stack = np.sign(c - emaF) + np.sign(emaF - emaS) + np.sign(emaS - emaS.shift(10))
    align = stack.abs() / 3.0
    atrexpN = ((atr / atr.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    strength = 10.0 * (er.fillna(0) + adxN.fillna(0) + align.fillna(0) + atrexpN.fillna(0)) / 4.0
    return strength, stack


def analyze(name, csv, cost, start="2018-01-01"):
    d5 = load_mt5_csv(csv).loc[start:]
    if "volume" not in d5.columns: d5["volume"] = 1.0
    d1 = d5.resample("60min").agg(AGG).dropna()
    st1, stack1 = strength_1h(d1)
    st = st1.shift(1).reindex(d5.index, method="ffill").values
    dv = np.sign(stack1).shift(1).reindex(d5.index, method="ffill").values
    c = d5["close"].values; h = d5["high"].values; l = d5["low"].values
    atr = ta.atr(d5["high"], d5["low"], d5["close"], 14).values
    yr = d5.index.year.values
    n = len(c); rows = []
    for t in range(200, n - K - 1):
        if np.isnan(atr[t]) or atr[t] <= 0 or np.isnan(st[t]): continue
        fret = (c[t + K] - c[t]) / atr[t]
        tgt = c[t] + TGT * atr[t]; stp = c[t] - STP * atr[t]; R = None
        for j in range(t + 1, t + 1 + K):
            if l[j] <= stp: R = -STP; break
            if h[j] >= tgt: R = TGT; break
        if R is None: R = (c[t + K] - c[t]) / atr[t]
        rows.append((st[t], dv[t], fret, R, cost / atr[t], yr[t]))
    df = pd.DataFrame(rows, columns=["st1", "dir1", "fret", "R", "costR", "yr"])
    df["Rnet"] = df.R - df.costR; df["win"] = df.R > 0; df["stop"] = df.R == -STP
    print(f"\n===== {name} 5m long =====  n={len(df)}  (cost {cost}/ATR med={df.costR.median():.3f}R)")
    print(f"  {'bucket':>18}{'n':>8}{'fwd_ret':>9}{'2:1 meanR':>11}{'net':>8}{'win%':>6}{'stop%':>6}")
    def row(tag, m):
        g = df[m]
        if len(g) < 50: print(f"  {tag:>18}{len(g):>8}  too few"); return
        print(f"  {tag:>18}{len(g):>8}{g.fret.mean():>+9.3f}{g.R.mean():>+11.3f}{g.Rnet.mean():>+8.3f}"
              f"{g.win.mean()*100:>5.0f}%{g.stop.mean()*100:>5.0f}%")
    up5 = (df.dir1 > 0) & (df.st1 >= 5); upw = (df.dir1 > 0) & (df.st1 < 5)
    row("all (beta)", df.index >= 0)
    row("UP & >=5", up5)
    row("UP & weak <5", upw)
    row("DOWN", df.dir1 < 0)
    print("  per-year fwd_ret [UP>=5 / UP<5 / diff]:")
    for y in sorted(df.yr.unique()):
        a = df[up5 & (df.yr == y)].fret; b = df[upw & (df.yr == y)].fret
        if len(a) > 30 and len(b) > 30:
            print(f"    {y}: {a.mean():+.3f} / {b.mean():+.3f} / {a.mean()-b.mean():+.3f}"
                  + ("  <-- weak wins" if a.mean() < b.mean() else ""))


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv", 0.6)
    analyze("BTC", "data/vantage_btcusd_m5.csv", 15.0, start="2018-10-01")
