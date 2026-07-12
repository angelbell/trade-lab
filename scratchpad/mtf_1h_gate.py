"""'1h strength >=5 -> is a 5m LONG better?'  MTF conditioning (NOT forecasting strength -- the
1h trend is a SLOW state that persists over many 5m bars, so conditioning on it is contemporaneous).
Compute a causal 1h trend-strength (0-10) from the last COMPLETED 1h bar; map to 5m. Bucket 5m long
outcomes (+2ATR / -1ATR, K=48) by 1h strength band x 1h direction. Key checks: does 'up & strong'
beat 'up & weak' (strength adds beyond direction?) and beat the overall 5m-long beta (edge, not drift?)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
K, TGT, STP = 48, 2.0, 1.0


def strength_1h(d1):
    c = d1["close"]; atr = ta.atr(d1["high"], d1["low"], d1["close"], 14)
    er = c.diff(20).abs() / c.diff().abs().rolling(20).sum()
    [_p, _m, adx] = [ta.dmi(14, 14)[i] for i in range(3)] if False else (None, None, ta.adx(d1["high"], d1["low"], d1["close"], 14)["ADX_14"])
    adxN = ((adx - 15) / 25).clip(0, 1)
    emaF = c.ewm(span=20, adjust=False).mean(); emaS = c.ewm(span=50, adjust=False).mean()
    s1 = np.sign(c - emaF); s2 = np.sign(emaF - emaS); s3 = np.sign(emaS - emaS.shift(10))
    stack = s1 + s2 + s3
    align = stack.abs() / 3.0
    atrexp = (atr / atr.rolling(100).mean())
    atrexpN = ((atrexp - 0.8) / 0.7).clip(0, 1)
    strength = 10.0 * (er.fillna(0) + adxN.fillna(0) + align.fillna(0) + atrexpN.fillna(0)) / 4.0
    return strength, stack   # stack sign = direction


def analyze(name, csv, start="2018-01-01"):
    d5 = load_mt5_csv(csv).loc[start:]
    if "volume" not in d5.columns: d5["volume"] = 1.0
    d1 = d5.resample("60min").agg(AGG).dropna()
    st1, stack1 = strength_1h(d1)
    # use the LAST COMPLETED 1h bar (shift 1), map onto 5m
    st = st1.shift(1).reindex(d5.index, method="ffill")
    dirp = np.sign(stack1).shift(1).reindex(d5.index, method="ffill")
    c = d5["close"].values; h = d5["high"].values; l = d5["low"].values
    atr = ta.atr(d5["high"], d5["low"], d5["close"], 14).values
    stv = st.values; dv = dirp.values
    n = len(c); rows = []
    for t in range(200, n - K - 1):
        if np.isnan(atr[t]) or atr[t] <= 0 or np.isnan(stv[t]): continue
        tgt = c[t] + TGT * atr[t]; stp = c[t] - STP * atr[t]; R = None
        for j in range(t + 1, t + 1 + K):
            if l[j] <= stp: R = -STP; break
            if h[j] >= tgt: R = TGT; break
        if R is None: R = (c[t + K] - c[t]) / atr[t]
        rows.append((stv[t], dv[t], R))
    df = pd.DataFrame(rows, columns=["st1", "dir1", "R"])
    df["win"] = df.R > 0; df["stop"] = df.R == -STP
    print(f"\n===== {name} 5m long, gated by 1h strength =====  n={len(df)}")
    print(f"  overall (beta): win={df.win.mean()*100:.0f}% stop={df.stop.mean()*100:.0f}% meanR={df.R.mean():+.3f}")
    print(f"  {'1h bucket':>22}{'n':>8}{'win%':>7}{'stop%':>7}{'meanR':>9}")
    def row(tag, m):
        g = df[m]
        if len(g) < 50: print(f"  {tag:>22}{len(g):>8}   too few"); return
        print(f"  {tag:>22}{len(g):>8}{g.win.mean()*100:>6.0f}%{g.stop.mean()*100:>6.0f}%{g.R.mean():>+9.3f}")
    row("UP & strong>=7", (df.dir1 > 0) & (df.st1 >= 7))
    row("UP & mid 5-7", (df.dir1 > 0) & (df.st1 >= 5) & (df.st1 < 7))
    row("UP & weak <5", (df.dir1 > 0) & (df.st1 < 5))
    row("FLAT (dir 0)", df.dir1 == 0)
    row("DOWN (any strength)", df.dir1 < 0)
    row(">> UP & strength>=5", (df.dir1 > 0) & (df.st1 >= 5))


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv")
    analyze("BTC", "data/vantage_btcusd_m5.csv", start="2018-10-01")
