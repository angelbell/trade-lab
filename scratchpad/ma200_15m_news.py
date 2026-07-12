"""News-spike exposure of the 15m 200MA bounce. No econ calendar in-repo -> proxy
US-data risk by UTC hour buckets (12:30=8:30ET NFP/CPI, 14:00=10:00ET ISM, FOMC pm).
(1) per-time-bucket win/meanR (is the US-data window worse?)
(2) stop OVERSHOOT = (stop - stopbar_low)/ATR on losing trades = slippage exposure."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
idx = d.index; iv = idx.values; SK, W = 20, 30
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
dd = d.resample("1D").agg(AGG).dropna(); dsm = dd.close.rolling(200).mean()
dgate = (dsm > dsm.shift(10)).shift(1).fillna(False).reindex(idx, method="ffill").fillna(False).values
r8 = d.resample("8H").agg(AGG).dropna(); g8s = r8.close > r8.close.rolling(200).mean()
g8 = g8s.shift(1).fillna(False).reindex(idx, method="ffill").fillna(False).values

cand, cnt = [], 0
for s in range(222, len(c) - 1):
    if np.isnan(sma[s - 1]) or np.isnan(a[s - 1]) or a[s - 1] <= 0: continue
    if c[s - 1] > sma[s - 1] + 1.5 * a[s - 1]: cnt = 0
    if not (sma[s - 1] > sma[s - 1 - SK] and l[s] <= sma[s - 1]): continue
    cnt += 1
    win = slice(s - W, s); ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
    vel = ((ext - sma[s - 1]) / a[s - 1]) / max(bd, 1)
    cand.append((s, iv[s], sma[s - 1], a[s - 1], vel, cnt,
                 e20[s - 1] > e20[s - 1 - 5], dgate[s], g8[s]))
A = pd.DataFrame(cand, columns=["s", "ts", "limit", "atr", "vel", "atk", "u5", "dg", "g8"])
A["ts_ns"] = pd.to_datetime(A.ts).astype("int64")

m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values.astype("datetime64[ns]").astype("int64")
o5, h5, l5, c5 = (m5[k].values for k in ("open", "high", "low", "close"))
COST = 0.40

def bucket(hr):
    if 12 <= hr < 15: return "US-data 12-15"
    if 7 <= hr < 12: return "London 07-12"
    if 15 <= hr < 22: return "US-pm 15-22"
    return "Asia 22-07"

def sim(df):
    rows, busy = [], -1
    for r in df.itertuples():
        if r.ts_ns <= busy: continue
        p0 = int(np.searchsorted(t5, r.ts_ns))
        if p0 >= len(t5): continue
        lim = r.limit; sd = r.atr; fj = None
        for j in range(p0, min(p0 + 12, len(t5))):
            if l5[j] <= lim: fj = j; break
        if fj is None: continue
        e = min(o5[fj], lim); stop = e - sd; tgt = e + sd
        R = None; over = 0.0; end = min(fj + 1500, len(t5))
        for j in range(fj, end):
            if l5[j] <= stop:
                R = -1.0; over = (stop - l5[j]) / sd; break       # how far past stop the bar dug
            if h5[j] >= tgt: R = 1.0; break
        if R is None: R = (c5[end - 1] - e) / sd
        hr = pd.Timestamp(t5[fj]).hour
        rows.append((bucket(hr), R - COST / sd, R, over))
        busy = r.ts_ns
    return pd.DataFrame(rows, columns=["buck", "R", "Rraw", "over"])

def report(name, df):
    t = sim(df)
    print(f"\n== {name} (n={len(t)}) ==")
    order = ["Asia 22-07", "London 07-12", "US-data 12-15", "US-pm 15-22"]
    print(f"  {'bucket':<16} n   win  meanR   | stop-overshoot(ATR) med/p90  (loss n)")
    for b in order:
        g = t[t.buck == b]
        if len(g) == 0: continue
        loss = g[g.Rraw <= -1.0]
        ov = loss.over.values
        om = f"{np.median(ov):.2f}/{np.percentile(ov,90):.2f}" if len(ov) else "  -  "
        print(f"  {b:<16} {len(g):>3} {(g.R>0).mean()*100:>3.0f}% {g.R.mean():+.3f}   | {om:>14}  (n{len(loss)})")
    # overall overshoot tail across all losers
    allloss = t[t.Rraw <= -1.0].over.values
    big = (allloss > 0.25).mean() * 100 if len(allloss) else 0
    print(f"  ALL losers: overshoot med={np.median(allloss):.2f} p90={np.percentile(allloss,90):.2f} "
          f"ATR; >0.25ATR past stop={big:.0f}% of losses")

report("raw 1st-touch", A[A.atk == 1])
report("gated+V (edge pop)", A[(A.atk==1) & A.u5 & A.dg & A.g8].pipe(lambda x: x[x.vel>=x.vel.quantile(0.667)]))
