"""巡行幅 (excursion) of the 15m 200MA long bounce, measured on M5 path in ATR units.
Per the verification order: bounce-rate -> selectability -> EXCURSION -> then RR.
For each population we report, in ATR units (median/std/quantiles):
  MFE   = max favorable (high-e)/ATR over the forward window (unconditional reach)
  MAE   = max adverse  (e-low)/ATR  (how deep it digs first)
  rMFE  = peak (high-e)/ATR reached BEFORE first hitting the 1.0-ATR stop
          = the realizable upside given a 1ATR stop -> the CEILING on RR."""
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
HMAX = 480     # forward window in M5 bars (~40h) for the unconditional reach

def excursion(df):
    mfe, mae, rmfe = [], [], []
    for r in df.itertuples():
        p0 = int(np.searchsorted(t5, r.ts_ns))
        if p0 >= len(t5): continue
        lim = r.limit; e = None; sd = r.atr; end = min(p0 + HMAX, len(t5))
        peak = -9; deep = 0; rpeak = -9; stopped = False
        for j in range(p0, end):
            if e is None:
                if l5[j] <= lim: e = min(o5[j], lim); stop = e - sd
                else: continue
            peak = max(peak, (h5[j] - e) / sd)
            deep = max(deep, (e - l5[j]) / sd)
            if not stopped:
                if l5[j] <= stop: stopped = True
                else: rpeak = max(rpeak, (h5[j] - e) / sd)
        if e is None: continue
        mfe.append(peak); mae.append(deep); rmfe.append(max(rpeak, 0.0))
    return np.array(mfe), np.array(mae), np.array(rmfe)

def show(tag, df):
    if len(df) < 20: print(f"  {tag:<22} (n={len(df)} too few)"); return
    mfe, mae, rmfe = excursion(df)
    def q(x): return f"med={np.median(x):.2f} std={x.std():.2f} [p25={np.percentile(x,25):.2f} p75={np.percentile(x,75):.2f} p90={np.percentile(x,90):.2f}]"
    print(f"  {tag:<22} n={len(mfe)}")
    print(f"      MFE  (ATR): {q(mfe)}")
    print(f"      MAE  (ATR): {q(mae)}")
    print(f"      rMFE before 1ATR-stop: {q(rmfe)}  | reached>=1.0R: {(rmfe>=1.0).mean()*100:.0f}%  >=1.5R: {(rmfe>=1.5).mean()*100:.0f}%  >=2.0R: {(rmfe>=2.0).mean()*100:.0f}%")

print("巡行幅 of the 15m 200MA long bounce (M5 path, ATR units, window ~40h)\n")
show("raw 1st-touch", A[A.atk == 1])
show("1st + steep-V", A[(A.atk == 1) & (A.vel >= A[A.atk==1].vel.quantile(0.667))])
sel = A[(A.atk == 1) & A.u5 & A.dg & A.g8]
show("gated (D+8H+u5)", sel)
show("gated + steep-V", sel[sel.vel >= sel.vel.quantile(0.667)])
