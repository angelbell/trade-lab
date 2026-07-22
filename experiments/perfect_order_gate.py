"""One-shot verdict: does PERFECT ORDER (multi-MA stack alignment, the JP "don't
fight the trend" filter) beat a single trend-MA gate? Test as the gate on the
gold-15m bounce-long (our most-developed edge), M5-resolved, news-skip+slip.
PO(daily) = daily EMA20>EMA50>EMA100 all rising (price above) = trend aligned."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import cdd_R

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
idx = d.index; iv = idx.values; SK, W = 20, 30
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
dd = d.resample("1D").agg(AGG).dropna()
dsm = dd.close.rolling(200).mean()
single = (dsm > dsm.shift(10)).shift(1).fillna(False).reindex(idx, method="ffill").fillna(False).values
# Perfect Order on daily EMAs (20>50>100, rising, price>fast)
e_f, e_m, e_s = (dd.close.ewm(span=n, adjust=False).mean() for n in (20, 50, 100))
po = ((e_f > e_m) & (e_m > e_s) & (e_f > e_f.shift(3)) & (dd.close > e_f))
po = po.shift(1).fillna(False).reindex(idx, method="ffill").fillna(False).values
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
                 e20[s - 1] > e20[s - 1 - 5], single[s], po[s], g8[s]))
A = pd.DataFrame(cand, columns=["s","ts","ma","atr","vel","atk","u5","single","po","g8"])
A = A[(A.atk == 1) & A.u5 & A.g8]
A = A[A.vel >= A.vel.quantile(0.667)].reset_index(drop=True)
A["ts_ns"] = pd.to_datetime(A.ts).astype("int64")

m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values.astype("datetime64[ns]").astype("int64")
o5, h5, l5, c5 = (m5[k].values for k in ("open","high","low","close"))
COST = 0.40

def sim(df):
    res, busy = [], -1
    for r in df.itertuples():
        if r.ts_ns <= busy: continue
        if 12 <= pd.Timestamp(r.ts_ns).hour < 15: continue
        p0 = int(np.searchsorted(t5, r.ts_ns))
        if p0 >= len(t5): continue
        lim = r.ma; fj = None
        for j in range(p0, min(p0 + 12, len(t5))):
            if l5[j] <= lim: fj = j; break
        if fj is None: continue
        e = min(o5[fj], lim); sd = r.atr; stop = e - sd; tgt = e + sd
        R = None; over = 0.0; end = min(fj + 1500, len(t5))
        for j in range(fj, end):
            if l5[j] <= stop: R = -1.0; over = (stop - l5[j]) / sd; break
            if h5[j] >= tgt: R = 1.0; break
        if R is None: R = (c5[end-1]-e)/sd
        if R == -1.0: R -= 0.5 * over
        res.append((t5[fj], R - COST/sd)); busy = r.ts_ns
    t = pd.DataFrame(res, columns=["time","R"]); t["time"] = pd.to_datetime(t["time"]); return t

def line(tag, df):
    t = sim(df); x = t.R.values
    if len(x) < 12: print(f"  {tag:<26} n={len(x)} too few"); return
    span = (t.time.iloc[-1]-t.time.iloc[0]).days/365.25
    pf = x[x>0].sum()/abs(x[x<=0].sum()) if (x<=0).any() else 9.0
    print(f"  {tag:<26} n={len(x):>3} {len(x)/span:>4.1f}/yr win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} "
          f"meanR={x.mean():+.3f} CDD={cdd_R(x,span)[2]:+.2f}")

print("PERFECT ORDER vs single trend gate (gold 15m bounce-long):")
line("no-gate", A)
line("single (dSMA200-up)", A[A.single])
line("Perfect Order (daily)", A[A.po])
line("PO AND single", A[A.po & A.single])
