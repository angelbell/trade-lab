"""5m 200MA bounce (same method as 15m, dropped one TF). Signal on 5m;
limit @ confirmed 5m-200MA; daily+8H trend gate; first-touch + steep-V + 20MAup.
Execution: (1) full 19yr on 5m-OHLC atomic fills [INFLATED — 5m bar can't resolve
intra-bar order, same trap 15m-OHLC had]; (2) recent 0.56yr 5m-OHLC vs 1m-RESOLVED
to MEASURE the inflation and read the honest 5m number. tgt1.0/stop1.0, cost$0.30 (real RAW)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import cdd_R

d = load_mt5_csv("data/vantage_xauusd_m5.csv"); cl = d["close"]
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
    u5 = e20[s - 1] > e20[s - 1 - 5]
    if not (cnt == 1 and u5 and dgate[s] and g8[s]): continue
    cand.append((s, iv[s], sma[s - 1], a[s - 1], vel))
C = pd.DataFrame(cand, columns=["s", "ts", "limit", "atr", "vel"])
C = C[C.vel >= C.vel.quantile(0.667)].reset_index(drop=True)
print(f"5m signals (1st+V+u5+D+8H): {len(C)}  span {C.ts.min()} -> {C.ts.max()}")

COST = 0.30
C["ts_ns"] = pd.to_datetime(C.ts).astype("int64")
def sim(df, ts_arr, oa, ha, la, ca):
    """generic limit-fill sim on whatever sub-bar arrays are passed (5m or 1m).
    ts_arr = int64-ns timestamps of those sub-bars."""
    tns = ts_arr.astype("datetime64[ns]").astype("int64")
    res, busy = [], -1
    for r in df.itertuples():
        if r.ts_ns <= busy: continue
        p0 = int(np.searchsorted(tns, r.ts_ns))
        if p0 >= len(tns): continue
        lim = r.limit; e = None; end = min(p0 + 1500, len(tns))
        for j in range(p0, end):
            if e is None:
                if la[j] <= lim:
                    e = min(oa[j], lim); sd = r.atr; stop = e - sd; tgt = e + sd
                    if la[j] <= stop: res.append((tns[j], -1.0 - COST/sd)); busy = tns[j]; break
                    if ha[j] >= tgt: res.append((tns[j], 1.0 - COST/sd)); busy = tns[j]; break
                continue
            if la[j] <= stop: res.append((tns[j], -1.0 - COST/sd)); busy = tns[j]; break
            if ha[j] >= tgt:  res.append((tns[j], 1.0 - COST/sd)); busy = tns[j]; break
        else:
            if e is not None: res.append((tns[end-1], (ca[end-1]-e)/sd - COST/sd)); busy = tns[end-1]
    out = pd.DataFrame(res, columns=["time", "R"]); out["time"] = pd.to_datetime(out["time"])
    return out

def stat(tag, t):
    if len(t) < 8: print(f"  {tag:<30} n={len(t)} too few"); return
    x = t.R.values; t = t.assign(y=pd.to_datetime(t.time).dt.year)
    span = (t.time.iloc[-1]-t.time.iloc[0]).days/365.25
    pf = x[x>0].sum()/abs(x[x<=0].sum()) if (x<=0).any() else 9.0
    half = sorted(t.y.unique())[t.y.nunique()//2]
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    print(f"  {tag:<30} n={len(x):>4} {len(x)/span:>4.0f}/yr win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} "
          f"meanR={x.mean():+.3f} IS={t[t.y<half].R.mean():+.2f} OOS={t[t.y>=half].R.mean():+.2f} grn={grn}/{t.y.nunique()} CDD={cdd_R(x,span)[2]:+.2f}")

print("\n== (1) FULL 19yr, 5m-OHLC atomic execution [INFLATED] ==")
stat("5m-OHLC full", sim(C, iv, o, h, l, c))

# (2) recent window: 5m-OHLC vs 1m-resolved on the same signals
m1 = load_mt5_csv("data/vantage_xauusd_m1.csv")
t1 = m1.index.values; o1, h1, l1, c1 = (m1[k].values for k in ("open", "high", "low", "close"))
Crec = C[C.ts >= t1[0]].reset_index(drop=True)
print(f"\n== (2) recent {(t1[-1]-t1[0]).astype('timedelta64[D]').astype(int)}d window, same {len(Crec)} signals ==")
stat("5m-OHLC (recent)", sim(Crec, iv, o, h, l, c))
stat("1m-RESOLVED (recent)", sim(Crec, t1, o1, h1, l1, c1))
