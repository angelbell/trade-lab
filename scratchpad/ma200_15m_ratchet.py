"""R-step ratchet exit (+1R->breakeven, +2R->+1R, +3R->+2R, ...) and a NO-WEEKEND-
HOLD rule (force-close at the last bar before any >6h gap = Fri close). Gated+V long,
M5 path. Compares to fixed tgt1.0, and isolates the weekend rule's effect."""
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
    if not (cnt == 1 and (e20[s - 1] > e20[s - 1 - 5]) and dgate[s] and g8[s]): continue
    cand.append((s, iv[s], sma[s - 1], a[s - 1], vel))
C = pd.DataFrame(cand, columns=["s", "ts", "limit", "atr", "vel"])
C = C[C.vel >= C.vel.quantile(0.667)].reset_index(drop=True)
C["ts_ns"] = pd.to_datetime(C.ts).astype("int64")

m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values.astype("datetime64[ns]").astype("int64")
o5, h5, l5, c5 = (m5[k].values for k in ("open", "high", "low", "close"))
COST = 0.40; GAP = 6 * 3600 * 1_000_000_000  # 6h in ns = weekend/holiday gap

def trade(p0, e, sd, mode, T, weekend):
    stop = e - sd; peak = e; end = min(p0 + 1500, len(t5)); wk = 0
    for j in range(p0, end):
        if l5[j] <= stop: return (stop - e) / sd, wk          # stop (prior-peak, conservative)
        if mode == "fixed" and h5[j] >= e + T * sd: return T, wk
        peak = max(peak, h5[j])
        if mode == "ratchet":
            k = int((peak - e) / sd)                          # full R reached
            if k >= 1: stop = max(stop, e + (k - 1) * sd)     # +1R->BE, +2R->+1R, ...
        if weekend and (j + 1 >= len(t5) or t5[j + 1] - t5[j] > GAP):
            return (c5[j] - e) / sd, 1                        # force-close before weekend gap
    return (c5[end - 1] - e) / sd, wk

def run(mode, T=1.0, weekend=False):
    res, busy, nwk = [], -1, 0
    for r in C.itertuples():
        if r.ts_ns <= busy: continue
        p0 = int(np.searchsorted(t5, r.ts_ns))
        if p0 >= len(t5): continue
        lim = r.limit; fj = None
        for j in range(p0, min(p0 + 12, len(t5))):
            if l5[j] <= lim: fj = j; break
        if fj is None: continue
        e = min(o5[fj], lim); sd = r.atr
        R, wk = trade(fj, e, sd, mode, T, weekend); nwk += wk
        res.append((t5[fj], R - COST / sd)); busy = r.ts_ns
    t = pd.DataFrame(res, columns=["time", "R"]); t["time"] = pd.to_datetime(t["time"])
    return t, nwk

def show(tag, t, nwk=None):
    x = t.R.values; t = t.assign(y=t.time.dt.year)
    span = (t.time.iloc[-1] - t.time.iloc[0]).days / 365.25
    pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else 9.0
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    extra = f" wkClose={nwk}" if nwk is not None else ""
    print(f"  {tag:<28} n={len(x)} win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={x.mean():+.3f} "
          f"med={np.median(x):+.2f} std={x.std():.2f} totR={x.sum():+.0f} max={x.max():.1f} "
          f"CDD={cdd_R(x,span)[2]:+.2f} grn={grn}/{t.y.nunique()}{extra}")

print("ratchet & weekend-close on gated+V long (M5 path)\n")
t, _ = run("fixed", 1.0); show("fixed tgt1.0 [baseline]", t)
t, _ = run("ratchet"); show("R-ratchet (run)", t)
print()
t, w = run("fixed", 1.0, True); show("fixed tgt1.0 + no-weekend", t, w)
t, w = run("ratchet", weekend=True); show("R-ratchet + no-weekend", t, w)
