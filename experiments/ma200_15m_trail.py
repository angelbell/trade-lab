"""R1: does a runner-capturing exit (trail / scale-out) beat fixed tgt1.0 over the
PERIOD (totR / CAGR-DD), by harvesting the right tail (rMFE p90 ~10 ATR)?
Gated+V long population, M5 path. Conservative intrabar order: check stop (from
peak through prior bar) BEFORE updating peak with the current bar's high."""
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
COST = 0.40

def trade(p0, e, sd, mode, P):
    stop = e - sd; peak = e; half = False; locked = 0.0; rem = 1.0
    end = min(p0 + 1500, len(t5))
    for j in range(p0, end):
        # 1) stop check uses stop derived from peak through PRIOR bar (conservative)
        if l5[j] <= stop:
            return locked + rem * (stop - e) / sd
        # 2) scale-out target on this bar's high
        if mode == "scale" and not half and h5[j] >= e + P["t1"] * sd:
            locked += 0.5 * P["t1"]; rem = 0.5; half = True; stop = max(stop, e)
        # 3) fixed full target
        if mode == "fixed" and h5[j] >= e + P["T"] * sd:
            return P["T"]
        # 4) update peak + trailing stop
        peak = max(peak, h5[j])
        if mode in ("trail", "scale"):
            act = P["act"]; tr = P["tr"]
            if mode == "scale" and not half:
                pass  # trail only the remainder after scaling
            elif peak >= e + act * sd:
                stop = max(stop, peak - tr * sd)
        if mode == "be" and peak >= e + P["be"] * sd:
            stop = max(stop, e)
    return locked + rem * (c5[end - 1] - e) / sd

def run(mode, P):
    res, busy = [], -1
    for r in C.itertuples():
        if r.ts_ns <= busy: continue
        p0 = int(np.searchsorted(t5, r.ts_ns))
        if p0 >= len(t5): continue
        lim = r.limit
        # find M5 fill bar (low<=limit); entry=min(open,limit)
        fj = None
        for j in range(p0, min(p0 + 12, len(t5))):
            if l5[j] <= lim: fj = j; break
        if fj is None: continue
        e = min(o5[fj], lim); sd = r.atr
        R = trade(fj, e, sd, mode, P) - COST / sd
        res.append((t5[fj], R)); busy = int(np.searchsorted(t5, r.ts_ns))  # no-overlap by signal time
        busy = r.ts_ns
    t = pd.DataFrame(res, columns=["time", "R"]); t["time"] = pd.to_datetime(t["time"])
    return t

def show(tag, t):
    x = t.R.values; t = t.assign(y=t.time.dt.year)
    span = (t.time.iloc[-1] - t.time.iloc[0]).days / 365.25
    pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else 9.0
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    print(f"  {tag:<26} n={len(x)} win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={x.mean():+.3f} "
          f"med={np.median(x):+.2f} std={x.std():.2f} totR={x.sum():+.0f} max={x.max():.1f} "
          f">=3R={ (x>=3).mean()*100:>3.0f}% CDD={cdd_R(x,span)[2]:+.2f} grn={grn}/{t.y.nunique()}")

print("R1 exits on gated+V long (M5 path). Baseline first.\n")
show("fixed tgt1.0", run("fixed", {"T": 1.0}))
show("fixed tgt1.5", run("fixed", {"T": 1.5}))
show("fixed tgt2.0", run("fixed", {"T": 2.0}))
print()
show("trail act1.0 tr1.0", run("trail", {"act": 1.0, "tr": 1.0}))
show("trail act1.0 tr1.5", run("trail", {"act": 1.0, "tr": 1.5}))
show("trail act1.0 tr2.0", run("trail", {"act": 1.0, "tr": 2.0}))
show("trail act1.5 tr1.5", run("trail", {"act": 1.5, "tr": 1.5}))
show("trail act2.0 tr2.0", run("trail", {"act": 2.0, "tr": 2.0}))
print()
show("scale.5@1 trail1.5", run("scale", {"t1": 1.0, "act": 1.0, "tr": 1.5}))
show("scale.5@1 trail2.0", run("scale", {"t1": 1.0, "act": 1.0, "tr": 2.0}))
