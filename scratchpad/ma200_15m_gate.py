"""B. Regime GATE on the M5-resolved LIMIT-entry 15m 200MA long (tgt1.0/stop1.0).
Goal: turn OFF the dead years (2021/22/24) while keeping the good ones.
Gates computed on DAILY/WEEKLY resample, made causal (prior completed bar),
mapped to each 15m signal by as-of. Reports base vs each gate + per-year of the
best gate, plus a CAGR/DD random-keep% null so we don't mistake n-trimming for skill."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import cdd_R, block_resample

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
idx15 = d.index; SK, W = 20, 30

# ---- daily / weekly causal gates ----
dd = d.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
wk = d.resample("1W").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
def kama(s, n=10, f=2, sl=30):
    ch = s.diff(n).abs(); vol = s.diff().abs().rolling(n).sum(); er = (ch / vol).fillna(0)
    sc = (er * (2/(f+1) - 2/(sl+1)) + 2/(sl+1)) ** 2
    out = s.to_numpy(copy=True)
    for i in range(1, len(s)):
        if np.isnan(out[i-1]): out[i] = s.values[i]; continue
        out[i] = out[i-1] + sc.values[i] * (s.values[i] - out[i-1])
    return pd.Series(out, index=s.index)
dsm200 = dd.close.rolling(200).mean(); dsm50 = dd.close.rolling(50).mean()
dk = kama(dd.close)
GATES = {
    "none": pd.Series(True, index=dd.index),
    "D close>SMA200": (dd.close > dsm200),
    "D SMA200 up": (dsm200 > dsm200.shift(10)),
    "D KAMA up": (dk > dk.shift(3)),
    "D SMA50>200": (dsm50 > dsm200),
    "D close>SMA50": (dd.close > dsm50),
}
WGATES = {
    "W <=30MA*1.10": (wk.close <= wk.close.rolling(30).mean() * 1.10),
    "W close>30MA": (wk.close > wk.close.rolling(30).mean()),
}
gate_lut = {}
for nm, g in GATES.items():
    gg = g.shift(1).fillna(False)                       # prior completed DAY
    gate_lut[nm] = gg.reindex(idx15, method="ffill").fillna(False).values
for nm, g in WGATES.items():
    gg = g.shift(1).fillna(False)
    gate_lut[nm] = gg.reindex(idx15, method="ffill").fillna(False).values

# ---- 15m signal collection (strict causal) ----
cand, cnt = [], 0
for s in range(222, len(c) - 1):
    if np.isnan(sma[s - 1]) or np.isnan(a[s - 1]) or a[s - 1] <= 0: continue
    if c[s - 1] > sma[s - 1] + 1.5 * a[s - 1]: cnt = 0
    if not (sma[s - 1] > sma[s - 1 - SK] and l[s] <= sma[s - 1]): continue
    cnt += 1
    win = slice(s - W, s); ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
    vel = ((ext - sma[s - 1]) / a[s - 1]) / max(bd, 1)
    cand.append((s, sma[s - 1], a[s - 1], vel, cnt, e20[s - 1] > e20[s - 1 - 5]))
C = pd.DataFrame(cand, columns=["s", "limit", "atr", "vel", "atk", "u5"])
C = C[(C.atk == 1) & C.u5]
C = C[C.vel >= C.vel.quantile(0.667)].reset_index(drop=True)

m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values; o5, h5, l5, c5 = (m5[k].values for k in ("open", "high", "low", "close"))
TGT, KST, COST = 1.0, 1.0, 0.40
iv = idx15.values

def sim(df):
    res, busy_t = [], np.datetime64("1900-01-01")
    for r in df.itertuples():
        ts = iv[r.s]
        if ts <= busy_t: continue
        p0 = int(np.searchsorted(t5, ts))
        if p0 >= len(t5): continue
        lim = r.limit; e = None; end = min(p0 + 900, len(t5))
        for j in range(p0, end):
            if e is None:
                if l5[j] <= lim:
                    e = min(o5[j], lim); sd = KST * r.atr; stop = e - sd; tgt = e + TGT * r.atr
                    if l5[j] <= stop: res.append((t5[j], -1.0 - COST/sd)); busy_t = t5[j]; break
                    if h5[j] >= tgt: res.append((t5[j], TGT/KST - COST/sd)); busy_t = t5[j]; break
                continue
            if l5[j] <= stop: res.append((t5[j], -1.0 - COST/sd)); busy_t = t5[j]; break
            if h5[j] >= tgt:  res.append((t5[j], TGT/KST - COST/sd)); busy_t = t5[j]; break
        else:
            if e is not None:
                res.append((t5[end-1], (c5[end-1]-e)/sd - COST/sd)); busy_t = t5[end-1]
    return pd.DataFrame(res, columns=["time", "R"])

def stats(t):
    x = t.R.values; t = t.assign(y=pd.to_datetime(t.time).dt.year)
    yrs = sorted(t.y.unique()); half = yrs[len(yrs)//2]
    pf = x[x>0].sum()/abs(x[x<=0].sum()) if (x<=0).any() else 9.0
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    span = (t.time.iloc[-1]-t.time.iloc[0]).days/365.25
    cdd = cdd_R(x, span)[2]
    return len(x), (x>0).mean()*100, pf, x.mean(), t[t.y<half].R.mean(), t[t.y>=half].R.mean(), grn, t.y.nunique(), cdd

base = sim(C)
print("B. regime gate on M5-resolved LIMIT long (tgt1.0/stop1.0):")
print(f"  {'GATE':<16} n   win  PF   meanR   IS    OOS   grn   CAGR/DD")
for nm in ["none","D close>SMA200","D SMA200 up","D KAMA up","D SMA50>200","D close>SMA50","W <=30MA*1.10","W close>30MA"]:
    keep = gate_lut[nm][C.s.values]
    t = sim(C[keep].reset_index(drop=True))
    if len(t) < 8: print(f"  {nm:<16} n={len(t)} too few"); continue
    n, w, pf, mr, is_, oos, g, ny, cdd = stats(t)
    print(f"  {nm:<16} {n:>3} {w:>3.0f}% {pf:.2f} {mr:+.3f} {is_:+.2f} {oos:+.2f} {g}/{ny}  {cdd:+.2f}")

# random keep-% null for the best structural gate (so we don't reward pure n-trimming)
best = "D close>SMA200"
keep = gate_lut[best][C.s.values]; tb = sim(C[keep].reset_index(drop=True))
obs_cdd = stats(tb)[-1]; kp = keep.mean()
rng = np.random.default_rng(0); span = (base.time.iloc[-1]-base.time.iloc[0]).days/365.25
nul = []
for _ in range(1000):
    m = rng.random(len(C)) < kp
    tt = sim(C[m].reset_index(drop=True))
    if len(tt) >= 8: nul.append(cdd_R(tt.R.values, span)[2])
nul = np.array(nul)
print(f"\n random keep~{kp*100:.0f}% null: gate '{best}' CAGR/DD={obs_cdd:+.2f} vs null med={np.median(nul):+.2f} "
      f"pctile={(nul<obs_cdd).mean()*100:.0f}%")
print(f"\nper-year (gate={best}):")
keep = gate_lut[best][C.s.values]; t = sim(C[keep].reset_index(drop=True)); t = t.assign(y=pd.to_datetime(t.time).dt.year)
for y, g in t.groupby("y"):
    gx = g.R.values
    print(f"  {y}: n={len(gx):>3} win={(gx>0).mean()*100:>3.0f}% sumR={gx.sum():+6.1f} meanR={gx.mean():+.2f}")
