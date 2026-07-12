"""Overfit audit on the M5-resolved, daily+intermediate-TF-gated 15m 200MA long.
Flagship = D-SMA200up & 8H close>SMA200, limit@MA, tgt1.0/stop1.0, M5 fills.
Grid (for DSR trial-haircut + PBO/CSCV) = gateTF{2H,4H,8H} x gateType{c>SMA200,
SMA200up,c>SMA50,SMA50>200} stacked on D-SMA200up, x target{0.75,1.0,1.5} x
stop{1.0,1.5}. Plus bootstrap CAGR/DD CI + mean-removed null."""
import sys, itertools; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
idx15 = d.index; iv = idx15.values; SK, W = 20, 30
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

dd = d.resample("1D").agg(AGG).dropna()
dsm200 = dd.close.rolling(200).mean()
dgate = (dsm200 > dsm200.shift(10)).shift(1).fillna(False).reindex(idx15, method="ffill").fillna(False).values
def tgset(close):
    s200 = close.rolling(200).mean(); s50 = close.rolling(50).mean()
    return {"c>SMA200": close > s200, "SMA200up": s200 > s200.shift(10),
            "c>SMA50": close > s50, "SMA50>200": s50 > s200}
ltf = {}
for tf in ["2H", "4H", "8H"]:
    r = d.resample(tf).agg(AGG).dropna()
    for nm, g in tgset(r.close).items():
        ltf[f"{tf} {nm}"] = g.shift(1).fillna(False).reindex(idx15, method="ffill").fillna(False).values

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
C = C[(C.atk == 1) & C.u5]; C = C[C.vel >= C.vel.quantile(0.667)].reset_index(drop=True)

m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values; o5, h5, l5, c5 = (m5[k].values for k in ("open", "high", "low", "close"))
COST = 0.40

def sim(df, TGT, KST):
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

sv = C.s.values
cols, srs = {}, []; cid = 0
for tf, ty, TGT, KST in itertools.product(["2H","4H","8H"],
        ["c>SMA200","SMA200up","c>SMA50","SMA50>200"], (0.75,1.0,1.5), (1.0,1.5)):
    keep = dgate[sv] & ltf[f"{tf} {ty}"][sv]
    if keep.sum() < 40: continue
    t = sim(C[keep].reset_index(drop=True), TGT, KST)
    if len(t) < 30: continue
    m = t.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()
    cols[f"c{cid}"] = m; srs.append(t.R.mean()/t.R.std(ddof=1)); cid += 1
M = pd.concat(cols, axis=1).fillna(0.0).values; V = float(np.var(srs))
print(f"grid configs={M.shape[1]} months={M.shape[0]} V_SR={V:.4f}")

keep = dgate[sv] & ltf["8H c>SMA200"][sv]
tf = sim(C[keep].reset_index(drop=True), 1.0, 1.0); R = tf.R.values
yrs = (tf.time.iloc[-1]-tf.time.iloc[0]).days/365.25
_, sr, g1, g4 = psr(R, 0.0)
print(f"\nflagship D+8H c>SMA200 (tgt1.0/stop1.0): n={len(R)} meanR={R.mean():+.3f} SR/tr={sr:.3f} "
      f"t={sr*np.sqrt(len(R)):.2f} skew={g1:+.2f} kurt={g4:.1f} freq={len(R)/yrs:.1f}/yr")
print("A. DSR: " + "  ".join(f"@{N}={psr(R, sr0(N, V))[0]:.2f}" for N in (1,10,50,100,200,400)))
pbo, oosm, pl = cscv(M)
print(f"B. PBO={pbo:.2f} (IS-best OOS-SR={oosm:+.2f}, P(OOS loss)={pl:.2f})")
rng = np.random.default_rng(0); obs = cdd_R(R, yrs)[2]
boot = np.array([cdd_R(block_resample(R, 20, rng), yrs)[2] for _ in range(2000)])
nr = R - R.mean(); nul = np.array([cdd_R(block_resample(nr, 20, rng), yrs)[2] for _ in range(2000)])
print(f"C. CAGR/DD obs={obs:+.2f} bootCI[5/50/95]={np.percentile(boot,5):+.2f}/{np.percentile(boot,50):+.2f}/"
      f"{np.percentile(boot,95):+.2f} null p={(nul>=obs).mean():.3f}")
