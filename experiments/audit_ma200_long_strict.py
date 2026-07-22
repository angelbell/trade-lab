"""STRICT-CAUSAL overfit audit on the LONG 20MA-gated 15m 200MA limit-bounce.
All indicator conditions use CONFIRMED [s-1] values (no same-bar-close peek):
the limit sits at the prior-bar 200MA; fills mid-bar s when low[s]<=sma[s-1];
trend/20MA-gate/V all evaluated at s-1. This is the realizable, mechanizable form."""
import sys, itertools; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values; a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
idx = d.index; SK, W = 20, 30

# strict-causal candidate collection (all conditions confirmed at s-1; limit @ sma[s-1])
cand, cnt = [], 0
for s in range(222, len(c) - 1):
    if np.isnan(sma[s - 1]) or np.isnan(a[s - 1]) or a[s - 1] <= 0:
        continue
    if c[s - 1] > sma[s - 1] + 1.5 * a[s - 1]:
        cnt = 0
    if not (sma[s - 1] > sma[s - 1 - SK] and l[s] <= sma[s - 1]):   # limit @ confirmed MA, mid-bar fill
        continue
    cnt += 1
    win = slice(s - W, s)                                          # PRIOR window (confirmed)
    ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
    vel = ((ext - sma[s - 1]) / a[s - 1]) / max(bd, 1)             # descent to the fill level
    cand.append((s, min(o[s], sma[s - 1]), a[s - 1], vel, cnt,
                 e20[s - 1] > e20[s - 1 - 3], e20[s - 1] > e20[s - 1 - 5], e20[s - 1] > e20[s - 1 - 8]))
C = pd.DataFrame(cand, columns=["s", "e", "atr", "vel", "atk", "u3", "u5", "u8"])
C = C[C.atk == 1]

def sim(df, tgtA, kATR):
    res, busy = [], -1
    for r in df.itertuples():
        s = r.s
        if s <= busy: continue
        e = r.e; sd = kATR * r.atr; stop = e - sd; tgt = e + tgtA * r.atr; R = None; xj = min(s + 300, len(c) - 1)
        for j in range(s, min(s + 300, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= tgt: R = tgtA / kATR; xj = j; break
        if R is None: R = (c[xj] - e) / sd
        res.append((idx[s], R - 0.40 / sd)); busy = xj
    return pd.DataFrame(res, columns=["time", "R"])

cols, srs = {}, []; cid = 0
for vcut, lb, tgt, kst in itertools.product((0.5, 0.667, 0.75), (3, 5, 8), (1.0, 1.25, 1.5), (0.75, 1.0, 1.5)):
    sub = C[C[f"u{lb}"]]; sub = sub[sub.vel >= sub.vel.quantile(vcut)]
    if len(sub) < 30: continue
    t = sim(sub, tgt, kst)
    if len(t) < 30: continue
    m = t.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()
    cols[f"c{cid}"] = m; srs.append(t.R.mean() / t.R.std(ddof=1)); cid += 1
M = pd.concat(cols, axis=1).fillna(0.0).values; V = float(np.var(srs))
print(f"STRICT grid configs={M.shape[1]} months={M.shape[0]} V_SR={V:.4f}")

fs = C[C.u5]; fs = fs[fs.vel >= fs.vel.quantile(0.667)]
tf = sim(fs, 1.5, 1.0); R = tf.R.values; yrs = (tf.time.iloc[-1] - tf.time.iloc[0]).days / 365.25
_, sr, g1, g4 = psr(R, 0.0)
print(f"\nflagship STRICT LONG: n={len(R)} meanR={R.mean():+.3f} SR/tr={sr:.3f} t={sr*np.sqrt(len(R)):.2f} skew={g1:+.2f} kurt={g4:.1f}")
print("A. DSR: " + "  ".join(f"@{N}={psr(R, sr0(N, V))[0]:.2f}" for N in (1, 10, 50, 100, 200, 400)))
pbo, oosm, pl = cscv(M)
print(f"B. PBO={pbo:.2f} (IS-best OOS-SR={oosm:+.2f}, P(OOS loss)={pl:.2f})")
rng = np.random.default_rng(0); obs = cdd_R(R, yrs)[2]
boot = np.array([cdd_R(block_resample(R, 20, rng), yrs)[2] for _ in range(2000)])
nr = R - R.mean(); nul = np.array([cdd_R(block_resample(nr, 20, rng), yrs)[2] for _ in range(2000)])
print(f"C. CAGR/DD obs={obs:+.2f} bootCI[5/50/95]={np.percentile(boot,5):+.2f}/{np.percentile(boot,50):+.2f}/{np.percentile(boot,95):+.2f} null p={(nul>=obs).mean():.3f}")
