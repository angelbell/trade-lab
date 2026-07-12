"""Overfit audit (DSR / PBO-CSCV / bootstrap-CI+null) on the LONG 20MA-gated
15m 200MA limit-bounce. Grid = zone x Vcut x 20MA-lookback x target x stop."""
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
up = {lb: np.concatenate([np.zeros(lb, bool), e20[lb:] > e20[:-lb]]) for lb in (3, 5, 8)}

# collect candidate LONG touches once: limit at the MA, fills ONLY if price reaches it
# (low <= sma[s-1]); entry = min(open, MA). No fabricated fills.
cand, cnt = [], 0
for s in range(221, len(c) - 1):
    if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0: continue
    if c[s] > sma[s] + 1.5 * a[s]: cnt = 0
    if not (sma[s] > sma[s - SK] and l[s] <= sma[s - 1]): continue   # price actually reached the MA limit
    cnt += 1
    win = slice(s - W + 1, s + 1); ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
    vel = ((ext - l[s]) / a[s]) / max(bd, 1)
    cand.append((s, min(o[s], sma[s - 1]), l[s], sma[s], a[s], vel, cnt,
                 up[3][s], up[5][s], up[8][s]))
C = pd.DataFrame(cand, columns=["s", "e", "lo", "sma", "atr", "vel", "atk", "u3", "u5", "u8"])
C = C[C.atk == 1]                                      # first-touch always

def sim(df, tgtA, stop_usd):
    res, busy = [], -1
    for r in df.itertuples():
        s = r.s
        if s <= busy: continue
        e = r.e; stop = e - stop_usd; tgt = e + tgtA * r.atr; R = None; xj = min(s + 300, len(c) - 1)
        for j in range(s, min(s + 300, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= tgt: R = (tgt - e) / stop_usd; xj = j; break
        if R is None: R = (c[xj] - e) / stop_usd
        res.append((idx[s], R - 0.40 / stop_usd)); busy = xj
    return pd.DataFrame(res, columns=["time", "R"])

# build grid (limit fixed at MA; vary Vcut x 20MA-lookback x target x stop)
cols, srs = {}, []; cid = 0
for vcut, lb, tgt, stp in itertools.product((0.5, 0.667, 0.75), (3, 5, 8),
                                            (1.0, 1.25, 1.5), (2.5, 3.0, 4.0)):
    sub = C[C[f"u{lb}"]]
    sub = sub[sub.vel >= sub.vel.quantile(vcut)]
    if len(sub) < 30: continue
    t = sim(sub, tgt, stp)
    if len(t) < 30: continue
    m = t.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()
    cols[f"c{cid}"] = m; srs.append(t.R.mean() / t.R.std(ddof=1)); cid += 1
M = pd.concat(cols, axis=1).fillna(0.0).values
V = float(np.var(srs))
print(f"grid configs={M.shape[1]}  months={M.shape[0]}  V_SR={V:.4f}")

# flagship: limit@MA / vcut0.667 / 20ma5 / tgt1.5 / stop3
fs = C[C.u5]; fs = fs[fs.vel >= fs.vel.quantile(0.667)]
tf = sim(fs, 1.5, 3.0); R = tf.R.values
yrs = (tf.time.iloc[-1] - tf.time.iloc[0]).days / 365.25
_, sr, g1, g4 = psr(R, 0.0); tstat = sr * np.sqrt(len(R))
print(f"\nflagship LONG: n={len(R)} meanR={R.mean():+.3f} SR/tr={sr:.3f} t={tstat:.2f} skew={g1:+.2f} kurt={g4:.1f}")
print("A. DSR (trial-count haircut):  " + "  ".join(
    f"@{N}={psr(R, sr0(N, V))[0]:.2f}" for N in (1, 10, 50, 100, 200, 400)))
pbo, oosm, ploss = cscv(M)
print(f"B. PBO via CSCV = {pbo:.2f}  (IS-best mean OOS-SR={oosm:+.2f}, P(OOS loss)={ploss:.2f})  [<0.2 robust, ~0.5 noise]")
rng = np.random.default_rng(0)
obs = cdd_R(R, yrs)[2]
boot = np.array([cdd_R(block_resample(R, 20, rng), yrs)[2] for _ in range(2000)])
nullR = R - R.mean(); nul = np.array([cdd_R(block_resample(nullR, 20, rng), yrs)[2] for _ in range(2000)])
p = (nul >= obs).mean()
print(f"C. CAGR/DD obs={obs:+.2f}  bootCI[5/50/95]={np.percentile(boot,5):+.2f}/{np.percentile(boot,50):+.2f}/{np.percentile(boot,95):+.2f}  null p={p:.3f}")
