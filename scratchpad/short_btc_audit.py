"""A8 BTC-short full verification: (1) per-year, (2) +-1 plateau (zz-k x frac x PDL-threshold),
(3) DSR + block-bootstrap CAGR/DD + mean-removed null + PBO/CSCV over the config universe,
(4) time-exclusivity & monthly corr vs the BTC 15m LONG leg (kama4h+PDH-soft)."""
import os, sys, warnings, itertools
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from research.overfit_audit import cscv, psr, sr0, cdd_R, block_resample
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from short_mirror_15m import invert

d = load_mt5_csv("data/vantage_btcusd_m15.csv")
cnt = d.groupby(d.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 80]
d15 = resample(d[d.index.date >= ok.index[0]], "15min")
inv = invert(d15)
C = 2 * d15["high"].max()
span = (d15.index[-1]-d15.index[0]).days/365.25
pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
import pandas_ta as ta
atr15 = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1).values

CACHE = {}
def leg(zz=2.0, frac=0.3, gate="1d"):
    key = (zz, frac, gate)
    if key in CACHE: return CACHE[key]
    extra = {}
    if gate == "1d": extra = dict(gate_kama=14)
    elif gate == "4h": extra = dict(gate_kama=14, gate_kama_tf="240min")
    t = run(inv, SimpleNamespace(**{**BASE, **extra, "zz_k": zz, "pullback_frac": frac}))
    Rn = t["R"].values - 15.0/t["risk"].values
    pos = d15.index.get_indexer(t["time"])
    e_real = C - t["e_px"].values
    CACHE[key] = (Rn, pd.DatetimeIndex(t["time"]), e_real, pos)
    return CACHE[key]

def stats(Rn, ts):
    yr = ts.year.values; half = np.median(yr)
    pf = Rn[Rn>0].sum()/abs(Rn[Rn<=0].sum()) if (Rn<=0).any() else 9.9
    eq = np.cumsum(Rn); dd = (np.maximum.accumulate(eq)-eq).max()
    return dict(n=len(Rn), pf=pf, mr=Rn.mean(), IS=Rn[yr<half].mean(), OOS=Rn[yr>=half].mean(),
                rdd=Rn.sum()/max(dd,1e-9), tot=Rn.sum())

# (1) per-year of the top cells
print("(1) per-year totR:")
for tag, gate, thr in [("kama1d∩PDL", "1d", 0.0), ("kama4h∩PDL", "4h", 0.0)]:
    Rn, ts, e_real, pos = leg(gate=gate)
    m = e_real < pdl[pos] - thr*atr15[pos]
    yr = ts.year.values
    print(f"   {tag}: " + "  ".join(f"{y}:{Rn[m & (yr==y)].sum():+.0f}({(m&(yr==y)).sum()})"
          for y in np.unique(yr)))

# (2) plateau
print("\n(2) plateau (kama1d ∩ PDL):")
print(f"   {'cell':<24}{'n':>5}{'PF':>6}{'meanR':>8}{'IS/OOS':>13}{'ret/DD':>7}")
grid_cfgs = []
for zz in (1.5, 2.0, 2.5):
    for frac in ((0.2, 0.3, 0.4) if zz == 2.0 else (0.3,)):
        Rn, ts, e_real, pos = leg(zz=zz, frac=frac)
        for thr in ((-0.5, 0.0, 0.5) if (zz == 2.0 and frac == 0.3) else (0.0,)):
            m = e_real < pdl[pos] - thr*atr15[pos]
            s = stats(Rn[m], ts[m]); grid_cfgs.append((f"zz{zz}/f{frac}/thr{thr}", Rn[m], ts[m]))
            print(f"   zz{zz}/f{frac}/thr{thr:+.1f}      "[:24] + f"{s['n']:>5}{s['pf']:>6.2f}{s['mr']:>+8.3f}"
                  f"{s['IS']:>+7.2f}/{s['OOS']:+.2f}{s['rdd']:>7.2f}")

# (3) audit: DSR + bootstrap + null on central cell; PBO over universe
Rn, ts, e_real, pos = leg()
m0 = e_real < pdl[pos]
cand = Rn[m0]; ts0 = ts[m0]
mon = ts0.to_period("M")
cfgs = {}
for tag, gate in [("none", "none"), ("1d", "1d"), ("4h", "4h")]:
    R2, t2, e2, p2 = leg(gate=gate)
    for pd_on in (False, True):
        mm = (e2 < pdl[p2]) if pd_on else np.ones(len(R2), bool)
        cfgs[f"{tag}{'+pdl' if pd_on else ''}"] = (R2[mm], t2[mm])
for tag, R2, t2 in grid_cfgs:
    cfgs[tag] = (R2, t2)
midx = pd.period_range(ts.min().to_period("M"), ts.max().to_period("M"), freq="M")
cols, srs = {}, []
for k, (R2, t2) in cfgs.items():
    if len(R2) < 20: continue
    s = pd.Series(R2, index=pd.DatetimeIndex(t2).to_period("M")).groupby(level=0).sum().reindex(midx, fill_value=0.0)
    cols[k] = s; srs.append(R2.mean()/R2.std(ddof=1))
M = pd.concat(cols, axis=1); V = float(np.var(srs))
pbo, oos_sr, ploss = cscv(M.values)
_, sr, sk, ku = psr(cand, 0.0)
print(f"\n(3) audit central cell (kama1d∩PDL n={len(cand)}, SR={sr:.3f}, skew={sk:.2f}):")
print("    DSR: " + "  ".join(f"@{N}={psr(cand, sr0(N, V))[0]:.2f}" for N in (1, 10, 20, 50)))
rng = np.random.default_rng(7)
obs = cdd_R(cand, span)[2]
boot = np.array([cdd_R(block_resample(cand, 5, rng), span)[2] for _ in range(3000)])
nul = np.array([cdd_R(block_resample(cand-cand.mean(), 5, rng), span)[2] for _ in range(3000)])
print(f"    CAGR/DD obs={obs:.2f} CI[{np.percentile(boot,5):.2f},{np.percentile(boot,95):.2f}] null p={(nul>=obs).mean():.3f}")
print(f"    PBO={pbo:.2f} ({M.shape[1]}cfg x {M.shape[0]}mo, IS-best OOS-SR={oos_sr:+.2f}, P(OOSloss)={ploss:.2f})")

# (4) exclusivity vs the long leg
tl = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3}))
Rl = tl["R"].values - 15.0/tl["risk"].values
pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
abl = tl["e_px"].values > pdh[d15.index.get_indexer(tl["time"])]
Rl_soft = Rl * np.where(abl, 1.0, 0.5)
tsl = pd.DatetimeIndex(tl["time"])
ms = pd.Series(cand, index=ts0.to_period("M")).groupby(level=0).sum().reindex(midx, fill_value=0.0)
ml = pd.Series(Rl_soft, index=tsl.to_period("M")).groupby(level=0).sum().reindex(midx, fill_value=0.0)
print(f"\n(4) vs LONG (kama4h+PDHsoft): monthly corr={np.corrcoef(ms, ml)[0,1]:+.2f}")
long_days = set(tsl.date); short_days = set(ts0.date)
print(f"    same-day overlap: {len(long_days & short_days)}/{len(short_days)} short-days "
      f"({len(long_days & short_days)/max(len(short_days),1)*100:.0f}%)")
both = ms + ml
def rdd(x):
    eq = np.cumsum(x); dd = (np.maximum.accumulate(eq)-eq).max()
    return x.sum()/max(dd,1e-9)
print(f"    totR/DD: long={rdd(ml.values):.2f} short={rdd(ms.values):.2f} 合成={rdd(both.values):.2f}")
