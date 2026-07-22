"""PDH audit script A: BTC 15m leg. (1) overfit: CSCV/PBO over the selector-config universe
+ DSR at honest trial counts (~28 cells explored today) + block-bootstrap CI & mean-removed
null on the candidate (kama4h & e>PDH). (2) SOFT version: inside-range weight w ladder
(monotone?). (3) f-ladder: 1-year capital-multiplier bootstrap (monthly resample) for
full / hard-PDH / soft0.5 at f=1/2/3%."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE, kama_up
from research.overfit_audit import cscv, psr, sr0, cdd_R, block_resample
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
d15 = resample(d, "15min")
span = (d15.index[-1]-d15.index[0]).days/365.25
t = run(d15, SimpleNamespace(**{**BASE, "pullback_frac": 0.3}))     # ungated base
Rn = t["R"].values - 15.0/t["risk"].values
pos = d15.index.get_indexer(t["time"])
atr = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1).values
dh = d15["high"].resample("1D").max().dropna()
pdh = dh.shift(1).reindex(d15.index, method="ffill").values
d20 = dh.rolling(20).max().shift(1).reindex(d15.index, method="ffill").values
e = t["e_px"].values
g1, g4 = kama_up(d15, "1D")[pos], kama_up(d15, "240min")[pos]
ab = (e > pdh[pos]) & ~np.isnan(pdh[pos])
mon = t["time"].dt.to_period("M")

# ---- (1) CSCV/PBO over selector x gate configs ----
CFG = {}
for gt, gm in [("k1d", g1), ("k4h", g4)]:
    CFG[f"{gt}"] = gm
    CFG[f"{gt}&pdh"] = gm & ab
    CFG[f"{gt}&pdh+.5"] = gm & (e > pdh[pos]+0.5*atr[pos])
    CFG[f"{gt}&pdh-.5"] = gm & (e > pdh[pos]-0.5*atr[pos])
    CFG[f"{gt}&20d"] = gm & (e > d20[pos]) & ~np.isnan(d20[pos])
    CFG[f"{gt}&inside"] = gm & ~ab
midx = pd.period_range(mon.min(), mon.max(), freq="M")
cols, srs = {}, []
for k, m in CFG.items():
    s = pd.Series(Rn[m], index=mon[m]).groupby(level=0).sum().reindex(midx, fill_value=0.0)
    cols[k] = s; srs.append(Rn[m].mean()/Rn[m].std(ddof=1))
M = pd.concat(cols, axis=1)
V = float(np.var(srs))
pbo, oos_sr, ploss = cscv(M.values)
rng = np.random.default_rng(1)
pbo_n, _, _ = cscv(rng.standard_normal(M.shape))
print(f"(1) PBO/CSCV: {M.shape[1]}cfg x {M.shape[0]}mo  PBO={pbo:.2f}  IS-best OOS-SR={oos_sr:+.2f}"
      f"  P(OOSloss)={ploss:.2f}  [noise {pbo_n:.2f}]  V_SR={V:.5f}")
cand = Rn[g4 & ab]
_, sr, g1_, g4_ = psr(cand, 0.0)
Ns = [1, 10, 28, 50, 100]
print(f"    DSR (kama4h&PDH n={len(cand)} SR={sr:.3f} skew={g1_:.2f}): "
      + "  ".join(f"@{N}={psr(cand, sr0(N, V))[0]:.2f}" for N in Ns))
rngb = np.random.default_rng(7)
obs = cdd_R(cand, span)[2]
boot = np.array([cdd_R(block_resample(cand, 5, rngb), span)[2] for _ in range(3000)])
nul = np.array([cdd_R(block_resample(cand - cand.mean(), 5, rngb), span)[2] for _ in range(3000)])
print(f"    CAGR/DD obs={obs:.2f}  CI5/50/95={np.percentile(boot,5):.2f}/{np.percentile(boot,50):.2f}/"
      f"{np.percentile(boot,95):.2f}  null p={(nul>=obs).mean():.3f}")

# ---- (2) soft ladder on kama4h ----
print("\n(2) SOFT ladder (kama4h leg; inside-range weight w):")
base4 = g4
for w in (0.0, 0.33, 0.5, 0.75, 1.0):
    wt = np.where(ab[base4], 1.0, w)
    r = Rn[base4] * wt
    eq = np.cumsum(r); dd = (np.maximum.accumulate(eq)-eq).max()
    ny = base4.sum()/span if w > 0 else (base4 & ab).sum()/span
    print(f"    w={w:.2f}: N/yr={ny:5.1f}  totR/yr={r.sum()/span:+5.1f}  maxDD={dd:5.1f}R"
          f"  ret/DD={r.sum()/dd:5.2f}")

# ---- (3) f-ladder: 1-year multiplier bootstrap (monthly resample) ----
print("\n(3) f-ladder (1yr multiplier, monthly bootstrap 4000):")
streams = {"full(kama4h)": (g4, None), "hard PDH": (g4 & ab, None), "soft0.5": (g4, 0.5)}
for tag, (m, w) in streams.items():
    if w is None: r = Rn[m]; tm = mon[m]
    else:
        r = Rn[m] * np.where(ab[m], 1.0, w); tm = mon[m]
    ms = pd.Series(r, index=tm).groupby(level=0).sum().reindex(midx, fill_value=0.0).values
    for f in (0.01, 0.02, 0.03):
        mult = np.array([np.prod(1 + f*ms[rng.integers(0, len(ms), 12)]) for _ in range(4000)])
        print(f"    {tag:<14} f={f*100:.0f}%: med={np.median(mult):.2f} sd={mult.std():.2f} "
              f"p10={np.percentile(mult,10):.2f} p90={np.percentile(mult,90):.2f} "
              f"P(2x)={(mult>=2).mean()*100:2.0f}% P(半減)={(mult<=0.5).mean()*100:.1f}%")
