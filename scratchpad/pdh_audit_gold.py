"""PDH audit B+C for GOLD.
B: gold 15m -- ext-cap vs PDH head-to-head on the sma150+slope base (frac0.25, $0.3):
   {none / cap8 / PDH / both} full cards + equal-keep null + per-year + DSR of winner.
C: gold 5m (canonical walk $0.3+slip0.27) -- DSR + block-bootstrap CAGR/DD null of the
   PDH-filtered leg, SOFT ladder (inside weight w), f-ladder full vs hard vs soft0.5."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from research.overfit_audit import psr, sr0, cdd_R, block_resample
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

# ============ B. gold 15m: cap vs PDH ============
gold = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
d15 = resample(gold, "15min")
span = (d15.index[-1]-d15.index[0]).days/365.25
dc = d15["close"].resample("1D").last().dropna()
sma = dc.rolling(150).mean()
ext = ((dc-sma)/sma*100.0).shift(1).reindex(d15.index, method="ffill").values
t = run(d15, SimpleNamespace(**{**BASE, "daily_sma":150, "daily_slope_k":10, "pullback_frac":0.25}))
Rn = t["R"].values - 0.3/t["risk"].values
pos = d15.index.get_indexer(t["time"])
dh = d15["high"].resample("1D").max().dropna()
pdh = dh.shift(1).reindex(d15.index, method="ffill").values
e = t["e_px"].values
cap_ok = ~(ext[pos] > 8.0)                      # ext-cap keeps these
ab = (e > pdh[pos]) & ~np.isnan(pdh[pos])
yr = t["time"].dt.year.values; half = np.median(yr)

def rd(x):
    eq = np.cumsum(x); dd = (np.maximum.accumulate(eq)-eq).max()
    return x.sum()/max(dd,1e-9), dd

rng = np.random.default_rng(7)
def nullpct(m, trials=1000):
    real = rd(Rn[m])[0]
    ns_ = [rd(Rn[np.sort(rng.choice(len(Rn), m.sum(), replace=False))])[0] for _ in range(trials)]
    return (np.array(ns_) < real).mean()*100

print(f"B. GOLD 15m cap-vs-PDH (base sma150+slope, N={len(Rn)}, {span:.1f}yr)")
for tag, m in [("none", np.ones(len(Rn), bool)), ("cap8のみ", cap_ok),
               ("PDHのみ", ab), ("両方(cap&PDH)", cap_ok & ab)]:
    r = Rn[m]; pf = r[r>0].sum()/abs(r[r<=0].sum())
    ratio, dd = rd(r)
    print(f"   {tag:<14} n={m.sum():4d}({m.mean()*100:3.0f}%) meanR={r.mean():+.3f} PF={pf:4.2f}"
          f" IS/OOS={Rn[m&(yr<half)].mean():+.2f}/{Rn[m&(yr>=half)].mean():+.2f}"
          f" totR/yr={r.sum()/span:+5.1f} DD={dd:5.1f}R ret/DD={ratio:5.2f} null%={nullpct(m):3.0f}")
m = ab
print("   per-year PDHのみ: " + "  ".join(f"{y}:{Rn[m&(yr==y)].sum():+.0f}" for y in np.unique(yr)))

# ============ C. gold 5m audit ============
src = open("scratchpad/pullback_5m_realcost.py").read()
ns = {}
exec(src.split("\nB5r4  = build")[0], ns)
build, eval_pull, stats = ns["build"], ns["eval_pull"], ns["stats"]
B = build(None, 4.0)
d5 = B["d"]
span5 = (d5.index[-1]-d5.index[0]).days/365.25
dh5 = d5["high"].resample("1D").max().dropna()
pdh5 = dh5.shift(1).reindex(d5.index, method="ffill").values
lab = np.array([ep > pdh5[i] and not np.isnan(pdh5[i]) for (i, ep, *_ ) in B["entries"]])
FR = lambda ee, ss, H: ee - 0.25*(ee-ss)
above = dict(B); above["entries"] = [en for en, L in zip(B["entries"], lab) if L]
tr_a, _ = eval_pull(above, FR, 0.3, stop_slip=0.27)
Ra = np.array([r for _, r in tr_a]); ta_ = pd.DatetimeIndex([x for x,_ in tr_a])
_, sr, sk, ku = psr(Ra, 0.0)
V = 0.0139  # cross-config SR variance from the BTC universe (same exploration family)
print(f"\nC. GOLD 5m PDH leg audit (n={len(Ra)}, SR={sr:.3f}, skew={sk:.2f})")
print("   DSR: " + "  ".join(f"@{N}={psr(Ra, sr0(N, V))[0]:.2f}" for N in (1,10,28,50,100)))
obs = cdd_R(Ra, span5)[2]
boot = np.array([cdd_R(block_resample(Ra,5,rng), span5)[2] for _ in range(3000)])
nul = np.array([cdd_R(block_resample(Ra-Ra.mean(),5,rng), span5)[2] for _ in range(3000)])
print(f"   CAGR/DD obs={obs:.2f}  CI5/50/95={np.percentile(boot,5):.2f}/{np.percentile(boot,50):.2f}/"
      f"{np.percentile(boot,95):.2f}  null p={(nul>=obs).mean():.3f}")
# soft ladder: run full leg, weight inside trades -- need per-trade label of the FULL eval
tr_f, _ = eval_pull(B, FR, 0.3, stop_slip=0.27)
# reconstruct label per trade: map fill time -> originating entry unknown; instead evaluate
# below-side separately and combine streams (deployment: two order books, one slot each side)
belw = dict(B); belw["entries"] = [en for en, L in zip(B["entries"], lab) if not L]
tr_b, _ = eval_pull(belw, FR, 0.3, stop_slip=0.27)
Rb = np.array([r for _, r in tr_b]); tb_ = pd.DatetimeIndex([x for x,_ in tr_b])
print("   SOFT ladder (above 1.0x + inside w; separate slots):")
midx = pd.period_range(min(ta_.min(), tb_.min()).to_period("M"),
                       max(ta_.max(), tb_.max()).to_period("M"), freq="M")
ma = pd.Series(Ra, index=ta_.to_period("M")).groupby(level=0).sum().reindex(midx, fill_value=0.0).values
mb = pd.Series(Rb, index=tb_.to_period("M")).groupby(level=0).sum().reindex(midx, fill_value=0.0).values
for w in (0.0, 0.33, 0.5, 1.0):
    r_all = np.concatenate([Ra, w*Rb])
    ordi = np.argsort(np.concatenate([ta_.values, tb_.values]))
    rs = r_all[ordi]
    eq = np.cumsum(rs); dd = (np.maximum.accumulate(eq)-eq).max()
    print(f"     w={w:.2f}: totR/yr={rs.sum()/span5:+5.1f}  maxDD={dd:5.1f}R  ret/DD={rs.sum()/dd:5.2f}")
print("   f-ladder (1yr multiplier, monthly bootstrap 4000):")
for tag, msr in [("hard PDH", ma), ("soft0.5", ma+0.5*mb), ("full", ma+mb)]:
    for f in (0.02, 0.03):
        mult = np.array([np.prod(1 + f*msr[rng.integers(0, len(msr), 12)]) for _ in range(4000)])
        print(f"     {tag:<9} f={f*100:.0f}%: med={np.median(mult):.2f} sd={mult.std():.2f} "
              f"p10={np.percentile(mult,10):.2f} p90={np.percentile(mult,90):.2f} "
              f"P(2x)={(mult>=2).mean()*100:2.0f}% P(半減)={(mult<=0.5).mean()*100:.1f}%")
