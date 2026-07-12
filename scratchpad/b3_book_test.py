"""B3 book-level judgment: does adding the gold 15m deep-bounce (frac0.9) to gold_bo15m
improve the composite CAGR/DD? Two framings: SUBSTITUTION (total risk held at 1%: w_bo+w_bn=1)
and ADD-ON (gold_bo 1% + bounce w%). Monthly R corr reported (annual n=8 was too coarse).
All net $0.3; bounce includes slip0.5; span = true-M5 2018-09+."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from bounce_b3_gauntlet import build, one
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo

gold = load_mt5_csv("data/vantage_xauusd_m5.csv")
df = gold.loc["2018-09-14":].resample("15min").agg(
    {"open":"first","high":"max","low":"min","close":"last"}).dropna()
span = (df.index[-1] - df.index[0]).days / 365.25

t_bo = run_bo(df, SimpleNamespace(**{**BASE, "daily_sma":150, "daily_slope_k":10,
                                     "ext_cap":8.0, "pullback_frac":0.25}))
Rbo = t_bo["R"].values - 0.3 / t_bo["risk"].values
ex_bo = t_bo["time"] + pd.to_timedelta(t_bo["hold"] * 15, unit="m")
bo = list(zip(ex_bo, Rbo))

B = build(df)
bn = [(t1, r) for _, t1, r in one(B, 0.90)]

def cagr_dd(events, span):
    ev = sorted(events, key=lambda x: x[0])
    eq = np.cumprod([1 + wr for _, wr in ev])
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (1 / span) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)

print(f"span {span:.1f}yr  gold_bo n={len(bo)} meanR={Rbo.mean():+.3f}  "
      f"bounce(f0.9) n={len(bn)} meanR={np.mean([r for _,r in bn]):+.3f}")
mbo = pd.Series([r for _,r in bo], index=pd.DatetimeIndex([t for t,_ in bo])).resample("ME").sum()
mbn = pd.Series([r for _,r in bn], index=pd.DatetimeIndex([t for t,_ in bn])).resample("ME").sum()
idx = mbo.index.union(mbn.index)
a, b = mbo.reindex(idx, fill_value=0), mbn.reindex(idx, fill_value=0)
print(f"monthly R corr = {np.corrcoef(a, b)[0,1]:+.2f}   annual corr = "
      f"{np.corrcoef(a.resample('YE').sum(), b.resample('YE').sum())[0,1]:+.2f}")

print(f"\n{'mix (f=1% total)':<28}{'CAGR%':>7}{'maxDD%':>8}{'CAGR/DD':>9}")
for wbo, wbn, tag in [(1.0,0.0,"gold_bo only"),(0.75,0.25,"75/25 substitution"),
                      (0.5,0.5,"50/50 substitution"),(0.0,1.0,"bounce only")]:
    ev = [(t, 0.01*wbo*r) for t,r in bo] + [(t, 0.01*wbn*r) for t,r in bn]
    c,d,r = cagr_dd(ev, span); print(f"{tag:<28}{c:>7.1f}{d:>8.1f}{r:>9.2f}")
print(f"\n{'add-on (gold 1% + bounce w)':<28}{'CAGR%':>7}{'maxDD%':>8}{'CAGR/DD':>9}")
for wbn in (0.25, 0.5, 1.0):
    ev = [(t, 0.01*r) for t,r in bo] + [(t, 0.01*wbn*r) for t,r in bn]
    c,d,r = cagr_dd(ev, span); print(f"gold 1% + bounce {wbn:.2f}%{'':<8}{c:>7.1f}{d:>8.1f}{r:>9.2f}")

print(f"\n{'f-raise comparison':<28}{'CAGR%':>7}{'maxDD%':>8}{'CAGR/DD':>9}")
for f, tag in [(0.015,"gold_bo f1.5%"),(0.02,"gold_bo f2.0%")]:
    ev=[(t,f*r) for t,r in bo]; c,d,r=cagr_dd(ev,span); print(f"{tag:<28}{c:>7.1f}{d:>8.1f}{r:>9.2f}")
ev=[(t,0.01*r) for t,r in bo]+[(t,0.01*r) for t,r in bn]
c,d,r=cagr_dd(ev,span); print(f"{'gold1% + bounce1% (再掲)':<28}{c:>7.1f}{d:>8.1f}{r:>9.2f}")
