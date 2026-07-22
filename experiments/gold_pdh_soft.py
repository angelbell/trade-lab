"""Last untested gold-PDH cell: SOFT sizing (inside=0.5x) on top of the gold 15m canon
(hard-AND was over-filtering; soft interpolates). Plus: does PDH fix the 2026 bleed
(-12.8R watch item)? Canon = sma150+slope+extcap8, frac0.25, net $0.3."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

d15 = resample(load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
span = (d15.index[-1]-d15.index[0]).days/365.25
t = run(d15, SimpleNamespace(**{**BASE, "daily_sma":150, "daily_slope_k":10,
                                "ext_cap":8.0, "pullback_frac":0.25}))
Rn = t["R"].values - 0.3/t["risk"].values
pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
ab = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
yr = t["time"].dt.year.values
def card(tag, r):
    pf = r[r>0].sum()/abs(r[r<=0].sum())
    eq = np.cumsum(r); dd = (np.maximum.accumulate(eq)-eq).max()
    print(f"  {tag:<22} totR/yr={r.sum()/span:+5.1f}  PF={pf:4.2f}  maxDD={dd:5.1f}R  ret/DD={r.sum()/dd:5.2f}")
card("canon (現行)", Rn)
card("canon∩PDH ハード", Rn[ab])
card("canon PDHソフト0.5", Rn*np.where(ab, 1.0, 0.5))
print("\n  per-year totR (canon / soft0.5 / 新値圏側 / レンジ内側):")
for y in np.unique(yr):
    m = yr == y
    print(f"   {y}: {Rn[m].sum():+6.1f} / {(Rn*np.where(ab,1,0.5))[m].sum():+6.1f} / "
          f"{Rn[m & ab].sum():+6.1f}({(m&ab).sum()}) / {Rn[m & ~ab].sum():+6.1f}({(m&~ab).sum()})")
