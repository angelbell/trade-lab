"""PDH selector transfer map: does 'entry above prior-day high' select better breakout
trades on other instruments/TFs? Cells: gold 15m (canon / no-extcap), gold 1h (gold_bo),
BTC 4h (book leg), USDJPY 15m + 1h (base edge is DEAD -> tests 'filters cannot create').
Metric per cell: base meanR / subset meanR+PF+IS/OOS / equal-keep random-drop null %ile."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

def null_pct(base, m, trials=1000, seed=7):
    def rd(x):
        eq = np.cumsum(x); dd = (np.maximum.accumulate(eq)-eq).max()
        return x.sum()/max(dd,1e-9)
    real = rd(base[m]); rng = np.random.default_rng(seed)
    ns_ = [rd(base[np.sort(rng.choice(len(base), m.sum(), replace=False))]) for _ in range(trials)]
    return (np.array(ns_) < real).mean()*100

CELLS = [
 ("gold 15m canon",  "data/vantage_xauusd_m5.csv", "15min", "2018-09-14", 0.3,
  dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0, pullback_frac=0.25)),
 ("gold 15m NO-extcap","data/vantage_xauusd_m5.csv","15min","2018-09-14", 0.3,
  dict(daily_sma=150, daily_slope_k=10, pullback_frac=0.25)),
 ("gold 1h gold_bo", "data/vantage_xauusd_h1.csv", "1h", None, 0.3,
  dict(daily_sma=150, daily_slope_k=10, rr=3.0, pullback_frac=0.0)),
 ("BTC 4h book",     "data/vantage_btcusd_h1.csv", "4h", None, 15.0,
  dict(gate_kama=14, rr=2.0, fwd=300, pullback_frac=0.0)),
 ("USDJPY 15m ungated","data/vantage_usdjpy_m15.csv","15min", None, 0.012,
  dict(pullback_frac=0.0)),
 ("USDJPY 1h ungated","data/vantage_usdjpy_h1.csv","1h", None, 0.012,
  dict(pullback_frac=0.0)),
]
print(f"{'cell':<20}{'N':>5}{'baseR':>7} | {'n>PDH':>6}{'%':>4}{'meanR':>7}{'PF':>5}"
      f"{'IS/OOS':>12}{'null%':>6} | {'≤PDH R':>7}")
for tag, csv, tf, start, rt, extra in CELLS:
    d = load_mt5_csv(csv)
    if start: d = d.loc[start:]
    dtf = resample(d, tf)
    t = run(dtf, SimpleNamespace(**{**BASE, **extra}))
    if len(t) < 30:
        print(f"{tag:<20} too few ({len(t)})"); continue
    Rn = t["R"].values - rt/t["risk"].values
    pos = dtf.index.get_indexer(t["time"])
    dh = dtf["high"].resample("1D").max().dropna()
    pdh = dh.shift(1).reindex(dtf.index, method="ffill").values
    e = t["e_px"].values
    yr = t["time"].dt.year.values; half = np.median(yr)
    m = (e > pdh[pos]) & ~np.isnan(pdh[pos])
    r = Rn[m]; rc = Rn[~m]
    pf = r[r>0].sum()/abs(r[r<=0].sum()) if (r<=0).any() else 9.9
    pct = null_pct(Rn, m)
    print(f"{tag:<20}{len(Rn):>5}{Rn.mean():>+7.3f} | {m.sum():>6}{m.mean()*100:>4.0f}"
          f"{r.mean():>+7.3f}{pf:>5.2f}{Rn[m&(yr<half)].mean():>+6.2f}/"
          f"{Rn[m&(yr>=half)].mean():+.2f}{pct:>6.0f} | {rc.mean():>+7.3f}")
