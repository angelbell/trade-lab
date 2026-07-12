"""Does prior-day-high / recent-daily-high structure SELECT better breakout trades?
Labels on the canon 15m legs (post-hoc): entry vs PDH (above = new-daily-high air /
below = inside yesterday's range), |H1-PDH|<=1ATR confluence, entry vs 20-day high.
Any promising subset must beat the equal-keep random-drop null on totR/maxDD."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd, pandas_ta as ta
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
    return real, (np.array(ns_) < real).mean()*100

for name, csv, extra, rt, start in [
    ("GOLD", "data/vantage_xauusd_m5.csv",
     dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0, pullback_frac=0.25), 0.3, "2018-09-14"),
    ("BTC", "data/vantage_btcusd_m15.csv",
     dict(gate_kama=14, pullback_frac=0.3), 15.0, "2018-10-01")]:
    d = load_mt5_csv(csv).loc[start:]
    d15 = resample(d, "15min")
    span = (d15.index[-1]-d15.index[0]).days/365.25
    t = run(d15, SimpleNamespace(**{**BASE, **extra}))
    Rn = t["R"].values - rt/t["risk"].values
    pos = d15.index.get_indexer(t["time"])
    atr = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1).values
    dh = d15["high"].resample("1D").max().dropna()
    pdh = dh.shift(1).reindex(d15.index, method="ffill").values
    d20 = dh.rolling(20).max().shift(1).reindex(d15.index, method="ffill").values
    e = t["e_px"].values
    a_e, pdh_e, d20_e = atr[pos], pdh[pos], d20[pos]
    labs = {"entry>PDH (新値圏)": e > pdh_e,
            "entry<=PDH (レンジ内)": e <= pdh_e,
            "|entry-PDH|<=1ATR (合流)": np.abs(e-pdh_e) <= a_e,
            "entry>20日高値": e > d20_e,
            "entry<=20日高値": e <= d20_e}
    print(f"\n===== {name} 15m leg (net ${rt}, {span:.1f}yr, N={len(Rn)}, "
          f"base meanR={Rn.mean():+.3f}) =====")
    for tag, m in labs.items():
        m = m & ~np.isnan(pdh_e) & ~np.isnan(a_e)
        if m.sum() < 30: print(f"  {tag:<26} n={m.sum()} few"); continue
        r = Rn[m]; pf = r[r>0].sum()/abs(r[r<=0].sum())
        real, pct = null_pct(Rn, m)
        print(f"  {tag:<26} n={m.sum():4d}({m.mean()*100:3.0f}%)  meanR={r.mean():+.3f}  PF={pf:4.2f}"
              f"  totR/yr={r.sum()/span:+5.1f}  totR/DD={real:5.2f}  null%ile={pct:4.0f}")
