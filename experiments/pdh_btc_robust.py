"""Robustness of the BTC 'entry above prior-day high' selector: per-year, IS/OOS,
and does it hold on the kama4h-gated (C1 attack) variant too? Plus threshold plateau
(entry vs PDH +/- 0.5 ATR shifts)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
d15 = resample(d, "15min")
span = (d15.index[-1]-d15.index[0]).days/365.25
atr = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1).values
dh = d15["high"].resample("1D").max().dropna()
pdh = dh.shift(1).reindex(d15.index, method="ffill").values

for gtag, extra in [("kama1d(canon)", dict(gate_kama=14, pullback_frac=0.3)),
                    ("kama4h(C1)", dict(gate_kama=14, gate_kama_tf="240min", pullback_frac=0.3))]:
    t = run(d15, SimpleNamespace(**{**BASE, **extra}))
    Rn = t["R"].values - 15.0/t["risk"].values
    pos = d15.index.get_indexer(t["time"])
    e = t["e_px"].values
    yr = t["time"].dt.year.values
    half = np.median(yr)
    print(f"\n===== BTC {gtag} N={len(Rn)} =====")
    for thr_tag, m in [("e>PDH", e > pdh[pos]),
                       ("e>PDH+0.5ATR", e > pdh[pos]+0.5*atr[pos]),
                       ("e>PDH-0.5ATR", e > pdh[pos]-0.5*atr[pos])]:
        m = m & ~np.isnan(pdh[pos]) & ~np.isnan(atr[pos])
        r = Rn[m]
        pf = r[r>0].sum()/abs(r[r<=0].sum())
        eq = np.cumsum(r); dd = (np.maximum.accumulate(eq)-eq).max()
        print(f"  {thr_tag:<14} n={m.sum():4d} N/yr={m.sum()/span:4.1f} meanR={r.mean():+.3f} PF={pf:4.2f}"
              f" IS/OOS={r[(yr<half)&m[:len(yr)]].mean() if False else Rn[m&(yr<half)].mean():+.2f}"
              f"/{Rn[m&(yr>=half)].mean():+.2f} totR/yr={r.sum()/span:+5.1f} maxDD={dd:4.1f}R ret/DD={r.sum()/dd:5.2f}")
    m = (e > pdh[pos]) & ~np.isnan(pdh[pos])
    print("  per-year e>PDH totR: " + "  ".join(
        f"{y}:{Rn[m&(yr==y)].sum():+.0f}" for y in np.unique(yr) if (m&(yr==y)).sum()>0))
    print("  per-year e<=PDH totR: " + "  ".join(
        f"{y}:{Rn[~m&(yr==y)].sum():+.0f}" for y in np.unique(yr)))
