"""BTC 5m breakout+pullback recompute at real spread sensitivity ($15 canon vs $10 tight).
Cell per the 2026-07-02 kill: Pattern B / zz2 / ema80 / RR4 / daily-KAMA gate / frac0.3."""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

d = load_mt5_csv("data/vantage_btcusd_m5.csv")
cnt = d.groupby(d.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 150]
d = d[d.index.date >= ok.index[0]]
d5 = resample(d, "5min")
span = (d5.index[-1] - d5.index[0]).days / 365.25
t = run(d5, SimpleNamespace(**{**BASE, "tf": "5min", "gate_kama": 14, "pullback_frac": 0.3}))
yr = t["time"].dt.year.values
print(f"span {d5.index[0].date()}->{d5.index[-1].date()} ({span:.1f}yr)  N={len(t)}  "
      f"cost/risk med @$15: {np.median(15/t['risk']):.1%}")
half = np.median(yr)
for rt in (0.0, 10.0, 15.0):
    Rn = t["R"].values - rt / t["risk"].values
    pf = Rn[Rn > 0].sum() / abs(Rn[Rn <= 0].sum())
    eq = np.cumsum(Rn); dd = (np.maximum.accumulate(eq) - eq).max()
    g = sum(Rn[yr == y].sum() > 0 for y in np.unique(yr))
    print(f"  $"f"{rt:4.0f}: N/yr={len(Rn)/span:5.1f} win={(Rn>0).mean()*100:4.1f}% PF={pf:4.2f} "
          f"meanR={Rn.mean():+.3f} IS/OOS={Rn[yr<half].mean():+.2f}/{Rn[yr>=half].mean():+.2f} "
          f"totR/yr={Rn.sum()/span:+5.1f} maxDD={dd:5.1f}R ret/DD={Rn.sum()/dd:5.2f} grn={g}/{len(np.unique(yr))}")
