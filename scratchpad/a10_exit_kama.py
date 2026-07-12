"""a10_exit_kama.py -- proposals A10: evaluate the implemented-but-unmeasured --exit-kama
flag (regime-flip exit: bail at close when the exit-TF KAMA turns down mid-trade).

Cells: BTC 15m long leg (gate kama4h, rally-limit frac0.3, net $15) -- the target of the
proposal -- sweep exit KAMA length {10,14,20} x exit TF {1D, 240min} vs OFF.
Secondary: gold 15m canon leg (SMA150 gate + ext-cap8, frac0.25, net $0.3) exit 14/1D.
Read: N/yr (rotation: earlier exits free the slot), PF, meanR, totR/yr, maxDD-R, hold med.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE


def card(tag, t, rt, span):
    Rn = t["R"].values - rt / t["risk"].values
    ts = pd.DatetimeIndex(t["time"])
    yr = ts.year.values
    half = np.median(yr)
    pf = Rn[Rn > 0].sum() / abs(Rn[Rn <= 0].sum())
    eq = np.cumsum(Rn)
    dd = (np.maximum.accumulate(eq) - eq).max()
    g = sum(Rn[yr == y].sum() > 0 for y in np.unique(yr))
    print(f"  {tag:<24} N/yr={len(Rn)/span:5.1f} win={(Rn>0).mean()*100:4.1f}% PF={pf:4.2f} "
          f"meanR={Rn.mean():+.3f} IS/OOS={Rn[yr<half].mean():+.2f}/{Rn[yr>=half].mean():+.2f} "
          f"totR/yr={Rn.sum()/span:+5.1f} DD={dd:5.1f}R grn={g}/{len(np.unique(yr))} "
          f"hold_med={np.median(t['hold']):.2f}d")


def main():
    b = load_mt5_csv("data/vantage_btcusd_m15.csv")
    cnt = b.groupby(b.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= 80]
    d15 = resample(b[b.index.date >= ok.index[0]], "15min")
    span = (d15.index[-1] - d15.index[0]).days / 365.25
    cell = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3}
    print(f"===== BTC 15m long (kama4h, frac0.3, net $15) {span:.1f}yr =====")
    card("exit OFF (現行)", run(d15, SimpleNamespace(**cell)), 15.0, span)
    for tf in ("1D", "240min"):
        for n in (10, 14, 20):
            t = run(d15, SimpleNamespace(**{**cell, "exit_kama": n, "exit_kama_tf": tf}))
            card(f"exit KAMA{n} {tf}", t, 15.0, span)

    g = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
    g15 = resample(g, "15min")
    span = (g15.index[-1] - g15.index[0]).days / 365.25
    gcell = {**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0, "pullback_frac": 0.25}
    print(f"\n===== GOLD 15m canon (SMA150+extcap8, frac0.25, net $0.3) {span:.1f}yr =====")
    card("exit OFF (現行)", run(g15, SimpleNamespace(**gcell)), 0.3, span)
    for n in (10, 14, 20):
        t = run(g15, SimpleNamespace(**{**gcell, "exit_kama": n}))
        card(f"exit KAMA{n} 1D", t, 0.3, span)


if __name__ == "__main__":
    main()
