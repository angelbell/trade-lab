"""btc15m_session_audit.py -- time-of-day / weekend EV audit of the BTC 15m validated cell.

Cell (passed gauntlet 2026-07-02): Pattern B / zz-k2 / trend-ema80 / RR4 / daily KAMA(14)-rising
gate / pullback-limit frac0.3 (stop+target at market levels) / net $15 rt absolute.
Span = genuine 15m density 2018-10-01+. Regenerates the trade list via breakout_wave.run()
(cost=0, $15 applied post-hoc on the risk column) -- MUST reproduce N=614 / meanR+0.322 /
PF1.37 before any bucket is read. Then: entry-hour (UTC) and weekday buckets of net R.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

ARGS = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0,
            pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80,
            bo_window=20, tp_mode="rr", rr=4.0, atr=14, cost=0.0, swap_pct=0.0,
            fwd=500, peryear=False, start=None, end=None, daily_sma=0,
            daily_slope_k=0, risk=0.01, gate_kama=14, pullback_frac=0.3, tf="15min", csv="")

RT = 15.0  # $ round-trip

def main():
    d = load_mt5_csv("data/vantage_btcusd_m15.csv")
    d = d[d.index >= "2018-10-01"]
    t = run(d, SimpleNamespace(**ARGS))
    Rn = t["R"].values - RT / t["risk"].values
    pf = Rn[Rn > 0].sum() / abs(Rn[Rn <= 0].sum())
    print(f"\nREGRESSION: N={len(t)} meanR={Rn.mean():+.3f} PF={pf:.2f} "
          f"win={(Rn>0).mean()*100:.1f}%  (target N=614 / +0.322 / 1.37 / 22.3%)")

    hr = t["time"].dt.hour.values
    wd = t["time"].dt.dayofweek.values  # 0=Mon .. 6=Sun
    print("\nentry-hour (UTC) buckets, net$15:  hour: n  meanR  totR")
    for h in range(24):
        m = hr == h
        if m.sum() >= 5:
            print(f"  {h:02d}: n={m.sum():3d}  meanR={Rn[m].mean():+.3f}  totR={Rn[m].sum():+6.1f}")
    print("\nweekday buckets (0=Mon..6=Sun):")
    for w in range(7):
        m = wd == w
        if m.sum() >= 3:
            print(f"  {w}: n={m.sum():3d}  meanR={Rn[m].mean():+.3f}  totR={Rn[m].sum():+6.1f}")
    wk = wd >= 5
    print(f"\nweekend entries: n={wk.sum()}  meanR={Rn[wk].mean():+.3f}  totR={Rn[wk].sum():+.1f}")
    print(f"weekday entries: n={(~wk).sum()}  meanR={Rn[~wk].mean():+.3f}  totR={Rn[~wk].sum():+.1f}")

if __name__ == "__main__":
    main()
