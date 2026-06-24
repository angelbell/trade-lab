"""wick_fade.py -- validate the PRICE-ACTION reversal lead: pin-bar / rejection-wick FADE on USDJPY.

The climax test left one whiff: on USDJPY 4h, fading a new-L-bar extreme that prints a long rejection wick
(price-action, NOT volume) gave meanR +0.08 / PF 1.13 / MFE-MAE 1.22 / IS+0.03~OOS+0.13. USDJPY is the
managed/mean-reverting instrument where a BB+RSI fade already worked, so a pin-bar fade is plausible.

Falsifier (up front): real edge = a PLATEAU across the TF ladder (not a lone 4h spike) AND across params
(wick length / extreme lookback / RR) AND per-year green spread AND IS~OOS AND survives cost. A lone-4h
spike, cost death, or one-era concentration => kill. tick-volume irrelevant here (price-action only).
In-sample; live-forward arbitrates.
  .venv/bin/python research/wick_fade.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample
from research.climax_reversal import trades          # mode="A" = wick-only fade
from research.portfolio_kama import cagr_dd

SPLIT = 2018


def line(tag, t, rr):
    if len(t) < 10:
        print(f"  {tag:<20} n={len(t)} (too few)"); return
    be = 100 / (1 + rr)
    cg, dd, cdd, _ = cagr_dd(t[["time", "R"]])
    is_ = t[t.time.dt.year < SPLIT].R.mean(); oos = t[t.time.dt.year >= SPLIT].R.mean()
    mm = t.mfe.mean() / max(t.mae.mean(), 1e-9)
    print(f"  {tag:<20} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f}(BE{be:.0f}) meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.1f} MFE/MAE={mm:4.2f} CAGR/DD={cdd:5.2f} | IS={is_:+.2f} OOS={oos:+.2f}")


def main():
    h1 = load_mt5_csv("data/vantage_usdjpy_h1.csv")
    m5 = load_mt5_csv("data/vantage_usdjpy_m5.csv")
    rr = 2.0
    print("== 1. TF LADDER (USDJPY wick-fade, default L10/wick1.0/RR2) -- is 4h a spike or a plateau? ==")
    for tf, d in [("5m", resample(m5, "5min")), ("15m", resample(m5, "15min")), ("1h", h1),
                  ("2h", resample(h1, "2h")), ("4h", resample(h1, "4h")), ("8h", resample(h1, "8h")),
                  ("1d", resample(h1, "1d"))]:
        line(tf, trades(d, "A", rr=rr), rr)

    d4 = resample(h1, "4h")
    print("\n== 2. PARAM PLATEAU on 4h (neighbors must agree) ==")
    print(" wick_k (rejection length, xATR):")
    for wk in (0.7, 1.0, 1.3, 1.6):
        line(f"  wick_k={wk}", trades(d4, "A", wick_k=wk, rr=rr), rr)
    print(" L (extreme lookback):")
    for L in (6, 10, 14, 20):
        line(f"  L={L}", trades(d4, "A", L=L, rr=rr), rr)
    print(" RR:")
    for r in (1.0, 1.5, 2.0, 2.5, 3.0):
        line(f"  RR{r}", trades(d4, "A", rr=r), r)

    print("\n== 3. PER-YEAR (4h default) ==")
    t = trades(d4, "A", rr=rr)
    by = t.groupby(t.time.dt.year).R.agg(["count", "sum"])
    print("   " + "  ".join(f"{y}:{r['sum']:+.0f}({int(r['count'])})" for y, r in by.iterrows()))
    print(f"   green {int((by['sum']>0).sum())}/{len(by)} yrs")

    print("\n== 4. COST STRESS (4h) ==")
    for c in (0.0002, 0.0005, 0.001):
        line(f"  cost={c*100:.2f}%", trades(d4, "A", rr=rr, cost=c), rr)
    print("\n  verdict: plateau across TF AND params AND green per-year AND cost-survival => real; else kill.")


if __name__ == "__main__":
    main()
