"""scaleout_transfer.py -- Phase A: does the SCALE-OUT exit (validated on the expansion leg) TRANSFER
to the existing breakout legs (gold_bo, btc_bo_kama)? breakout_wave.run already supports scale-out
(tp1_frac/tp1_rr/tp1_be), so this swaps each leg's fixed-RR exit for half@2R -> runner@4R (noBE, the
expansion-validated recipe; stop = the leg's own STRUCTURAL stop).

Falsifier (up front): adopt per-leg ONLY if CAGR/DD up AND DD-driven (maxDD/streak down) AND IS~OOS held.
A fixed-RR4 column isolates whether scale-out (not just a higher target) is the cause. If all legs fail,
scale-out is expansion-specific. In-sample; live-forward arbitrates.
  .venv/bin/python research/scaleout_transfer.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG
from research.portfolio_kama import cagr_dd, kama_gate_btc

SPLIT = 2022


def leg(csv, tf, base_rr, fwd, extra, scale=False, rr4=False, kama=False):
    ov = {**CFG, "csv": "x", "tf": tf, "fwd": fwd, **extra}
    ov["rr"] = 4.0 if (scale or rr4) else base_rr
    if scale:
        ov.update(tp1_frac=0.5, tp1_rr=2.0, tp1_be=0)        # half@2R -> runner@4R, no breakeven move
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):           # run_bo prints a summary; mute it
        t = run_bo(resample(load_mt5_csv(csv), tf), SimpleNamespace(**ov))
    t = t[["time", "R"]].copy()
    return kama_gate_btc(t) if kama else t


def streak(R):
    m = c = 0
    for x in R:
        c = c + 1 if x < 0 else 0; m = max(m, c)
    return m


def line(tag, t):
    cg, dd, cdd, _ = cagr_dd(t)
    is_ = t[t.time.dt.year < SPLIT].R.sum(); oos = t[t.time.dt.year >= SPLIT].R.sum()
    print(f"  {tag:<24} n={len(t):>4} meanR={t.R.mean():+5.2f} totR={t.R.sum():>+6.1f} "
          f"maxDD%={dd:4.1f} CAGR/DD={cdd:5.2f} loss{streak(t.R.values):>3} | IS={is_:+5.1f} OOS={oos:+5.1f}")


def main():
    GOLD = ("data/vantage_xauusd_h1.csv", "1h", 3.0, 500, {"daily_sma": 150, "daily_slope_k": 10})
    BTC = ("data/vantage_btcusd_h1.csv", "4h", 2.0, 300, {})
    print("== Phase A: scale-out (half@2R->run@4R) transfer to existing breakout legs ==")

    print("\n  -- gold_bo (current fixed RR3) --")
    line("fixed RR3 (current)", leg(*GOLD))
    line("fixed RR4 (control)", leg(*GOLD, rr4=True))
    line("scale-out 2R/4R", leg(*GOLD, scale=True))

    print("\n  -- btc_bo_kama (current fixed RR2 + KAMA gate) --")
    line("fixed RR2 (current)", leg(*BTC, kama=True))
    line("fixed RR4 (control)", leg(*BTC, rr4=True, kama=True))
    line("scale-out 2R/4R", leg(*BTC, scale=True, kama=True))

    print("\n  verdict: adopt per-leg only if CAGR/DD up AND maxDD/streak down AND IS~OOS, vs BOTH its")
    print("           current config AND fixed-RR4 (else the change is just a higher target, not scale-out).")


if __name__ == "__main__":
    main()
