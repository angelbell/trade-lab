"""gold_bo_scaleout.py -- does a SCALE-OUT exit (bank part at RR1, run the rest to RR3, stop->BE)
beat the all-or-nothing RRn exit on the adopted breakout legs and on the 2-leg book?

Honest prior (from the project): "let winners run RR2-3 >> 1:1" and the retest kill both say the
edge lives in the FAT TAIL (the few breaks that run). Scale-out clips that tail, so it should CUT
CAGR. The only way it earns its keep is if it cuts maxDD MORE than CAGR => higher CAGR/DD. Judge
on CAGR/DD (and book maxDD), never on win% or meanR alone.

  base      = current adopted exit (RR3 gold / RR2 btc, all-or-nothing)
  scaleout  = take tp1_frac at tp1_rr, move stop to BE, run remainder to the same final tgt
Grid over tp1_frac x tp1_rr; BE on. Reuses breakout_wave eval (getattr scale-out path) = same
entries/gate/cost/overlap, only the exit changes.

  .venv/bin/python research/gold_bo_scaleout.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc

GOLD = dict(csv="x", tf="1h", rr=3.0, fwd=500, daily_sma=150, daily_slope_k=10)
BTC = dict(csv="x", tf="4h", rr=2.0, fwd=300)


def gold(**extra):
    d = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h")
    return run_bo(d, SimpleNamespace(**{**CFG, **GOLD, **extra}))[["time", "R"]]


def btc(**extra):
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    t = run_bo(d, SimpleNamespace(**{**CFG, **BTC, **extra}))[["time", "R"]]
    return kama_gate_btc(t)


def stats(t, risk=0.01):
    t = t.sort_values("time")
    eq = (1 + risk * t.R).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t.time.iloc[-1] - t.time.iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    return dict(n=len(t), win=(t.R > 0).mean() * 100, mr=t.R.mean(),
                cagr=cagr, dd=dd, cdd=cagr / max(dd, 1e-9))


def line(name, t):
    s = stats(t)
    print(f"  {name:<26} n={s['n']:>4}  win={s['win']:>3.0f}%  meanR={s['mr']:+.3f}  "
          f"CAGR={s['cagr']:+5.1f}%  maxDD={s['dd']:4.1f}%  CAGR/DD={s['cdd']:5.2f}")


def book(g, b):
    # adopted inv-vol weights (gold 0.79% / btc 1.21%, 2% budget)
    return pd.concat([g.assign(R=g.R * 0.79), b.assign(R=b.R * 1.21)])


def main():
    grid = [(f, rr) for f in (0.3, 0.5, 0.7) for rr in (1.0, 1.5)]

    print("\n=== GOLD bo: base RR3 vs scale-out (bank tp1_frac @ tp1_rr, stop->BE, run to RR3) ===")
    g_base = gold()
    line("base RR3 (all-or-nothing)", g_base)
    g_so = {}
    for f, rr in grid:
        t = gold(tp1_frac=f, tp1_rr=rr, tp1_be=1)
        g_so[(f, rr)] = t
        line(f"scaleout f={f} tp1={rr}R", t)

    print("\n=== BTC bo+KAMA: base RR2 vs scale-out (run to RR2) ===")
    b_base = btc()
    line("base RR2 (all-or-nothing)", b_base)
    b_so = {}
    for f, rr in grid:
        rr_use = min(rr, 1.0)  # tp1 must be < final RR2; keep tp1 at 1.0R for BTC
        t = btc(tp1_frac=f, tp1_rr=1.0, tp1_be=1)
        b_so[(f, 1.0)] = t
    for f in (0.3, 0.5, 0.7):
        line(f"scaleout f={f} tp1=1.0R", b_so[(f, 1.0)])

    print("\n=== 2-leg BOOK (inv-vol 0.79/1.21): base vs scale-out (the DD-reduction thesis) ===")
    line("book base", book(g_base, b_base))
    for f in (0.3, 0.5, 0.7):
        line(f"book scaleout f={f}", book(g_so[(f, 1.0)], b_so[(f, 1.0)]))
    print("\n  ref: book base CAGR/DD ~1.88 (adopted). PASS = scale-out RAISES book CAGR/DD via lower maxDD.")


if __name__ == "__main__":
    main()
