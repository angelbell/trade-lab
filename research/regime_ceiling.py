"""regime_ceiling.py -- how big a lever is REGIME SELECTION ("when to run the bot")?

The community system's claimed edge = a human decides WHICH regime to deploy a gold-long
trend bot in. We can't replicate the human's judgment, but we can bound its CEILING:
take a FIXED gold-long edge (validated 1H breakout, entries unchanged) and apply 3
deploy-decisions, then compare CAGR/DD:
  (a) always-on   : take every signal (bot always running)
  (b) mechanical  : daily SMA150+slope gate (our fixed rule, shift(1) = no lookahead)
  (c) oracle      : deploy only when gold's FORWARD M-day return was up (LOOKAHEAD = the
                    ceiling of a perfect human regime-timer)
(b)-(a) = what a fixed rule recovers. (c)-(b) = headroom a perfect human could still add
on top of the fixed rule = how plausible "discretionary regime timing is the edge" is.

  .venv/bin/python research/regime_ceiling.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

CFG = dict(pattern="B", sl_mode="line", sl_buf=0.25, swing="zigzag", zz_k=2.0, pivot_n=5,
           renko_k=2.0, mom_fast=12, mom_slow=26, trend_ema=80, bo_window=20, tp_mode="rr",
           atr=14, cost=0.001, swap_pct=0.0, peryear=False, start=None, end=None,
           daily_sma=0, daily_slope_k=0, wave="all", rr=3.0, fwd=500, risk=0.01)
M = 20  # oracle regime horizon (trading days ~ 1 month)


def metrics(t, risk=0.01):
    t = t.sort_values("time")
    if len(t) == 0:
        return None
    eq = (1 + risk * t["R"]).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    return dict(n=len(t), totR=t.R.sum(), meanR=t.R.mean(), cagr=cagr, dd=dd,
                cdd=cagr / max(dd, 1e-9), ret=(eq.iloc[-1] - 1) * 100)


def show(name, t):
    m = metrics(t)
    if m is None:
        print(f"  {name:<22} no trades"); return
    print(f"  {name:<22} n={m['n']:>4}  totR={m['totR']:+5.0f}  meanR={m['meanR']:+.2f}  "
          f"CAGR={m['cagr']:+5.1f}%  maxDD={m['dd']:4.1f}%  CAGR/DD={m['cdd']:5.2f}")
    by = t.assign(y=t.time.dt.year).groupby("y").R.sum()
    print("       per-year totR: " + " ".join(f"{y}:{v:+.0f}" for y, v in by.items()))


def main():
    csv = "data/vantage_xauusd_h1.csv"
    d = resample(load_mt5_csv(csv), "1h")
    args = SimpleNamespace(**{**CFG, "csv": csv, "tf": "1h"})
    print(f"\n=== regime-selection CEILING test :: gold 1H breakout (entries FIXED) ===")
    t = run(d, args)                                   # always-on base (no daily gate)
    if t is None:
        print("no trades"); return
    t = t.sort_values("time").reset_index(drop=True)

    dc = d["close"].resample("1D").last().dropna()
    sma = dc.rolling(150).mean()
    mech = ((dc > sma) & (sma > sma.shift(10))).shift(1)            # no lookahead
    oracle = (dc.shift(-M) > dc)                                    # LOOKAHEAD = ceiling
    mech_at = mech.reindex(t.time, method="ffill").fillna(False).values
    oracle_at = oracle.reindex(t.time, method="ffill").fillna(False).values

    print()
    show("(a) always-on", t)
    show("(b) mechanical gate", t[mech_at])
    show("(c) oracle (lookahead)", t[oracle_at])
    print(f"\n  gate keep-rate: mechanical {mech_at.mean()*100:.0f}%  oracle {oracle_at.mean()*100:.0f}%  "
          f"(of {len(t)} always-on signals)")


if __name__ == "__main__":
    main()
