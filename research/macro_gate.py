"""macro_gate.py -- does a PRICE-EXTERNAL regime gate (DXY = the dollar) capture headroom that
price-based gates (KAMA) could not? gold breakout, deploy long only when the DOLLAR is WEAK.

regime_headroom.py found the mech->oracle headroom is NOT predictable from gold's own price
features. Hypothesis: the missing forward-regime info lives OUTSIDE price -- for gold, in the
dollar (DXY) and real yields. This tests a DXY-weakness gate on gold 1H breakout vs always-on,
the gold-price KAMA gate, and the oracle ceiling. DXY data starts 2020 (thin IS = caveat).

Dollar-weak definitions (all daily, shift(1) = no lookahead; gold is long-only so weak$ = tailwind):
  dxy<SMA50   : DXY below its 50d SMA (below medium trend)
  dxy KAMA dn : adaptive DXY trend falling
  dxy ret20<0 : DXY 20-day return negative

  .venv/bin/python research/macro_gate.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG, metrics, at
from research.regime_adaptive import kama


def show(name, t):
    m = metrics(t)
    if m is None:
        print(f"  {name:<20} (too few)"); return
    by = t.assign(y=t.time.dt.year).groupby("y").R.sum()
    pys = " ".join(f"{y}:{v:+.0f}" for y, v in by.items())
    print(f"  {name:<20} n={m['n']:>3} CAGR={m['cagr']:+5.1f}% DD={m['dd']:4.1f}% "
          f"CAGR/DD={m['cdd']:5.2f} | IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | {pys}")


def main():
    gcsv = "data/vantage_xauusd_h1.csv"
    d = resample(load_mt5_csv(gcsv), "1h")
    t = run(d, SimpleNamespace(**{**CFG, "csv": gcsv, "tf": "1h"})).sort_values("time").reset_index(drop=True)

    dxy = load_mt5_csv("data/vantage_usdx.r_h1.csv")["close"].resample("1D").last().dropna()
    start = dxy.index[0]
    t = t[t.time >= start].reset_index(drop=True)        # restrict gold trades to DXY-available era

    # gold's own price KAMA gate (reference)
    gdc = d["close"].resample("1D").last().dropna()
    gk = kama(gdc, 14)
    gold_kama_up = (gk > gk.shift(1)).shift(1)

    # DXY weakness gates (dollar down = gold tailwind)
    dsma = dxy < dxy.rolling(50).mean()
    dk = kama(dxy, 14)
    dxy_kama_dn = dk < dk.shift(1)
    dxy_ret20 = dxy < dxy.shift(20)

    # oracle (forward gold regime) ceiling
    oracle = gdc.shift(-20) > gdc

    print(f"\n=== MACRO gate: gold 1H breakout, DXY era {start.date()}->{t.time.iloc[-1].date()} "
          f"(IS<2022 thin!) ===")
    show("always-on", t)
    show("gold KAMA (price)", t[at(gold_kama_up, t.time)])
    show("DXY<SMA50", t[at(dsma.shift(1), t.time)])
    show("DXY KAMA falling", t[at(dxy_kama_dn.shift(1), t.time)])
    show("DXY ret20<0", t[at(dxy_ret20.shift(1), t.time)])
    show("ORACLE (gold fwd)", t[at(oracle, t.time)])


if __name__ == "__main__":
    main()
