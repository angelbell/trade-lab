"""macro_gate_yields.py -- does a REAL-YIELDS (DFII10) regime gate add edge to gold breakout
that price/DXY could not? gold long only when the 10y real yield is FALLING (gold tailwind).

DXY failed (it's gold's mirror = redundant). Real yields are the more ORTHOGONAL gold driver
(gold can rise on safe-haven/CB demand even when yields are flat) AND have longer history
(2003+, covers all gold trades = thicker sample than DXY's 2020+). This is the cleaner macro test.

Falling-yield definitions (daily, shift(1) = no lookahead; gold long-only so falling$real = tailwind):
  ry ret20<0 : 10y real yield 20d change negative
  ry<SMA50   : real yield below its 50d trend
  ry KAMA dn : adaptive real-yield trend falling

  .venv/bin/python research/macro_gate_yields.py
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


def load_fred(path):
    s = pd.read_csv(path)
    s.columns = ["date", "y"]
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize("UTC")
    s["y"] = pd.to_numeric(s["y"], errors="coerce")
    return s.dropna().set_index("date")["y"]


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

    ry = load_fred("data/dfii10_raw.csv")
    gdc = d["close"].resample("1D").last().dropna()
    gk = kama(gdc, 14); gold_kama_up = (gk > gk.shift(1)).shift(1)
    oracle = gdc.shift(-20) > gdc

    ry_ret20 = (ry < ry.shift(20))            # yield falling over 20d
    ry_sma = (ry < ry.rolling(50).mean())     # yield below trend
    rk = kama(ry, 14); ry_kama_dn = (rk < rk.shift(1))

    print(f"\n=== MACRO gate: gold 1H breakout + 10y REAL-YIELD (DFII10) | full history, IS<2022 ===")
    show("always-on", t)
    show("gold KAMA (price)", t[at(gold_kama_up, t.time)])
    show("realyld ret20<0", t[at(ry_ret20.shift(1), t.time)])
    show("realyld <SMA50", t[at(ry_sma.shift(1), t.time)])
    show("realyld KAMA dn", t[at(ry_kama_dn.shift(1), t.time)])
    show("ORACLE (gold fwd)", t[at(oracle, t.time)])


if __name__ == "__main__":
    main()
