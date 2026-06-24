"""regime_kama_legs.py -- does the KAMA(14)-rising regime gate generalize across ALL book legs?

Validated on BTC breakout (0.61->1.4) and seen helping gold breakout (0.35->0.72). This applies
the SAME daily-KAMA(14)-rising gate (shift1, no-lookahead) to the pullback legs too, to map how
far the adaptive gate transfers: gold/BTC x breakout/pullback. always-on vs +kama, IS/OOS.

  .venv/bin/python research/regime_kama_legs.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG, metrics, at
from research.regime_adaptive import kama

KN = 14
PB = dict(side="long", ema_fast=20, ema_slow=80, slope_k=6, filter="slope", er_period=14,
          swap_pct=0.0, daily_ema=0, exit_sma=0, exit_ma_type="sma", peryear=False,
          no_overlap=True, entry_trigger="close", fill_at_close=True, rr=3.0,
          min_stop_atr=0.5, atr=14, fwd=90, cost=0.001, trend_ma_type="sma", fast_ma_type="ema")


def kama_gate(d, times):
    dc = d["close"].resample("1D").last().dropna()
    km = kama(dc, KN)
    return at((km > km.shift(1)).shift(1), times)          # KAMA rising, no lookahead


def line(name, t):
    a = metrics(t); g = metrics(t[t.g])
    if a is None or g is None:
        print(f"  {name:<22} (too few)"); return
    flag = "OK" if g["cdd"] > a["cdd"] else "-- no gain"
    print(f"  {name:<22} always-on {a['cdd']:5.2f} (IS{a['isr']:+.2f}/OOS{a['oos']:+.2f}, n{a['n']:>3})  ->  "
          f"+kama {g['cdd']:5.2f} (IS{g['isr']:+.2f}/OOS{g['oos']:+.2f}, n{g['n']:>3})  {flag}")


def main():
    print(f"\n=== KAMA({KN})-rising gate across book legs :: always-on -> +kama (CAGR/DD, IS<2022) ===\n")
    # breakout legs
    for name, csv, tf, rr, fwd in [("GOLD breakout 1H", "data/vantage_xauusd_h1.csv", "1h", 3.0, 500),
                                   ("BTC breakout 4H", "data/vantage_btcusd_h1.csv", "4h", 2.0, 300)]:
        d = resample(load_mt5_csv(csv), tf)
        t = run_bo(d, SimpleNamespace(**{**CFG, "csv": csv, "tf": tf, "rr": rr, "fwd": fwd}))
        t = t.sort_values("time").reset_index(drop=True); t["g"] = kama_gate(d, t.time)
        line(name, t)
    # pullback legs
    for name, csv, tf in [("BTC pullback 4H", "data/vantage_btcusd_h1.csv", "4h"),
                          ("GOLD pullback 4H", "data/vantage_xauusd_h1.csv", "4h")]:
        d = resample(load_mt5_csv(csv), tf)
        t = run_pb(d, "long", SimpleNamespace(**{**PB, "csv": csv, "tf": tf}), 0.0)
        t = t.sort_values("time").reset_index(drop=True); t["g"] = kama_gate(d, t.time)
        line(name, t)
    print("\n  read: a TRANSFERABLE adaptive gate helps (or at least doesn't hurt) across legs.")


if __name__ == "__main__":
    main()
