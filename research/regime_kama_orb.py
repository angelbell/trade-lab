"""regime_kama_orb.py -- does the daily-KAMA regime gate help H17 (gold M15 ORB + 1H gate)?

H17 ALREADY has a 1H-trend gate (trades align to 1H EMA80). Principle: a gate helps only if it
supplies regime context the strategy LACKS -> likely redundant here (+ a daily gate was found
"too heavy" for an M15 exec TF). But the 1H gate is a lower-TF LEVEL check while daily-KAMA is a
higher-TF persistence check, so test rather than assume. Two-sided gate: daily KAMA rising ->
longs only / falling -> shorts only. IS + VAL only (TEST stays sealed).

  .venv/bin/python research/regime_kama_orb.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.scalp_lab import orb_signals, htf_trend_gate, backtest, metrics, SPLITS
from research.regime_adaptive import kama
from research.regime_gate_lab import at

P = dict(asia_start_h=0, asia_end_h=7, bo_start_h=7, bo_end_h=11, force_exit_h=20,
         rr=1.0, buf_atr=0.0, sl_buf_atr=0.0, max_range_atr=0.0, min_range_atr=0.0,
         no_tp=True, fade=False, dir="both", sl_frac=1.0, rsi_max=100.0, box_trend_max=1.0,
         htf_tf="1h", htf_ema=80, htf_slope_k=0, cost=1.4, stop_slip=0.0)


def htf_kama_gate(d, dir_, p, kn):
    """REPLACE the 1H EMA80-level gate with a 1H KAMA-slope gate: KAMA rising -> longs only,
    falling -> shorts only. shift(1) on the HTF = no lookahead (same idiom as htf_trend_gate)."""
    ck = d["close"].resample(p.htf_tf, label="left", closed="left").last().dropna()
    km = kama(ck, kn)
    up = (km > km.shift(1)).shift(1).reindex(d.index, method="ffill").fillna(False).values
    dn = (km < km.shift(1)).shift(1).reindex(d.index, method="ffill").fillna(False).values
    g = dir_.copy()
    g[(g > 0) & ~up] = 0
    g[(g < 0) & ~dn] = 0
    return g


def show(label, t):
    m = metrics(t)
    if m is None:
        print(f"  {label:<22} no trades"); return
    print(f"  {label:<22} n={m['n']:>4} net={m['net']:+7.0f}p win={m['win']:>3.0f}% "
          f"PF={m['pf']:4.2f} maxDD={m['dd']:4.1f}%")


def main():
    p = SimpleNamespace(**P)
    csv = "data/vantage_xauusd_m15.csv"
    raw = load_mt5_csv(csv)
    print("\n=== H17 ORB: 1H gate = EMA80-level (current) vs KAMA-slope (replace) | TEST sealed ===")
    for split in ("is", "val"):
        s, e = SPLITS[split]
        d = raw.loc[s:e]
        dir0, sl, tp = orb_signals(d, p)
        # (a) current: 1H EMA80 level gate
        dE, slE, tpE = htf_trend_gate(d, dir0.copy(), sl, tp, p)
        print(f"\n  --- {split.upper()} ({s}..{e}) ---")
        show("EMA80 level (now)", backtest(d, dE, slE, tpE, p))
        # (b) replace with 1H KAMA-slope gate, length sweep
        for kn in (8, 14, 20):
            dK = htf_kama_gate(d, dir0.copy(), p, kn)
            show(f"KAMA{kn} slope", backtest(d, dK, sl, tp, p))


if __name__ == "__main__":
    main()
