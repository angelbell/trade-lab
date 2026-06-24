"""Trim-outlier robustness check for the ATR-Trail trade distribution.

Question: is the edge broad-based, or carried by a handful of lucky big wins?
Method: take the full trade list, then drop the N biggest wins AND N biggest
losses, and recompute the stats. If PF stays > 1 after trimming, the edge is
robust; if it collapses, it was outlier-driven (fragile).
"""

import argparse

import numpy as np
import pandas as pd

from src.data_loader import fetch_ohlcv
from src.strategy import compute_signals_atr_trail, run_portfolio


def pf_of(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = pnl[pnl < 0].abs().sum()
    return gp / gl if gl > 0 else float("inf")


def stats_block(label: str, pnl: pd.Series) -> None:
    n = len(pnl)
    wins = (pnl > 0).sum()
    print(f"--- {label}  (n={n}) ---")
    print(f"  Total PnL   : {pnl.sum():10.2f}")
    print(f"  Profit Factor: {pf_of(pnl):9.3f}")
    print(f"  Win Rate    : {wins / n * 100:9.1f} %")
    print(f"  Mean trade  : {pnl.mean():10.2f}")
    print(f"  Median trade: {pnl.median():10.2f}")
    print(f"  Best / Worst: {pnl.max():10.2f} / {pnl.min():.2f}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end",   default="2025-01-01")
    p.add_argument("--tf",    default="4h")
    p.add_argument("--ema",   type=int,   default=200)
    p.add_argument("--adx",   type=float, default=20.0)
    p.add_argument("--trim",  type=int,   default=5, help="drop N biggest wins AND N biggest losses")
    args = p.parse_args()

    freq_map = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(args.tf.lower(), args.tf)

    data = fetch_ohlcv("BTC/USDT", args.tf, args.start, args.end)
    le, lx, se, sx = compute_signals_atr_trail(
        data["close"], data["high"], data["low"],
        ema_period=args.ema, adx_threshold=args.adx,
    )
    pf = run_portfolio(data["close"], data["high"], data["low"],
                       le, lx, se, sx, atr_mult_sl=0.0, freq=freq)

    pnl = pf.trades.records_readable["PnL"].reset_index(drop=True)
    pnl_sorted = pnl.sort_values()

    print(f"\n4h ATR-Trail + EMA{args.ema} + ADX>{args.adx}  ({args.start}→{args.end})\n")
    stats_block("FULL", pnl)

    n = args.trim
    # drop n smallest (biggest losses) and n largest (biggest wins)
    trimmed = pnl_sorted.iloc[n:-n] if len(pnl) > 2 * n else pnl_sorted
    stats_block(f"TRIMMED (drop {n} biggest wins + {n} biggest losses)", trimmed)

    print("Dropped biggest LOSSES :",
          [round(x, 1) for x in pnl_sorted.iloc[:n].tolist()])
    print("Dropped biggest WINS   :",
          [round(x, 1) for x in pnl_sorted.iloc[-n:].tolist()])


if __name__ == "__main__":
    main()
