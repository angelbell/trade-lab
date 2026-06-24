"""Validate the structure_sma strategy on VANTAGE's own BTCUSD feed.

We validated the edge on Binance/UTC, but we trade on Vantage -- whose
broker-server-time bars and gaps make the strategy trade ~2x as often and
erode the edge (the MT5 Strategy Tester showed 2022 going negative). This
script re-runs the SAME locked strategy on the broker's exported H1 history
(via export_history.mq5), so we judge the edge on the feed we actually trade.

Steps:
  1. Load the MT5 CSV (broker-server-time bars kept as the clock).
  2. Per-calendar-year breakdown (trades / PF / return / MaxDD), so we can
     line it up against the MT5 Strategy Tester runs and confirm the Python
     pipeline reproduces them (==> the divergence is the data, settled).
  3. Full-period stats + buy & hold, with the same fees/slippage baked in.

Run: .venv/bin/python validate_vantage.py --csv data/vantage_btcusd_h1.csv
"""

import argparse

import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv
from src.strategy import compute_signals_structure, run_portfolio, profit_factor

LOCK = dict(entry_mode="breakout", sma_regime=150, htf_timeframe="4h")
FREQ = "1h"


def stats(pf) -> dict:
    val = pf.value()
    ret = val.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(365 * 24) if ret.std() > 0 else float("nan")
    return dict(
        trades=int(pf.trades.count()),
        pf=profit_factor(pf),
        total=val.iloc[-1] / val.iloc[0] - 1,
        maxdd=(val / val.cummax() - 1).min(),
        sharpe=sharpe,
    )


def run_slice(d: pd.DataFrame, fractal_n: int, atr_mult_sl: float):
    le, lx, se, sx = compute_signals_structure(
        d["close"], d["high"], d["low"], fractal_n=fractal_n, **LOCK
    )
    return run_portfolio(d["close"], d["high"], d["low"], le, lx, se, sx,
                         atr_mult_sl=atr_mult_sl, freq=FREQ)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/vantage_btcusd_h1.csv",
                   help="MT5-exported H1 CSV (from export_history.mq5)")
    p.add_argument("--fractal-n", type=int, default=3)
    p.add_argument("--atr-sl", type=float, default=2.0)
    args = p.parse_args()

    d = load_mt5_csv(args.csv)
    print(f"\nLoaded {len(d):,} H1 bars  {d.index[0]} -> {d.index[-1]}  "
          f"(broker-server time)\n")

    print(f"=== Per-year (Vantage feed, n={args.fractal_n}, atr_sl={args.atr_sl}, "
          f"LEVEL breakout) ===")
    print(f"{'year':>5} {'trades':>7} {'PF':>6} {'return':>8} {'MaxDD':>7} {'Sharpe':>7}  vs MT5 tester")
    mt5_ref = {2022: "28t/0.78/-4%", 2023: "32t/1.66/+14.5%", 2024: "32t/1.35/+7.6%"}
    for yr in range(d.index[0].year, d.index[-1].year + 1):
        dy = d.loc[f"{yr}-01-01":f"{yr}-12-31"]
        if len(dy) < 24 * 200:   # skip partial years with <~200 days
            continue
        pf = run_slice(dy, args.fractal_n, args.atr_sl)
        s = stats(pf)
        ref = mt5_ref.get(yr, "")
        print(f"{yr:>5} {s['trades']:>7} {s['pf']:>6.2f} {s['total']:>8.1%} "
              f"{s['maxdd']:>7.1%} {s['sharpe']:>7.2f}  {ref}")

    # full period + buy & hold
    pf = run_slice(d, args.fractal_n, args.atr_sl)
    s = stats(pf)
    bh = d["close"].iloc[-1] / d["close"].iloc[0] - 1
    bh_dd = (d["close"] / d["close"].cummax() - 1).min()
    print(f"\n=== Full period ===")
    print(f"  strategy : trades={s['trades']}  PF={s['pf']:.2f}  return={s['total']:+.1%}  "
          f"MaxDD={s['maxdd']:.1%}  Sharpe={s['sharpe']:.2f}")
    print(f"  buy&hold : return={bh:+.1%}  MaxDD={bh_dd:.1%}")


if __name__ == "__main__":
    main()
