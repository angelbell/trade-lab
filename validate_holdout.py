"""Out-of-sample HOLDOUT validation of the winning structure_sma strategy.

The WFO (2022-2025) found the edge and CONVERGED on sma_regime=150 / breakout.
This script LOCKS those params and tests them on 2025-01-01 → today — data the
optimizer never saw — then reports the things that actually decide whether to
trade it live:

  1. Locked-param holdout backtest vs buy & hold (PF, Sharpe, MaxDD, return).
  2. A small fractal_n × atr_mult_sl sweep on the holdout — NOT to optimize, but
     to confirm the result is broad (not a knife-edge that only one combo hits).
  3. Vantage SWAP (overnight financing) drag: CFDs charge ~ a few bp/day on the
     notional you hold. We subtract it from the equity curve to see if the edge
     survives realistic carry.
  4. Aggregate stats: max drawdown, and a bootstrap "risk of ruin" estimate
     (probability the equity ever halves) by resampling the trade-PnL sequence.
"""

import argparse

import numpy as np
import pandas as pd

from src.data_loader import fetch_ohlcv
from src.strategy import compute_signals_structure, run_portfolio, profit_factor

# LOCKED converged params from the 2022-2025 WFO (the optimizer's modal winner).
LOCK = dict(entry_mode="breakout", sma_regime=150, htf_timeframe="4h")
FREQ = "1h"
FREQ_MAP = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def annualization(freq: str) -> float:
    bars_per_year = {"15min": 365 * 96, "1h": 365 * 24, "4h": 365 * 6, "1D": 365}
    return np.sqrt(bars_per_year[freq])


def equity_stats(pf, freq: str, label: str) -> dict:
    val = pf.value()
    ret = val.pct_change().dropna()
    ann = annualization(freq)
    sharpe = ret.mean() / ret.std() * ann if ret.std() > 0 else float("nan")
    total = val.iloc[-1] / val.iloc[0] - 1
    dd = (val / val.cummax() - 1).min()
    pf_v = profit_factor(pf)
    n = pf.trades.count()
    return dict(label=label, total=total, sharpe=sharpe, maxdd=dd, pf=pf_v, trades=int(n))


def apply_swap(pf, position_frac: pd.Series, daily_bp: float, freq: str):
    """Subtract overnight financing from the equity curve.

    daily_bp = financing cost in basis points per day on held notional.
    position_frac = fraction of equity in the market each bar (|position|>0 -> 1).
    """
    val = pf.value().copy()
    in_mkt = (position_frac.abs() > 0).reindex(val.index).fillna(False).astype(float)
    bars_per_day = {"15min": 96, "1h": 24, "4h": 6, "1D": 1}[freq]
    per_bar = (daily_bp / 1e4) / bars_per_day
    drag = (in_mkt * per_bar).cumsum()
    return val * (1 - drag)


def risk_of_ruin(pnl: np.ndarray, init_cash: float, ruin_frac: float = 0.5,
                 n_boot: int = 2000, horizon: int | None = None) -> float:
    """Bootstrap: resample the trade-PnL sequence; fraction of paths whose
    running equity ever drops to <= ruin_frac * init_cash."""
    if len(pnl) == 0:
        return float("nan")
    horizon = horizon or len(pnl)
    rng = np.random.default_rng(42)
    ruined = 0
    floor = ruin_frac * init_cash
    for _ in range(n_boot):
        sample = rng.choice(pnl, size=horizon, replace=True)
        eq = init_cash + np.cumsum(sample)
        if eq.min() <= floor:
            ruined += 1
    return ruined / n_boot


def run_one(data, fractal_n, atr_mult_sl, freq):
    le, lx, se, sx = compute_signals_structure(
        data["close"], data["high"], data["low"],
        fractal_n=fractal_n, **LOCK,
    )
    pf = run_portfolio(data["close"], data["high"], data["low"], le, lx, se, sx,
                       atr_mult_sl=atr_mult_sl, freq=freq)
    return pf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-01", help="holdout start (unseen)")
    p.add_argument("--end",   default="2026-06-09")
    p.add_argument("--tf",    default="1h")
    p.add_argument("--swap-bp", type=float, default=2.0,
                   help="Vantage overnight financing, bp/day on notional")
    args = p.parse_args()
    freq = FREQ_MAP[args.tf.lower()]

    print(f"\n=== HOLDOUT {args.start} → {args.end}  ({args.tf} exec / {LOCK['htf_timeframe']} struct / "
          f"daily SMA{LOCK['sma_regime']} gate, breakout) ===\n")
    data = fetch_ohlcv("BTC/USDT", args.tf, args.start, args.end)

    # buy & hold baseline
    bh_total = data["close"].iloc[-1] / data["close"].iloc[0] - 1
    bh_dd = (data["close"] / data["close"].cummax() - 1).min()
    print(f"Buy & Hold: return {bh_total:+.1%}   MaxDD {bh_dd:.1%}\n")

    # 1) + 2) sweep fractal_n x atr_mult_sl on the holdout (robustness, not tuning)
    print(f"{'fractal_n':>9} {'atr_sl':>7} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'trades':>7}")
    base_pf = None
    for fractal_n in (3, 5):
        for atr_mult_sl in (2.0, 3.0):
            pf = run_one(data, fractal_n, atr_mult_sl, freq)
            s = equity_stats(pf, freq, "")
            star = ""
            if fractal_n == 3 and atr_mult_sl == 2.0:
                base_pf = pf
                star = "  <- base"
            print(f"{fractal_n:>9} {atr_mult_sl:>7.1f} {s['pf']:>6.2f} {s['sharpe']:>7.2f} "
                  f"{s['maxdd']:>7.1%} {s['total']:>8.1%} {s['trades']:>7d}{star}")

    # 3) Vantage swap drag on the base config
    print(f"\n--- Vantage swap drag ({args.swap_bp} bp/day) on base (n=3, sl=2.0) ---")
    pos = base_pf.asset_flow().cumsum()  # net position over time (proxy for in-market)
    val_raw = base_pf.value()
    val_swap = apply_swap(base_pf, pos, args.swap_bp, freq)
    print(f"  Return  no-swap {val_raw.iloc[-1]/val_raw.iloc[0]-1:+.1%}   "
          f"with-swap {val_swap.iloc[-1]/val_swap.iloc[0]-1:+.1%}")
    dd_swap = (val_swap / val_swap.cummax() - 1).min()
    print(f"  MaxDD   with-swap {dd_swap:.1%}")

    # 4) risk of ruin via trade-PnL bootstrap
    pnl = base_pf.trades.records_readable["PnL"].to_numpy()
    ror = risk_of_ruin(pnl, init_cash=10_000.0, ruin_frac=0.5)
    print(f"\n--- Risk of ruin (equity ever halves, bootstrap n=2000) ---")
    print(f"  trades={len(pnl)}  mean PnL={pnl.mean():+.1f}  median={np.median(pnl):+.1f}  "
          f"P(ruin to 50%)={ror:.1%}")


if __name__ == "__main__":
    main()
