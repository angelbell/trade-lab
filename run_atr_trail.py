"""Straight backtest of ceyhun's ATR Trailing Stop strategy — no optimization.

Pine Script defaults: ap1=5, af1=0.5, ap2=10, af2=3.0.
Faithful to the original: always in market, flip long/short on Trail1×Trail2 cross.
No extra hard SL (the strategy IS the stop).
"""

import argparse

import vectorbt as vbt

from src.data_loader import fetch_ohlcv
from src.strategy import (
    FEES,
    SLIPPAGE,
    compute_signals_atr_trail,
    profit_factor,
    run_portfolio,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end",   default="2025-01-01")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--tf",    default="15m")
    p.add_argument("--ap1", type=int,   default=5)
    p.add_argument("--af1", type=float, default=0.5)
    p.add_argument("--ap2", type=int,   default=10)
    p.add_argument("--af2", type=float, default=3.0)
    p.add_argument("--ema", type=int,   default=200, help="EMA trend filter period (0=off)")
    p.add_argument("--sl",  type=float, default=0.0, help="Hard SL = ATR*mult (0=off)")
    p.add_argument("--adx", type=float, default=0.0, help="ADX trend-strength threshold (0=off)")
    args = p.parse_args()

    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min",
                "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"}
    freq = freq_map.get(args.tf.lower(), args.tf)

    print(f"Fetching {args.symbol} {args.tf} {args.start} → {args.end} …")
    data = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  Loaded {len(data):,} bars")
    print(f"Params: ap1={args.ap1} af1={args.af1} ap2={args.ap2} af2={args.af2} "
          f"ema={args.ema} adx={args.adx} sl={args.sl}\n")

    le, lx, se, sx = compute_signals_atr_trail(
        data["close"], data["high"], data["low"],
        ap1=args.ap1, af1=args.af1, ap2=args.ap2, af2=args.af2,
        ema_period=args.ema, adx_threshold=args.adx,
    )

    if args.sl and args.sl > 0:
        # hard ATR-based stop on top of the trail-cross exits
        pf = run_portfolio(
            data["close"], data["high"], data["low"],
            le, lx, se, sx,
            atr_mult_sl=args.sl, freq=freq,
        )
    else:
        pf = vbt.Portfolio.from_signals(
            close=data["close"],
            entries=le, exits=lx,
            short_entries=se, short_exits=sx,
            fees=FEES, slippage=SLIPPAGE,
            init_cash=10_000.0, freq=freq,
        )

    n_trades = pf.trades.count()
    print("=" * 50)
    print(f"  Total Return   : {pf.total_return() * 100:8.2f} %")
    print(f"  Profit Factor  : {profit_factor(pf):8.3f}")
    print(f"  Sharpe Ratio   : {pf.sharpe_ratio():8.3f}")
    print(f"  Max Drawdown   : {pf.max_drawdown() * 100:8.2f} %")
    print(f"  Win Rate       : {pf.trades.win_rate() * 100:8.2f} %")
    print(f"  Total Trades   : {n_trades:8d}")
    print("=" * 50)

    # buy & hold benchmark
    bh_ret = (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100
    print(f"  Buy & Hold     : {bh_ret:8.2f} %")


if __name__ == "__main__":
    main()
