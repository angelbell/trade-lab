"""Dead-simple 200-day MA timing — long-only, flat below the MA.

The hypothesis: every active strategy lost to buy & hold. So instead of
fighting the trend, just RIDE it and step aside during crashes.

  close > MA(period)  → hold long
  close < MA(period)  → flat (cash)

One parameter (period). Almost no room to curve-fit. Compared to buy & hold.
"""

import argparse

import vectorbt as vbt

from src.data_loader import fetch_ohlcv
from src.strategy import FEES, SLIPPAGE, profit_factor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end",   default="2025-01-01")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--tf",    default="1d")
    p.add_argument("--period", type=int, default=200)
    p.add_argument("--ma",    default="sma", help="sma | ema")
    args = p.parse_args()

    freq_map = {"1d": "1D", "1w": "1W", "4h": "4h", "1h": "1h"}
    freq = freq_map.get(args.tf.lower(), args.tf)

    print(f"Fetching {args.symbol} {args.tf} {args.start} → {args.end} …")
    data = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  Loaded {len(data):,} bars\n")

    close = data["close"]
    if args.ma == "ema":
        ma = close.ewm(span=args.period, adjust=False).mean().shift(1)
    else:
        ma = close.rolling(args.period).mean().shift(1)

    above = (close > ma).fillna(False)
    entries = above & (~above.shift(1, fill_value=False))   # cross up
    exits   = (~above) & above.shift(1, fill_value=False)    # cross down

    pf = vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits,
        fees=FEES, slippage=SLIPPAGE, init_cash=10_000.0, freq=freq,
        direction="longonly",
    )

    bh_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
    print(f"=== {args.ma.upper()}({args.period}) timing, long-only, {args.tf} ===")
    print(f"  Total Return : {pf.total_return() * 100:8.2f} %")
    print(f"  Profit Factor: {profit_factor(pf):8.3f}")
    print(f"  Sharpe Ratio : {pf.sharpe_ratio():8.3f}")
    print(f"  Max Drawdown : {pf.max_drawdown() * 100:8.2f} %")
    print(f"  Win Rate     : {pf.trades.win_rate() * 100:8.2f} %")
    print(f"  Trades       : {pf.trades.count():8d}")
    print(f"  --- vs ---")
    print(f"  Buy & Hold   : {bh_ret:8.2f} %  (DD see below)")
    bh = vbt.Portfolio.from_holding(close, init_cash=10_000.0, freq=freq)
    print(f"  B&H Sharpe   : {bh.sharpe_ratio():8.3f}")
    print(f"  B&H Max DD   : {bh.max_drawdown() * 100:8.2f} %")


if __name__ == "__main__":
    main()
