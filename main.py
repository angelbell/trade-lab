"""Walk-forward optimization pipeline entry point.

Usage:
    python main.py --start 2022-01-01 --end 2024-01-01
    python main.py --start 2022-01-01 --end 2024-01-01 --htf-tf 4h --htf-period 50
    python main.py --start 2022-01-01 --end 2024-01-01 --htf-tf 1h --htf-period 50
"""

import argparse

from src.data_loader import fetch_ohlcv
from src.report import print_oos_table, print_wfe_summary
from src.wfo import run_wfo


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC/USDT WFO Pipeline")
    parser.add_argument("--start",    default="2022-01-01", help="Data start date (inclusive)")
    parser.add_argument("--end",      default="2024-01-01", help="Data end date (exclusive)")
    parser.add_argument("--symbol",   default="BTC/USDT")
    parser.add_argument("--tf",       default="1d",  help="Timeframe (e.g. 15m, 1h, 1d)")
    parser.add_argument("--is-days",  type=int, default=180, help="In-sample window days")
    parser.add_argument("--oos-days", type=int, default=30,  help="Out-of-sample window days")
    parser.add_argument("--step",       type=int, default=30,   help="Step size in days")
    parser.add_argument("--htf-tf",   default="1W",         help="HTF regime timeframe (e.g. 1h, 1W)")
    parser.add_argument("--strategy", default="williams",   help="Strategy: williams | donchian")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} {args.tf} from {args.start} to {args.end} …")
    data = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  Loaded {len(data):,} bars ({data.index[0]} → {data.index[-1]})")

    print(f"\nRunning WFO: IS={args.is_days}d  OOS={args.oos_days}d  step={args.step}d")
    # map ccxt timeframe string to vectorbt freq string
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min",
                "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"}
    freq = freq_map.get(args.tf.lower(), args.tf)

    results = run_wfo(
        data,
        is_days=args.is_days,
        oos_days=args.oos_days,
        step_days=args.step,
        htf_timeframe=args.htf_tf,
        strategy=args.strategy,
        freq=freq,
    )

    print_oos_table(results)
    print_wfe_summary(results)


if __name__ == "__main__":
    main()
