"""ema_pullback.py -- 20/80 EMA pullback-continuation strategy (4H).

The user's discretionary method, mechanised for screening:

  Trend   : the 80 EMA SLOPE. Up-slope = uptrend, down-slope = downtrend.
            When the slope is flat/shallow (|slope/ATR per bar| < --slope-thresh)
            it's a RANGE -> stand aside, no trades.
  Pullback: in an uptrend, wait for price to CLOSE back below the 20 EMA
            (a counter-trend dip); mirror for downtrend.
  Entry   : the moment price reclaims the 20 EMA in the trend direction.
            Backtest approximation (closed bars can't see intrabar): the first
            bar whose HIGH crosses the 20 EMA after a pullback, filled at the
            20 EMA level -- close to a real intrabar stop-entry.
  Stop    : the pullback's pierce extreme (the low of the dip for longs), with
            a --min-stop-atr floor for when the dip is too shallow.
  Target  : --rr * stop-distance (default 1:1).

Output: MFE/MAE excursion ratio (entry-edge screen) + the actual 1:1 outcome
(win% / mean R / total R) with an IS/OOS split and per-year R, swept across a
few slope-filter thresholds so you can see if "skip the ranges" earns its keep.
R is in each trade's OWN risk units (a 1:1 win = +1R, a stop = -1R).

Run:
  .venv/bin/python ema_pullback.py --csv data/vantage_xauusd_h1.csv --tf 4h --side both
  .venv/bin/python ema_pullback.py --csv data/vantage_usdjpy_h1.csv --tf 4h --side both
"""

import argparse

import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule.lower() in ("1h", "h1", ""):
        return df
    o = {"4h": "4h", "h4": "4h", "1d": "1D", "d1": "1D"}.get(rule.lower(), rule)
    return pd.DataFrame({
        "open":  df["open"].resample(o).first(),
        "high":  df["high"].resample(o).max(),
        "low":   df["low"].resample(o).min(),
        "close": df["close"].resample(o).last(),
    }).dropna()


def efficiency_ratio(c: np.ndarray, n: int) -> np.ndarray:
    """Kaufman Efficiency Ratio: |net change over n| / sum(|bar-to-bar change|).
    ~0 = choppy/range (long path, no progress), ~1 = clean trend."""
    er = np.full(len(c), np.nan)
    absdiff = np.abs(np.diff(c, prepend=c[0]))
    vol = pd.Series(absdiff).rolling(n).sum().values
    net = np.full(len(c), np.nan)
    net[n:] = np.abs(c[n:] - c[:-n])
    er[n:] = np.where(vol[n:] > 0, net[n:] / vol[n:], np.nan)
    return er


def run(d: pd.DataFrame, side: str, args, thr: float):
    """Composable engine entry point — logic lives in src/engine/ (detect_ema /
    gates.ema_* / walk.walk_ema / stats.summarize_ema). This wrapper keeps the
    historical call signature. Bit-identity guarded by invariants/engine_tieback.py
    and invariants/engine_golden.py — re-run both before trusting any engine edit."""
    from src.engine.compat import run_ema_compat   # lazy: avoids import cycle
    return run_ema_compat(d, side, args, thr)


def main() -> None:
    p = argparse.ArgumentParser(description="20/80 EMA pullback-continuation screener")
    p.add_argument("--csv", required=True)
    p.add_argument("--tf", default="4h")
    p.add_argument("--side", default="both", choices=["long", "short", "both"])
    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--ema-slow", type=int, default=80)
    p.add_argument("--slope-k", type=int, default=6, help="bars to measure 80EMA slope")
    p.add_argument("--trend-ma-type", default="ema", choices=["ema", "sma"],
                   help="MA type for the trend/slow line (the Pine trendType toggle)")
    p.add_argument("--fast-ma-type", default="ema", choices=["ema", "sma"],
                   help="MA type for the fast/pullback line")
    p.add_argument("--filter", default="slope", choices=["slope", "er"],
                   help="range filter: slope=80EMA slope magnitude, er=Kaufman efficiency ratio")
    p.add_argument("--er-period", type=int, default=14, help="lookback for efficiency ratio")
    p.add_argument("--swap-pct", type=float, default=0.0,
                   help="daily swap cost as %% of NOTIONAL per calendar day held (e.g. 0.096 ~= -35%%/yr)")
    p.add_argument("--gate-tf", default="",
                   help="HTF gate timeframe (e.g. 1D, 1W, 4h; empty=off)")
    p.add_argument("--gate-type", default="ema-slope",
                   choices=["ema-slope", "sma-slope", "kama-rising"],
                   help="HTF gate type: ema-slope / sma-slope (slope direction) / kama-rising (adaptive)")
    p.add_argument("--gate-n", type=int, default=14,
                   help="MA period for the HTF gate (default 14)")
    p.add_argument("--daily-ema", type=int, default=0,
                   help="legacy alias: --gate-tf 1D --gate-type ema-slope --gate-n N (0=off)")
    p.add_argument("--exit-sma", type=int, default=0,
                   help="trend-failure exit: close the trade when price closes across this MA (0=off, e.g. 200)")
    p.add_argument("--exit-ma-type", default="sma", choices=["sma", "ema"],
                   help="MA type for the --exit-sma trend-failure exit")
    p.add_argument("--peryear", action="store_true", help="show per-year totR breakdown")
    p.add_argument("--no-overlap", action="store_true",
                   help="one position at a time (faithful to a real strategy; skip signals while in a trade)")
    p.add_argument("--entry-trigger", default="close", choices=["close", "touch"],
                   help="close=wait for confirmed close across 20EMA; touch=resting stop fills intrabar (Pine/live)")
    p.add_argument("--fill-at-close", action="store_true",
                   help="confirmed mode: fill at the reclaim bar's CLOSE (realistic) instead of the 20EMA level (idealized)")
    p.add_argument("--rr", type=float, default=1.0, help="reward:risk")
    p.add_argument("--min-stop-atr", type=float, default=0.5, help="min stop dist (ATR) floor")
    p.add_argument("--atr", type=int, default=14)
    p.add_argument("--fwd", type=int, default=30, help="forward bars for excursion / outcome")
    p.add_argument("--cost", type=float, default=0.001, help="round-trip cost fraction")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    # backward compat: --daily-ema N → --gate-tf 1D --gate-type ema-slope --gate-n N
    if args.daily_ema > 0 and not args.gate_tf:
        args.gate_tf = "1D"
        args.gate_type = "ema-slope"
        args.gate_n = args.daily_ema

    d = load_mt5_csv(args.csv)
    if args.start or args.end:
        d = d.loc[args.start:args.end]
    d = resample(d, args.tf)
    thrs = [0.0, 0.05, 0.10, 0.15, 0.20] if args.filter == "slope" \
        else [0.0, 0.20, 0.30, 0.40, 0.50]
    gate_tag = f"  gate={args.gate_tf}/{args.gate_type}({args.gate_n})" if args.gate_tf else "  gate=off"
    print(f"\n=== EMA{args.ema_fast}/{args.ema_slow} pullback  {args.csv}  TF={args.tf}  "
          f"RR={args.rr}  filter={args.filter}"
          f"{'('+str(args.er_period)+')' if args.filter=='er' else ''}{gate_tag} ===")
    print(f"  {len(d):,} {args.tf} bars  {d.index[0].date()} -> {d.index[-1].date()}  "
          f"(thr sweep: 0=all {'trends' if args.filter=='slope' else 'efficiency'}, "
          f"higher={'steeper' if args.filter=='slope' else 'cleaner trend'} only)")

    sides = ["long", "short"] if args.side == "both" else [args.side]
    for s in sides:
        print(f"\n  ----- {s.upper()} -----")
        for thr in thrs:
            run(d, s, args, thr)


if __name__ == "__main__":
    main()
