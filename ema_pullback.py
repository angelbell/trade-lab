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
import pandas_ta as ta

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


def run(d: pd.DataFrame, side: str, args, thr: float) -> None:
    ef = (d["close"].rolling(args.ema_fast).mean() if args.fast_ma_type == "sma"
          else d["close"].ewm(span=args.ema_fast, adjust=False).mean()).values
    es = (d["close"].rolling(args.ema_slow).mean() if args.trend_ma_type == "sma"
          else d["close"].ewm(span=args.ema_slow, adjust=False).mean()).values
    a  = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    K, N = args.slope_k, args.fwd

    slope = np.full(len(c), np.nan)
    slope[K:] = (es[K:] - es[:-K]) / (K * np.where(a[K:] > 0, a[K:], np.nan))  # per-bar, ATR-norm
    er = efficiency_ratio(c, args.er_period) if args.filter == "er" else None

    # higher-TF (daily) trend gate: require the DAILY EMA to slope the trade's
    # way. Uses YESTERDAY's completed daily slope (shift 1) -> no lookahead.
    # optional trend-failure exit: close the trade if price closes back across
    # a long MA (e.g. the 200). Checked AFTER stop/target (those are intrabar).
    exit_ma = None
    if args.exit_sma > 0:
        s = d["close"].rolling(args.exit_sma).mean() if args.exit_ma_type == "sma" \
            else d["close"].ewm(span=args.exit_sma, adjust=False).mean()
        exit_ma = s.values

    # HTF gate is optional; tolerate callers (e.g. get_legs) whose args predate it.
    gate_tf = getattr(args, "gate_tf", "")
    gate_type = getattr(args, "gate_type", "ema-slope")
    gate_n = getattr(args, "gate_n", 14)
    gate_up = gate_dn = None
    if gate_tf:
        gc = d["close"].resample(gate_tf).last().dropna()
        if gate_type == "kama-rising":
            from research.regime_adaptive import kama as _kama
            gm = _kama(gc, gate_n)
        elif gate_type == "ema-slope":
            gm = gc.ewm(span=gate_n, adjust=False).mean()
        else:  # sma-slope
            gm = gc.rolling(gate_n).mean()
        rising = (gm > gm.shift(1)).shift(1).reindex(d.index, method="ffill").fillna(0).astype(bool)
        gate_up = rising.values
        gate_dn = (~rising).values
    # backward compat alias
    daily_up = gate_up; daily_dn = gate_dn

    # --- walk the bars: detect pullback then reclaim-entry ---
    entries = []   # (i, entry_px, stop_px)
    state, ext = 0, None
    for i in range(K + 1, len(c) - 1):
        if np.isnan(slope[i]) or np.isnan(a[i]) or a[i] <= 0:
            continue
        # direction from 80EMA slope sign; gate on slope-magnitude OR efficiency
        dir_ok = (slope[i] > 0) if side == "long" else (slope[i] < 0)
        if args.filter == "er":
            gate = (not np.isnan(er[i])) and (er[i] >= thr)
            trending = dir_ok and gate
        else:  # slope-magnitude filter (original)
            trending = (slope[i] >= thr) if side == "long" else (slope[i] <= -thr)
        if daily_up is not None:
            trending = trending and (daily_up[i] if side == "long" else daily_dn[i])
        if not trending:
            state, ext = 0, None
            continue
        # entry trigger:
        #   "close" (default) = wait for a bar to CLOSE back across the 20EMA
        #                       (ignores intrabar pokes that fail) -> confirmed.
        #   "touch"           = a resting stop at the 20EMA fills the instant
        #                       price reclaims it intrabar, even if the bar
        #                       closes back the wrong side (Pine/live behaviour).
        if side == "long":
            if args.entry_trigger == "touch" and state == 1 and h[i] >= ef[i]:
                stop = min(ext if ext is not None else l[i], l[i])
                e = ef[i]
                if e - stop < args.min_stop_atr * a[i]:
                    stop = e - args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None
            elif c[i] < ef[i]:                      # pullback: below 20EMA
                state = 1; ext = l[i] if ext is None else min(ext, l[i])
            elif state == 1 and h[i] >= ef[i]:      # confirmed reclaim -> enter
                e = c[i] if args.fill_at_close else ef[i]   # close = realistic, ema = idealized
                stop = min(ext, l[i])
                if e - stop < args.min_stop_atr * a[i]:
                    stop = e - args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None
        else:
            if args.entry_trigger == "touch" and state == 1 and l[i] <= ef[i]:
                stop = max(ext if ext is not None else h[i], h[i])
                e = ef[i]
                if stop - e < args.min_stop_atr * a[i]:
                    stop = e + args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None
            elif c[i] > ef[i]:
                state = 1; ext = h[i] if ext is None else max(ext, h[i])
            elif state == 1 and l[i] <= ef[i]:
                e = c[i] if args.fill_at_close else ef[i]
                stop = max(ext, h[i])
                if stop - e < args.min_stop_atr * a[i]:
                    stop = e + args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None

    # --- evaluate excursion + 1:1 outcome ---
    # In --no-overlap mode hold ONE position at a time (faithful to a real
    # strategy): skip any entry whose bar falls before the prior trade's exit.
    mfe, mae, trades = [], [], []
    busy_until = -1
    for (i, e, stop) in entries:
        if args.no_overlap and i <= busy_until:
            continue
        risk = abs(e - stop)
        fh, fl = h[i + 1:i + 1 + N], l[i + 1:i + 1 + N]
        if risk <= 0 or len(fh) == 0:
            continue
        exit_j = min(i + N, len(c) - 1)           # default: timeout bar
        if side == "long":
            mfe.append((fh.max() - e) / a[i]); mae.append((e - fl.min()) / a[i])
            tgt, R = e + args.rr * risk, None
            for j in range(i + 1, min(i + 1 + N, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt:  R = args.rr; exit_j = j; break
                if exit_ma is not None and not np.isnan(exit_ma[j]) and c[j] < exit_ma[j]:
                    R = (c[j] - e) / risk; exit_j = j; break        # closed below 200 -> bail
            if R is None: R = (c[exit_j] - e) / risk
        else:
            mfe.append((e - fl.min()) / a[i]); mae.append((fh.max() - e) / a[i])
            tgt, R = e - args.rr * risk, None
            for j in range(i + 1, min(i + 1 + N, len(c))):
                if h[j] >= stop: R = -1.0; exit_j = j; break
                if l[j] <= tgt:  R = args.rr; exit_j = j; break
                if exit_ma is not None and not np.isnan(exit_ma[j]) and c[j] > exit_ma[j]:
                    R = (e - c[j]) / risk; exit_j = j; break        # closed above 200 -> bail
            if R is None: R = (e - c[exit_j]) / risk
        R -= args.cost / risk * e          # round-trip cost in R units
        # calendar days held (incl. weekends) -> daily swap cost on NOTIONAL.
        hold_days = (d.index[exit_j] - d.index[i]).total_seconds() / 86400.0
        if args.swap_pct > 0:
            R -= (args.swap_pct / 100.0) * (e / risk) * hold_days  # notional/risk * %/day * days
        trades.append((d.index[i], R, hold_days))
        busy_until = exit_j

    if not trades:
        print(f"  thr={thr:>4.2f}: no entries"); return None
    mfe, mae = np.array(mfe), np.array(mae)
    ratio = mfe.mean() / mae.mean() if mae.mean() > 0 else float("inf")
    t = pd.DataFrame(trades, columns=["time", "R", "hold"]); t["y"] = t["time"].dt.year
    yrs = sorted(t["y"].unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t[t["y"] < half]["R"] if half else t["R"]
    oosr = t[t["y"] >= half]["R"] if half else t["R"]
    tag = "EDGE" if ratio >= 1.2 else "marg" if ratio >= 1.0 else "DEAD"
    print(f"  thr={thr:>4.2f}  n={len(t):>4}  MFE/MAE={ratio:>4.2f}[{tag}]  "
          f"win={(t['R']>0).mean()*100:>3.0f}%  meanR={t['R'].mean():+.2f}  "
          f"totR={t['R'].sum():+6.0f}  | IS={isr.mean():+.2f} OOS={oosr.mean():+.2f}"
          f"  | hold(d) med={t['hold'].median():.1f} max={t['hold'].max():.1f}"
          + (f"  [swap {args.swap_pct}%/d ON]" if args.swap_pct > 0 else ""))
    if args.peryear:
        pos = sum(1 for _, g in t.groupby("y") if g["R"].sum() > 0)
        print("       per-year totR: " + " ".join(
            f"{y}:{g['R'].sum():+.0f}(n{len(g)})" for y, g in t.groupby("y"))
            + f"   [{pos}/{t['y'].nunique()} yrs +]")
    return t


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
