"""jpy_ema_pullback_orig.py -- faithful mechanization of the user's old TradingView
"EMA Pullback Strategy [Original]" (USDJPY), to falsify it on real data.

Rules (as written in the Pine):
  EMA50 + EMA200 trend filter (optional). pivot(5,5) swing high/low -> fib50 = mid.
  LONG  : close>EMA50 AND low<=EMA50 (touched it)  AND  close within +-1% of fib50
          AND bullish pin bar  AND  (close>EMA200 or filter off).  SHORT mirror.
  pin bar (AS WRITTEN -- looks INVERTED, see --fix-pin):
     bull = close>open and (high-close) > 2*(open-low)     # big UPPER wick (?!)
     bear = close<open and (close-low)  > 2*(high-open)     # big LOWER wick (?!)
  SL = entry -/+ 1.5*ATR ; TP = entry +/- 1.5*ATR*2.5 (RR 2.5). 1 trade/day, 1 pos.

Fill = NEXT bar open (Pine default). intrabar SL/TP. cost in pips (round trip).
R per trade = +2.5 (TP) / -1 (SL) / mark-to-close if unresolved, minus cost/risk.

  .venv/bin/python research/jpy_ema_pullback_orig.py --csv data/vantage_usdjpy_m1.csv --resample 5min
  .venv/bin/python research/jpy_ema_pullback_orig.py --csv data/vantage_usdjpy_h1.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

PIP = 0.01  # USDJPY


def pivots(high, low, L):
    n = len(high)
    sh = np.full(n, np.nan); sl = np.full(n, np.nan)
    cur_h = np.nan; cur_l = np.nan
    for i in range(2 * L, n):
        c = i - L
        if high[c] == high[i - 2 * L:i + 1].max():
            cur_h = high[c]
        if low[c] == low[i - 2 * L:i + 1].min():
            cur_l = low[c]
        sh[i] = cur_h; sl[i] = cur_l
    return sh, sl


def run(d, args):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    ema = ta.ema(d["close"], args.ema_len).values
    ema200 = ta.ema(d["close"], 200).values
    atr = ta.atr(d["high"], d["low"], d["close"], length=args.atr_len).values
    n = len(c)
    sh, sl_sw = pivots(h, l, args.pivot_len)
    fib = np.where(~np.isnan(sh) & ~np.isnan(sl_sw), (sh + sl_sw) / 2, np.nan)

    if args.fix_pin:    # corrected hammer / shooting-star shapes
        bull = (c > o) & ((np.minimum(o, c) - l) > 2 * (h - np.maximum(o, c)))
        bear = (c < o) & ((h - np.maximum(o, c)) > 2 * (np.minimum(o, c) - l))
    else:               # AS WRITTEN in the user's Pine
        bull = (c > o) & ((h - c) > 2 * (o - l))
        bear = (c < o) & ((c - l) > 2 * (h - o))

    near_fib = (~np.isnan(fib)) & (c <= fib * (1 + args.fib_tol)) & (c >= fib * (1 - args.fib_tol))
    trendUp = (c > ema200) | (not args.use_ema200)
    trendDn = (c < ema200) | (not args.use_ema200)
    longE = (c > ema) & (l <= ema) & near_fib & bull & trendUp
    shortE = (c < ema) & (h >= ema) & near_fib & bear & trendDn

    day = (d.index + pd.Timedelta(hours=args.day_offset_h)).normalize().values
    cost = args.cost_pips * PIP
    trades = []
    pos = 0; e_px = stop = tp = 0.0; ei = 0; cur_day = None; day_n = 0
    for i in range(n - 1):
        if day[i] != cur_day:
            cur_day = day[i]; day_n = 0
        if pos != 0:
            hit = None
            if pos > 0:
                if l[i] <= stop: hit = stop
                elif h[i] >= tp: hit = tp
            else:
                if h[i] >= stop: hit = stop
                elif l[i] <= tp: hit = tp
            if hit is not None:
                g = (hit - e_px) if pos > 0 else (e_px - hit)
                trades.append((d.index[ei], pos, g / PIP - cost / PIP, (i - ei)))
                pos = 0
        if pos != 0:
            continue
        if day_n >= args.max_per_day:
            continue
        sig = 1 if longE[i] else (-1 if shortE[i] else 0)
        if sig == 0 or np.isnan(atr[i]):
            continue
        e_px = o[i + 1]; ei = i + 1; pos = sig; day_n += 1
        ref = c[i]                                   # TV sets SL/TP from the SIGNAL-bar close
        risk = atr[i] * args.sl_mult
        stop = ref - risk if sig > 0 else ref + risk
        tp = ref + risk * args.rr if sig > 0 else ref - risk * args.rr

    t = pd.DataFrame(trades, columns=["t", "dir", "pips", "bars"])
    if len(t) == 0:
        print("  no trades"); return
    risk_pips = (t.pips[t.pips < 0].abs().mean()) if (t.pips < 0).any() else 1.0
    R = t.pips / risk_pips
    wins = t.pips[t.pips > 0]; loss = t.pips[t.pips < 0]
    pf = wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() else float("inf")
    eq = (1 + 0.01 * R).cumprod(); dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    t["y"] = t.t.dt.year
    yrs = sorted(t.y.unique()); half = yrs[len(yrs)//2] if len(yrs) > 1 else None
    isr = t.pips[t.y < half] if half else t.pips
    oos = t.pips[t.y >= half] if half else t.pips
    print(f"  n={len(t):>4}  win={(t.pips>0).mean()*100:>3.0f}%  PF={pf:4.2f}  "
          f"net={t.pips.sum():+7.0f}p  meanR={R.mean():+.2f}  maxDD={dd:4.1f}%  "
          f"| IS_net={isr.sum():+.0f} OOS_net={oos.sum():+.0f}  avgW={wins.mean() if len(wins) else 0:+.1f} avgL={loss.mean() if len(loss) else 0:+.1f}")
    if args.peryear and len(yrs) > 1:
        print("       per-year net(p): " + " ".join(f"{y}:{g.pips.sum():+.0f}({len(g)})" for y, g in t.groupby("y")))
    if args.list:
        print("   --- trades (entry time UTC | dir | pips | bars-held) ---")
        for _, r in t.iterrows():
            print(f"   {r.t}  {'LONG ' if r['dir']>0 else 'SHORT'}  {r.pips:+6.1f}p  {int(r.bars)}b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--resample", default=None, help="e.g. 5min (resample M1->M5)")
    ap.add_argument("--ema-len", type=int, default=50)
    ap.add_argument("--pivot-len", type=int, default=5)
    ap.add_argument("--use-ema200", type=lambda s: s != "0", default=True)
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--sl-mult", type=float, default=1.5)
    ap.add_argument("--rr", type=float, default=2.5)
    ap.add_argument("--max-per-day", type=int, default=1)
    ap.add_argument("--day-offset-h", type=int, default=0, help="shift the day boundary (h) for the 1/day rule")
    ap.add_argument("--fib-tol", type=float, default=0.01, help="fib50 band (fraction). 0.01=+-1%% (as written)")
    ap.add_argument("--fix-pin", action="store_true", help="use corrected hammer/star pin-bar shapes")
    ap.add_argument("--cost-pips", type=float, default=0.8, help="round-trip cost in pips")
    ap.add_argument("--peryear", action="store_true")
    ap.add_argument("--list", action="store_true", help="print every trade (to diff vs TradingView)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    d = load_mt5_csv(a.csv)
    if a.resample:
        d = pd.DataFrame({"open": d["open"].resample(a.resample).first(),
                          "high": d["high"].resample(a.resample).max(),
                          "low":  d["low"].resample(a.resample).min(),
                          "close":d["close"].resample(a.resample).last()}).dropna()
    if a.start:
        d = d.loc[a.start:]
    if a.end:
        d = d.loc[:a.end]
    print(f"\n=== {a.csv} {a.resample or 'native'}  {d.index[0]} -> {d.index[-1]}  "
          f"({len(d):,} bars)  ema{a.ema_len}/200 piv{a.pivot_len} SL{a.sl_mult}ATR RR{a.rr} "
          f"fibtol{a.fib_tol} {'FIXEDpin' if a.fix_pin else 'origpin'} cost{a.cost_pips}p ===")
    run(d, a)


if __name__ == "__main__":
    main()
