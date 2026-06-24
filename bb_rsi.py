"""bb_rsi.py -- faithful backtest of the user's TradingView "BB+RSI" scalper.

Ports the Pine v6 strategy (Bollinger Bands + RSI, always-in-market reverse):

  basis  = SMA(close, length)         length=21
  dev    = mult * STDEV(close, length) mult=2.0
  upper/lower = basis +/- dev
  rsi    = RSI(close, rsiLength)       rsiLength=14
  buyEntry  = crossover(close, lower)  and rsi <= oversold   (30)
  sellEntry = crossunder(close, upper) and rsi >= overbought (70)

The Pine version has NO stop-loss / take-profit: it is stop-and-REVERSE -- a
long is closed only when the opposite (short) signal fires, and vice versa, so
it is always in the market once started (confirmed by the user's trade list:
"long close" rows carry a BBandSE signal). We replicate that exactly.

Realism the TradingView test omitted:
  - fills at the NEXT bar's OPEN after the signal bar closes (no lookahead;
    slightly conservative vs Pine's intrabar stop fill).
  - SPREAD + SLIPPAGE charged on every fill (the dominant cost of a 1-min FX
    scalper -- a few-pip mean-reverter lives or dies on this).

P&L is reported in PIPS (USDJPY: 1 pip = 0.01). The whole point is to see what
happens OUTSIDE the 12-day June range the user could see on TradingView.

Run:
  .venv/bin/python bb_rsi.py --csv data/vantage_usdjpy_m1.csv --spread-pips 0.3
  .venv/bin/python bb_rsi.py --csv data/vantage_usdjpy_m1.csv --start 2026-06-01 --end 2026-06-13   # validate vs TV
"""

import argparse

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv

PIP = 0.01  # USDJPY


def run(d: pd.DataFrame, args) -> None:
    c = d["close"]
    basis = c.rolling(args.length).mean()
    dev = args.mult * c.rolling(args.length).std(ddof=0)       # Pine ta.stdev = population
    upper = (basis + dev).values
    lower = (basis - dev).values
    rsi = ta.rsi(c, args.rsi_len).values
    cv = c.values
    op = d["open"].values
    hi = d["high"].values
    lo = d["low"].values
    idx = d.index

    # --- optional filters (the user's idea: direction gate + 乖離率) ---------
    # direction: only mean-revert WITH the higher trend (buy dips in an uptrend,
    # sell rallies in a downtrend) -> stop fighting trends, which is what made
    # the no-stop tail (500-pip MAE) blow out. 0 = off.
    tup = tdn = None
    if args.trend_ma > 0:
        tma = c.ewm(span=args.trend_ma, adjust=False).mean().values
        tup, tdn = cv > tma, cv < tma
    # deviation %: only fade when price is at least X% away from the basis
    # (a "乖離率" gate -> take only stretched extremes). 0 = off.
    dev_pct = np.abs(cv - basis.values) / basis.values * 100.0

    # regime gate (the (A) idea): only mean-revert when the market is RANGING;
    # sit out strong trends (the source of the 500-pip MAE). range_ok[i]=True
    # means "calm enough to fade". off = always allowed.
    range_ok = np.ones(len(cv), bool)
    if args.regime == "er":
        # Kaufman Efficiency Ratio: |net move| / sum|bar moves| over n. low = range.
        n = args.regime_period
        absd = np.abs(np.diff(cv, prepend=cv[0]))
        vol = pd.Series(absd).rolling(n).sum().values
        net = np.full(len(cv), np.nan); net[n:] = np.abs(cv[n:] - cv[:-n])
        er = np.where(vol > 0, net / vol, np.nan)
        range_ok = er <= args.regime_thresh           # fade only when ER is low
    elif args.regime == "adx":
        adx = ta.adx(d["high"], d["low"], c, length=args.regime_period)
        col = [x for x in adx.columns if x.startswith("ADX")][0]
        range_ok = adx[col].values <= args.regime_thresh
    elif args.regime == "slope":
        slowma = c.ewm(span=args.regime_period, adjust=False).mean().values
        sl = np.full(len(cv), np.nan)
        k = args.regime_period
        sl[k:] = np.abs(slowma[k:] - slowma[:-k]) / cv[k:] * 100.0   # % move over k bars
        range_ok = sl <= args.regime_thresh           # fade only when the trend is flat

    # Pine crossover/crossunder on closed bars
    buy = np.zeros(len(cv), bool)
    sell = np.zeros(len(cv), bool)
    for i in range(1, len(cv)):
        if np.isnan(lower[i]) or np.isnan(rsi[i]):
            continue
        xover = cv[i] > lower[i] and cv[i - 1] <= lower[i - 1]
        xunder = cv[i] < upper[i] and cv[i - 1] >= upper[i - 1]
        b = xover and rsi[i] <= args.oversold
        s = xunder and rsi[i] >= args.overbought
        if args.dev_pct > 0:
            b = b and dev_pct[i] >= args.dev_pct
            s = s and dev_pct[i] >= args.dev_pct
        if tup is not None:
            b = b and tup[i]      # long only with uptrend
            s = s and tdn[i]      # short only with downtrend
        if not range_ok[i]:       # (A) regime gate: skip when trending
            b = s = False
        buy[i] = b; sell[i] = s

    cost_pips = args.spread_pips + args.slippage_pips          # charged per fill
    atr = ta.atr(d["high"], d["low"], c, length=args.atr_len).values
    pos, e_px, e_i, stop_px, cur_slp = 0, None, None, None, args.r_pips
    trades = []   # (entry_time, exit_time, dir, pips_net, mfe_pips, mae_pips, sl_pips_used)

    def excursion(a, b, direction, ep):
        seg_h = hi[a:b + 1]; seg_l = lo[a:b + 1]
        if len(seg_h) == 0:
            return 0.0, 0.0
        if direction > 0:
            return (seg_h.max() - ep) / PIP, (ep - seg_l.min()) / PIP
        return (ep - seg_l.min()) / PIP, (seg_h.max() - ep) / PIP

    def close(exit_i, exit_px):
        mfe, mae = excursion(e_i, exit_i, pos, e_px)
        gross = (exit_px - e_px) / PIP if pos > 0 else (e_px - exit_px) / PIP
        trades.append((idx[e_i], idx[exit_i], pos, gross - cost_pips, mfe, mae, cur_slp))

    for i in range(len(cv) - 1):
        # 1. hard stop-loss (intrabar) -- caps the MAE the no-stop version lacked
        if pos != 0 and stop_px is not None:
            if pos > 0 and lo[i] <= stop_px:
                close(i, stop_px); pos = 0; stop_px = None
            elif pos < 0 and hi[i] >= stop_px:
                close(i, stop_px); pos = 0; stop_px = None
        # 2. signal -> reverse (fill next-bar open)
        want = 1 if buy[i] else (-1 if sell[i] else 0)
        if want == 0 or want == pos:
            continue
        fill = op[i + 1]
        if pos != 0:
            close(i + 1, fill)
        # stop distance: ATR-based (volatility-adaptive) overrides fixed pips
        if args.sl_atr_mult > 0 and not np.isnan(atr[i + 1]):
            sl_dist = args.sl_atr_mult * atr[i + 1]
        elif args.sl_pips > 0:
            sl_dist = args.sl_pips * PIP
        else:
            sl_dist = 0.0
        pos, e_px, e_i = want, fill, i + 1
        stop_px = (fill - sl_dist if want > 0 else fill + sl_dist) if sl_dist > 0 else None
        cur_slp = sl_dist / PIP if sl_dist > 0 else args.r_pips

    if not trades:
        print("  no trades"); return
    t = pd.DataFrame(trades, columns=["t_in", "t_out", "dir", "pips", "mfe", "mae", "slp"])
    t["y"] = t["t_in"].dt.tz_localize(None).dt.to_period("M")
    wins = t[t["pips"] > 0]["pips"]; losses = t[t["pips"] < 0]["pips"]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    # max drawdown in pips
    eq = t["pips"].cumsum().values
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq).max() if len(eq) else 0.0
    # max drawdown in %: size each trade so its OWN SL distance (t.slp) risks
    # risk_pct of equity, compound. This is true constant-risk sizing -> with an
    # ATR stop the position auto-shrinks when volatility (and the stop) widen.
    rets = (t["pips"].values / t["slp"].values) * (args.risk_pct / 100.0)
    eqc = np.cumprod(1.0 + rets)
    peakc = np.maximum.accumulate(eqc)
    ddpct = ((peakc - eqc) / peakc).max() * 100.0 if len(eqc) else 0.0
    totret = (eqc[-1] - 1.0) * 100.0 if len(eqc) else 0.0
    print(f"  trades={len(t):>4}  net={t['pips'].sum():+7.1f} pips  "
          f"win={(t['pips']>0).mean()*100:>3.0f}%  PF={pf:4.2f}  "
          f"avgW={wins.mean():+.1f} avgL={losses.mean() if len(losses) else 0:+.1f}  "
          f"maxDD={dd:.0f}pips/{ddpct:.1f}%  ret={totret:+.0f}%  "
          f"worstMAE={t['mae'].max():.0f}p  worstTr={t['pips'].min():+.0f}p"
          f"  [SLmed={t['slp'].median():.0f}p@{args.risk_pct:g}%]")
    if args.bymonth:
        print("      by month: " + "  ".join(
            f"{m}:{g['pips'].sum():+.0f}({len(g)})" for m, g in t.groupby("y")))


def main() -> None:
    p = argparse.ArgumentParser(description="BB+RSI 1-min mean-reversion scalper backtest")
    p.add_argument("--csv", required=True)
    p.add_argument("--length", type=int, default=21)
    p.add_argument("--mult", type=float, default=2.0)
    p.add_argument("--rsi-len", type=int, default=14)
    p.add_argument("--oversold", type=float, default=30)
    p.add_argument("--overbought", type=float, default=70)
    p.add_argument("--spread-pips", type=float, default=0.3, help="round-trip spread cost per fill (pips)")
    p.add_argument("--slippage-pips", type=float, default=0.0)
    p.add_argument("--trend-ma", type=int, default=0,
                   help="direction filter: only long above / short below this EMA (0=off)")
    p.add_argument("--dev-pct", type=float, default=0.0,
                   help="乖離率 filter: only enter when |close-basis|/basis%% >= this (0=off)")
    p.add_argument("--regime", default="off", choices=["off", "er", "adx", "slope"],
                   help="(A) range gate: fade ONLY in ranges, sit out trends")
    p.add_argument("--regime-period", type=int, default=50, help="lookback for the regime gate")
    p.add_argument("--regime-thresh", type=float, default=0.3,
                   help="fade only when metric <= this (ER ~0.3, ADX ~20, slope ~0.05)")
    p.add_argument("--sl-pips", type=float, default=0.0,
                   help="hard stop-loss in pips (0=off, stop-and-reverse only) -- caps the MAE")
    p.add_argument("--risk-pct", type=float, default=1.0, help="%% equity risked per 1R (for DD%%)")
    p.add_argument("--r-pips", type=float, default=30.0,
                   help="pips that equal 1R when no SL is set (for the DD%% conversion)")
    p.add_argument("--sl-atr-mult", type=float, default=0.0,
                   help="ATR-based stop: SL = this * ATR at entry (overrides --sl-pips; 0=off)")
    p.add_argument("--atr-len", type=int, default=14, help="ATR length for the ATR stop")
    p.add_argument("--bymonth", action="store_true")
    p.add_argument("--pip", type=float, default=0.01,
                   help="pip size of the instrument (USDJPY 0.01, EURUSD/most majors 0.0001)")
    p.add_argument("--resample", default="", help="resample the loaded bars up to this TF (e.g. 4h/1D); empty=native")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    global PIP
    PIP = args.pip

    d = load_mt5_csv(args.csv)
    if args.resample:
        d = d.resample(args.resample, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    if args.start or args.end:
        d = d.loc[args.start:args.end]
    print(f"\n=== BB+RSI  {args.csv}  BB({args.length},{args.mult}) RSI({args.rsi_len},"
          f"{args.oversold:.0f}/{args.overbought:.0f})  spread={args.spread_pips}+slip{args.slippage_pips}pips ===")
    print(f"  {len(d):,} M1 bars  {d.index[0]} -> {d.index[-1]}")
    run(d, args)


if __name__ == "__main__":
    main()
