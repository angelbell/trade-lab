"""ut_bot.py -- faithful mechanization of the TradingView "UT Bot Alerts" indicator.

UT Bot = an ATR trailing-stop STOP-AND-REVERSE trend follower:
  nLoss = a * ATR(c)  (a = "key value" sensitivity; c = ATR period, default a=1 c=10)
  xATRTrailingStop ratchets (Chandelier-style); flip LONG when src crosses above the stop,
  SHORT when it crosses below. ema(src,1)==src so `above`=crossover(src,stop). No fixed SL/TP --
  the trailing stop / opposite signal IS the exit. Always in the market (long or short).

Tested as-written: next-bar-open fill, sequential stop (no lookahead), cost charged per round trip.
R is normalized by the system's own stop distance nLoss at entry (so meanR/PF are comparable).
Falsification: all-signals base, meanR/PF, IS/OOS, a- & c-PLATEAU sweep, per-year, long/short split,
cost stress. Pine atr()=RMA(tr) -> pandas_ta.atr default (rma) matches.

  .venv/bin/python research/ut_bot.py --csv data/vantage_xauusd_h1.csv --tf 1h
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample


def ut_signals(d, a, c):
    src = d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=c).values
    n = len(src)
    nLoss = a * atr
    stop = np.zeros(n)
    for i in range(1, n):
        ps = stop[i - 1]
        if np.isnan(nLoss[i]):
            stop[i] = 0.0
            continue
        if src[i] > ps and src[i - 1] > ps:
            stop[i] = max(ps, src[i] - nLoss[i])
        elif src[i] < ps and src[i - 1] < ps:
            stop[i] = min(ps, src[i] + nLoss[i])
        elif src[i] > ps:
            stop[i] = src[i] - nLoss[i]
        else:
            stop[i] = src[i] + nLoss[i]
    buy = np.zeros(n, bool); sell = np.zeros(n, bool)
    for i in range(1, n):
        buy[i] = src[i] > stop[i] and src[i - 1] <= stop[i - 1]               # crossover(src,stop)
        sell[i] = stop[i] > src[i] and stop[i - 1] <= src[i - 1]              # crossover(stop,src)
    return stop, buy, sell, nLoss


def trades(d, a, c, cost_usd, warmup=50):
    op = d["open"].values; tm = d.index
    stop, buy, sell, nLoss = ut_signals(d, a, c)
    n = len(op)
    sigs = sorted([(i, 1) for i in np.where(buy)[0] if i > warmup] +
                  [(i, -1) for i in np.where(sell)[0] if i > warmup])
    rows = []; prev = None
    for i, dr in sigs:
        if prev is None:
            prev = (i, dr); continue
        if dr == prev[1]:
            continue                                     # same direction -> already in it
        ei, xi = prev[0] + 1, i + 1
        if xi < n and nLoss[prev[0]] > 0:
            e, x = op[ei], op[xi]
            R = prev[1] * (x - e) / nLoss[prev[0]] - cost_usd / nLoss[prev[0]]
            rows.append((tm[ei], R, prev[1]))
        prev = (i, dr)
    return pd.DataFrame(rows, columns=["time", "R", "dir"])


def stat(t, tag):
    if len(t) == 0:
        print(f"  {tag:<22} n=0"); return
    w = t.R[t.R > 0].sum(); l = t.R[t.R < 0].sum()
    pf = w / abs(l) if l else float("inf")
    isr = t[t.time.dt.year < 2022].R; oos = t[t.time.dt.year >= 2022].R
    print(f"  {tag:<22} n={len(t):>5} win={(t.R>0).mean()*100:>3.0f}% PF={pf:4.2f} "
          f"meanR={t.R.mean():+.3f} totR={t.R.sum():+6.0f} | IS={isr.mean():+.3f} OOS={oos.mean():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_xauusd_h1.csv")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--a", type=float, default=3.0)
    ap.add_argument("--c", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.30, help="round-trip cost in USD (gold)")
    args = ap.parse_args()
    d = resample(load_mt5_csv(args.csv), args.tf)
    print(f"\n=== UT Bot  {os.path.basename(args.csv)} {args.tf}  {d.index[0].date()}->{d.index[-1].date()}"
          f"  cost={args.cost}usd/rt ===")
    print(f"  [meanR>0 & PF>1 = edge; flip system so ~50% win is fine if winners>losers]")

    print(f"\n  -- user's pick: a={args.a} c={args.c} (long/short split = beta check) --")
    t = trades(d, args.a, args.c, args.cost)
    stat(t, f"a{args.a} all")
    stat(t[t.dir == 1], f"a{args.a} LONG")
    stat(t[t.dir == -1], f"a{args.a} SHORT")

    print(f"\n  -- a PLATEAU sweep (c={args.c}; real edge = neighbors agree) --")
    for a in (1, 2, 3, 4, 5, 6):
        stat(trades(d, a, args.c, args.cost), f"a={a}")

    print(f"\n  -- c (ATR period) sweep (a={args.a}) --")
    for c in (5, 7, 10, 14, 20):
        stat(trades(d, args.a, c, args.cost), f"c={c}")

    print(f"\n  -- per-year (a={args.a} c={args.c}) --")
    for y, g in t.assign(y=t.time.dt.year).groupby("y"):
        print(f"    {y}: n={len(g):>4} totR={g.R.sum():+6.1f} meanR={g.R.mean():+.3f}")


if __name__ == "__main__":
    main()
