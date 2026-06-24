"""pmax.py -- faithful mechanization of PMax "Profit Maximizer" (Kivanc Ozbilgic).

PMax = SuperTrend applied to a MOVING AVERAGE instead of price:
  MAvg = MA(src, length)  (src=hl2 default; MA type selectable)
  longStop/shortStop = MAvg -/+ mult*ATR(periods), ratcheting (Chandelier-style on the MA)
  dir flips when MAvg crosses its trailing stop; PMax = dir==1 ? longStop : shortStop
  buy = crossover(MAvg, PMax), sell = crossunder(MAvg, PMax)  -> a STOP-AND-REVERSE trend flip.
Same family as UT Bot / SuperTrend, just double-smoothed (price -> MA -> ATR stop). No repaint.

Tested as-written: next-bar-open fill, sequential stop (no lookahead), cost/round-trip, R normalized
by the system's own stop distance (mult*ATR at entry) so it's directly comparable to UT Bot.

  .venv/bin/python research/pmax.py --csv data/vantage_xauusd_h1.csv --tf 1h
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.ut_bot import trades as ut_trades


def getMA(s, length, mav):
    if mav == "sma":  return s.rolling(length).mean()
    if mav == "ema":  return s.ewm(span=length, adjust=False).mean()
    if mav == "wwma": return s.ewm(alpha=1 / length, adjust=False).mean()
    if mav == "wma":
        w = np.arange(1, length + 1)
        return s.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    if mav == "zlema":
        lag = int(round(length / 2))
        return (s + (s - s.shift(lag))).ewm(span=length, adjust=False).mean()
    if mav == "tma":
        import math
        return s.rolling(math.ceil(length / 2)).mean().rolling(math.floor(length / 2) + 1).mean()
    raise ValueError(mav)


def pmax_lines(d, length, mult, periods, mav, src="hl2"):
    s = (d["high"] + d["low"]) / 2 if src == "hl2" else d["close"]
    MAvg = getMA(s, length, mav).values
    atr = ta.atr(d["high"], d["low"], d["close"], length=periods).values
    n = len(MAvg)
    longS = np.full(n, np.nan); shortS = np.full(n, np.nan); dirr = np.ones(n); pmax = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(MAvg[i]) or np.isnan(atr[i]):
            continue
        ls = MAvg[i] - mult * atr[i]; ss = MAvg[i] + mult * atr[i]
        lsp = longS[i - 1] if i > 0 and not np.isnan(longS[i - 1]) else ls
        ssp = shortS[i - 1] if i > 0 and not np.isnan(shortS[i - 1]) else ss
        ls = max(ls, lsp) if MAvg[i] > lsp else ls
        ss = min(ss, ssp) if MAvg[i] < ssp else ss
        longS[i] = ls; shortS[i] = ss
        pd_ = dirr[i - 1] if i > 0 else 1
        d_ = 1 if (pd_ == -1 and MAvg[i] > ssp) else (-1 if (pd_ == 1 and MAvg[i] < lsp) else pd_)
        dirr[i] = d_; pmax[i] = ls if d_ == 1 else ss
    return MAvg, pmax, atr * mult


def pmax_trades(d, length, mult, periods, mav, cost_usd, warmup=50):
    MAvg, pmax, nLoss = pmax_lines(d, length, mult, periods, mav)
    op = d["open"].values; tm = d.index; n = len(op)
    buy = np.zeros(n, bool); sell = np.zeros(n, bool)
    for i in range(1, n):
        if np.isnan(pmax[i]) or np.isnan(pmax[i - 1]):
            continue
        buy[i] = MAvg[i] > pmax[i] and MAvg[i - 1] <= pmax[i - 1]
        sell[i] = MAvg[i] < pmax[i] and MAvg[i - 1] >= pmax[i - 1]
    sigs = sorted([(i, 1) for i in np.where(buy)[0] if i > warmup] +
                  [(i, -1) for i in np.where(sell)[0] if i > warmup])
    rows = []; prev = None
    for i, dr in sigs:
        if prev is None:
            prev = (i, dr); continue
        if dr == prev[1]:
            continue
        ei, xi = prev[0] + 1, i + 1
        if xi < n and nLoss[prev[0]] > 0:
            R = prev[1] * (op[xi] - op[ei]) / nLoss[prev[0]] - cost_usd / nLoss[prev[0]]
            rows.append((tm[ei], R, prev[1]))
        prev = (i, dr)
    return pd.DataFrame(rows, columns=["time", "R", "dir"])


def stat(t, tag):
    if len(t) == 0:
        print(f"  {tag:<24} n=0"); return
    w = t.R[t.R > 0].sum(); l = t.R[t.R < 0].sum(); pf = w / abs(l) if l else 9.9
    isr = t[t.time.dt.year < 2022].R; oos = t[t.time.dt.year >= 2022].R
    print(f"  {tag:<24} n={len(t):>5} win={(t.R>0).mean()*100:>3.0f}% PF={pf:4.2f} "
          f"meanR={t.R.mean():+.3f} totR={t.R.sum():+6.0f} | IS={isr.mean():+.3f} OOS={oos.mean():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_xauusd_h1.csv")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--length", type=int, default=10)
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--periods", type=int, default=10)
    ap.add_argument("--mav", default="ema")
    ap.add_argument("--cost", type=float, default=0.30)
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    print(f"\n=== PMax  {os.path.basename(a.csv)} {a.tf}  {d.index[0].date()}->{d.index[-1].date()}  "
          f"MA={a.mav}{a.length} ATR{a.periods}x{a.mult}  cost={a.cost}usd ===")

    print(f"\n  -- 1. BETA CHECK: PMax default (mult={a.mult}) long/short split --")
    t = pmax_trades(d, a.length, a.mult, a.periods, a.mav, a.cost)
    stat(t, "PMax all"); stat(t[t.dir == 1], "PMax LONG"); stat(t[t.dir == -1], "PMax SHORT")

    print(f"\n  -- mult sweep (plateau?) --")
    for m in (2, 3, 4, 5):
        stat(pmax_trades(d, a.length, m, a.periods, a.mav, a.cost), f"PMax mult={m}")

    print(f"\n  -- 2. HEAD-TO-HEAD vs UT Bot (same data/TF/cost, R both stop-normalized) --")
    for m in (2, 3, 4):
        stat(pmax_trades(d, a.length, m, a.periods, a.mav, a.cost), f"PMax mult={m}")
    for aa in (2, 3, 4):
        ut = ut_trades(d, aa, 10, a.cost)
        stat(ut, f"UTBot a={aa}")
        stat(ut[ut.dir == 1], f"UTBot a={aa} LONG")


if __name__ == "__main__":
    main()
