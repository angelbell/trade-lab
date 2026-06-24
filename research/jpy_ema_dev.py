"""jpy_ema_dev.py -- pure EMA-deviation (乖離率) fade: long when close is thr% BELOW EMA, short
when thr% ABOVE; exit on revert to EMA (mean) or 1.5ATR stop. Sweep EMA len x dev threshold,
report net @ real spreads + IS/OOS. (Same 'distance-from-MA' axis as BB-dev; EMA-skinned.)

  .venv/bin/python research/jpy_ema_dev.py --csv data/vantage_usdjpy_m5.csv --start 2018-06-01
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

PIP = 0.01
SL_ATR, MAXH = 1.5, 400


def sim(d, elen, thr):
    c = d["close"]; cv = c.values; op = d["open"].values; hi = d["high"].values; lo = d["low"].values
    ema = c.ewm(span=elen, adjust=False).mean().values
    atr = ta.atr(d["high"], d["low"], c, length=14).values
    yr = d.index.year.values
    n = len(cv)
    longsig = cv < ema * (1 - thr)
    shortsig = cv > ema * (1 + thr)
    rows = []; busy = -1
    for i in range(n - 1):
        if i + 1 <= busy or np.isnan(atr[i + 1]):
            continue
        dr = 1 if longsig[i] else (-1 if shortsig[i] else 0)
        if dr == 0:
            continue
        ei = i + 1; e_px = op[ei]; stop = e_px - dr * SL_ATR * atr[ei]; ex = None
        for j in range(ei + 1, min(ei + 1 + MAXH, n)):
            if dr > 0 and lo[j] <= stop: ex = stop; break
            if dr < 0 and hi[j] >= stop: ex = stop; break
            if dr > 0 and hi[j] >= ema[j]: ex = ema[j]; break     # revert to EMA = TP
            if dr < 0 and lo[j] <= ema[j]: ex = ema[j]; break
            busy = j
        if ex is None:
            ex = cv[min(ei + MAXH, n - 1)]; busy = min(ei + MAXH, n - 1)
        else:
            busy = j
        rows.append((yr[ei], (ex - e_px) / PIP * dr))
    return pd.DataFrame(rows, columns=["y", "g"])


def pf(g, sp=0.0):
    gn = g - sp; w = gn[gn > 0].sum(); l = gn[gn < 0].sum()
    return w / abs(l) if l else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_usdjpy_m5.csv")
    ap.add_argument("--start", default="2018-06-01")
    a = ap.parse_args()
    d = load_mt5_csv(a.csv).loc[a.start:]
    print(f"\n=== EMA-deviation fade  {os.path.basename(a.csv)}  {d.index[0].date()}->{d.index[-1].date()} ===")
    print(f"  {'cfg':<16} {'n':>6} {'win':>4} {'gross':>5} {'net@0.5':>7} {'net@1.0':>7} {'net@1.9':>7} {'IS@1.0':>6} {'OOS@1.0':>7}")
    for elen in (20, 50):
        for thr in (0.001, 0.002, 0.003):
            t = sim(d, elen, thr)
            if len(t) < 50:
                print(f"  ema{elen}/{thr*100:.1f}%   (too few n={len(t)})"); continue
            isr, oos = t[t.y < 2022].g, t[t.y >= 2022].g
            print(f"  ema{elen}/dev{thr*100:.1f}%  {len(t):>6} {(t.g>0).mean()*100:>3.0f}% {pf(t.g):>5.2f} "
                  f"{pf(t.g,0.5):>7.2f} {pf(t.g,1.0):>7.2f} {pf(t.g,1.9):>7.2f} {pf(isr,1.0):>6.2f} {pf(oos,1.0):>7.2f}")


if __name__ == "__main__":
    main()
