"""jpy_fade_exits.py -- EXIT study for the USDJPY BB+RSI fade on low TF.

Entry fixed (BB21/2 + RSI14 fade). Vary only the EXIT (ATR1.5 stop is the loss cap in all):
  oppband : ride to the OPPOSITE band (current stop-and-reverse proxy)
  mean    : TP at the BB basis (classic mean-reversion exit)
  tpN     : TP at entry +/- N pips (fixed small target)
  timeN   : time stop -- exit at close after N bars
One position at a time (no-overlap), next-bar-open fill. Reports net-of-spread PF so we see
which exit makes the thin gross edge actually beat cost on a low TF.

  .venv/bin/python research/jpy_fade_exits.py --csv data/vantage_usdjpy_m1.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

PIP = 0.01
LEN, MULT, RSL, OSL, OBL, SL_ATR, MAXH = 21, 2.0, 14, 30, 70, 1.5, 720


def run_exit(d, mode, spread):
    c = d["close"]; cv = c.values; op = d["open"].values; hi = d["high"].values; lo = d["low"].values
    basis = c.rolling(LEN).mean().values
    dev = MULT * c.rolling(LEN).std(ddof=0).values
    upper, lower = basis + dev, basis - dev
    rsi = ta.rsi(c, RSL).values
    atr = ta.atr(d["high"], d["low"], c, length=14).values
    n = len(cv)
    xo = (cv > lower) & np.r_[False, cv[:-1] <= lower[:-1]]
    xu = (cv < upper) & np.r_[False, cv[:-1] >= upper[:-1]]
    buy = xo & (rsi <= OSL); sell = xu & (rsi >= OBL)
    buy[:LEN] = sell[:LEN] = False

    tp_pips = int(mode[2:]) if mode.startswith("tp") else 0
    t_bars = int(mode[4:]) if mode.startswith("time") else 0
    g = []; busy = -1
    for i in range(n - 1):
        if not (buy[i] or sell[i]) or i + 1 <= busy:
            continue
        dr = 1 if buy[i] else -1
        e_px = op[i + 1]; ei = i + 1
        if np.isnan(atr[ei]):
            continue
        stop = e_px - dr * SL_ATR * atr[ei]
        tp = (e_px + dr * tp_pips * PIP) if tp_pips else np.nan
        ex_px = None
        for j in range(ei + 1, min(ei + 1 + MAXH, n)):
            if dr > 0 and lo[j] <= stop: ex_px = stop; break
            if dr < 0 and hi[j] >= stop: ex_px = stop; break
            if mode == "oppband":
                if dr > 0 and hi[j] >= upper[j]: ex_px = upper[j]; break
                if dr < 0 and lo[j] <= lower[j]: ex_px = lower[j]; break
            elif mode == "mean":
                if dr > 0 and hi[j] >= basis[j]: ex_px = basis[j]; break
                if dr < 0 and lo[j] <= basis[j]: ex_px = basis[j]; break
            elif tp_pips:
                if dr > 0 and hi[j] >= tp: ex_px = tp; break
                if dr < 0 and lo[j] <= tp: ex_px = tp; break
            elif t_bars and (j - ei) >= t_bars:
                ex_px = cv[j]; break
            busy = j
        if ex_px is None:
            ex_px = cv[min(ei + MAXH, n - 1)]
            busy = min(ei + MAXH, n - 1)
        else:
            busy = j
        g.append((ex_px - e_px) / PIP * dr - spread)
    g = np.array(g)
    if len(g) == 0:
        return None
    w = g[g > 0]; l = g[g < 0]
    pf = w.sum() / abs(l.sum()) if l.sum() else float("inf")
    return dict(n=len(g), win=(g > 0).mean() * 100, pf=pf,
                aw=w.mean() if len(w) else 0, al=l.mean() if len(l) else 0, net=g.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_usdjpy_m1.csv")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--spread", type=float, default=0.5)
    a = ap.parse_args()
    d = load_mt5_csv(a.csv).loc[a.start:]
    print(f"\n=== USDJPY fade EXIT study  {os.path.basename(a.csv)}  {d.index[0].date()}->{d.index[-1].date()}  "
          f"spread={a.spread}p ===")
    print(f"  {'exit':<9} {'n':>6} {'win%':>5} {'PF':>5} {'avgW':>6} {'avgL':>6} {'net(p)':>8}")
    for mode in ["oppband", "mean", "tp3", "tp5", "tp8", "tp15", "time30", "time120"]:
        m = run_exit(d, mode, a.spread)
        if m:
            print(f"  {mode:<9} {m['n']:>6} {m['win']:>5.0f} {m['pf']:>5.2f} {m['aw']:>+6.1f} "
                  f"{m['al']:>+6.1f} {m['net']:>+8.0f}")


if __name__ == "__main__":
    main()
