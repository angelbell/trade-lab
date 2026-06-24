"""jpy_fade_confirm.py -- pre-registered: does a REVERSAL-CANDLE confirmation rescue the JPY fade?

Baseline: fade fires on the band-cross+RSI bar, enter next open.
Confirm : require the NEXT bar to close in the reversal direction (band touch + reversal candle),
          enter the bar AFTER that (i+2 open). Filters fake touches; costs 1-2 bars of entry price.
oppband exit + 1.5ATR stop (the established setup). Judge net@ REAL spreads (0.5/1.0/1.9) + IS/OOS.
PASS = confirmed subset net@1.0 PF >= 1.1 and IS/OOS stable; else reject.

  .venv/bin/python research/jpy_fade_confirm.py --csv data/vantage_usdjpy_m5.csv --start 2018-06-01
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

PIP = 0.01
LEN, MULT, RSL, OSL, OBL, SL_ATR, MAXH = 21, 2.0, 14, 30, 70, 1.5, 400


def sim(d, confirm):
    c = d["close"]; cv = c.values; op = d["open"].values; hi = d["high"].values; lo = d["low"].values
    basis = c.rolling(LEN).mean().values
    dev = MULT * c.rolling(LEN).std(ddof=0).values
    upper, lower = basis + dev, basis - dev
    rsi = ta.rsi(c, RSL).values
    atr = ta.atr(d["high"], d["low"], c, length=14).values
    yr = d.index.year.values
    n = len(cv)
    xo = (cv > lower) & np.r_[False, cv[:-1] <= lower[:-1]]
    xu = (cv < upper) & np.r_[False, cv[:-1] >= upper[:-1]]
    buy = xo & (rsi <= OSL); sell = xu & (rsi >= OBL)
    buy[:LEN] = sell[:LEN] = False

    rows = []; busy = -1
    for i in range(n - 2):
        dr = 0
        if confirm:
            if buy[i] and cv[i + 1] > cv[i]:   dr = 1
            elif sell[i] and cv[i + 1] < cv[i]: dr = -1
            ei = i + 2                          # enter after the confirmation bar
        else:
            if buy[i]:   dr = 1
            elif sell[i]: dr = -1
            ei = i + 1
        if dr == 0 or ei <= busy or ei >= n or np.isnan(atr[ei]):
            continue
        e_px = op[ei]; stop = e_px - dr * SL_ATR * atr[ei]; ex = None
        for j in range(ei + 1, min(ei + 1 + MAXH, n)):
            if dr > 0 and lo[j] <= stop: ex = stop; break
            if dr < 0 and hi[j] >= stop: ex = stop; break
            if dr > 0 and hi[j] >= upper[j]: ex = upper[j]; break
            if dr < 0 and lo[j] <= lower[j]: ex = lower[j]; break
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


def report(tag, t):
    isr, oos = t[t.y < 2022].g, t[t.y >= 2022].g
    print(f"  {tag:<12} n={len(t):>5} win={(t.g>0).mean()*100:>3.0f}% gross={pf(t.g):.2f}  "
          f"net@0.5={pf(t.g,0.5):.2f} net@1.0={pf(t.g,1.0):.2f} net@1.9={pf(t.g,1.9):.2f}  "
          f"| IS@1.0={pf(isr,1.0):.2f} OOS@1.0={pf(oos,1.0):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_usdjpy_m5.csv")
    ap.add_argument("--start", default="2018-06-01")
    a = ap.parse_args()
    d = load_mt5_csv(a.csv).loc[a.start:]
    print(f"\n=== reversal-candle confirm test  {os.path.basename(a.csv)}  {d.index[0].date()}->{d.index[-1].date()} ===")
    report("baseline", sim(d, False))
    report("confirm", sim(d, True))


if __name__ == "__main__":
    main()
