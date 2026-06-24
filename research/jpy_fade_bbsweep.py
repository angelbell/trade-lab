"""jpy_fade_bbsweep.py -- does sweeping BB length/mult rescue the USDJPY fade? (oppband exit)
Judge on net@0.5 PF + IS/OOS + plateau. Expectation: gross moves a bit, net stays <1 (sub-spread).

  .venv/bin/python research/jpy_fade_bbsweep.py --csv data/vantage_usdjpy_m5.csv --start 2018-06-01
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

PIP = 0.01
RSL, OSL, OBL, SL_ATR, MAXH = 14, 30, 70, 1.5, 400


def sim(d, length, mult):
    c = d["close"]; cv = c.values; op = d["open"].values; hi = d["high"].values; lo = d["low"].values
    basis = c.rolling(length).mean().values
    dev = mult * c.rolling(length).std(ddof=0).values
    upper, lower = basis + dev, basis - dev
    rsi = ta.rsi(c, RSL).values
    atr = ta.atr(d["high"], d["low"], c, length=14).values
    yr = d.index.year.values
    n = len(cv)
    xo = (cv > lower) & np.r_[False, cv[:-1] <= lower[:-1]]
    xu = (cv < upper) & np.r_[False, cv[:-1] >= upper[:-1]]
    buy = xo & (rsi <= OSL); sell = xu & (rsi >= OBL)
    buy[:length] = sell[:length] = False
    rows = []; busy = -1
    for i in range(n - 1):
        if not (buy[i] or sell[i]) or i + 1 <= busy:
            continue
        dr = 1 if buy[i] else -1
        e_px = op[i + 1]; ei = i + 1
        if np.isnan(atr[ei]):
            continue
        stop = e_px - dr * SL_ATR * atr[ei]; ex = None
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
    print(f"  {tag:<14} n={len(t):>5} grossPF={pf(t.g):.2f}  net@0.5={pf(t.g,0.5):.2f}  "
          f"IS={pf(isr,0.5):.2f} OOS={pf(oos,0.5):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_usdjpy_m5.csv")
    ap.add_argument("--start", default="2018-06-01")
    a = ap.parse_args()
    d = load_mt5_csv(a.csv).loc[a.start:]
    print(f"\n=== BB-param sweep  {os.path.basename(a.csv)}  {d.index[0].date()}->{d.index[-1].date()} (oppband exit) ===")
    print("  -- length sweep (mult 2.0) --")
    for L in (10, 14, 21, 34, 50):
        report(f"len{L}/m2.0", sim(d, L, 2.0))
    print("  -- mult sweep (len 21) --")
    for M in (1.5, 2.0, 2.5, 3.0):
        report(f"len21/m{M}", sim(d, 21, M))


if __name__ == "__main__":
    main()
