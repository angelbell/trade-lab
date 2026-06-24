"""jpy_fade_dig.py -- the gross USDJPY M1 BB+RSI fade has PF 1.15 (a real but thin edge).
Don't dismiss it as "sub-spread"; DIG: where does the gross edge CONCENTRATE? If a subset is
fat enough to beat spread AND holds IS/OOS, it's tradeable. If the edge is uniformly thin, dead.

Records per trade: entry hour (broker tz), deviation% from BB basis, year, GROSS pips (no cost).
Then gross PF by hour / dev-quartile / year, and -- for the fattest subset -- net-of-spread + IS/OOS.

  .venv/bin/python research/jpy_fade_dig.py
"""
import os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

PIP = 0.01
LEN, MULT, RSL, OS, OB = 21, 2.0, 14, 30, 70
SL_ATR = 1.5


def simulate(d):
    c = d["close"]; cv = c.values; op = d["open"].values; hi = d["high"].values; lo = d["low"].values
    basis = c.rolling(LEN).mean().values
    dev = MULT * c.rolling(LEN).std(ddof=0).values
    upper, lower = basis + dev, basis - dev
    rsi = ta.rsi(c, RSL).values
    atr = ta.atr(d["high"], d["low"], c, length=14).values
    devpct = np.abs(cv - basis) / basis * 100.0
    hour = d.index.hour.values; year = d.index.year.values
    n = len(cv)
    # vectorized crossover/under
    xo = (cv > lower) & (np.r_[False, cv[:-1] <= lower[:-1]])
    xu = (cv < upper) & (np.r_[False, cv[:-1] >= upper[:-1]])
    buy = xo & (rsi <= OS); sell = xu & (rsi >= OB)
    buy[:LEN] = sell[:LEN] = False

    rows = []
    pos = 0; e_px = stop = 0.0; e_i = 0
    for i in range(n - 1):
        if pos != 0:                                  # intrabar stop
            if (pos > 0 and lo[i] <= stop) or (pos < 0 and hi[i] >= stop):
                g = (stop - e_px) / PIP if pos > 0 else (e_px - stop) / PIP
                rows.append((hour[e_i], devpct[e_i], year[e_i], g)); pos = 0
        want = 1 if buy[i] else (-1 if sell[i] else 0)
        if want == 0 or want == pos:
            continue
        fill = op[i + 1]
        if pos != 0:
            g = (fill - e_px) / PIP if pos > 0 else (e_px - fill) / PIP
            rows.append((hour[e_i], devpct[e_i], year[e_i], g))
        sd = SL_ATR * atr[i + 1] if not np.isnan(atr[i + 1]) else 0.0
        pos, e_px, e_i = want, fill, i + 1
        stop = (fill - sd if want > 0 else fill + sd) if sd > 0 else np.nan
    return pd.DataFrame(rows, columns=["hour", "devpct", "year", "g"])


def pf(g):
    w = g[g > 0].sum(); l = g[g < 0].sum()
    return w / abs(l) if l != 0 else float("inf")


def net_pf(g, spread):
    gn = g - spread
    w = gn[gn > 0].sum(); l = gn[gn < 0].sum()
    return w / abs(l) if l != 0 else float("inf")


def main():
    d = load_mt5_csv("data/vantage_usdjpy_m1.csv")
    t = simulate(d)
    print(f"\n=== USDJPY M1 fade gross-edge DIG  ({len(t):,} trades, {d.index[0].date()}->{d.index[-1].date()}) ===")
    print(f"  ALL gross PF={pf(t.g):.2f}  net@0.5sp PF={net_pf(t.g,0.5):.2f}  n={len(t)}  win={(t.g>0).mean()*100:.0f}%")

    print("\n  -- gross PF by ENTRY HOUR (broker tz) --")
    for h, gp in t.groupby("hour"):
        if len(gp) > 100:
            print(f"    h{h:>2}: n={len(gp):>4} grossPF={pf(gp.g):.2f}  net@0.5={net_pf(gp.g,0.5):.2f}  meanG={gp.g.mean():+.2f}")

    print("\n  -- gross PF by DEVIATION quartile --")
    t["dq"] = pd.qcut(t.devpct, 4, labels=["Q1lo", "Q2", "Q3", "Q4hi"])
    for q, gp in t.groupby("dq", observed=True):
        print(f"    {q:>5}: n={len(gp):>5} grossPF={pf(gp.g):.2f}  net@0.5={net_pf(gp.g,0.5):.2f}")

    print("\n  -- gross PF by YEAR --")
    print("    " + "  ".join(f"{y}:{pf(gp.g):.2f}" for y, gp in t.groupby("year")))

    # ---- IS/OOS HOUR validation (pick good hours on IS, test on OOS) ----
    is_t, oos_t = t[t.year < 2022], t[t.year >= 2022]
    print("\n  -- per-hour net@0.5 PF: IS(2018-21) vs OOS(2022-26) --")
    good = []
    for h in range(24):
        gi = is_t[is_t.hour == h].g; go = oos_t[oos_t.hour == h].g
        if len(gi) < 50:
            continue
        pi, po = net_pf(gi, 0.5), net_pf(go, 0.5)
        mark = "  <-IS-good" if pi >= 1.10 else ""
        if pi >= 1.10:
            good.append(h)
        print(f"    h{h:>2}: IS n={len(gi):>4} netPF={pi:.2f}   OOS n={len(go):>4} netPF={po:.2f}{mark}")
    print(f"\n  IS-selected hours {good}")
    sub = oos_t[oos_t.hour.isin(good)]
    allnet = net_pf(oos_t.g, 0.5)
    print(f"  >>> OOS, IS-selected hours: n={len(sub)} net@0.5 PF={net_pf(sub.g,0.5):.2f}  (OOS all-hours PF={allnet:.2f})")
    sub_lo = sub[sub.devpct <= sub.devpct.median()]
    print(f"  >>> OOS, IS-hours + low-deviation half: n={len(sub_lo)} net@0.5 PF={net_pf(sub_lo.g,0.5):.2f}")


if __name__ == "__main__":
    main()
