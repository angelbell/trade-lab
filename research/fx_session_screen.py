"""fx_session_screen.py -- CHEAP screen: does USDJPY have exploitable TIME-OF-DAY / session structure?

FX is session-driven (Tokyo/London/NY). Lower-TF FX only beats cost where a session concentrates a LARGE
move (e.g. London open). Before building any session strategy, screen whether hour-of-day / day-of-week
carries volatility AND directional structure at all. Broker-server time is the clock (HTF bins align).

Falsifier (up front): structure must be (a) clear in the VOLATILITY profile (some hours much more active --
identifies London/NY) AND (b) show a directional/continuation tendency in those windows that a fixed-cost
trade could exploit. Flat profile or only-vol-no-direction => no session edge to build on. Descriptive
screen only; in-sample.
  .venv/bin/python research/fx_session_screen.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv


def screen(name, d):
    c = d["close"]
    ret = np.log(c / c.shift(1))
    rng = (d["high"] - d["low"]) / c                      # bar range (vol proxy), fraction
    hh = d.index.hour
    print(f"\n== {name}  (broker-server hour; {d.index.min().date()}..{d.index.max().date()}) ==")
    print(f"  {'hr':>3} {'n':>6} {'meanRet(bps)':>13} {'|ret|(bps)':>11} {'range(bps)':>11} {'up%':>6}")
    g = pd.DataFrame({"h": hh, "ret": ret.values, "rng": rng.values}).dropna()
    for h in range(24):
        s = g[g.h == h]
        if len(s) < 50:
            continue
        print(f"  {h:>3} {len(s):>6} {s.ret.mean()*1e4:>+13.2f} {s.ret.abs().mean()*1e4:>11.1f} "
              f"{s.rng.mean()*1e4:>11.1f} {(s.ret>0).mean()*100:>5.1f}%")
    # day-of-week directional + range
    print(f"\n  day-of-week:  {'dow':>4} {'n':>6} {'meanRet(bps)':>13} {'|ret|(bps)':>11} {'up%':>6}")
    dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    gd = pd.DataFrame({"d": d.index.dayofweek, "ret": ret.values}).dropna()
    daily = np.log(c.resample("1D").last()).diff()        # day-level direction
    dd = pd.DataFrame({"d": daily.index.dayofweek, "r": daily.values}).dropna()
    for k in range(7):
        s = dd[dd.d == k]
        if len(s) < 20:
            continue
        print(f"               {dn[k]:>4} {len(s):>6} {s.r.mean()*1e4:>+13.2f} {s.r.abs().mean()*1e4:>11.1f} "
              f"{(s.r>0).mean()*100:>5.1f}%")


def main():
    for name, csv in [("USDJPY 1h", "data/vantage_usdjpy_h1.csv"),
                      ("USDX 1h", "data/vantage_usdx.r_h1.csv")]:
        screen(name, load_mt5_csv(csv))
    print("\n  read: a VOL hump at certain hours = the active session (London/NY). A directional/up% skew")
    print("        there, or a day-of-week skew, is the raw material for a session/calendar strategy.")


if __name__ == "__main__":
    main()
