"""fx_session_breakout.py -- Asian-range -> London/NY session breakout on USDJPY (the cost-aware low-TF FX play).

The session screen showed clear VOLATILITY structure (vol doubles into the London/NY window, hr9-17 broker
time) but weak directional structure. The cost-aware way to use that: trade the BREAKOUT of the quiet Asian
range during the high-vol session, where moves are big enough to clear spread.

Rule (confirmed-close, the lab's law): per day, range = [min low, max high] over the Asian window. During
the breakout window, the FIRST bar that CLOSES beyond the range -> enter (long above / short below). Stop =
k*ATR, target = RR*risk, intrabar SL/TP, one trade/day, exit by end of day.

Falsifier (up front): PASS = cost-after meanR>0 AND IS~OOS AND it BEATS the same breakout in a RANDOM/quiet
window (proves the SESSION matters, not just 'breakouts work') AND the window choice plateaus (+-hours). A
quiet-window control that does as well => it's not a session edge. In-sample; live-forward arbitrates.
  .venv/bin/python research/fx_session_breakout.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

SPLIT = 2018


def atr(d, n=14):
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def run(d, asia=(0, 8), bo=(9, 17), eod=22, k=1.0, rr=2.0, cost=0.0003, fade=False):
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = atr(d).values
    hr = d.index.hour.values
    day = d.index.normalize()
    rows = []
    # per-day asian range
    in_asia = (hr >= asia[0]) & (hr < asia[1])
    dfA = pd.DataFrame({"day": day, "h": h, "l": l, "in": in_asia})
    aH = dfA[dfA["in"]].groupby("day")["h"].max()
    aL = dfA[dfA["in"]].groupby("day")["l"].min()
    in_bo = (hr >= bo[0]) & (hr < bo[1])
    traded = set()
    i = 0
    n = len(d)
    while i < n - 1:
        dd = day[i]
        if in_bo[i] and dd not in traded and dd in aH.index and np.isfinite(a[i]) and a[i] > 0:
            rh, rl = aH[dd], aL[dd]
            brokeUp = c[i] > rh
            brokeDn = c[i] < rl
            isL = brokeDn if fade else brokeUp           # fade: break DOWN -> go long (false-break fade)
            isS = brokeUp if fade else brokeDn
            if isL or isS:
                e = c[i]                                   # confirmed-close entry
                stop = e - k * a[i] if isL else e + k * a[i]
                risk = abs(e - stop)
                tgt = e + rr * risk if isL else e - rr * risk
                R = None
                j = i + 1
                while j < n and not (day[j] != dd and hr[j] >= eod) and (day[j] == dd or hr[j] < eod):
                    if (l[j] <= stop) if isL else (h[j] >= stop):
                        R = -1; break
                    if (h[j] >= tgt) if isL else (l[j] <= tgt):
                        R = rr; break
                    if day[j] != dd and hr[j] >= asia[1]:  # safety: next day's session reached
                        break
                    j += 1
                if R is None:
                    jj = min(j, n - 1)
                    R = ((c[jj] - e) if isL else (e - c[jj])) / risk
                R -= cost * e / risk
                rows.append((d.index[i], "L" if isL else "S", R))
                traded.add(dd)
        i += 1
    return pd.DataFrame(rows, columns=["time", "side", "R"])


def line(tag, t, rr=2.0):
    if len(t) < 20:
        print(f"  {tag:<26} n={len(t)} (too few)"); return
    be = 100 / (1 + rr)
    w, ll = t[t.R > 0].R.sum(), -t[t.R < 0].R.sum()
    is_ = t[t.time.dt.year < SPLIT].R.mean(); oos = t[t.time.dt.year >= SPLIT].R.mean()
    print(f"  {tag:<26} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f}(BE{be:.0f}) meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.1f} PF={w/max(ll,1e-9):4.2f} | IS={is_:+.2f} OOS={oos:+.2f}")


def main():
    d = load_mt5_csv("data/vantage_usdjpy_h1.csv")
    rr = 2.0
    print("== USDJPY Asian-range -> session breakout (1h, confirmed close, ATR stop, RR2, cost0.03%) ==")
    print(" SESSION windows (asia=0-8):")
    line("London 9-13", run(d, bo=(9, 13), rr=rr), rr)
    line("NY 14-18", run(d, bo=(14, 18), rr=rr), rr)
    line("London+NY 9-17", run(d, bo=(9, 17), rr=rr), rr)
    print(" CONTROL (breakout in QUIET hours -- if ~equal, session doesn't matter):")
    line("quiet 19-24", run(d, bo=(19, 24), eod=8, rr=rr), rr)
    line("late-asia 2-7", run(d, bo=(2, 7), eod=8, rr=rr), rr)

    print("\n == best window: per-year, plateau, cost ==")
    base = run(d, bo=(9, 17), rr=rr)
    by = base.groupby(base.time.dt.year).R.sum()
    print("  per-year: " + " ".join(f"{y}:{v:+.0f}" for y, v in by.items()) + f"  (green {int((by>0).sum())}/{len(by)})")
    print(" window-start shift (plateau?):")
    for s in (8, 9, 10, 11):
        line(f"  bo {s}-17", run(d, bo=(s, 17), rr=rr), rr)
    print(" RR sweep:")
    for r in (1.5, 2.0, 2.5, 3.0):
        line(f"  RR{r}", run(d, bo=(9, 17), rr=r), r)
    print(" cost stress (bo 9-17):")
    for cc in (0.0003, 0.0006, 0.001):
        line(f"  cost={cc*100:.2f}%", run(d, bo=(9, 17), rr=rr, cost=cc), rr)
    print("\n  verdict: session windows must beat the QUIET control AND meanR>0 after cost AND IS~OOS AND plateau.")


if __name__ == "__main__":
    main()
