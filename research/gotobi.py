"""Idea 5 — Gotobi (五十日) forced-flow seasonality on USDJPY.

Structural counterparty: on 5/10/15/20/25/month-end ("gotobi") dates, Japanese
importers settle invoices and BUY USD into the 9:55 JST Tokyo fix (仲値). This is
MANDATE-driven, price-insensitive flow → USDJPY tends to drift UP through Tokyo
morning into the fix, and can revert after. We TEST it, not assume it.

Clock: data is broker-server time (UTC+2/+3). Volume profile puts the Tokyo fix
at ~server hour 3 (9:55 JST = ~02:55-03:55 server). So:
  PRE-FIX  window = server [0,4)  -- long the Tokyo-AM drift into the fix
  POST-FIX window = server [4,8)  -- the reversion leg
For a server date D, hours 0-4 map to JST 06-11 same date, so server-date == Tokyo
date for this window (tagging gotobi by server-date day-of-month is correct here).

Pre-registered, falsification-first:
  - ALL-SIGNALS base: every gotobi date, no selection. Compare to non-gotobi.
  - meanR in PIPS vs a realistic USDJPY round-trip spread (~1.2 pip) -- sub-cost = dead.
  - PERMUTATION p: shuffle the gotobi labels across dates (keeps count) -> is the
    gotobi mean distinguishable from a random same-size set of dates?
  - PER-YEAR: is it one-era (pre-2015, the famous decay) or persistent?
  - PLATEAU: sweep the window edges; a real flow effect is smooth, not a knife-edge.

Run:  .venv/bin/python research/gotobi.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_mt5_csv  # noqa: E402

RNG = np.random.default_rng(20260620)
PIP = 0.01  # USDJPY


def window_returns(d: pd.DataFrame, h0: int, h1: int) -> pd.DataFrame:
    """Per server-date: long return (pips) over server-hours [h0,h1):
    open of first bar with hour>=h0 to close of last bar with hour<h1."""
    sub = d[(d.index.hour >= h0) & (d.index.hour < h1)].copy()
    sub["date"] = sub.index.normalize()
    g = sub.groupby("date")
    op = g["open"].first()
    cl = g["close"].last()
    ret = (cl - op) / PIP
    return pd.DataFrame({"ret_pips": ret}).dropna()


def gotobi_dates(dates: pd.DatetimeIndex, weekend_adjust: bool) -> pd.Series:
    """Boolean Series over `dates`: is each date a gotobi settlement date?
    Targets: dom in {5,10,15,20,25} and the last calendar day of the month.
    weekend_adjust: if the target lands on Sat/Sun, settlement shifts to the
    preceding business day (the realistic convention; JP bank holidays ignored)."""
    dates = pd.DatetimeIndex(dates)
    avail = set(dates.normalize())
    flags = pd.Series(False, index=dates)
    targets = []
    for d in pd.date_range(dates.min(), dates.max(), freq="D"):
        dom = d.day
        last = (d + pd.offsets.MonthEnd(0)).day
        if dom in (5, 10, 15, 20, 25) or dom == last:
            t = d
            if weekend_adjust:
                while t.weekday() >= 5:           # Sat=5,Sun=6 -> step back
                    t -= pd.Timedelta(days=1)
            targets.append(t.normalize())
    targets = set(targets) & avail
    return pd.Series(dates.normalize().isin(targets), index=dates)


def perm_p(ret: pd.Series, is_g: pd.Series, n=5000):
    obs = ret[is_g].mean()
    k = int(is_g.sum())
    arr = ret.values
    idx = np.arange(len(arr))
    null = np.array([arr[RNG.choice(idx, k, replace=False)].mean() for _ in range(n)])
    return obs, (null >= obs).mean(), null.mean()


def report(tag, ret: pd.Series, is_g: pd.Series, cost=1.2):
    g, ng = ret[is_g], ret[~is_g]
    obs, p, nullm = perm_p(ret, is_g)
    print(f"  [{tag}]  gotobi n={len(g)}  mean={g.mean():+.2f}pip  win(long)={(g>0).mean()*100:.0f}%"
          f"   non-gotobi mean={ng.mean():+.2f}pip   diff={g.mean()-ng.mean():+.2f}")
    print(f"        net of {cost}pip round-trip: {g.mean()-cost:+.2f}pip   "
          f"perm-p(vs random dates)={p:.3f}  (null mean {nullm:+.2f})")
    by = g.groupby(g.index.year).mean()
    pos = (by > 0).sum()
    print("        per-year mean(pip): " + " ".join(f"{y}:{v:+.1f}" for y, v in by.items())
          + f"   [{pos}/{len(by)} yrs +]")


def main():
    d = load_mt5_csv("data/vantage_usdjpy_h1.csv")
    d = d[d.index.year >= 2011]
    print(f"USDJPY H1  {d.index.min().date()} -> {d.index.max().date()}  ({len(d)} bars)\n")

    print("=" * 78)
    print("PRE-FIX long drift, server[0,4)  (Tokyo AM into the 9:55 fix)")
    print("=" * 78)
    pre = window_returns(d, 0, 4)["ret_pips"]
    for wa in (False, True):
        is_g = gotobi_dates(pre.index, weekend_adjust=wa)
        report(f"gotobi {'weekend-adj' if wa else 'exact-date'}", pre, is_g)

    print("\n" + "=" * 78)
    print("Window PLATEAU sweep (weekend-adjusted gotobi, long drift)")
    print("=" * 78)
    for (h0, h1) in [(0, 3), (0, 4), (0, 5), (22, 4), (1, 4)]:
        r = window_returns(d, h0, h1)["ret_pips"] if h0 < h1 else None
        if r is None:   # wrap window 22->4 handled separately
            sub = d[(d.index.hour >= 22) | (d.index.hour < 4)].copy()
            sub["date"] = (sub.index + pd.Timedelta(hours=4)).normalize()  # group by the morning date
            g = sub.groupby("date")
            r = ((g["close"].last() - g["open"].first()) / PIP).dropna()
        is_g = gotobi_dates(r.index, weekend_adjust=True)
        g = r[is_g]
        diff = g.mean() - r[~is_g].mean()
        print(f"  server[{h0:2d},{h1}) n={len(g):4d}  gotobi mean={g.mean():+.2f}pip  "
              f"diff vs non={diff:+.2f}  net-1.2={g.mean()-1.2:+.2f}")

    print("\n" + "=" * 78)
    print("POST-FIX reversion, server[4,8)  (does the demand unwind / fade short?)")
    print("=" * 78)
    post = window_returns(d, 4, 8)["ret_pips"]
    is_g = gotobi_dates(post.index, weekend_adjust=True)
    report("gotobi post-fix (long sign)", post, is_g)
    print("  (negative gotobi mean here => a SHORT reversion edge after the fix)")


if __name__ == "__main__":
    main()
