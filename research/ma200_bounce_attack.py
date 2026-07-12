"""ma200_bounce_attack.py -- EMA vs SMA for the 200MA, and ATTACK-COUNT
(does the Nth touch bounce differently?) on the touch-and-hold 200MA bounce.

Two questions the user raised:
  1. Is the 200MA EMA or SMA -- which gives the better bounce edge?
  2. Attack count: how many times has price tested the MA since it last escaped
     clearly above it? Bin trades by attack# (1st / 2nd / 3rd+) and read the
     bounce probability (win% / meanR) -- and whether selecting a good bin fixes
     the lumpiness (the standalone CAGR/DD=0.09 problem), not just meanR.

The bin selection is itself a filter, so the good bin must beat the RANDOM-DROP
NULL on CAGR/DD (n-trimming can lift meanR for free; it cannot fix lumpiness).

Exit = fractal structural target (戻り高値), long-only, +V-avoid approach filter.
Run:  .venv/bin/python research/ma200_bounce_attack.py
"""
import os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.ma200_bounce import resample, stats
from research.ma200_bounce_fractal import fractals, simulate
from research.portfolio_kama import cagr_dd

RNG = np.random.default_rng(11)


def find(d, ma_type="ema", emalen=200, slopeK=20, tol=0.25, atrlen=14,
         swingW=30, pivp=3, reset=1.5):
    c = d["close"]
    ma = (c.rolling(emalen).mean() if ma_type == "sma"
          else c.ewm(span=emalen, adjust=False).mean()).values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, cl = d["open"].values, d["high"].values, d["low"].values, c.values
    _, lows = fractals(h, l, pivp)
    out, cnt = [], 0
    for s in range(max(slopeK, swingW) + 1, len(cl) - 1):
        if np.isnan(ma[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        if cl[s] > ma[s] + reset * a[s]:      # escaped clearly above -> reset attack count
            cnt = 0
        if not (ma[s] > ma[s - slopeK] and l[s] <= ma[s] + tol * a[s]
                and cl[s] > ma[s] and cl[s] > o[s]):
            continue
        cnt += 1                               # this is attack #cnt since last escape
        e = o[s + 1]; stop = l[s]
        if e - stop < 0.5 * a[s]:
            stop = e - 0.5 * a[s]
        if e <= stop:
            continue
        win = slice(s - swingW + 1, s + 1)
        sh_rel = int(np.argmax(h[win])); swing_hi = h[win][sh_rel]
        bars_down = swingW - 1 - sh_rel
        vel = ((swing_hi - l[s]) / a[s]) / max(bars_down, 1)
        out.append(dict(i=s + 1, e=e, stop=stop, vel=vel, bars_down=bars_down,
                        target=swing_hi, attack=cnt))
    return out, lows


def cdd_of(t):
    if len(t) < 5:
        return np.nan
    return cagr_dd(t)[2]


def drop_null_cdd(base_t, k, real_cdd, n_iter=1500):
    if len(base_t) <= k or k < 5:
        return np.nan
    R = base_t.sort_values("time").reset_index(drop=True)
    vals = []
    for _ in range(n_iter):
        idx = np.sort(RNG.choice(len(R), size=k, replace=False))
        vals.append(cagr_dd(R.iloc[idx])[2])
    return (real_cdd > np.array(vals)).mean() * 100


def run(name, csv, tf):
    d = resample(load_mt5_csv(csv), tf)
    print(f"\n########## {name} {tf} long  (target exit, +V-avoid) ##########")
    for ma_type in ("ema", "sma"):
        rows, lows = find(d, ma_type)
        nv = [r for r in rows if r["bars_down"] >= 4 and r["vel"] <= 0.6]
        if len(nv) < 12:
            print(f"  {ma_type.upper()}: too few ({len(nv)})"); continue
        base = simulate(d, nv, "target", lows)
        st = stats(base); cdd = cdd_of(base)
        print(f"\n  === 200{ma_type.upper()} ===  n={st['n']} win={st['win']:.0f}% "
              f"meanR={st['meanR']:+.2f} IS={st['IS']:+.2f} OOS={st['OOS']:+.2f} "
              f"grn={st['green']}/{st['nyr']} CAGR/DD={cdd:.2f}")
        # attack-count breakdown
        print(f"    {'attack#':<9}{'n':>5}{'win%':>6}{'meanR':>8}{'IS':>7}{'OOS':>7}{'CAGR/DD':>9}{'cddNull':>9}")
        bins = [("1", lambda x: x == 1), ("2", lambda x: x == 2), ("3+", lambda x: x >= 3)]
        for tag, fn in bins:
            sub = [r for r in nv if fn(r["attack"])]
            if len(sub) < 5:
                print(f"    {tag:<9}{len(sub):>5}   (too few)"); continue
            t = simulate(d, sub, "target", lows); s2 = stats(t); cd = cdd_of(t)
            nullp = drop_null_cdd(base, s2["n"], cd)
            ns = f"{nullp:>8.0f}" if nullp == nullp else "       -"
            print(f"    {tag:<9}{s2['n']:>5}{s2['win']:>6.0f}{s2['meanR']:>8.2f}"
                  f"{s2['IS']:>7.2f}{s2['OOS']:>7.2f}{cd:>9.2f}{ns}")


if __name__ == "__main__":
    for inst, csv in [("GOLD", "data/vantage_xauusd_h1.csv"),
                      ("BTC", "data/vantage_btcusd_h1.csv")]:
        for tf in ("4h", "8h", "1d"):
            run(inst, csv, tf)
