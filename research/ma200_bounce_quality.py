"""ma200_bounce_quality.py -- does APPROACH-QUALITY + horizontal-level confluence
separate the tradeable 200EMA bounces from the noise?

Extends the touch-and-hold 200EMA bounce (ma200_bounce.py) with two discretionary
quality filters the user flagged, mechanised causally:

  V-attack avoidance : from the recent swing high into the touch, measure the
      descent VELOCITY = (drop/ATR) / bars_down. A sharp few-bar plunge (a "V",
      a falling knife) is rejected; keep only gradual/rounded pullbacks
      (bars_down >= minBars AND velocity <= maxVel).
  Horizontal-level   : cluster confirmed swing pivots (fractal, ±p bars, only
      those confirmed BEFORE the signal) into price levels; a level with
      >= minTouch members is "important". Require the touch to sit within
      levelTol*ATR of such a level (200EMA + horizontal S/R confluence).

The KEY test is not "did meanR go up" (any trim can do that) -- it is the
RANDOM-DROP NULL: keep the same number of trades drawn at RANDOM from the base
and see if the filtered meanR beats that distribution. <~90%ile => just n-trimming.

Run:  .venv/bin/python research/ma200_bounce_quality.py
"""
import os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.ma200_bounce import resample, simulate, stats

RNG = np.random.default_rng(7)


def pivots(h, l, p=3):
    """Fractal swing highs/lows. Returns list of (confirm_idx, price). A pivot at
    bar t is confirmed at t+p (causal: usable only from t+p onward)."""
    out = []
    for t in range(p, len(h) - p):
        if h[t] == max(h[t - p:t + p + 1]):
            out.append((t + p, h[t]))
        if l[t] == min(l[t - p:t + p + 1]):
            out.append((t + p, l[t]))
    return out


def find_signals_feat(d, side, emalen=200, slopeK=20, tol=0.25, atrlen=14,
                      swingW=30, pivp=3):
    """Touch-and-hold bounces + per-signal features (velocity, near_level)."""
    ema = d["close"].ewm(span=emalen, adjust=False).mean().values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    piv = pivots(h, l, pivp)
    piv_idx = np.array([x[0] for x in piv]); piv_px = np.array([x[1] for x in piv])
    out = []
    for s in range(max(slopeK, swingW) + 1, len(c) - 1):
        if np.isnan(ema[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        if side == "long":
            ok = ema[s] > ema[s - slopeK] and l[s] <= ema[s] + tol * a[s] and c[s] > ema[s] and c[s] > o[s]
        else:
            ok = ema[s] < ema[s - slopeK] and h[s] >= ema[s] - tol * a[s] and c[s] < ema[s] and c[s] < o[s]
        if not ok:
            continue
        # entry/stop
        if side == "long":
            e = o[s + 1]; stop = l[s]
            if e - stop < 0.5 * a[s]: stop = e - 0.5 * a[s]
            if e <= stop: continue
        else:
            e = o[s + 1]; stop = h[s]
            if stop - e < 0.5 * a[s]: stop = e + 0.5 * a[s]
            if stop <= e: continue
        # --- approach velocity (V-attack) ---
        win = slice(s - swingW + 1, s + 1)
        if side == "long":
            sh_rel = int(np.argmax(h[win])); swing = h[win][sh_rel]
            bars_down = swingW - 1 - sh_rel
            drop = swing - l[s]
        else:
            sl_rel = int(np.argmin(l[win])); swing = l[win][sl_rel]
            bars_down = swingW - 1 - sl_rel
            drop = h[s] - swing
        vel = (drop / a[s]) / max(bars_down, 1)
        # --- nearest important horizontal level (causal pivots only) ---
        touch_px = l[s] if side == "long" else h[s]
        mask = (piv_idx <= s - 1) & (piv_idx >= s - 1 - 300)
        levp = piv_px[mask]
        near = False; best_touch = 0
        if len(levp):
            # greedy cluster around the touch price within levelTol; count members
            for ltol in (0.5,):  # base proximity; touch-count below
                pass
            # cluster: any level center (a pivot) with >=minTouch pivots within tol*ATR of it AND near the touch
            for center in levp:
                if abs(center - touch_px) <= 0.6 * a[s]:
                    cnt = int(np.sum(np.abs(levp - center) <= 0.5 * a[s]))
                    best_touch = max(best_touch, cnt)
            near = best_touch >= 2
        out.append(dict(i=s + 1, e=e, stop=stop, vel=vel, bars_down=bars_down,
                        near_level=near, level_touches=best_touch))
    return out, ema, a


def to_sig(rows):
    return [(r["i"], r["e"], r["stop"]) for r in rows]


def rand_drop_null(base_t, k, real_mean, n_iter=2000):
    """%ile of real (filtered) meanR vs keeping k random base trades."""
    R = base_t["R"].values
    if len(R) <= k or k < 5:
        return np.nan
    means = np.array([RNG.choice(R, size=k, replace=False).mean() for _ in range(n_iter)])
    return (real_mean > means).mean() * 100


def report(name, csv, tf, side, rr, maxVel=0.6, minBars=4, fwd=200, cost=0.001):
    d = resample(load_mt5_csv(csv), tf)
    rows, ema, a = find_signals_feat(d, side, slopeK=20, tol=0.25)
    if len(rows) < 12:
        print(f"{name} {tf} {side} RR{rr}: too few base ({len(rows)})"); return
    base = simulate(d, to_sig(rows), side, rr, fwd, cost)
    no_v = [r for r in rows if r["bars_down"] >= minBars and r["vel"] <= maxVel]
    lvl = [r for r in rows if r["near_level"]]
    both = [r for r in rows if r["bars_down"] >= minBars and r["vel"] <= maxVel and r["near_level"]]
    print(f"\n=== {name} {tf} {side} RR{rr}  (V: bars>={minBars} & vel<={maxVel}; level: >=2 touches) ===")
    print(f"{'variant':<14}{'n':>5}{'win%':>6}{'meanR':>8}{'IS':>7}{'OOS':>7}{'grn':>6}{'drop%ile':>9}")
    for tag, rws in [("base", rows), ("+V-avoid", no_v), ("+level", lvl), ("+both", both)]:
        if len(rws) < 5:
            print(f"{tag:<14}{len(rws):>5}   (too few)"); continue
        t = simulate(d, to_sig(rws), side, rr, fwd, cost)
        st = stats(t)
        dn = rand_drop_null(base, st["n"], st["meanR"]) if tag != "base" else np.nan
        ds = f"{dn:>8.0f}" if dn == dn else "       -"
        print(f"{tag:<14}{st['n']:>5}{st['win']:>6.0f}{st['meanR']:>8.2f}"
              f"{st['IS']:>7.2f}{st['OOS']:>7.2f}{st['green']:>3}/{st['nyr']:<2}{ds}")


if __name__ == "__main__":
    for inst, csv, tf in [("GOLD", "data/vantage_xauusd_h1.csv", "1h"),
                          ("GOLD", "data/vantage_xauusd_h1.csv", "4h"),
                          ("GOLD", "data/vantage_xauusd_h1.csv", "8h"),
                          ("BTC", "data/vantage_btcusd_h1.csv", "4h"),
                          ("BTC", "data/vantage_btcusd_h1.csv", "1d")]:
        for rr in (2.0, 3.0):
            report(inst, csv, tf, "long", rr)
