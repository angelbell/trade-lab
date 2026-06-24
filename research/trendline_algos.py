"""trendline_algos.py -- FINAL line-algo probe: does a SMARTER line escape the gold_bo redundancy?

Hold EVERYTHING constant (ZigZag pivots, structural stop, RR2 exit, confirmed-close break, cost) and
vary ONLY the line selector, so any delta is the algorithm:
  last2  = the two most recent descending highs (naive control)
  ransac = max-inlier line robust to outliers (= real_trendline.active_resistance, the touch-count max)
  hull   = upper convex-hull last edge ("rubber-band" line price never crossed; near parameter-free)

Cheap-kill (same tests that killed every prior line): mfe/mae vs random-long + corr(trade-R, gold_bo).
Prior: better line -> MORE redundant (0.72->0.89), because "break a well-fit resistance in a trend" IS
"structure broke up" = what gold_bo trades. If hull/ransac stay redundant -> CLOSE the line-algo search.

  .venv/bin/python research/trendline_algos.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag
from research.regime_gate_lab import metrics
from research.portfolio_kama import get_legs
from research.real_trendline import active_resistance, detect, make_trades, mm

RNG = np.random.default_rng(7)
CSV, TF = "data/vantage_xauusd_h1.csv", "1h"
ZZ, TOL, MT, M, W, BUF = 2.0, 0.5, 2, 6, 24, 0.05


def _fill(n, highs, idx, s, xa, ya, slope, x0, y0, wid, w):
    end = highs[idx + 1][0] if idx + 1 < len(highs) else n
    rng = np.arange(highs[idx][0], end)
    slope[rng] = s; x0[rng] = xa; y0[rng] = ya; wid[rng] = w


def last2_resistance(d, atr, sw, M_):
    """naive: line through the two most recent (descending) confirmed highs."""
    n = len(d); slope = np.full(n, np.nan); x0 = np.full(n, np.nan); y0 = np.full(n, np.nan)
    wid = np.full(n, -1, int); highs = [(c, p, pr) for (c, p, pr, k) in sw if k == +1]; w = 0
    for idx in range(1, len(highs)):
        xa, ya = highs[idx - 1][1], highs[idx - 1][2]
        xb, yb = highs[idx][1], highs[idx][2]
        if xb > xa and yb < ya:                       # descending
            _fill(n, highs, idx, (yb - ya) / (xb - xa), xa, ya, slope, x0, y0, wid, w); w += 1
    return y0 + slope * (np.arange(n) - x0), wid


def _upper_hull(pts):
    """upper convex hull of (x,y) sorted by x; keep strictly clockwise turns."""
    h = []
    for p in pts:
        while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) -
                               (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) >= 0:
            h.pop()
        h.append(p)
    return h


def hull_resistance(d, atr, sw, M_):
    """convex-hull: active line = LAST upper-hull edge (descending) over the last M highs."""
    n = len(d); slope = np.full(n, np.nan); x0 = np.full(n, np.nan); y0 = np.full(n, np.nan)
    wid = np.full(n, -1, int); highs = [(c, p, pr) for (c, p, pr, k) in sw if k == +1]; w = 0
    for idx in range(len(highs)):
        pts = [(p, pr) for (_, p, pr) in highs[max(0, idx - M_ + 1):idx + 1]]
        pts = sorted(pts)
        if len(pts) < 2:
            continue
        hull = _upper_hull(pts)
        if len(hull) < 2:
            continue
        (xa, ya), (xb, yb) = hull[-2], hull[-1]       # last (rightmost) upper-hull edge
        if xb <= xa:
            continue
        s = (yb - ya) / (xb - xa)
        if s > 0:                                     # need a descending resistance
            continue
        _fill(n, highs, idx, s, xa, ya, slope, x0, y0, wid, w); w += 1
    return y0 + slope * (np.arange(n) - x0), wid


def rand_pctile(d, recent_low, atr, n_trades, gated_cdd, draws=150):
    valid = np.arange(60, len(d) - 2)
    vals = []
    for _ in range(draws):
        bars = np.sort(RNG.choice(valid, n_trades, replace=False))
        m = metrics(make_trades(d, list(bars), recent_low, atr, rr=2.0))
        if m: vals.append(m["cdd"])
    vals = np.array(vals)
    return (vals < gated_cdd).mean() * 100 if len(vals) else np.nan


def main():
    d = resample(load_mt5_csv(CSV), TF)
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    h, l, n = d["high"].values, d["low"].values, len(d)
    sw = swings_zigzag(h, l, atr, ZZ)
    recent_low = np.full(n, np.nan)
    for (c, p, pr, k) in sw:
        if k == -1 and c < n: recent_low[c] = pr
    recent_low = pd.Series(recent_low).ffill().values

    legs = get_legs(); gb = legs["gold_bo"]; cstart = gb.time.min()
    gbc = gb[gb.time >= cstart]
    def ann(x): return x.assign(y=x.time.dt.year).groupby("y").R.sum()
    def mon(x): return x.assign(m=x.time.dt.to_period("M")).groupby("m").R.sum()

    print(f"trendline_algos -- gold {TF} (n={n})  vary ONLY the line; cheap-kill: mfe/mae vs random + corr(gold_bo)")
    print(f"  random-long mfe/mae baseline (beta): {mm(d, list(np.sort(RNG.choice(np.arange(60,n-2),800,replace=False))), atr)[3]:.2f}")
    print(f"\n  {'selector':<8}{'nEntry':>7}{'mfe/mae':>8}{'CAGR/DD':>8}{'rand%':>6}{'corr_yr':>8}{'corr_mo':>8}  {'IS/OOS':>12}")
    for name, fn in [("last2", last2_resistance), ("ransac", active_resistance), ("hull", hull_resistance)]:
        if name == "ransac":
            L, wid = fn(d, atr, sw, TOL, MT, M)        # active_resistance signature
        else:
            L, wid = fn(d, atr, sw, M)
        ents = detect(d, atr, L, wid, "break", W, BUF, TOL)
        _, _, _, mr = mm(d, ents, atr)
        t = make_trades(d, ents, recent_low, atr, rr=2.0)
        m = metrics(t)
        if m is None:
            print(f"  {name:<8} n={len(ents)} too few"); continue
        tc = t[t.time >= cstart]
        cy = cm = np.nan
        if len(tc) >= 10:
            A = pd.concat([ann(tc), ann(gbc)], axis=1).fillna(0)
            Mo = pd.concat([mon(tc), mon(gbc)], axis=1).fillna(0)
            cy = A.iloc[:, 0].corr(A.iloc[:, 1]); cm = Mo.iloc[:, 0].corr(Mo.iloc[:, 1])
        pct = rand_pctile(d, recent_low, atr, m["n"], m["cdd"])
        print(f"  {name:<8}{m['n']:>7}{mr:>8.2f}{m['cdd']:>8.2f}{pct:>6.0f}{cy:>+8.2f}{cm:>+8.2f}  "
              f"{m['isr']:>+.2f}/{m['oos']:>+.2f}")
    print("\n  refs: gold_bo CAGR/DD=1.09. PASS = mfe/mae > random AND corr_yr < ~0.4 (not redundant).")
    print("  if hull & ransac both stay redundant (corr>=~0.6) -> line-algo search CLOSED.")


if __name__ == "__main__":
    main()
