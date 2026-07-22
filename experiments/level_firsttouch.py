"""Head-to-head: 'first touch of the 200SMA' vs 'first touch of an important
HORIZONTAL LEVEL', both in a rising-200SMA uptrend, +V-avoid, target=戻り高値.

Does trading the level (many levels -> more signals) give the same first-touch
edge with more n than the single 200SMA? Attacks the thinness病巣 (n=30)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.ma200_bounce import resample, stats
from research.ma200_bounce_fractal import fractals, simulate
from research.portfolio_kama import cagr_dd

RNG = np.random.default_rng(11)


def levels_signals(d, slopeK=20, tol=0.25, atrlen=14, swingW=30, pivp=3,
                   cool=20, reset=1.5):
    sma = d["close"].rolling(200).mean().values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    highs, lows = fractals(h, l, pivp)
    pidx = np.array([x[0] for x in highs] + [x[0] for x in lows])
    ppx = np.array([x[1] for x in highs] + [x[1] for x in lows])
    out = []
    for s in range(max(slopeK, swingW, 200) + 1, len(c) - 1):
        if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        # uptrend context: 200SMA rising AND price above it
        if not (sma[s] > sma[s - slopeK] and c[s] > sma[s]):
            continue
        # find an important level near the touch (>=2 prior pivots clustered)
        mask = (pidx <= s - 1) & (pidx >= s - 1 - 300)
        levp = ppx[mask]
        if not len(levp):
            continue
        best_center, best_cnt = None, 0
        for center in levp:
            if abs(center - l[s]) <= 0.6 * a[s]:
                cnt = int(np.sum(np.abs(levp - center) <= 0.5 * a[s]))
                if cnt > best_cnt:
                    best_cnt, best_center = cnt, center
        if best_cnt < 2 or best_center is None:
            continue
        L = best_center
        # touch-and-hold this level, bullish
        if not (l[s] <= L + tol * a[s] and c[s] > L and c[s] > o[s]):
            continue
        # FIRST touch: price did not reach this level in the prior `cool` bars
        if l[s - cool:s].min() <= L + tol * a[s]:
            continue
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
                        target=swing_hi))
    return out, lows


def drop_null_cdd(base_t, k, real_cdd, n_iter=1500):
    if len(base_t) <= k or k < 5:
        return np.nan
    R = base_t.sort_values("time").reset_index(drop=True)
    vals = [cagr_dd(R.iloc[np.sort(RNG.choice(len(R), size=k, replace=False))])[2]
            for _ in range(n_iter)]
    return (real_cdd > np.array(vals)).mean() * 100


for inst, csv in [("GOLD", "data/vantage_xauusd_h1.csv"),
                  ("BTC", "data/vantage_btcusd_h1.csv")]:
    print(f"\n######## {inst}: first-touch of an important HORIZONTAL LEVEL (uptrend, V-avoid, target) ########")
    print(f"{'TF':<5}{'n':>5}{'win%':>6}{'meanR':>8}{'IS':>7}{'OOS':>7}{'grn':>7}{'CAGR/DD':>9}{'cddNull':>9}")
    for tf in ("1h", "4h", "8h", "1d"):
        d = resample(load_mt5_csv(csv), tf)
        rows, lows = levels_signals(d)
        nv = [r for r in rows if r["bars_down"] >= 4 and r["vel"] <= 0.6]
        if len(nv) < 8:
            print(f"{tf:<5}{len(nv):>5}   (too few)"); continue
        t = simulate(d, nv, "target", lows); st = stats(t)
        c, dd, cdd, _ = cagr_dd(t.sort_values("time"))
        # null: vs random-equal-n drop from the FULL (no V-avoid) base set
        full = simulate(d, rows, "target", lows)
        nullp = drop_null_cdd(full, len(t), cdd)
        ns = f"{nullp:>8.0f}" if nullp == nullp else "       -"
        print(f"{tf:<5}{st['n']:>5}{st['win']:>6.0f}{st['meanR']:>8.2f}{st['IS']:>7.2f}"
              f"{st['OOS']:>7.2f}{st['green']:>3}/{st['nyr']:<3}{cdd:>9.2f}{ns}")
