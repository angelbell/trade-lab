"""Raise the QUALITY of the 200MA touch: test new touch-quality features, each
against the random-drop CAGR/DD null (so a winner = real separation, not n-trim).

Discovery base = gold 4h+8h 200SMA touches (V-avoid, all touches, target exit) to
get enough n; pre-registered 'good' direction per feature. A feature is a real
quality lever only if its good-side beats the CAGR/DD null on BOTH TFs.

Features (causal):
  A slope_steep : (sma[s]-sma[s-K])/(K*ATR)         -- steeper rising MA = stronger trend (high=good)
  B depth       : (recentHigh-sma[s])/ATR over swingW -- how far above MA before the touch (gradient)
  C clean       : # of last N bars whose low entered the MA zone -- chop count (low=good, clean approach)
  D react       : (close-low)/range at the touch bar  -- close position / wick rejection (high=good)
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.ma200_bounce import resample, stats
from research.ma200_bounce_fractal import fractals, simulate
from research.portfolio_kama import cagr_dd

RNG = np.random.default_rng(13)


def touches(raw, tf, slopeK=20, zoneW=0.5, atrlen=14, swingW=30, cleanN=10):
    d = resample(raw, tf)
    sma = d["close"].rolling(200).mean().values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    _, lows = fractals(h, l, 3)
    rows, cnt = [], 0
    for s in range(max(slopeK, swingW, 200) + 1, len(c) - 1):
        if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        z = zoneW * a[s]
        if c[s] > sma[s] + 1.5 * a[s]:
            cnt = 0
        if not (sma[s] > sma[s - slopeK] and c[s] > sma[s]):
            continue
        if not (l[s] <= sma[s] + z and c[s] > sma[s] - z and c[s] > o[s]):
            continue
        cnt += 1
        e = o[s + 1]; stop = l[s]
        if e - stop < 0.5 * a[s]:
            stop = e - 0.5 * a[s]
        if e <= stop:
            continue
        win = slice(s - swingW + 1, s + 1)
        sh_rel = int(np.argmax(h[win])); swing_hi = h[win][sh_rel]
        bars_down = swingW - 1 - sh_rel
        vel = ((swing_hi - l[s]) / a[s]) / max(bars_down, 1)
        rng = max(h[s] - l[s], 1e-9)
        feats = dict(
            slope_steep=(sma[s] - sma[s - slopeK]) / (slopeK * a[s]),
            depth=(swing_hi - sma[s]) / a[s],
            clean=int(np.sum(l[s - cleanN:s] <= sma[s - cleanN:s] + z)),
            react=(c[s] - l[s]) / rng,
        )
        rows.append(dict(i=s + 1, e=e, stop=stop, vel=vel, bars_down=bars_down,
                         target=swing_hi, attack=cnt, **feats))
    return d, rows, lows


def drop_null_cdd(base_t, k, real_cdd, n_iter=1500):
    if len(base_t) <= k or k < 5:
        return np.nan
    R = base_t.sort_values("time").reset_index(drop=True)
    vals = [cagr_dd(R.iloc[np.sort(RNG.choice(len(R), size=k, replace=False))])[2]
            for _ in range(n_iter)]
    return (real_cdd > np.array(vals)).mean() * 100


def vavoid(rows):
    return [r for r in rows if r["bars_down"] >= 4 and r["vel"] <= 0.6]


def feat_test(d, rows, lows, feat, good="high"):
    """Keep top tercile by good direction; report vs the V-avoid base + null."""
    base = simulate(d, rows, "target", lows)
    vals = np.array([r[feat] for r in rows])
    if good == "high":
        thr = np.quantile(vals, 2 / 3); keep = [r for r in rows if r[feat] >= thr]
    else:
        thr = np.quantile(vals, 1 / 3); keep = [r for r in rows if r[feat] <= thr]
    if len(keep) < 6:
        return f"{feat:<12}{good:<5} (too few)"
    t = simulate(d, keep, "target", lows); st = stats(t)
    c, dd, cdd, _ = cagr_dd(t.sort_values("time"))
    nullp = drop_null_cdd(base, len(t), cdd)
    # also gradient: corr of feature with R over the base
    bt = simulate(d, rows, "target", lows)
    fr = pd.DataFrame({"f": vals[:len(bt)], "R": bt["R"].values[:len(vals)]})
    corr = fr["f"].corr(fr["R"])
    return (f"{feat:<12}{good:<5} n={st['n']:<4} win={st['win']:>3.0f}% meanR={st['meanR']:+.2f} "
            f"IS={st['IS']:+.2f} OOS={st['OOS']:+.2f} grn={st['green']}/{st['nyr']} "
            f"CAGR/DD={cdd:+.2f} null={nullp:>3.0f} | corr(f,R)={corr:+.2f}")


for tf in ("4h", "8h"):
    raw = load_mt5_csv("data/vantage_xauusd_h1.csv")
    d, rows, lows = touches(raw, tf)
    nv = vavoid(rows)
    bt = simulate(d, nv, "target", lows); st = stats(bt); c, dd, cdd, _ = cagr_dd(bt.sort_values("time"))
    print(f"\n#### GOLD {tf} 200SMA touch (V-avoid, all) base: n={st['n']} meanR={st['meanR']:+.2f} "
          f"IS={st['IS']:+.2f} OOS={st['OOS']:+.2f} CAGR/DD={cdd:+.2f} ####")
    for feat, good in [("slope_steep", "high"), ("depth", "high"), ("depth", "low"),
                       ("clean", "low"), ("react", "high")]:
        print("  " + feat_test(d, nv, lows, feat, good))
