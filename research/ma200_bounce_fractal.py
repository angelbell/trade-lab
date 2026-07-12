"""ma200_bounce_fractal.py -- 200EMA touch-and-hold bounce with FRACTAL-STRUCTURE
exits (no fixed RR) + the approach-quality / horizontal-level filters.

Per the user: don't fix RR -- let the FRACTAL structure decide the trade. Two
structural exits, both no-fixed-RR:
  target : take profit at the swing high we pulled back FROM (the fractal high in
           the approach window = 戻り高値); stop at the touch fractal low.
  trail  : ratchet the stop up to each new CONFIRMED fractal low as it forms;
           exit on the structure break (price takes out the last fractal low).
           Lets winners run to the next structure break.

R is in each trade's OWN risk units (risk = entry - initial fractal-low stop), so
the realised R VARIES per trade -- we read its distribution, not a preset multiple.

Filters (causal): V-attack avoidance (gradual approach only) + important
horizontal level (>=2 prior swing-pivot touches). Each is checked against the
RANDOM-DROP NULL (does it beat keeping the same count at random?).

NB: 1h/4h/8h/1d here are all SWING/positional, NOT scalp (scalp = 1m/5m, a
separate test). Run:  .venv/bin/python research/ma200_bounce_fractal.py
"""
import os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.ma200_bounce import resample, stats

RNG = np.random.default_rng(7)


def fractals(h, l, p=3):
    """Confirmed fractal highs/lows. Returns (highs, lows) as lists of
    (confirm_idx, price); a pivot at t is confirmed (usable) from t+p."""
    highs, lows = [], []
    for t in range(p, len(h) - p):
        if h[t] == max(h[t - p:t + p + 1]):
            highs.append((t + p, h[t]))
        if l[t] == min(l[t - p:t + p + 1]):
            lows.append((t + p, l[t]))
    return highs, lows


def find_signals_feat(d, side, emalen=200, slopeK=20, tol=0.25, atrlen=14,
                      swingW=30, pivp=3):
    ema = d["close"].ewm(span=emalen, adjust=False).mean().values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    highs, lows = fractals(h, l, pivp)
    pidx = np.array([x[0] for x in highs] + [x[0] for x in lows])
    ppx = np.array([x[1] for x in highs] + [x[1] for x in lows])
    out = []
    for s in range(max(slopeK, swingW) + 1, len(c) - 1):
        if np.isnan(ema[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        if side != "long":
            continue  # long-only (the live side; FX/short died in the base screen)
        if not (ema[s] > ema[s - slopeK] and l[s] <= ema[s] + tol * a[s]
                and c[s] > ema[s] and c[s] > o[s]):
            continue
        e = o[s + 1]; stop = l[s]
        if e - stop < 0.5 * a[s]:
            stop = e - 0.5 * a[s]
        if e <= stop:
            continue
        win = slice(s - swingW + 1, s + 1)
        sh_rel = int(np.argmax(h[win])); swing_hi = h[win][sh_rel]
        bars_down = swingW - 1 - sh_rel
        drop = swing_hi - l[s]
        vel = (drop / a[s]) / max(bars_down, 1)
        touch_px = l[s]
        mask = (pidx <= s - 1) & (pidx >= s - 1 - 300)
        levp = ppx[mask]; best = 0
        for center in levp:
            if abs(center - touch_px) <= 0.6 * a[s]:
                best = max(best, int(np.sum(np.abs(levp - center) <= 0.5 * a[s])))
        out.append(dict(i=s + 1, e=e, stop=stop, vel=vel, bars_down=bars_down,
                        near_level=best >= 2, target=swing_hi))
    return out, ema, a, lows


def simulate(d, rows, mode, lows, fwd=300, cost=0.001):
    """mode='target' (exit at the pulled-back-from swing high) or 'trail'
    (ratchet stop to each new confirmed fractal low; exit on structure break)."""
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    low_idx = np.array([x[0] for x in lows]); low_px = np.array([x[1] for x in lows])
    res, busy = [], -1
    for r in rows:
        i, e, stop0, tgt = r["i"], r["e"], r["stop"], r["target"]
        if i <= busy:
            continue
        risk = e - stop0
        if risk <= 0:
            continue
        exit_j = min(i + fwd, len(c) - 1); R = None; stop = stop0
        for j in range(i, min(i + fwd, len(c))):
            if l[j] <= stop:
                R = (stop - e) / risk; exit_j = j; break
            if mode == "target":
                if tgt > e and h[j] >= tgt:
                    R = (tgt - e) / risk; exit_j = j; break
            else:  # trail: raise stop to the highest confirmed fractal low below price
                cand = low_px[(low_idx <= j) & (low_px < c[j]) & (low_px > stop)]
                if len(cand):
                    stop = cand.max()
        if R is None:
            R = (c[exit_j] - e) / risk
        R -= cost / risk * e
        res.append((d.index[i], R, (tgt - e) / risk))   # also log structural R-avail
        busy = exit_j
    return pd.DataFrame(res, columns=["time", "R", "Ravail"])


def rand_drop_null(base_R, k, real_mean, n_iter=2000):
    if len(base_R) <= k or k < 5:
        return np.nan
    means = np.array([RNG.choice(base_R, size=k, replace=False).mean() for _ in range(n_iter)])
    return (real_mean > means).mean() * 100


def report(name, csv, tf, mode):
    d = resample(load_mt5_csv(csv), tf)
    rows, ema, a, lows = find_signals_feat(d, "long")
    if len(rows) < 12:
        print(f"{name} {tf}: too few ({len(rows)})"); return
    base = simulate(d, rows, mode, lows)
    no_v = [r for r in rows if r["bars_down"] >= 4 and r["vel"] <= 0.6]
    lvl = [r for r in rows if r["near_level"]]
    both = [r for r in rows if r["bars_down"] >= 4 and r["vel"] <= 0.6 and r["near_level"]]
    rav = base["Ravail"].median()
    print(f"\n=== {name} {tf} long  exit={mode}  (median structural R avail={rav:.1f}) ===")
    print(f"{'variant':<12}{'n':>5}{'win%':>6}{'meanR':>8}{'totR':>7}{'IS':>7}{'OOS':>7}{'grn':>6}{'drop%ile':>9}")
    for tag, rws in [("base", rows), ("+V-avoid", no_v), ("+level", lvl), ("+both", both)]:
        if len(rws) < 5:
            print(f"{tag:<12}{len(rws):>5}   (too few)"); continue
        t = simulate(d, rws, mode, lows); st = stats(t)
        dn = rand_drop_null(base["R"].values, st["n"], st["meanR"]) if tag != "base" else np.nan
        ds = f"{dn:>8.0f}" if dn == dn else "       -"
        print(f"{tag:<12}{st['n']:>5}{st['win']:>6.0f}{st['meanR']:>8.2f}{st['totR']:>7.0f}"
              f"{st['IS']:>7.2f}{st['OOS']:>7.2f}{st['green']:>3}/{st['nyr']:<2}{ds}")


if __name__ == "__main__":
    for inst, csv in [("GOLD", "data/vantage_xauusd_h1.csv"),
                      ("BTC", "data/vantage_btcusd_h1.csv")]:
        for tf in ("1h", "4h", "8h", "1d"):
            for mode in ("target", "trail"):
                report(inst, csv, tf, mode)
