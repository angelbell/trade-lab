"""Rebuilt horizontal-level detector: lines = HIGHER-TIMEFRAME swing pivots
(日足/週足の高安), per the user. Test whether a GOOD line detector gives the
horizontal level a real, independent bounce edge -- and more signals than the
single 200SMA (fixing the thinness).

Bounce = in a 200SMA uptrend, price (low) touches an HTF pivot line and HOLDS
(close above, bullish), non-V approach, exit = 戻り高値 target. Long-only.
Three variants compared: 200SMA-firsttouch / HTF-level / confluence(both).
Causal: an HTF pivot is usable only after it confirms (t+p HTF bars)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.ma200_bounce import resample, stats
from research.ma200_bounce_fractal import fractals, simulate
from research.portfolio_kama import cagr_dd

RNG = np.random.default_rng(11)


def htf_pivot_lines(raw, level_tf, p):
    """HTF swing highs/lows as (confirm_time, price), causal (confirm at t+p)."""
    H = resample(raw, level_tf)
    hi, lo, idx = H["high"].values, H["low"].values, H.index
    lines = []
    for t in range(p, len(hi) - p):
        if hi[t] == max(hi[t - p:t + p + 1]):
            lines.append((idx[t + p], hi[t]))
        if lo[t] == min(lo[t - p:t + p + 1]):
            lines.append((idx[t + p], lo[t]))
    lines.sort()
    return np.array([x[0] for x in lines]), np.array([x[1] for x in lines])


def signals(raw, tf, level_tf, p=3, slopeK=20, tol=0.25, atrlen=14, swingW=30,
            lookbk_days=365, mode="htf", cool=20, zoneW=0.5):
    """mode: 'sma' = 200SMA first-touch; 'htf' = HTF-line touch;
    'conf' = HTF-line touch that is ALSO near the 200SMA (confluence).
    Support is a ZONE, not a line: a touch = low enters [level-z, level+z]
    (z=zoneW*ATR) and HOLD = close does not close below the zone bottom."""
    d = resample(raw, tf)
    sma = d["close"].rolling(200).mean().values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    _, lows_fr = fractals(h, l, 3)
    ct, cp = htf_pivot_lines(raw, level_tf, p)
    ct_ns = ct.astype("datetime64[ns]")
    tindex = d.index.values.astype("datetime64[ns]")
    look = np.timedelta64(lookbk_days, "D")
    out = []
    sma_cnt = 0
    for s in range(max(slopeK, swingW, 200) + 1, len(c) - 1):
        if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        ts = tindex[s]
        if not (sma[s] > sma[s - slopeK] and c[s] > sma[s]):  # 200SMA uptrend context
            continue
        if c[s] > sma[s] + 1.5 * a[s]:
            sma_cnt = 0
        z = zoneW * a[s]                            # support ZONE half-width
        # candidate touched level (low dips into the zone from above)
        near_sma = l[s] <= sma[s] + z
        # active HTF lines (confirmed, within lookback)
        hi_i = np.searchsorted(ct_ns, ts)
        lo_i = np.searchsorted(ct_ns, ts - look)
        cand = cp[lo_i:hi_i]
        hit_htf = None
        if len(cand):
            dist = np.abs(cand - l[s])
            j = int(np.argmin(dist))
            if dist[j] <= z:
                hit_htf = cand[j]
        # select by mode
        if mode == "sma":
            if not near_sma:
                continue
            level = sma[s]
        elif mode == "htf":
            if hit_htf is None:
                continue
            level = hit_htf
        else:  # confluence: HTF line near AND near the 200SMA
            if hit_htf is None or not near_sma:
                continue
            level = hit_htf
        # HOLD: don't close below the zone bottom (level - z), and bullish
        if not (c[s] > level - z and c[s] > o[s]):
            continue
        if mode == "sma":
            sma_cnt += 1
            attack = sma_cnt
        else:
            # first-touch of THIS zone: not reached in prior `cool` bars
            attack = 1 if l[s - cool:s].min() > level + z else 2
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
                        target=swing_hi, attack=attack))
    return d, out, lows_fr


def drop_null_cdd(base_t, k, real_cdd, n_iter=1500):
    if len(base_t) <= k or k < 5:
        return np.nan
    R = base_t.sort_values("time").reset_index(drop=True)
    vals = [cagr_dd(R.iloc[np.sort(RNG.choice(len(R), size=k, replace=False))])[2]
            for _ in range(n_iter)]
    return (real_cdd > np.array(vals)).mean() * 100


def row(d, rows, lows, label, base_full=None):
    nv = [r for r in rows if r["bars_down"] >= 4 and r["vel"] <= 0.6]
    if len(nv) < 6:
        print(f"  {label:<26} n={len(nv):<3} (too few)"); return None
    t = simulate(d, nv, "target", lows); st = stats(t)
    c, dd, cdd, _ = cagr_dd(t.sort_values("time"))
    nullp = drop_null_cdd(base_full, len(t), cdd) if base_full is not None else np.nan
    ns = f"{nullp:>4.0f}" if nullp == nullp else "   -"
    print(f"  {label:<26} n={st['n']:<4} win={st['win']:>3.0f}% meanR={st['meanR']:+.2f} "
          f"IS={st['IS']:+.2f} OOS={st['OOS']:+.2f} grn={st['green']}/{st['nyr']} "
          f"CAGR/DD={cdd:+.2f} cddNull={ns}")
    return t


CELLS = [("GOLD", "data/vantage_xauusd_h1.csv", "8h", "1W"),
         ("GOLD", "data/vantage_xauusd_h1.csv", "4h", "1D"),
         ("GOLD", "data/vantage_xauusd_h1.csv", "8h", "1D"),
         ("BTC", "data/vantage_btcusd_h1.csv", "4h", "1D"),
         ("BTC", "data/vantage_btcusd_h1.csv", "8h", "1W")]
for inst, csv, tf, ltf in CELLS:
    raw = load_mt5_csv(csv)
    print(f"\n#### {inst} trade={tf} lines={ltf} -- support ZONE width sweep (HTF-line, all touches, V-avoid) ####")
    for zoneW in (0.25, 0.5, 0.75, 1.0):
        d, htf_all, lows = signals(raw, tf, ltf, mode="htf", zoneW=zoneW)
        base_full = simulate(d, htf_all, "target", lows) if len(htf_all) >= 6 else None
        row(d, htf_all, lows, f"zoneW={zoneW}", base_full)
