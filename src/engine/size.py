"""Size overlays — the WEIGHTS stage of the engine pipeline. Every function maps
(bars d, trade table t) → a per-trade multiplier array; multipliers compose by
multiplication. Nothing here changes the trade set: entries/exits stay canonical,
only the bet size moves (structural law 9 family: strength/regime/context are
SIZE variables, never targets or exits).

All levels use confirmed-bar semantics (shift(1) + ffill) = no lookahead:
prior-day high/low, the last CONFIRMED 4H swing high, yesterday's completed
daily SMA, and ICT labels computed strictly before the fill bar.

Lifted verbatim from the measured evidence scripts (kept frozen in experiments/):
book_integration.py (PDH soft), stack_size_btc15mL.py (ladder / daily regime),
ict_size_transplant.py (compute_labels). Guarded by invariants/size_tieback.py —
array-identical against those originals on the canonical btc15m_L trade set.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta

from breakout_wave import swings_zigzag, swings_pivot


def bar_idx(d, t):
    """Bar positions of each trade's fill time in d.index."""
    return d.index.get_indexer(pd.DatetimeIndex(t["time"]))


# ---------------------------------------------------------------- prior-day levels

def pdh_series(d):
    """Prior day's high, per bar (shift(1) = yesterday's completed day)."""
    return d["high"].resample("1D").max().dropna().shift(1).reindex(d.index, method="ffill").values


def pdl_series(d):
    """Prior day's low, per bar."""
    return d["low"].resample("1D").min().dropna().shift(1).reindex(d.index, method="ffill").values


def pdh_soft(d, t, mult=0.5):
    """PDH soft size (the adopted btc15m_L rule): full size when the fill price is
    above the prior day's high, `mult` below it. Returns (W, above_pdh)."""
    pdh = pdh_series(d)
    ab = t["e_px"].values > pdh[bar_idx(d, t)]
    return np.where(ab, 1.0, mult), ab


def pdl_break_mask(d, t_inv, C):
    """Short-mirror hard filter (btc15m_S): keep only trades whose REAL fill price
    (C − inverted e_px) is below the prior day's low. C = the inversion constant."""
    pdl = pdl_series(d)
    return (C - t_inv["e_px"].values) < pdl[d.index.get_indexer(pd.DatetimeIndex(t_inv["time"]))]


# ---------------------------------------------------------------- PDH × HH4H ladder

def hh4h_series(d15):
    """Last CONFIRMED 4H ZigZag swing high, expanded to the LTF index.
    shift(1) after confirmation = no lookahead (a 4H swing is only knowable on
    the close of its confirming 4H bar)."""
    h4 = d15.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = ta.atr(h4["high"], h4["low"], h4["close"], 14).values
    sw = swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0)
    s = pd.Series(np.nan, index=h4.index)
    for (ci, pi, px, kind) in sw:
        if kind == +1:
            s.iloc[ci] = px
    return s.ffill().shift(1).reindex(d15.index, method="ffill").values


def pdh_hh4h_ladder(d15, t, ii=None):
    """The measured size ladder (E1 component 1): fill above BOTH the prior-day
    high and the last confirmed 4H swing high = ×1.0, above one = ×0.5,
    above neither = ×0.25. Returns (W, above_pdh, above_hh4)."""
    if ii is None:
        ii = bar_idx(d15, t)
    pdh = pdh_series(d15)
    hh4 = hh4h_series(d15)
    e_px = t["e_px"].values
    above_pdh = e_px > pdh[ii]
    above_hh4 = np.where(np.isfinite(hh4[ii]), e_px > hh4[ii], False)
    both = above_pdh & above_hh4
    one = above_pdh ^ above_hh4
    W = np.where(both, 1.0, np.where(one, 0.5, 0.25))
    return W, above_pdh, above_hh4


# ---------------------------------------------------------------- daily regime

def daily_regime_mult(d, t, ii=None, sma_n=150, mult=0.75):
    """Law 9b: when yesterday's completed daily close is BELOW its SMA, size down
    (never skip — the correct treatment of a weak regime with positive EV is a
    smaller bet, not abstention). Returns (W, down_flags)."""
    if ii is None:
        ii = bar_idx(d, t)
    dly = d["close"].resample("1D").last().dropna()
    sma = dly.rolling(sma_n).mean()
    down = (dly < sma).shift(1)
    down_bar = down.reindex(d.index, method="ffill").fillna(False).values
    down_at = down_bar[ii]
    return np.where(down_at, mult, 1.0), down_at


# ---------------------------------------------------------------- ICT labels

def _bullish_fvg_size(hi, lo, atr_val, s, e, min_atr):
    """3-bar bullish FVG scan on [s, e]: candle3.low > candle1.high opens the gap
    band; size = gap / ATR; the largest gap >= min_atr is returned (else None).
    Fixed at generation time (candle3 close); later fills don't undo existence."""
    best = None
    edges = None
    for i in range(s, e - 1):
        c1_hi, c3_lo = hi[i], lo[i + 2]
        if c3_lo > c1_hi:
            size = (c3_lo - c1_hi) / atr_val
            if size >= min_atr and (best is None or size > best):
                best = size
                edges = (c1_hi, c3_lo)
    return best, edges


def compute_labels(d, t, ii, X):
    """ICT context labels per trade, looking back X bars strictly BEFORE the fill:
    label A = a liquidity hunt (fractal low(2,2) or prior-day low undercut) followed
    by a close-reclaim of the swept level; label B = a bullish FVG (>= 0.15 ATR) in
    the break leg. Measured: label-ABSENT trades are the ~0-EV subset (2026-07-17)."""
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    n = len(d)

    sw = swings_pivot(h, l, 2)
    frac_lo = [(ci, price) for (ci, pi, price, kind) in sw if kind == -1]
    frac_lo.sort()
    confirms = np.array([x[0] for x in frac_lo])
    prices = np.array([x[1] for x in frac_lo])

    pdl_daily = pd.Series(l, index=d.index).resample("1D").min().shift(1)
    pdl_arr = pdl_daily.reindex(d.index, method="ffill").values

    base_bars = t["base_bars"].values

    labelA = np.zeros(len(ii), dtype=bool)
    labelB = np.zeros(len(ii), dtype=bool)
    for k, b in enumerate(ii):
        lo_win = max(0, b - X)
        if lo_win >= b:
            continue
        # ---- Label A: hunt + reclaim ----
        candidates = []
        idx = np.searchsorted(confirms, b) - 1
        while idx >= 0 and confirms[idx] >= lo_win:
            candidates.append(prices[idx]); idx -= 1
            if len(candidates) >= 3:
                break
        if np.isfinite(pdl_arr[b]):
            candidates.append(pdl_arr[b])
        hit = False
        for lvl in candidates:
            if (l[lo_win:b] < lvl).any() and b >= 1 and c[b - 1] > lvl:
                hit = True; break
        labelA[k] = hit
        # ---- Label B: FVG in break leg ----
        bb = base_bars[k]
        if np.isfinite(bb) and bb > 2:
            leg_start = max(0, int(b - bb))
            A = atr[b - 1] if b >= 1 and np.isfinite(atr[b - 1]) else np.nan
            if np.isfinite(A) and A > 0 and leg_start < b - 1:
                sz, edges = _bullish_fvg_size(h, l, A, leg_start, min(b, n - 2), 0.15)
                labelB[k] = sz is not None
    return labelA, labelB


def ict_label_mult(d, t, ii=None, x=48, weak=0.5, label="AB"):
    """ICT-label size (E1 component 3): label-absent trades sized at `weak`.
    label = "AB" (btc15m_L: both required), "A" (gold15m: sweep only — the FVG
    label is REVERSED on gold, do not use it there), or "B"."""
    if ii is None:
        ii = bar_idx(d, t)
    labelA, labelB = compute_labels(d, t, ii, x)
    flag = {"AB": labelA & labelB, "A": labelA, "B": labelB}[label]
    return np.where(flag, 1.0, weak), flag
