"""trend_birth_meter.py -- SCORING HARNESS for causal trend-birth detectors: earliness vs precision.

Ground truth (ex-post, peek allowed -- used ONLY for scoring, never as a detector input): a ZigZag
on CLOSES of the cell's timeframe with an ADAPTIVE reversal threshold of 4xATR14 measured at the
running extreme (the ATR value on the bar of the extreme, not today's). A pivot low->high pair
qualifies as a TRUE UP-LEG when (high_px - low_px) >= 8xATR14(at the pivot-low bar) AND duration
>= 10 units. BIRTH WINDOW of a leg = its first max(5, 20% of leg duration) units.

UNITS: intraday cells (15m / 1h) measure duration/birth/race horizon in BARS of that TF (the bar IS
the unit). The --daily mode reproduces the original daily-cell spec, where duration/birth are in
CALENDAR DAYS on a calendar-day resample of H1 (kept for comparability with the first report).

Detectors are plain functions state(dd) -> pd.Series[bool] on the cell's bar index, registered in
the DETECTORS dict (adding a detector = one function + one dict entry). A "fire" = the bar the
state flips OFF->ON. All detector state at bar t uses only data up to and including bar t (rolling/
EMA constructions are causal by construction). The only look-ahead in this file is the ground-truth
ZigZag, which is explicitly ex-post and used only to grade fires.

ATR14 convention (repo standard, e.g. research/gold_overextension.py): TR = max(H-L, |H-Cprev|,
|L-Cprev|), simple rolling(14).mean() (not Wilder), computed on the cell's own TF bars.

False-fire race: from the fire bar's close, which is touched first within 30 bars (starting NEXT
bar): close+2xATR14 (win) or close-2xATR14 (loss)? Both-in-one-bar = tie (excluded); neither within
horizon = none (excluded). Baseline = the same race started on EVERY bar (vectorized).

Usage:
  .venv/bin/python scratchpad/trend_birth_meter.py --smoke   # GOLD 15m, 2024 only (mechanics)
  .venv/bin/python scratchpad/trend_birth_meter.py           # all 4 intraday cells
  .venv/bin/python scratchpad/trend_birth_meter.py --daily   # original daily cells (calendar-day units)
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

SHORT = [3, 5, 8, 10, 12, 15]
LONG = [30, 35, 40, 45, 50, 60]
HORIZON = 30      # race horizon, in units (bars intraday / days daily)
ZZ_K = 4.0        # ZigZag reversal threshold, xATR14
LEG_MAG_K = 8.0   # true-leg magnitude floor, xATR14 at pivot low
LEG_MIN_DUR = 10  # true-leg duration floor, in units

INTRADAY_CELLS = [
    ("GOLD 15m", "data/vantage_xauusd_m15.csv"),
    ("GOLD 1h", "data/vantage_xauusd_h1.csv"),
    ("BTC 15m", "data/vantage_btcusd_m15.csv"),
    ("BTC 1h", "data/vantage_btcusd_h1.csv"),
]
DAILY_CELLS = [
    ("GOLD daily", "data/vantage_xauusd_h1.csv"),
    ("BTC daily", "data/vantage_btcusd_h1.csv"),
]

# ----------------------------------------------------------------------------------------------
# data plumbing
# ----------------------------------------------------------------------------------------------

def daily_ohlc(dh1):
    return pd.DataFrame({
        "open": dh1["open"].resample("1D").first(),
        "high": dh1["high"].resample("1D").max(),
        "low": dh1["low"].resample("1D").min(),
        "close": dh1["close"].resample("1D").last(),
    }).dropna()


def atr14(dd, n=14):
    h, l, c = dd["high"], dd["low"], dd["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ----------------------------------------------------------------------------------------------
# ground-truth ZigZag (ex-post; scoring only) -- positions, unit-agnostic
# ----------------------------------------------------------------------------------------------

def zigzag_pivots(prices, a, k=ZZ_K):
    """ATR-threshold ZigZag on closes (numpy arrays). Returns [(type, pos), ...] alternating
    'low'/'high'; a pivot is confirmed once price retraces >= k*ATR (ATR taken AT the running
    extreme's own bar) from the running extreme."""
    n = len(prices)
    max_price = prices[0]; max_pos = 0
    min_price = prices[0]; min_pos = 0
    trend = 0
    pivots = []
    for i in range(1, n):
        px = prices[i]
        if px > max_price:
            max_price = px; max_pos = i
        if px < min_price:
            min_price = px; min_pos = i
        if trend <= 0:
            av = a[min_pos]
            if not np.isnan(av) and (px - min_price) >= k * av:
                pivots.append(("low", min_pos))
                trend = 1
                max_price = px; max_pos = i
        if trend >= 0:
            av = a[max_pos]
            if not np.isnan(av) and (max_price - px) >= k * av:
                pivots.append(("high", max_pos))
                trend = -1
                min_price = px; min_pos = i
    return pivots


def build_uplegs(pivots, prices, a, index, calendar_days=False):
    """True up-legs. duration/birth in BARS (positions), unless calendar_days=True (daily mode:
    duration = (high_date-low_date).days, birth window in calendar days -- the original spec)."""
    legs = []
    for i in range(len(pivots) - 1):
        t0, p0 = pivots[i]
        t1, p1 = pivots[i + 1]
        if t0 != "low" or t1 != "high":
            continue
        lo_px, hi_px = prices[p0], prices[p1]
        a0 = a[p0]
        if np.isnan(a0) or (hi_px - lo_px) < LEG_MAG_K * a0:
            continue
        if calendar_days:
            dur = (index[p1] - index[p0]).days
            if dur < LEG_MIN_DUR:
                continue
            birth_end_date = index[p0] + pd.Timedelta(days=max(5.0, 0.2 * dur))
            birth_end_pos = int(np.searchsorted(index, birth_end_date, side="right")) - 1
        else:
            dur = p1 - p0
            if dur < LEG_MIN_DUR:
                continue
            birth_end_pos = p0 + int(np.ceil(max(5.0, 0.2 * dur)))
        legs.append(dict(low_pos=p0, high_pos=p1, low_px=lo_px, high_px=hi_px, dur=dur,
                          birth_end_pos=min(birth_end_pos, p1), atr_at_low=a0,
                          low_date=index[p0], high_date=index[p1]))
    return legs


# ----------------------------------------------------------------------------------------------
# detectors (unchanged constructions, computed on the cell's own TF bars)
# ----------------------------------------------------------------------------------------------

def kama(close, n=14, fast=2, slow=30):
    ch = close.diff(n).abs()
    vol = close.diff().abs().rolling(n).sum()
    erc = (ch / vol).fillna(0).values
    fsc, ssc = 2 / (fast + 1), 2 / (slow + 1)
    sc = (erc * (fsc - ssc) + ssc) ** 2
    c = close.values; out = np.full(len(c), np.nan)
    seed = n
    if len(c) <= seed:
        return pd.Series(out, index=close.index)
    out[seed] = c[seed]
    for i in range(seed + 1, len(c)):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def gmma_emas(dc):
    return {s: dc.ewm(span=s, adjust=False).mean() for s in SHORT + LONG}


def d1_kama_rising(dd):
    km = kama(dd["close"], 14)
    return (km > km.shift(1)).fillna(False)


def d2_gmma_separation(dd):
    emas = gmma_emas(dd["close"])
    short_min = pd.concat([emas[s] for s in SHORT], axis=1).min(axis=1)
    long_max = pd.concat([emas[s] for s in LONG], axis=1).max(axis=1)
    return (short_min > long_max).fillna(False)


def d3_gmma_compress_cross(dd):
    dc = dd["close"]
    emas = gmma_emas(dc)
    short_df = pd.concat([emas[s] for s in SHORT], axis=1)
    long_df = pd.concat([emas[s] for s in LONG], axis=1)
    ms = short_df.mean(axis=1); ml = long_df.mean(axis=1)
    width = (long_df.max(axis=1) - long_df.min(axis=1)) / dc
    q33 = width.rolling(250).quantile(1 / 3)
    compressed = (width <= q33).fillna(False)
    cross_up = (ms > ml) & (ms.shift(1) <= ml.shift(1))
    cross_down = (ms < ml) & (ms.shift(1) >= ml.shift(1))
    compressed_recent = compressed.shift(1).rolling(10).max().fillna(0).astype(bool)
    fire_up = (cross_up & compressed_recent).fillna(False).values
    cd = cross_down.fillna(False).values
    state = np.zeros(len(dc), dtype=bool)
    on = False
    for i in range(len(dc)):
        if fire_up[i]:
            on = True
        elif on and cd[i]:
            on = False
        state[i] = on
    return pd.Series(state, index=dc.index)


def d4_donchian20(dd):
    donch_high = dd["high"].rolling(20).max().shift(1)
    return (dd["close"] > donch_high).fillna(False)


def d5_sma150_slope(dd):
    sma = dd["close"].rolling(150).mean()
    return (sma.diff(10) > 0).fillna(False)


DETECTORS = {
    "D1_KAMA_rising": d1_kama_rising,
    "D2_GMMA_separation": d2_gmma_separation,
    "D3_GMMA_compress_cross": d3_gmma_compress_cross,
    "D4_Donchian20_bo": d4_donchian20,
    "D5_SMA150_slope_turn": d5_sma150_slope,
}

# ----------------------------------------------------------------------------------------------
# race (vectorized over all bars once per cell; per-fire results are lookups into this)
# ----------------------------------------------------------------------------------------------

WIN, LOSS, TIE, NONE = 1, -1, 2, 0

def race_codes(close, high, low, a, horizon=HORIZON):
    """For every bar: first-touch race of close+/-2*ATR14 over the NEXT `horizon` bars.
    Returns int8 codes: WIN/LOSS/TIE/NONE (NONE also where ATR is NaN)."""
    n = len(close)
    win_px = close + 2 * a
    loss_px = close - 2 * a
    first_win = np.full(n, horizon + 1, dtype=np.int32)
    first_loss = np.full(n, horizon + 1, dtype=np.int32)
    for j in range(1, horizon + 1):
        hj = np.full(n, -np.inf); lj = np.full(n, np.inf)
        hj[: n - j] = high[j:]; lj[: n - j] = low[j:]
        hit_w = (hj >= win_px) & (first_win > horizon)
        first_win[hit_w] = j
        hit_l = (lj <= loss_px) & (first_loss > horizon)
        first_loss[hit_l] = j
    codes = np.full(n, NONE, dtype=np.int8)
    got_w = first_win <= horizon; got_l = first_loss <= horizon
    codes[got_w & (first_win < first_loss)] = WIN
    codes[got_l & (first_loss < first_win)] = LOSS
    codes[got_w & got_l & (first_win == first_loss)] = TIE
    codes[np.isnan(a)] = NONE
    return codes


def race_winpct(codes, sel=None):
    c = codes if sel is None else codes[sel]
    w = int((c == WIN).sum()); l = int((c == LOSS).sum())
    t = int((c == TIE).sum()); nn = int((c == NONE).sum())
    pct = 100 * w / (w + l) if (w + l) > 0 else np.nan
    return pct, w, l, t, nn


# ----------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------

def fires_from_state(state):
    s = state.fillna(False).astype(bool)
    prev = s.shift(1, fill_value=False)
    return np.flatnonzero((s & (~prev)).values)


def leg_of_pos(legs_low, legs_high, p):
    """Legs are ordered & non-overlapping -> searchsorted lookup. Returns leg idx or -1."""
    i = int(np.searchsorted(legs_low, p, side="right")) - 1
    if i >= 0 and p <= legs_high[i]:
        return i
    return -1


def score_detector(fire_pos, close, legs, codes, kama_first_fire, span_years):
    legs_low = np.array([l["low_pos"] for l in legs])
    legs_high = np.array([l["high_pos"] for l in legs])
    n_fires = len(fire_pos)
    hit_flags, birth_flags, missfracs, false_pos = [], [], [], []
    leg_hit = [False] * len(legs); leg_birth = [False] * len(legs)
    first_fire = [None] * len(legs)
    for p in fire_pos:
        i = leg_of_pos(legs_low, legs_high, p) if len(legs) else -1
        if i >= 0:
            leg = legs[i]
            hit_flags.append(True)
            leg_hit[i] = True
            if first_fire[i] is None:
                first_fire[i] = p
            b = p <= leg["birth_end_pos"]
            birth_flags.append(b)
            if b:
                leg_birth[i] = True
            frac = (close[p] - leg["low_px"]) / (leg["high_px"] - leg["low_px"])
            missfracs.append(min(max(frac, 0.0), 1.5))
        else:
            hit_flags.append(False); birth_flags.append(False)
            false_pos.append(p)
    diffs = [first_fire[i] - kama_first_fire[i] for i in range(len(legs))
             if first_fire[i] is not None and kama_first_fire[i] is not None]
    if false_pos:
        ff_win, w, l, _, _ = race_winpct(codes, np.array(false_pos, dtype=int))
    else:
        ff_win, w, l = np.nan, 0, 0
    return dict(
        fires=n_fires,
        fires_per_yr=n_fires / span_years if span_years > 0 else np.nan,
        hit_pct=100 * np.mean(hit_flags) if n_fires else np.nan,
        birth_pct=100 * np.mean(birth_flags) if n_fires else np.nan,
        missfrac_med=np.median(missfracs) if missfracs else np.nan,
        missfrac_std=np.std(missfracs) if missfracs else np.nan,
        recall_birth=100 * np.mean(leg_birth) if legs else np.nan,
        recall_leg=100 * np.mean(leg_hit) if legs else np.nan,
        earliness_med=np.median(diffs) if diffs else np.nan,
        falsefire_win=ff_win, ff_n=w + l)


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------

def run_cell(name, csv, daily=False, start=None, end=None):
    raw = load_mt5_csv(csv)
    dd = daily_ohlc(raw) if daily else raw
    if start or end:
        dd = dd.loc[start:end]
    close = dd["close"].values
    high = dd["high"].values
    low = dd["low"].values
    a = atr14(dd).values
    index = dd.index
    unit = "day" if daily else "bar"

    pivots = zigzag_pivots(close, a)
    legs = build_uplegs(pivots, close, a, index, calendar_days=daily)
    span_years = (index[-1] - index[0]).days / 365.25
    n_days = len(pd.unique(index.date))

    print(f"\n{'='*104}\n=== {name} ({os.path.basename(csv)}) bars={len(dd)} "
          f"[{index[0]} .. {index[-1]}]  span={span_years:.1f}yr  trading-days={n_days} ===")
    print(f"  pivots={len(pivots)}  TRUE up-legs={len(legs)}  (units: {unit}s; reversal {ZZ_K}xATR14, "
          f"leg >= {LEG_MAG_K}xATR14 & >= {LEG_MIN_DUR} {unit}s)")
    if legs:
        by_yr = pd.Series([l["low_date"].year for l in legs]).value_counts().sort_index()
        print("  legs/year (by pivot-low year): " + ", ".join(f"{y}:{c}" for y, c in by_yr.items()))
        durs = np.array([l["dur"] for l in legs])
        dur_hours = np.median([(l["high_date"] - l["low_date"]).total_seconds() / 3600 for l in legs])
        mags = np.median([(l["high_px"] - l["low_px"]) / l["atr_at_low"] for l in legs])
        print(f"  leg duration: med={np.median(durs):.0f} {unit}s (= {dur_hours:.0f} wall-clock hours ="
              f" {dur_hours/24:.1f} days)   leg magnitude: med={mags:.1f} xATR14")
    else:
        print("  ! no true up-legs found")

    codes = race_codes(close, high, low, a)
    base_pct, w, l, t, nn = race_winpct(codes)
    print(f"  baseline race (EVERY bar, +/-2xATR14, {HORIZON}-{unit} horizon): win%={base_pct:.1f}% "
          f"(wins={w} losses={l} ties={t} none={nn} n={len(dd)})")

    states = {dn: fn(dd) for dn, fn in DETECTORS.items()}
    fires = {dn: fires_from_state(s) for dn, s in states.items()}

    legs_low = np.array([lg["low_pos"] for lg in legs])
    legs_high = np.array([lg["high_pos"] for lg in legs])
    kama_first = [None] * len(legs)
    for p in fires["D1_KAMA_rising"]:
        i = leg_of_pos(legs_low, legs_high, p) if len(legs) else -1
        if i >= 0 and kama_first[i] is None:
            kama_first[i] = p

    results = {dn: score_detector(fp, close, legs, codes, kama_first, span_years)
               for dn, fp in fires.items()}

    per_day = (not daily) and ("15m" in name)
    print(f"  {'detector':<24}{'fires/yr':>9}" + ("  fires/day" if per_day else "") +
          f"{'hit%':>7}{'birth%':>8}{'missfrac med±std':>19}{'recall_birth%':>14}{'recall_leg%':>12}"
          f"{'earliness_vs_KAMA':>19}{'falsefire_win%':>18}")
    for dn, r in results.items():
        mf = f"{r['missfrac_med']:.2f}±{r['missfrac_std']:.2f}" if not np.isnan(r["missfrac_med"]) else "n/a"
        earl = f"{r['earliness_med']:+.0f}{unit[0]}" if not np.isnan(r["earliness_med"]) else "n/a"
        fd = f"{r['fires']/n_days:>10.2f}" if per_day else ""
        ffw = f"{r['falsefire_win']:.1f} (n={r['ff_n']})" if not np.isnan(r["falsefire_win"]) else "n/a"
        print(f"  {dn:<24}{r['fires_per_yr']:>9.2f}{fd}{r['hit_pct']:>7.1f}{r['birth_pct']:>8.1f}"
              f"{mf:>19}{r['recall_birth']:>14.1f}{r['recall_leg']:>12.1f}{earl:>19}{ffw:>18}")
    print(f"  (earliness in {unit}s, negative = earlier than D1 KAMA; baseline race win% = {base_pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="GOLD 15m, 2024 only")
    ap.add_argument("--daily", action="store_true", help="original daily cells (calendar-day units)")
    a = ap.parse_args()
    print("trend_birth_meter -- causal trend-birth detector scoring harness (earliness vs precision)")
    if a.smoke:
        print("[SMOKE TEST: GOLD 15m, 2024-01-01..2024-12-31]")
        run_cell("GOLD 15m", "data/vantage_xauusd_m15.csv", start="2024-01-01", end="2024-12-31")
        return
    if a.daily:
        for name, csv in DAILY_CELLS:
            run_cell(name, csv, daily=True)
        return
    for name, csv in INTRADAY_CELLS:
        run_cell(name, csv)


if __name__ == "__main__":
    main()
