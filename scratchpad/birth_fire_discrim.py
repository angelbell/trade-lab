"""birth_fire_discrim.py -- can CAUSAL features at fire time concentrate the true 15m KAMA flips?

Events = D1 KAMA(14)-rising OFF->ON flips on 15m bars (same construction as
scratchpad/trend_birth_meter.py, whose loading/KAMA/race/ground-truth code is imported directly).
Cells: GOLD m15 >= 2018-05-01, BTC m15 >= 2018-09-01 (dense-data windows only).

OUTCOME per event (primary): the +/-2xATR14(15m) first-touch race from the NEXT bar, 30-bar
horizon; win = +1R, loss = -1R; ties/unresolved are EXCLUDED from win%/meanR (decided events).
DIAGNOSTIC (ex-post, report-only, never used for selection): fire inside a true-leg birth window
(ground-truth ZigZag legs, identical params to trend_birth_meter).

Discipline copied from research/regime_discriminator.py:
  IS threshold/zone selection -> OOS apply -> equal-keep random-drop null -> per-year ON%.
Differences (this is event-level, not trade-level): IS = first half of EVENTS by time; keep-zone =
best IS bucket(s) by meanR (contiguous zones for terciles, single category for categoricals);
null = 1000 equal-count draws from the OOS decided events, %ile of kept OOS meanR.

CAUSAL features at the fire bar (confirmed info only; HTF states use only the last COMPLETED
higher-TF bar whose close is strictly before the fire bar's close):
  F1 session (fire-bar hour UTC: 0-8 Asia / 9-15 London / 16-23 NY)
  F2 1h KAMA(14)-rising state    F3 4h KAMA(14)-rising state
  F4 GMMA long-group width pct-rank in trailing 250 bars (terciles)
  F5 decline depth (90-bar high - close)/ATR14 (terciles)
  F6 extension (close - SMA150)/ATR14 (terciles)
  F7 weekly cycle: close vs 30-week SMA, last completed week (binary)
  F23 stack: F2 AND F3 both rising (binary, the mechanistic candidate)

PASS line (reported, not editorialized): OOS kept meanR > OOS base AND randDrop >= 90%ile, on
BOTH instruments with the SAME feature and SAME zone direction.

Usage:
  .venv/bin/python scratchpad/birth_fire_discrim.py --smoke   # GOLD m15, 2024 only, F1 only
  .venv/bin/python scratchpad/birth_fire_discrim.py           # full run, both cells
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from scratchpad.trend_birth_meter import (kama, atr14, zigzag_pivots, build_uplegs, race_codes,
                                          fires_from_state, d1_kama_rising, LONG, WIN, LOSS)

MIN_KEEP_IS = 30   # a keep-zone must retain at least this many IS decided events
MIN_KEEP_OOS = 12
NULL_TRIALS = 1000

CELLS = [
    ("GOLD m15", "data/vantage_xauusd_m15.csv", "2018-05-01"),
    ("BTC m15", "data/vantage_btcusd_m15.csv", "2018-09-01"),
]

AGGD = {"open": "first", "high": "max", "low": "min", "close": "last"}

# ----------------------------------------------------------------------------------------------
# causal feature builders: f(dd, fire_pos) -> (vals per fire, kind, labels)
#   kind 'cat' -> vals are integer category codes; kind 'ter' -> continuous, bucketed by IS terciles
# ----------------------------------------------------------------------------------------------

def _fire_close_times(dd, fire_pos):
    return (dd.index + pd.Timedelta("15min"))[fire_pos]


def _htf_kama_state(dd, span, fire_pos):
    """1/0 state of KAMA(14)-rising on the last COMPLETED span-bar strictly before fire close."""
    h = dd.resample(span).agg(AGGD).dropna()
    km = kama(h["close"], 14)
    valid = km.notna() & km.shift(1).notna()
    rising = np.where(valid, (km > km.shift(1)).astype(float), np.nan)
    close_t = h.index + pd.Timedelta(span)
    m = pd.merge_asof(pd.DataFrame({"t": _fire_close_times(dd, fire_pos)}),
                      pd.DataFrame({"t": close_t, "v": rising}).sort_values("t"),
                      on="t", direction="backward", allow_exact_matches=False)
    return m["v"].values


def f1_session(dd, fire_pos):
    hr = dd.index.hour.values[fire_pos]
    cat = np.where(hr <= 8, 0, np.where(hr <= 15, 1, 2)).astype(float)
    return cat, "cat", ["Asia0-8", "Lon9-15", "NY16-23"]


def f2_htf1h(dd, fire_pos):
    return _htf_kama_state(dd, "1h", fire_pos), "cat", ["1hKAMAdn", "1hKAMAup"]


def f3_htf4h(dd, fire_pos):
    return _htf_kama_state(dd, "4h", fire_pos), "cat", ["4hKAMAdn", "4hKAMAup"]


def f4_compression(dd, fire_pos):
    ldf = pd.concat([dd["close"].ewm(span=s, adjust=False).mean() for s in LONG], axis=1)
    width = (ldf.max(axis=1) - ldf.min(axis=1)) / dd["close"]
    rank = width.rolling(250).rank(pct=True)
    return rank.values[fire_pos], "ter", None


def f5_decline_depth(dd, fire_pos):
    a = atr14(dd)
    depth = (dd["high"].rolling(90).max() - dd["close"]) / a
    return depth.values[fire_pos], "ter", None


def f6_extension(dd, fire_pos):
    a = atr14(dd)
    ext = (dd["close"] - dd["close"].rolling(150).mean()) / a
    return ext.values[fire_pos], "ter", None


def f7_weekly_cycle(dd, fire_pos):
    w = dd["close"].resample("1W").last().dropna()
    sma = w.rolling(30).mean()
    above = pd.Series(np.where(sma.notna(), (w > sma).astype(float), np.nan), index=w.index).shift(1)
    m = pd.merge_asof(pd.DataFrame({"t": dd.index[fire_pos]}),
                      pd.DataFrame({"t": w.index, "v": above.values}).sort_values("t"),
                      on="t", direction="backward")
    return m["v"].values, "cat", ["belowW30", "aboveW30"]


def f23_stack(dd, fire_pos):
    v2 = _htf_kama_state(dd, "1h", fire_pos)
    v3 = _htf_kama_state(dd, "4h", fire_pos)
    both = np.where(np.isnan(v2) | np.isnan(v3), np.nan, ((v2 > 0.5) & (v3 > 0.5)).astype(float))
    return both, "cat", ["stackOFF", "stackON"]


FEATURES = {
    "F1_session": f1_session,
    "F2_1hKAMA": f2_htf1h,
    "F3_4hKAMA": f3_htf4h,
    "F4_compression": f4_compression,
    "F5_decline_depth": f5_decline_depth,
    "F6_extension": f6_extension,
    "F7_weekly_cycle": f7_weekly_cycle,
    "F23_stack_1h4h": f23_stack,
}

# ----------------------------------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------------------------------

def bucket_stats(R, birth, mask):
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, win=np.nan, mr=np.nan, b=np.nan)
    r = R[mask]
    return dict(n=n, win=100 * (r > 0).mean(), mr=r.mean(), b=100 * birth[mask].mean())


def run_feature(fname, vals, kind, labels, R, birth, years, is_mask, verbose=True):
    """R/birth/years/is_mask are per DECIDED event; vals aligned. Returns result dict or None."""
    valid = ~np.isnan(vals)
    v = vals; oos_mask = ~is_mask
    # ---- buckets on IS ----
    if kind == "ter":
        iv = v[is_mask & valid]
        if len(iv) < 3 * MIN_KEEP_IS:
            print(f"  {fname:<18} (too few valued IS events)"); return None
        q1, q2 = np.quantile(iv, [1 / 3, 2 / 3])
        code = np.where(~valid, np.nan, np.where(v <= q1, 0, np.where(v <= q2, 1, 2)))
        labels = [f"lo(<= {q1:.2f})", f"mid", f"hi(> {q2:.2f})"]
        zones = {"lo": [0], "mid": [1], "hi": [2], "lo+mid": [0, 1], "mid+hi": [1, 2]}
    else:
        code = v
        cats = sorted(int(c) for c in np.unique(v[valid]))
        zones = {labels[c]: [c] for c in cats}
    ncat = len(labels)
    # ---- IS bucket table ----
    rows = []
    for c in range(ncat):
        m = is_mask & valid & (code == c)
        rows.append((labels[c], bucket_stats(R, birth, m)))
    if verbose:
        tab = " | ".join(f"{lb} n={s['n']} w={s['win']:.0f}% mR={s['mr']:+.3f} b={s['b']:.0f}%"
                         if s["n"] else f"{lb} n=0" for lb, s in rows)
        print(f"  {fname:<18} IS: {tab}")
    # ---- keep-zone selection on IS meanR ----
    best = (None, -np.inf, None)
    for zname, zcats in zones.items():
        m = is_mask & valid & np.isin(code, zcats)
        if m.sum() < MIN_KEEP_IS:
            continue
        mr = R[m].mean()
        if mr > best[1]:
            best = (zname, mr, zcats)
    if best[0] is None:
        print(f"    -> no keep-zone reaches min IS n={MIN_KEEP_IS}"); return None
    zname, _, zcats = best
    # ---- OOS apply + equal-keep random-drop null ----
    keep_oos = oos_mask & valid & np.isin(code, zcats)
    base_oos = oos_mask
    kn = int(keep_oos.sum())
    oos_base_mr = R[base_oos].mean()
    if kn < MIN_KEEP_OOS:
        print(f"    -> zone={zname}: OOS kept n={kn} too few"); return None
    kept = R[keep_oos]
    kept_mr = kept.mean(); kept_win = 100 * (kept > 0).mean()
    rng = np.random.default_rng(0)
    pool = R[base_oos]
    nul = np.array([rng.choice(pool, kn, replace=False).mean() for _ in range(NULL_TRIALS)])
    pct = 100 * (nul < kept_mr).mean()
    ok = (kept_mr > oos_base_mr) and (pct >= 90)
    print(f"    -> zone={zname:<9} OOS kept n={kn} w={kept_win:.1f}% mR={kept_mr:+.4f} "
          f"vs base {oos_base_mr:+.4f}  randDrop={pct:.0f}%ile  {'PASS-cell' if ok else 'fail'}")
    # ---- per-year meanR of kept subset (full history, zone applied) + kept share ----
    keep_all = valid & np.isin(code, zcats)
    ys = sorted(np.unique(years))
    parts = []
    for y in ys:
        my = years == y
        mk = my & keep_all
        mr = R[mk].mean() if mk.sum() else np.nan
        share = 100 * mk.sum() / my.sum() if my.sum() else np.nan
        parts.append(f"{y}:{mr:+.2f}/{share:.0f}%" if mk.sum() else f"{y}:--/0%")
    print(f"    yr mR/ON%: " + " ".join(parts))
    return dict(zone=zname, zcats=tuple(zcats), delta=kept_mr - oos_base_mr, pct=pct,
                ok=ok, kept_mr=kept_mr, base_mr=oos_base_mr, kn=kn)


def run_cell(name, csv, start, end=None, features=FEATURES):
    dd = load_mt5_csv(csv).loc[start:end]
    close, high, low = dd["close"].values, dd["high"].values, dd["low"].values
    a = atr14(dd).values
    state = d1_kama_rising(dd)
    fire_pos = fires_from_state(state)
    codes = race_codes(close, high, low, a)

    pivots = zigzag_pivots(close, a)
    legs = build_uplegs(pivots, close, a, dd.index)
    legs_low = np.array([l["low_pos"] for l in legs])
    legs_high = np.array([l["high_pos"] for l in legs])
    legs_bend = np.array([l["birth_end_pos"] for l in legs])

    def birth_flag(p):
        i = int(np.searchsorted(legs_low, p, side="right")) - 1
        return bool(i >= 0 and p <= legs_high[i] and p <= legs_bend[i])

    birth_all = np.array([birth_flag(p) for p in fire_pos])
    c_ev = codes[fire_pos]
    decided = (c_ev == WIN) | (c_ev == LOSS)
    R_all = np.where(c_ev == WIN, 1.0, np.where(c_ev == LOSS, -1.0, np.nan))

    # decided-event arrays (protocol event set; ties/unresolved excluded from win%/meanR)
    dpos = fire_pos[decided]
    R = R_all[decided]
    birth = birth_all[decided]
    years = dd.index.year.values[dpos]
    n_dec = len(dpos)
    is_mask = np.zeros(n_dec, dtype=bool); is_mask[: n_dec // 2] = True  # first half by time
    split_t = dd.index[dpos[n_dec // 2]] if n_dec else None

    print(f"\n{'='*104}\n=== {name} ({os.path.basename(csv)} >= {start}) bars={len(dd)} "
          f"[{dd.index[0]} .. {dd.index[-1]}] ===")
    print(f"  true up-legs={len(legs)} (ground truth, diagnostic only)")
    yr_mr = " ".join(f"{y}:{R[years == y].mean():+.3f}(n{(years == y).sum()})"
                     for y in sorted(np.unique(years)))
    print(f"  BASE: fires={len(fire_pos)} decided={n_dec} (ties/unresolved excl.) "
          f"win%={100 * (R > 0).mean():.1f} meanR={R.mean():+.4f} birth%={100 * birth_all.mean():.1f}")
    print(f"  BASE per-year meanR (decided): {yr_mr}")
    print(f"  IS = first {n_dec // 2} decided events (until {split_t}), OOS = rest "
          f"(IS mR={R[is_mask].mean():+.4f}, OOS mR={R[~is_mask].mean():+.4f})")

    out = {}
    for fname, fn in features.items():
        vals_all, kind, labels = fn(dd, fire_pos)
        vals = np.asarray(vals_all, float)[decided]
        out[fname] = run_feature(fname, vals, kind, labels, R, birth, years, is_mask)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="GOLD m15 2024 only, F1 only")
    args = ap.parse_args()
    print("birth_fire_discrim -- causal fire-time features vs 15m KAMA-flip race outcome")
    print(f"(race +/-2xATR14, 30-bar horizon, next-bar start; null = {NULL_TRIALS} equal-keep draws "
          f"from OOS decided events)")
    if args.smoke:
        print("[SMOKE: GOLD m15 2024 only, F1_session only]")
        run_cell("GOLD m15 smoke", "data/vantage_xauusd_m15.csv", "2024-01-01", "2024-12-31",
                 features={"F1_session": f1_session})
        return
    res = {}
    for name, csv, start in CELLS:
        res[name] = run_cell(name, csv, start)
    # ---- summary matrix ----
    print(f"\n{'='*104}\nSUMMARY MATRIX  (cell entry: zone | OOS kept-base delta meanR | randDrop %ile | PASS-cell?)")
    print(f"  {'feature':<18}" + "".join(f"{n:>42}" for n in res))
    both_pass = []
    for fname in FEATURES:
        row = f"  {fname:<18}"
        cells = [res[n].get(fname) for n in res]
        for r in cells:
            row += f"{'(no gate)':>42}" if r is None else \
                f"{r['zone']:>12} d={r['delta']:+.4f} {r['pct']:>4.0f}%ile {'PASS' if r['ok'] else 'fail':>5}".rjust(42)
        print(row)
        if all(r is not None and r["ok"] for r in cells) and len({r["zcats"] for r in cells}) == 1:
            both_pass.append(fname)
    print(f"\n  PASS line (both instruments, same feature, same zone direction): "
          f"{', '.join(both_pass) if both_pass else 'NONE'}")


if __name__ == "__main__":
    main()
