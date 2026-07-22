"""
Script 1 (FROZEN SPEC): band-walk end -> bounce / continuation screen.

BB(20, on close), 1-sigma and 2-sigma bands. Three walk variants:
  1sig_M3, 1sig_M5, 2sig_M3 (a "strength" variant).
EVENT-A = walk end (first close back inside the 1-sigma band after a
  completed walk of >=M consecutive closes strictly outside the WALK band).
EVENT-B = pullback re-entry: after event A, first bar within 3*M bars whose
  low (up-walk) / high (down-walk) touches the mid-band and which closes
  back on the outside of mid (continuation reading), scored also against
  an UNQUALIFIED BASE (same touch-and-reclaim condition, no walk required).
Both A and B are raced BOTH ways (fade / continuation) per the shared spec.

IMPLEMENTATION DECISIONS (spec was ambiguous on these three points; flagged
here rather than guessed silently, mirroring the RESAMPLE_1H_FIX precedent
in pdh_approach_fade.py):

  1. EVENT-A reversion trigger is ALWAYS the 1-sigma band, even for the
     2sig_M3 variant (the spec's literal text names "upper 1 sigma band"
     for event A without re-stating a 2-sigma variant of it). This makes
     the three walk variants differ ONLY in walk-defining entry threshold
     (what counts as "outside"), not in the reversion/exit definition, so
     they're apples-to-apples comparable. A close falling from >2sigma
     necessarily passes back through 1sigma, so this just means the
     2sig_M3 event-A can fire a few bars after the 2sigma run itself ends.

  2. EVENT-B's 3*M-bar search window is defined as
     [event_A_idx, event_A_idx + 3*M - 1] inclusive (3*M bars total,
     starting AT the walk-end bar itself, since that bar could already
     dip into mid-band on the same close that ended the walk).

  3. The UNQUALIFIED BASE for event B additionally requires the touch to
     be a genuine "from above" (up-direction) / "from below" (down-
     direction) approach, i.e. prev close was already on the far side of
     mid, not just "low<=mid & close>=mid" in isolation (which can also
     fire on a bar living entirely below mid that never really "touched
     from above"). Implemented as prev_close vs prev_mid on the bar
     immediately preceding the touch bar.

  4. MFE/MAE (ATR units) is printed once per cell-pair, for the FADE side
     only (per spec: "...MFE med/sd MAE med/sd in ATR units for the fade
     side"); per-year delta and ER-tercile are printed for BOTH sides
     since delta differs materially by side and the sentence's plain
     reading covers all three extras generically -- only MFE/MAE is
     side-specific by construction (MAE_fade == MFE_cont and vice versa,
     so printing it twice would be redundant).

Race, control-beta, dedupe, and ER machinery are shared with
vwap_band_edge.py via _race_common.py (same K-bar barrier race, same
hour-matched control-beta baseline, same dedupe rule as
pdh_approach_fade.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv
from _race_common import (
    RNG_SEED, CAP_CTRL, resample_ohlcv, efficiency_ratio, dedupe,
    next_true_at_or_after, build_control_pool, race_matrix,
    process_cell, print_peryear, print_er_tercile, print_mfe_mae,
    print_summary_table,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

K_BY_TF = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}      # 24h worth of bars
RULE_BY_TF = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}


def compute_bb(close: pd.Series, n=20):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid, std


def runs_of_true(bool_arr: np.ndarray):
    idx = np.flatnonzero(bool_arr)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def find_walk_eventA(close, walk_band, M, reentry_band, direction):
    n = len(close)
    if direction == "up":
        above = (close > walk_band) & ~np.isnan(walk_band)
        reentry_true = (close <= reentry_band) & ~np.isnan(reentry_band)
    else:
        above = (close < walk_band) & ~np.isnan(walk_band)
        reentry_true = (close >= reentry_band) & ~np.isnan(reentry_band)
    runs = runs_of_true(above)
    next_reentry = next_true_at_or_after(reentry_true)
    out = []
    for s, e in runs:
        if (e - s + 1) >= M and e + 1 < n:
            j = next_reentry[e + 1]
            if j < n:
                out.append(int(j))
    if not out:
        return np.array([], dtype=int)
    return np.array(sorted(set(out)), dtype=int)


def find_eventB(eventA_idx, cond_bool, window):
    idx = np.asarray(eventA_idx, dtype=int)
    n = len(cond_bool)
    if len(idx) == 0:
        return np.array([], dtype=int)
    idx_v = idx[idx + window - 1 <= n - 1]
    if len(idx_v) == 0:
        return np.array([], dtype=int)
    offsets = np.arange(0, window)
    mat = idx_v[:, None] + offsets[None, :]
    cond_mat = cond_bool[mat]
    has = cond_mat.any(axis=1)
    if not has.any():
        return np.array([], dtype=int)
    first_off = cond_mat[has].argmax(axis=1)
    result = idx_v[has] + first_off
    return np.array(sorted(set(result.tolist())), dtype=int)


def find_base_eventB(close, mid, low, high, direction):
    n = len(close)
    prev_close = np.roll(close, 1)
    prev_mid = np.roll(mid, 1)
    ok = ~np.isnan(mid) & ~np.isnan(prev_mid)
    if direction == "up":
        cond = ok & (prev_close > prev_mid) & (low <= mid) & (close >= mid)
    else:
        cond = ok & (prev_close < prev_mid) & (high >= mid) & (close <= mid)
    cond[0] = False
    return np.flatnonzero(cond)


def process_instrument(instr, base_df, tfs, all_results):
    rng = np.random.default_rng(RNG_SEED)
    for tf in tfs:
        d = resample_ohlcv(base_df, RULE_BY_TF[tf])
        n = len(d)
        K = K_BY_TF[tf]
        if n < 200:
            print(f"{instr} {tf}: SKIP (only {n} bars after resample)")
            continue

        atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1)
        atr_v = atr.values
        mid, std = compute_bb(d["close"], 20)
        up1 = (mid + std).values
        dn1 = (mid - std).values
        up2 = (mid + 2 * std).values
        dn2 = (mid - 2 * std).values
        mid_v = mid.values
        er = efficiency_ratio(d["close"], K).values

        high = d["high"].values
        low = d["low"].values
        close_v = d["close"].values

        span_years = (d.index[-1] - d.index[0]).days / 365.25

        ctrl_idx = build_control_pool(d, atr_v, K, CAP_CTRL, rng)
        _, ctrl_first_up, ctrl_first_dn, _, _, _, _ = race_matrix(
            high, low, ctrl_idx, atr_v, close_v, K
        )
        ctrl_idx_v = ctrl_idx[ctrl_idx + K <= n - 1]

        print(f"--- {instr} {tf}  (n_bars={n}, span={span_years:.1f}yr, K={K}) ---")

        variants = [("1sig_M3", up1, dn1, 3), ("1sig_M5", up1, dn1, 5), ("2sig_M3", up2, dn2, 3)]
        qualified_1sigM3_B = {}  # (direction, side_name) -> res, for the base-comparison line

        for variant_name, up_walk_band, dn_walk_band, M in variants:
            extras = (variant_name == "1sig_M3")

            for direction, walk_band, reentry_band, fade_side, cont_side, mid_cond_kind in [
                ("UP", up_walk_band, up1, "short", "long", "up"),
                ("DN", dn_walk_band, dn1, "long", "short", "down"),
            ]:
                eventA = find_walk_eventA(close_v, walk_band, M, reentry_band, mid_cond_kind)
                eventA = dedupe(eventA, K)

                cid_base = f"{instr:<4} {tf:<3} {direction:<2} {variant_name:<8}"
                for side_name, side in [("fade", fade_side), ("cont", cont_side)]:
                    cid = f"{cid_base} A {side_name}"
                    res = process_cell(cid, d, atr_v, high, low, close_v, eventA, side, K,
                                        span_years, ctrl_idx_v, ctrl_first_up, ctrl_first_dn)
                    all_results.append(res)
                    if extras:
                        print_peryear(res)
                        print_er_tercile(res, er)
                        if side_name == "fade":
                            print_mfe_mae(res)

                # EVENT-B qualified (walk-conditioned)
                if mid_cond_kind == "up":
                    condB = (low <= mid_v) & (close_v > mid_v) & ~np.isnan(mid_v)
                else:
                    condB = (high >= mid_v) & (close_v < mid_v) & ~np.isnan(mid_v)
                eventB = find_eventB(eventA, condB, 3 * M)
                eventB = dedupe(eventB, K)

                for side_name, side in [("fade", fade_side), ("cont", cont_side)]:
                    cid = f"{cid_base} B {side_name}"
                    res = process_cell(cid, d, atr_v, high, low, close_v, eventB, side, K,
                                        span_years, ctrl_idx_v, ctrl_first_up, ctrl_first_dn)
                    all_results.append(res)
                    if variant_name == "1sig_M3":
                        qualified_1sigM3_B[(direction, side_name)] = res
                    if extras:
                        print_peryear(res)
                        print_er_tercile(res, er)
                        if side_name == "fade":
                            print_mfe_mae(res)

        # UNQUALIFIED BASE for event B (once per direction, no walk requirement)
        for direction, fade_side, cont_side, mid_cond_kind in [
            ("UP", "short", "long", "up"),
            ("DN", "long", "short", "down"),
        ]:
            base_idx = find_base_eventB(close_v, mid_v, low, high, mid_cond_kind)
            base_idx = dedupe(base_idx, K)
            cid_base = f"{instr:<4} {tf:<3} {direction:<2} {'BASE':<8}"
            base_res = {}
            for side_name, side in [("fade", fade_side), ("cont", cont_side)]:
                cid = f"{cid_base} B0 {side_name}"
                res = process_cell(cid, d, atr_v, high, low, close_v, base_idx, side, K,
                                    span_years, ctrl_idx_v, ctrl_first_up, ctrl_first_dn)
                all_results.append(res)
                base_res[side_name] = res
            # base vs 1sig_M3-qualified comparison line (does the walk
            # qualifier concentrate anything over the unqualified base?)
            cmp_parts = []
            for side_name in ["fade", "cont"]:
                r = qualified_1sigM3_B.get((direction, side_name))
                b = base_res.get(side_name)
                if r is not None and b is not None and not np.isnan(r["delta"]) and not np.isnan(b["delta"]):
                    cmp_parts.append(f"{side_name}: qualified={r['delta']:+.1f}pt vs base={b['delta']:+.1f}pt")
            if cmp_parts:
                print(f"    walk-qualifier check ({direction}): " + " | ".join(cmp_parts))

        print()


def main():
    all_results = []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 100)
    print("PRE-REGISTERED PREDICTIONS:")
    print("  - trend instruments (gold/BTC) band-walk FADE  ~ <= 0 (user suspicion + graveyard prior)")
    print("  - USDJPY fade small-positive (re-derivation of the known BB-class fade edge)")
    print("  - EVENT-B walk qualifier ~ no lift over the unqualified BASE (the qualifier-doesn't-create-edge law)")
    print("  - many cells printed; single-cell PASS is noise unless neighbors (variant/TF) agree (plateau)")
    print("=" * 100)
    print()

    print("=" * 100)
    print("GOLD  (vantage_xauusd_m5.csv, .loc['2018-09-14':]  -> 5m/15m/1h/4h)")
    print("=" * 100)
    gold = load_mt5_csv(os.path.join(DATA_DIR, "vantage_xauusd_m5.csv"))
    gold = gold.loc["2018-09-14":]
    process_instrument("GOLD", gold, ["5m", "15m", "1h", "4h"], all_results)

    print("=" * 100)
    print("BTC   (vantage_btcusd_m15.csv, density-guarded -> 15m/1h/4h)")
    print("=" * 100)
    btc = load_mt5_csv(os.path.join(DATA_DIR, "vantage_btcusd_m15.csv"))
    cnt = btc.groupby(btc.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    btc = btc[btc.index.date >= okd.index[0]]
    process_instrument("BTC", btc, ["15m", "1h", "4h"], all_results)

    print("=" * 100)
    print("USDJPY  (vantage_usdjpy_m1.csv -> 5m/15m ; vantage_usdjpy_h1.csv -> 1h/4h)")
    print("=" * 100)
    ujm1 = load_mt5_csv(os.path.join(DATA_DIR, "vantage_usdjpy_m1.csv"))
    process_instrument("USDJPY", ujm1, ["5m", "15m"], all_results)
    ujh1 = load_mt5_csv(os.path.join(DATA_DIR, "vantage_usdjpy_h1.csv"))
    process_instrument("USDJPY", ujh1, ["1h", "4h"], all_results)

    print_summary_table(all_results, "SCRIPT 1 SUMMARY: band-walk end -> bounce/continuation, all cells")


if __name__ == "__main__":
    main()
