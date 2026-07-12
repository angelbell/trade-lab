"""
Script 2 (FROZEN SPEC + AMENDMENT): daily-anchored VWAP +/- k*sigma band edge
fade/breakout screen.

Daily VWAP resets at each calendar-day boundary in the data's index tz (the
index is UTC-labeled BROKER SERVER time per src.data_loader.load_mt5_csv, so
day/hour boundaries here ARE server-time boundaries -- this is what makes the
WIN-B amendment below a direct "10:00 server time" cut, no extra tz math).
tp=(H+L+C)/3, w=volume (tick_volume) if present else 1 (flagged below).
sigma_vwap = sqrt(cumsum(w*(tp-vwap)^2)/cumsum(w)) within the day, using the
literal (slightly non-standard) spec formula: the per-bar squared deviation
uses that bar's OWN running vwap, not the day's final vwap -- fully causal,
implemented exactly as specified, not "corrected" to a textbook running
variance.

AMENDMENT (received after Script 1 was already run, applies to Script 2
only): the original "skip first 2 hours of the day" window is too lenient;
report TWO entry-window variants side by side per cell:
  WIN-A (original): events allowed once elapsed-since-day-start >= 2h.
  WIN-B (primary):  events allowed only once server-clock hour >= 10:00.
VWAP/sigma ACCUMULATION is unaffected in both variants (always from day
start); only the event/entry window differs. WIN-B is the main table (full
extras for k=2: per-year, ER-tercile, MFE/MAE); WIN-A is printed as a compact
comparison one-liner right under each WIN-B line. Beta baselines are
recomputed per window (hour-matched controls restricted to the same allowed
hours) -- NOT shared between WIN-A/WIN-B. Applied uniformly to all three
instruments including BTC (24h market; the session logic is FX/gold-
flavored, kept for comparability per instruction).

TF ladder is 5m/15m/1h ONLY (4h has <=6 bars/day -- meaningless for intraday
VWAP, per spec). k in {1,2}; k=2 gets full extras (in WIN-B), k=1 one-liners
only (both windows).

Race/control-beta/dedupe/ER machinery shared with bandwalk_exit_bounce.py
via _race_common.py.
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
    race_matrix, process_cell, print_peryear, print_er_tercile, print_mfe_mae,
    print_summary_table,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

K_BY_TF = {"5m": 288, "15m": 96, "1h": 24}       # 24h worth of bars (no 4h -- spec excludes it)
RULE_BY_TF = {"5m": "5min", "15m": "15min", "1h": "1h"}

WIN_ELAPSED_H = {"WIN-A": 2, "WIN-B": 10}


def compute_vwap_sigma(d: pd.DataFrame, vol_flag_printed: list):
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    if "volume" in d.columns:
        w = d["volume"].astype(float)
        if not vol_flag_printed:
            print("  [weight] using load_mt5_csv 'volume' (tick_volume proxy, not true "
                  "traded volume) as the VWAP weight -- a volume/tick-volume column IS "
                  "present, so the equal-weight fallback path is not used.")
            vol_flag_printed.append(True)
    else:
        w = pd.Series(1.0, index=d.index)
        if not vol_flag_printed:
            print("  [weight] NO volume column found -- falling back to equal-weight "
                  "(w=1) cumulative mean of typical price.")
            vol_flag_printed.append(True)
    day = d.index.normalize()
    tpw = tp * w
    cum_tpw = tpw.groupby(day).cumsum()
    cum_w = w.groupby(day).cumsum()
    vwap = cum_tpw / cum_w
    dev2 = w * (tp - vwap) ** 2
    cum_dev2 = dev2.groupby(day).cumsum()
    sigma = np.sqrt(cum_dev2 / cum_w)
    return vwap.values, sigma.values


def find_vwap_events(close, high, low, upper, lower, mask):
    n = len(close)
    prev_close = np.roll(close, 1)
    prev_upper = np.roll(upper, 1)
    prev_lower = np.roll(lower, 1)
    ok = mask & ~np.isnan(upper) & ~np.isnan(prev_upper) & ~np.isnan(lower) & ~np.isnan(prev_lower)
    ok[0] = False
    raw_upper = ok & (prev_close < prev_upper) & (high >= upper) & (close < upper)
    raw_lower = ok & (prev_close > prev_lower) & (low <= lower) & (close > lower)
    return np.flatnonzero(raw_upper), np.flatnonzero(raw_lower)


def build_control_pool_masked(d, atr_v, K, cap, rng, mask):
    candidates = np.flatnonzero(~np.isnan(atr_v) & mask)
    candidates = candidates[candidates + K <= len(d) - 1]
    hours = d.index.hour.values[candidates]
    picked = []
    for h in range(24):
        h_idx = candidates[hours == h]
        if len(h_idx) > cap:
            h_idx = rng.choice(h_idx, size=cap, replace=False)
        picked.append(h_idx)
    return np.concatenate(picked) if picked else np.array([], dtype=int)


def process_instrument(instr, base_df, tfs, all_results):
    rng = np.random.default_rng(RNG_SEED)
    vol_flag_printed = []
    for tf in tfs:
        d = resample_ohlcv(base_df, RULE_BY_TF[tf])
        n = len(d)
        K = K_BY_TF[tf]
        if n < 200:
            print(f"{instr} {tf}: SKIP (only {n} bars after resample)")
            continue

        atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1)
        atr_v = atr.values
        er = efficiency_ratio(d["close"], K).values
        vwap_v, sigma_v = compute_vwap_sigma(d, vol_flag_printed)

        high = d["high"].values
        low = d["low"].values
        close_v = d["close"].values

        elapsed = (d.index - d.index.normalize()).values
        masks = {
            name: (elapsed >= np.timedelta64(h, "h"))
            for name, h in WIN_ELAPSED_H.items()
        }

        span_years = (d.index[-1] - d.index[0]).days / 365.25

        ctrl = {}
        for wname, mask in masks.items():
            ctrl_idx = build_control_pool_masked(d, atr_v, K, CAP_CTRL, rng, mask)
            _, cf_up, cf_dn, _, _, _, _ = race_matrix(high, low, ctrl_idx, atr_v, close_v, K)
            ctrl_idx_v = ctrl_idx[ctrl_idx + K <= n - 1]
            ctrl[wname] = (ctrl_idx_v, cf_up, cf_dn)

        print(f"--- {instr} {tf}  (n_bars={n}, span={span_years:.1f}yr, K={K}) ---")

        for k in [1, 2]:
            upper = vwap_v + k * sigma_v
            lower = vwap_v - k * sigma_v
            extras = (k == 2)

            for edge, fade_side, brk_side in [("UPPER", "short", "long"), ("LOWER", "long", "short")]:
                eventsB_by_win = {}
                for wname, mask in masks.items():
                    up_idx, lo_idx = find_vwap_events(close_v, high, low, upper, lower, mask)
                    raw_idx = up_idx if edge == "UPPER" else lo_idx
                    ev = dedupe(raw_idx, K)
                    eventsB_by_win[wname] = ev

                for side_name, side in [("fade", fade_side), ("breakout", brk_side)]:
                    ctrl_idx_v, cf_up, cf_dn = ctrl["WIN-B"]
                    cid_b = f"{instr:<4} {tf:<3} k={k} {edge:<5} {side_name:<9} [WIN-B]"
                    res_b = process_cell(cid_b, d, atr_v, high, low, close_v,
                                          eventsB_by_win["WIN-B"], side, K, span_years,
                                          ctrl_idx_v, cf_up, cf_dn, print_line=True)
                    all_results.append(res_b)
                    if extras:
                        print_peryear(res_b)
                        print_er_tercile(res_b, er)
                        if side_name == "fade":
                            print_mfe_mae(res_b)

                    ctrl_idx_v_a, cf_up_a, cf_dn_a = ctrl["WIN-A"]
                    cid_a = f"{instr:<4} {tf:<3} k={k} {edge:<5} {side_name:<9} [WIN-A]"
                    res_a = process_cell(cid_a, d, atr_v, high, low, close_v,
                                          eventsB_by_win["WIN-A"], side, K, span_years,
                                          ctrl_idx_v_a, cf_up_a, cf_dn_a, print_line=False)
                    all_results.append(res_a)
                    print(f"    WIN-A cmp: n={res_a['n']:<6d} N/yr={res_a['n_yr']:>7.1f}  "
                          f"win={res_a['win_pct']:>5.1f}%  beta={res_a['beta_pct']:>5.1f}%  "
                          f"delta={res_a['delta']:>+6.1f}pt")
        print()


def main():
    all_results = []

    print("=" * 100)
    print("PRE-REGISTERED PREDICTIONS:")
    print("  - trend instruments (gold/BTC) VWAP-band FADE ~ <= 0 (user suspicion + graveyard prior)")
    print("  - USDJPY fade small-positive = re-derivation of the known BB-class fade edge (positive control)")
    print("  - many cells printed; single-cell PASS is noise unless neighbors (k / window) agree (plateau)")
    print("AMENDMENT: WIN-B (server >=10:00, ~London-open-on) is the primary window; WIN-A (>=2h elapsed,")
    print("  includes Asia session) is a lenient comparison. Prior: WIN-B should show CLEANER separation")
    print("  (Asia low-liquidity chop is deadweight/noise, same 'dead-window' pattern as the gold 15m book leg).")
    print("=" * 100)
    print()

    print("=" * 100)
    print("GOLD  (vantage_xauusd_m5.csv, .loc['2018-09-14':]  -> 5m/15m/1h)")
    print("=" * 100)
    gold = load_mt5_csv(os.path.join(DATA_DIR, "vantage_xauusd_m5.csv"))
    gold = gold.loc["2018-09-14":]
    process_instrument("GOLD", gold, ["5m", "15m", "1h"], all_results)

    print("=" * 100)
    print("BTC   (vantage_btcusd_m15.csv, density-guarded -> 15m/1h)")
    print("=" * 100)
    btc = load_mt5_csv(os.path.join(DATA_DIR, "vantage_btcusd_m15.csv"))
    cnt = btc.groupby(btc.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    btc = btc[btc.index.date >= okd.index[0]]
    process_instrument("BTC", btc, ["15m", "1h"], all_results)

    print("=" * 100)
    print("USDJPY  (vantage_usdjpy_m1.csv -> 5m/15m ; vantage_usdjpy_h1.csv -> 1h)  [positive control]")
    print("=" * 100)
    ujm1 = load_mt5_csv(os.path.join(DATA_DIR, "vantage_usdjpy_m1.csv"))
    process_instrument("USDJPY", ujm1, ["5m", "15m"], all_results)
    ujh1 = load_mt5_csv(os.path.join(DATA_DIR, "vantage_usdjpy_h1.csv"))
    process_instrument("USDJPY", ujh1, ["1h"], all_results)

    print_summary_table(all_results, "SCRIPT 2 SUMMARY: VWAP +/- k*sigma band edge fade/breakout, all cells (WIN-A + WIN-B)")


if __name__ == "__main__":
    main()
