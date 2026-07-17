"""scalp_null_bracket.py -- NULL-DISTRIBUTION measurement for "high-winrate near-target
scalp" ideas (RR 0.5-1.0). This is a ruler, not a strategy test: it measures what win
rate / meanR you get by entering RANDOMLY, so a later real signal's numbers can be
judged against this baseline instead of against 50%.

Frozen spec card, revision 2 (2026-07-14). Rev.2 deltas from rev.1 (coordinator
correction, applied BEFORE any production run):
  1. Exit/settlement resolution is ALWAYS on the symbol's 5-min bars, regardless of
     which TF the entry CANDIDATE comes from. Only the entry-candidate timestamps are
     selected at TF granularity (5min: every 5-min bar; 15min: every 15-min-boundary
     bar). Resolving stop/target on the resampled (coarse) bar was rev.1's bug: coarser
     bars manufacture more same-bar TP+SL conflicts, and the conservative same-bar rule
     (SL wins) then over-penalizes win rate purely as a resolution artifact, not a real
     effect. Fixed here by always walking 5-min bars for the touch/no-touch decision.
  2. Dropped the 1h arm ("1H is not a scalp" -- coordinator). Only 5min/15min remain.
  3. Added a per-year null breakdown (200 draws x 100 samples/draw, median+std), to
     show that a drifting instrument's null win rate is NOT flat across regimes (e.g.
     BTC bull years vs BTC 2022).
  4. Added holding-time-to-resolution (median, IQR 25/75) per cell, in hours, so a
     "scalp" claim can be checked against how long the bracket actually stays open.

Rules (unchanged core, all pre-registered, no lookahead):
  - Randomly choose an entry-candidate bar (uniform); entry price e = open of the NEXT
    eligible bar (i+1 at the entry TF's own grid -- never a same-bar fill).
  - long: TP = e+T, SL = e-S.  short: TP = e-T, SL = e+S.  S = T / RR.
  - Walk 5-min bars one at a time, starting at the entry bar itself (the entry bar's own
    remaining excursion after the open counts). Whichever level is touched first wins;
    if the SAME 5-min bar's high and low touch both levels, SL wins (conservative).
  - Max hold = 3 CALENDAR days (not bar count) from the entry bar's timestamp. If
    neither level is touched by then, exit at the close of the last in-window 5-min bar.
  - R = (realized_price_diff - cost) / S uniformly (realized_diff = +T on a TP touch,
    -S on an SL touch, signed close-vs-entry diff on a timeout close). This already
    matches the explicit override "SL hit -> R = -(S+cost)/S" since realized_diff=-S
    exactly on an SL touch.
  - No signal, no filter, no direction bias: this IS the null.
  - Fixed seed, deterministic iteration order -> reproducible bootstrap draws.

Instruments / TFs / brackets (all fixed, no sweep beyond the specified grid):
  gold   data/vantage_xauusd_m5.csv  sliced .loc["2018-09-14":]   T in {5.0,7.5,10.0} $/oz   cost=$0.30 RT
  usdjpy data/vantage_usdjpy_m5.csv  full period (see caveat printed at runtime: pre-1999
         is ~250 bars/yr = daily data mislabeled m5, same pattern as the documented gold H1
         issue; kept because the spec says "full period" -- flagged, not silently sliced)
  btc    data/vantage_btcusd_m5.csv  sliced .loc["2018-10-01":] (file only goes back to
         2019-01-01, so this is a no-op in practice -- reported at runtime)
  RR in {0.5, 0.7, 1.0}; side in {long, short}; TF (entry-candidate grid) in {5min, 15min}.

Cells = 3 symbols x 2 TF x 2 side x 3 T x 3 RR = 108.

Reuses: src.data_loader.load_mt5_csv, breakout_wave.resample (only to get the official
15-min bar-boundary timestamps for the 15min entry-candidate grid -- the actual touch/
no-touch walk never uses resampled OHLC, only raw 5-min bars, per fix #1 above).
NOTE on breakout_wave.resample: it treats rule in {"1h","h1",""} as a no-op sentinel
("already this TF, don't touch"), which would silently skip resampling here (our base
is 5-min, not 1h). We never hit that branch: our two TF strings are "5min"/"15min",
which fall through to the real OHLC-resample branch, so reuse is safe as-is.

Usage:
  .venv/bin/python scratchpad/scalp_null_bracket.py --smoke
  .venv/bin/python scratchpad/scalp_null_bracket.py 2>&1 | tee scratchpad/out_scalp_null_bracket.txt
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, like research/*.py

from src.data_loader import load_mt5_csv
from breakout_wave import resample

WINDOW_NS = 3 * 24 * 3600 * 10**9  # 3 calendar days, in ns (TF-independent per spec)
NS_PER_HOUR = 3.6e12

SYMBOLS = [
    # name, csv, slice_start (None = full period), T list, RT cost (price units)
    ("gold",   "data/vantage_xauusd_m5.csv", "2018-09-14", [5.0, 7.5, 10.0], 0.30),
    ("usdjpy", "data/vantage_usdjpy_m5.csv", None,          [0.50, 0.75, 1.00], 0.009),
    ("btc",    "data/vantage_btcusd_m5.csv", "2018-10-01",  [500.0, 750.0, 1000.0], 15.0),
]
RRS = [0.5, 0.7, 1.0]
TFS = ["5min", "15min"]
SIDES = [("long", 1), ("short", -1)]

BASE_SEED = 20260714


@numba.njit(cache=True)
def scan_entries(open_, high, low, close, time_ns, entry_pos, T, S, side, cost, window_ns):
    n_e = entry_pos.shape[0]
    n = open_.shape[0]
    R = np.empty(n_e, dtype=np.float64)
    hit = np.empty(n_e, dtype=np.bool_)
    hold_h = np.empty(n_e, dtype=np.float64)
    for k in range(n_e):
        pos = entry_pos[k]
        e = open_[pos]
        t_entry = time_ns[pos]
        deadline = t_entry + window_ns
        if side == 1:
            tp = e + T
            sl = e - S
        else:
            tp = e - T
            sl = e + S
        j = pos
        last_j = pos
        resolved = False
        outcome = 0.0
        while j < n and time_ns[j] <= deadline:
            hh = high[j]
            ll = low[j]
            if side == 1:
                sl_touch = ll <= sl
                tp_touch = hh >= tp
            else:
                sl_touch = hh >= sl
                tp_touch = ll <= tp
            if sl_touch:            # conservative: SL wins same-bar conflicts
                outcome = -S
                resolved = True
                last_j = j
                break
            if tp_touch:
                outcome = T
                resolved = True
                last_j = j
                break
            last_j = j
            j += 1
        if not resolved:
            if side == 1:
                outcome = close[last_j] - e
            else:
                outcome = e - close[last_j]
        R[k] = (outcome - cost) / S
        hit[k] = resolved
        hold_h[k] = (time_ns[last_j] - t_entry) / NS_PER_HOUR
    return R, hit, hold_h


def entry_positions_for_tf(df5: pd.DataFrame, tf: str) -> np.ndarray:
    """5-min-index positions of eligible ENTRY bars for the given candidate-TF grid.
    Entry price is always df5['open'] at the returned position (i+1 already applied:
    these ARE the "next eligible bar" positions, not the candidate/signal bars)."""
    if tf == "5min":
        return np.arange(1, len(df5), dtype=np.int64)
    dfT = resample(df5, tf)          # only used to get the official TF-boundary timestamps
    idx = df5.index.get_indexer(dfT.index[1:])
    idx = idx[idx >= 0]
    return idx.astype(np.int64)


def bootstrap_stats(R: np.ndarray, rng: np.random.Generator, n_draws: int, n_per_draw: int):
    n_e = len(R)
    draw_idx = rng.integers(0, n_e, size=(n_draws, n_per_draw))
    Rd = R[draw_idx]
    winrates = (Rd > 0).mean(axis=1)
    meanRs = Rd.mean(axis=1)
    pos_sum = np.where(Rd > 0, Rd, 0.0).sum(axis=1)
    neg_sum = np.where(Rd < 0, -Rd, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        PFs = np.where(neg_sum > 0, pos_sum / neg_sum, np.nan)
    return winrates, meanRs, PFs


def fmt_pct(x):
    return f"{x*100:.1f}%"


def run(args):
    seed_counter = [BASE_SEED]

    def next_rng():
        seed_counter[0] += 1
        return np.random.default_rng(seed_counter[0])

    symbols = SYMBOLS
    tfs = TFS
    rrs = RRS
    if args.smoke:
        symbols = [("usdjpy", "data/vantage_usdjpy_m5.csv", None, [0.75], 0.009)]
        tfs = ["5min"]
        rrs = [0.7]

    n_draws_full = 50 if args.smoke else 1000
    n_per_full = 50 if args.smoke else 200
    n_draws_yr = 20 if args.smoke else 200
    n_per_yr = 30 if args.smoke else 100

    main_rows = []
    yearly_rows = []

    for name, csv, slice_start, T_list, cost in symbols:
        t0 = _time.time()
        df_raw = load_mt5_csv(csv)
        actual_start_full, actual_end_full = df_raw.index[0], df_raw.index[-1]
        if slice_start is not None:
            df5 = df_raw.loc[slice_start:]
        else:
            df5 = df_raw
        print(f"# {name}: file span {actual_start_full.date()} -> {actual_end_full.date()}; "
              f"used span (after slice) {df5.index[0].date()} -> {df5.index[-1].date()}  "
              f"({len(df5):,} 5-min bars)", file=sys.stderr)

        open_ = df5["open"].values.astype(np.float64)
        high = df5["high"].values.astype(np.float64)
        low = df5["low"].values.astype(np.float64)
        close = df5["close"].values.astype(np.float64)
        time_ns = df5.index.values.astype("int64")

        for tf in tfs:
            entry_pos = entry_positions_for_tf(df5, tf)
            entry_years = df5.index[entry_pos].year.values

            for side_name, side_val in SIDES:
                for T in T_list:
                    for RR in rrs:
                        S = T / RR
                        R, hit, hold_h = scan_entries(
                            open_, high, low, close, time_ns, entry_pos,
                            T, S, side_val, cost, WINDOW_NS,
                        )
                        rng = next_rng()
                        winrates, meanRs, PFs = bootstrap_stats(R, rng, n_draws_full, n_per_full)
                        theo_wr = 1.0 / (1.0 + RR)
                        breakeven_wr = (S + cost) / (S + T)
                        timeout_rate = 1.0 - hit.mean()

                        row = dict(
                            symbol=name, tf=tf, side=side_name, T=T, RR=RR, S=S,
                            n_entries=len(R),
                            theo_wr=theo_wr, breakeven_wr=breakeven_wr,
                            wr_median=np.median(winrates), wr_std=winrates.std(),
                            wr_p5=np.percentile(winrates, 5), wr_p95=np.percentile(winrates, 95),
                            mr_median=np.median(meanRs),
                            mr_p5=np.percentile(meanRs, 5), mr_p95=np.percentile(meanRs, 95),
                            pf_median=np.nanmedian(PFs),
                            timeout_rate=timeout_rate,
                            hold_median=np.median(hold_h),
                            hold_p25=np.percentile(hold_h, 25),
                            hold_p75=np.percentile(hold_h, 75),
                        )
                        main_rows.append(row)
                        print(
                            f"{name} | {tf} | {side_name} | T={T} | RR={RR} | S={S:.4f} | "
                            f"n={len(R)} | theo_wr={fmt_pct(theo_wr)} | "
                            f"breakeven_wr={fmt_pct(breakeven_wr)} | "
                            f"wr_med={fmt_pct(row['wr_median'])} | wr_std={row['wr_std']*100:.2f}pt | "
                            f"wr_p5={fmt_pct(row['wr_p5'])} | wr_p95={fmt_pct(row['wr_p95'])} | "
                            f"meanR_med={row['mr_median']:+.4f} | meanR_p5={row['mr_p5']:+.4f} | "
                            f"meanR_p95={row['mr_p95']:+.4f} | PF_med={row['pf_median']:.3f} | "
                            f"timeout_rate={fmt_pct(timeout_rate)} | "
                            f"hold_med_h={row['hold_median']:.2f} | hold_p25_h={row['hold_p25']:.2f} | "
                            f"hold_p75_h={row['hold_p75']:.2f}"
                        )

                        if not args.smoke:
                            for yr in sorted(set(entry_years)):
                                mask = entry_years == yr
                                R_yr = R[mask]
                                if len(R_yr) < 30:
                                    continue
                                rng_yr = next_rng()
                                wr_yr, mr_yr, _ = bootstrap_stats(R_yr, rng_yr, n_draws_yr, n_per_yr)
                                yrow = dict(
                                    symbol=name, tf=tf, side=side_name, T=T, RR=RR, year=int(yr),
                                    n_entries=len(R_yr),
                                    wr_median=np.median(wr_yr), wr_std=wr_yr.std(),
                                    mr_median=np.median(mr_yr),
                                    theo_wr=theo_wr,
                                )
                                yearly_rows.append(yrow)
                                print(
                                    f"YR {name} | {tf} | {side_name} | T={T} | RR={RR} | "
                                    f"year={yr} | n={len(R_yr)} | theo_wr={fmt_pct(theo_wr)} | "
                                    f"wr_med={fmt_pct(yrow['wr_median'])} | "
                                    f"wr_std={yrow['wr_std']*100:.2f}pt | "
                                    f"meanR_med={yrow['mr_median']:+.4f}"
                                )
        print(f"# {name} done in {_time.time()-t0:.1f}s", file=sys.stderr)

    main_df = pd.DataFrame(main_rows)
    main_df["wr_minus_breakeven_pt"] = (main_df["wr_median"] - main_df["breakeven_wr"]) * 100
    print("\n=== SUMMARY: 実測ランダム勝率(中央値) - 損益ゼロ勝率  (percentage points) ===")
    for _, r in main_df.iterrows():
        print(f"{r['symbol']} | {r['tf']} | {r['side']} | T={r['T']} | RR={r['RR']} | "
              f"wr_minus_breakeven={r['wr_minus_breakeven_pt']:+.2f}pt")

    main_df.to_csv("scratchpad/scalp_null_bracket_main.csv", index=False)
    if yearly_rows:
        pd.DataFrame(yearly_rows).to_csv("scratchpad/scalp_null_bracket_yearly.csv", index=False)

    # pre-registered sanity checks
    print("\n=== 事前登録サニティチェック ===")
    uj = main_df[main_df["symbol"] == "usdjpy"]
    if len(uj):
        dev = (uj["wr_median"] - uj["theo_wr"]).abs() * 100
        print(f"USDJPY: |実測中央値 - 理論値| max={dev.max():.2f}pt, mean={dev.mean():.2f}pt "
              f"(基準: ±2pt以内)")
    btc = main_df[main_df["symbol"] == "btc"]
    if len(btc):
        btc_long = btc[btc["side"] == "long"]
        btc_short = btc[btc["side"] == "short"]
        print(f"BTC long:  wr_median - theo_wr = {(btc_long['wr_median']-btc_long['theo_wr']).mean()*100:+.2f}pt "
              f"(基準: 上振れするはず)")
        print(f"BTC short: wr_median - theo_wr = {(btc_short['wr_median']-btc_short['theo_wr']).mean()*100:+.2f}pt "
              f"(基準: 下振れするはず)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
