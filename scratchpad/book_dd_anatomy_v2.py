"""book_dd_anatomy_v2.py -- dissect the 6-leg book's REAL (intra-month) drawdown for the first
time, and settle whether booking R at ENTRY time (book_arbiter_v2.trade_book, today's convention)
vs at EXIT time (when the P&L actually lands in the account) changes the picture.

Frozen premises (not re-derived, imported):
  - scratchpad/btc_family_ext_throttle.py: build_base() -- 6 canonical legs (gold_bo, btc_bo_kama,
    btc_pull, gold15m, btc15m_L, btc15m_S). Re-derived HERE only to attach the "hold" column that
    build_base()/get_legs() drop; the trade-construction code itself is copy-pasted unchanged
    (see build_base_full() docstring below) and tied back leg-by-leg against build_base().
  - scratchpad/book_arbiter_v2.py: book_monthly_gen() (inv-vol/3%-budget weights, UNCHANGED) and
    trade_book() (entry-time trade-resolution arbiter, tie-back target CAGR 44.68%/DD 6.53%/
    CAGR-DD 6.84, n=1445, verified against a fresh --smoke run of that script before this one
    was written).
  - scratchpad/book_hh4h_weight_sweep.py: ROOT, OLD (3-leg), NEW (6-leg) basket lists.

Reconstructing exit timestamps (Step 0 premise, verified per-leg below):
  breakout_wave.run()/ema_pullback.run() both return a "hold" column = holding period in DAYS
  ((exit_bar_time - entry_bar_time).total_seconds()/86400), so exit_time = entry_time +
  pd.to_timedelta(hold, unit="D") recovers the exact decommission timestamp -- this is the exact
  inverse of how "hold" itself was computed, so there is no approximation/rounding beyond float64.
  research/portfolio_kama.get_legs() immediately slices to [["time","R"]], dropping "hold" before
  applying kama_gate_btc()/cycle_gate_pull(). Both gate fns only ever do `t[mask]` (boolean-index
  a DataFrame) -- see research/regime_gate_lab.at(). Row-masking a DataFrame does not touch other
  columns, so calling the SAME construction with the slice removed preserves "hold" through the
  gate untouched. get_legs_full() below is exactly research.portfolio_kama.get_legs() with the
  premature [["time","R"]] slices deleted -- no other line changed.

Equity-curve construction:
  - entry-accounted, trade-resolution: book_arbiter_v2.trade_book() as-is (existing, tied back).
  - entry-accounted, daily-bucketed:   per-leg weighted R bucketed by ENTRY calendar date, summed,
    compounded once per day-with-activity (same "sum-then-compound-once-per-period" convention as
    book_monthly_gen(), just at daily instead of monthly granularity). Computed as a clean
    isolation of "daily-bucket vs trade-resolution" from "entry vs exit timing" (both entry-dated).
  - exit-accounted, daily-bucketed:    identical construction, EXIT calendar date instead of entry.
    This is the version the spec card asks for (Step 0's literal instruction).
  - exit-accounted, trade-resolution:  same trades, sorted+compounded individually by exit
    timestamp (mirrors trade_book()'s method exactly, just re-timestamped) -- cross-check that the
    daily-bucketing choice itself isn't driving any entry-vs-exit difference.
  All four use the SAME trade set (legs' entries restricted to A0's own calendar window, exactly
  as trade_book() does) and the SAME per-leg weights w (from book_monthly_gen(legs, NEW)) -- only
  the accounting-time convention and bucketing granularity differ.

Run (smoke -- tie-back only, stops before Steps 1-3):
  .venv/bin/python scratchpad/book_dd_anatomy_v2.py --smoke
Run (full):
  .venv/bin/python scratchpad/book_dd_anatomy_v2.py
"""
import sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")

from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from short_mirror_15m import invert
from book_hh4h_weight_sweep import ROOT, OLD, NEW
from btc_family_ext_throttle import build_base as build_base_reference
from book_arbiter_v2 import book_monthly_gen, trade_book

BTC_LEGS = ["btc_bo_kama", "btc_pull", "btc15m_L", "btc15m_S"]
GOLD_LEGS = ["gold_bo", "gold15m"]


# =============================================================================
# leg construction, WITH hold preserved (same calls as get_legs()/build_base(), no logic changed)
# =============================================================================
def get_legs_full():
    """research.portfolio_kama.get_legs(), with the [["time","R"]] slices removed so "hold"
    survives kama_gate_btc()/cycle_gate_pull() (both are pure row-masks -- see module docstring)."""
    gold = run(resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv"), "1h"),
               SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                  "daily_sma": 150, "daily_slope_k": 10}))
    btc = run(resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h"),
              SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))
    btc_k = kama_gate_btc(btc)
    dbtc = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
    pb = run_pb(dbtc, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)
    pb = cycle_gate_pull(pb)
    return {"gold_bo": gold, "btc_bo_kama": btc_k, "btc_pull": pb}


def build_base_full():
    """Exactly btc_family_ext_throttle.build_base(), plus a parallel legs_hold dict (hold in
    days, same index as legs). Every construction line below is copy-pasted from build_base();
    tie-back to build_base() itself is asserted in main() before anything else runs."""
    with contextlib.redirect_stderr(io.StringIO()):
        full3 = get_legs_full()
        legs, legs_hold = {}, {}
        for k, t in full3.items():
            legs[k] = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))
            legs_hold[k] = pd.Series(t["hold"].values, index=pd.DatetimeIndex(t["time"]))

        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))
        legs_hold["gold15m"] = pd.Series(t["hold"].values, index=pd.DatetimeIndex(t["time"]))

        full = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv")
        d15 = resample(full.loc["2018-10-01":], "15min")
        inv = invert(d15); C = 2 * d15["high"].max()
        ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
        Rs = ts_["R"].values - 15.0 / ts_["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])
        legs_hold["btc15m_S"] = pd.Series(ts_["hold"].values[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5}))
        Rn = tL["R"].values - 15.0 / tL["risk"].values
        ei = d15.index.get_indexer(tL["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        base_w = np.where(tL["e_px"].values > pdh[ei], 1.0, 0.5)
        legs["btc15m_L"] = pd.Series(Rn * base_w, index=pd.DatetimeIndex(tL["time"]))
        legs_hold["btc15m_L"] = pd.Series(tL["hold"].values, index=pd.DatetimeIndex(tL["time"]))
    return legs, legs_hold


# =============================================================================
# daily-bucketed curve, either by entry date or by exit date (use_exit)
# =============================================================================
def per_leg_daily_by(legs, legs_hold, basket, w, st_ts, en_ts, use_exit):
    series = {}
    for k in basket:
        s, h = legs[k], legs_hold[k]
        mask = (s.index >= st_ts) & (s.index <= en_ts)
        entry_t = s.index[mask]
        wr = s.values[mask] * w[k]
        t_ = (entry_t + pd.to_timedelta(h.values[mask], unit="D")) if use_exit else entry_t
        df = pd.DataFrame({"d": t_.floor("D"), "wr": wr})
        series[k] = df.groupby("d")["wr"].sum()
    idx = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series.values()])))
    return pd.DataFrame({k: series[k].reindex(idx, fill_value=0.0) for k in basket})


def curve_stats(M):
    book_ret = M.sum(axis=1)
    eq = np.cumprod(1 + book_ret.values)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    days = (M.index[-1] - M.index[0]).total_seconds() / 86400.0
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd, eq, book_ret


def trade_res_by_exit(legs, legs_hold, basket, w, st_ts, en_ts):
    """Cross-check: same trades as trade_book(), re-timestamped by EXIT time, sorted +
    compounded individually (trade resolution, not daily-bucketed)."""
    parts = []
    for k in basket:
        s, h = legs[k], legs_hold[k]
        mask = (s.index >= st_ts) & (s.index <= en_ts)
        entry_t = s.index[mask]
        exit_t = entry_t + pd.to_timedelta(h.values[mask], unit="D")
        wr = s.values[mask] * w[k]
        parts.append(pd.Series(wr, index=exit_t))
    allr = pd.concat(parts).sort_index(kind="mergesort")
    eq = np.cumprod(1 + allr.values)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    days = (allr.index[-1] - allr.index[0]).total_seconds() / 86400.0
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd, len(allr)


# =============================================================================
# Step 1: simultaneous open-risk. Grid at hourly resolution (NOT daily-midnight snapshots) --
# several 15m legs hold sub-day (median hold well under 24h), so a once-a-day snapshot would
# systematically MISS most of their overlap with other legs. Distribution stats are still
# reported as requested; the grid is just fine enough not to undercount.
# =============================================================================
def open_risk_series(legs, legs_hold, basket, w, st_ts, en_ts, freq="1h"):
    leg_iv = {}
    max_exit = st_ts
    for k in basket:
        s, h = legs[k], legs_hold[k]
        mask = (s.index >= st_ts) & (s.index <= en_ts)
        entry_t = s.index[mask]
        exit_t = entry_t + pd.to_timedelta(h.values[mask], unit="D")
        order = np.argsort(entry_t.values)
        et, xt = entry_t.values[order], exit_t.values[order]
        leg_iv[k] = (et, xt)
        if len(xt):
            mx = pd.Timestamp(xt.max())
            if mx.tzinfo is None:
                mx = mx.tz_localize("UTC")
            max_exit = max(max_exit, mx)
    grid = pd.date_range(st_ts, max_exit, freq=freq, tz="UTC")
    gvals = grid.values
    total = np.zeros(len(grid))
    per_leg_frac = {}
    for k in basket:
        et, xt = leg_iv[k]
        idx = np.searchsorted(et, gvals, side="right") - 1
        openb = np.zeros(len(grid), dtype=bool)
        valid = idx >= 0
        openb[valid] = gvals[valid] < xt[idx[valid]]
        total += w[k] * openb
        per_leg_frac[k] = openb.mean()
    return grid, total, per_leg_frac


# =============================================================================
# Step 2: drawdown-episode finder off an arbitrary eq/dates series
# =============================================================================
def find_drawdowns(eq, dates):
    pk = np.maximum.accumulate(eq)
    ddser = (pk - eq) / pk
    n = len(eq)
    episodes = []
    i = 0
    while i < n:
        if ddser[i] <= 1e-12:
            i += 1
            continue
        start = i
        while i < n and ddser[i] > 1e-12:
            i += 1
        end = i - 1
        peak_idx = start - 1 if start > 0 else 0
        trough_idx = start + int(np.argmax(ddser[start:end + 1]))
        recovery_idx = i if i < n else None   # first index back at (or above) the prior peak
        episodes.append(dict(
            peak_idx=peak_idx, trough_idx=trough_idx, recovery_idx=recovery_idx, end_idx=end,
            peak_date=dates[peak_idx], trough_date=dates[trough_idx],
            recovery_date=dates[recovery_idx] if recovery_idx is not None else None,
            depth_pct=ddser[trough_idx] * 100))
    return episodes


def main(smoke=False):
    print("=" * 100)
    print("STEP 0a -- reconstruction fidelity: build_base_full() vs build_base() (reference), leg by leg")
    print("=" * 100)
    legs, legs_hold = build_base_full()
    with contextlib.redirect_stderr(io.StringIO()):
        ref_legs = build_base_reference()
    for k in NEW:
        a, b = legs[k], ref_legs[k]
        same_idx = a.index.equals(b.index)
        maxdiff = float(np.abs(a.values - b.values).max()) if same_idx else np.nan
        hold_n = legs_hold[k].notna().sum()
        print(f"  {k:<14} n={len(a):>5}  index-match(build_base)={same_idx}  max|R diff|={maxdiff:.3e}  "
              f"hold recovered for {hold_n}/{len(a)} trades  hold(d) med={legs_hold[k].median():.2f} "
              f"max={legs_hold[k].max():.1f}")
        if not same_idx or maxdiff > 1e-9:
            print(f"\n*** RECONSTRUCTION MISMATCH on leg '{k}' -- stopping, report this. ***")
            return
        if legs_hold[k].isna().any():
            print(f"\n*** hold NOT fully recoverable for leg '{k}' -- stopping, report this. ***")
            return
    print("\n  all 6 legs: exact tie-back to build_base(), hold fully recovered for every trade.\n")

    print("=" * 100)
    print("STEP 0b -- tie-back to book_arbiter_v2.trade_book() A0 (target: CAGR 44.68%, DD 6.53%, "
          "CAGR/DD 6.84, n=1445)")
    print("=" * 100)
    cNew, dNew, cdNew, allrA0, midxNew = trade_book(legs, NEW)
    _, _, _, _, w, midxW = book_monthly_gen(legs, NEW)
    print(f"  trade_book(legs, NEW) on THIS script's legs: CAGR={cNew:.2f}%  DD={dNew:.2f}%  "
          f"CAGR/DD={cdNew:.2f}  n={len(allrA0)}")
    if abs(cdNew - 6.84) > 0.02 or abs(dNew - 6.53) > 0.02 or abs(cNew - 44.68) > 0.05 or len(allrA0) != 1445:
        print("\n*** TIE-BACK MISMATCH -- stopping before proceeding. Report this. ***")
        return
    print("  tie-back PASSED.\n")

    st_ts = midxW[0].to_timestamp().tz_localize("UTC")
    en_ts = midxW[-1].to_timestamp(how="end").tz_localize("UTC")
    print(f"  A0 calendar window (by ENTRY time, from book_monthly_gen): {st_ts.date()} -> {en_ts.date()}  "
          f"({len(midxW)} months)")
    print("\n  per-leg inv-vol weight w_leg (account-% risked per 1R of that leg):")
    for k in NEW:
        print(f"    {k:<14} w={w[k]*100:.3f}%")

    if smoke:
        print("\n[--smoke] stopping here after tie-back checks (Steps 1-3 skipped).")
        return

    # =========================================================================
    # STEP 0 -- entry-accounted vs exit-accounted equity curves
    # =========================================================================
    print()
    print("=" * 100)
    print("STEP 0 -- entry-accounted vs exit-accounted book equity curve")
    print("=" * 100)
    M_entry = per_leg_daily_by(legs, legs_hold, NEW, w, st_ts, en_ts, use_exit=False)
    M_exit = per_leg_daily_by(legs, legs_hold, NEW, w, st_ts, en_ts, use_exit=True)
    c_e_daily, d_e_daily, cd_e_daily, eq_e_daily, _ = curve_stats(M_entry)
    c_x_daily, d_x_daily, cd_x_daily, eq_x_daily, _ = curve_stats(M_exit)
    c_x_trade, d_x_trade, cd_x_trade, n_x_trade = trade_res_by_exit(legs, legs_hold, NEW, w, st_ts, en_ts)

    print(f"  {'variant':<46}{'CAGR%':>8}{'maxDD%':>8}{'CAGR/DD':>9}{'span':>26}")
    print(f"  {'entry-accounted, trade-resolution (existing trade_book, tie-back)':<46}"
          f"{cNew:>8.2f}{dNew:>8.2f}{cdNew:>9.2f}"
          f"{'  ' + str(allrA0.index[0].date()) + ' -> ' + str(allrA0.index[-1].date()):>26}")
    print(f"  {'entry-accounted, daily-bucketed (isolation check)':<46}"
          f"{c_e_daily:>8.2f}{d_e_daily:>8.2f}{cd_e_daily:>9.2f}"
          f"{'  ' + str(M_entry.index[0].date()) + ' -> ' + str(M_entry.index[-1].date()):>26}")
    print(f"  {'exit-accounted, daily-bucketed  (SPEC-requested version)':<46}"
          f"{c_x_daily:>8.2f}{d_x_daily:>8.2f}{cd_x_daily:>9.2f}"
          f"{'  ' + str(M_exit.index[0].date()) + ' -> ' + str(M_exit.index[-1].date()):>26}")
    print(f"  {'exit-accounted, trade-resolution (cross-check)':<46}"
          f"{c_x_trade:>8.2f}{d_x_trade:>8.2f}{cd_x_trade:>9.2f}{'  n=' + str(n_x_trade):>26}")

    dd_gap = d_x_daily - dNew
    print(f"\n  maxDD gap (exit-accounted daily-bucketed minus entry-accounted trade-resolution) = "
          f"{dd_gap:+.2f} percentage points")
    use_exit_as_canon = abs(dd_gap) >= 0.5 or abs(cd_x_daily - cdNew) >= 0.3
    verdict = ("large enough to matter -> exit-accounted daily-bucketed curve is used as the CANON "
               "curve for Steps 1-3 below" if use_exit_as_canon else
               "small -> exit-accounted daily-bucketed curve is STILL used as canon for Steps 1-3 "
               "below (that is the object this investigation is about), but the difference from the "
               "entry-accounted convention is not large in this run")
    print(f"  verdict: {verdict}.")

    # =========================================================================
    # STEP 1 -- simultaneous open-risk
    # =========================================================================
    print()
    print("=" * 100)
    print("STEP 1 -- simultaneous open-risk (Sum of w_leg over trades open at time t, hourly grid, "
          "entries restricted to A0's window)")
    print("=" * 100)
    grid, total_risk, per_leg_frac = open_risk_series(legs, legs_hold, NEW, w, st_ts, en_ts, freq="1h")
    pct = total_risk * 100
    print(f"  n grid points (hourly) = {len(grid)}   span {grid[0].date()} -> {grid[-1].date()}")
    print(f"  median={np.median(pct):.3f}%  mean={np.mean(pct):.3f}%  std={np.std(pct):.3f}%  "
          f"p95={np.percentile(pct, 95):.3f}%  max={np.max(pct):.3f}%")
    over3 = (pct > 3.0).mean() * 100
    print(f"  fraction of hours with total open-risk > 3.00% (nominal budget) = {over3:.1f}%")
    print("\n  per-leg fraction of time open (of this leg's own presence, not weighted):")
    for k in NEW:
        print(f"    {k:<14} open {per_leg_frac[k]*100:5.1f}% of hours   w_leg={w[k]*100:.3f}%")

    # =========================================================================
    # STEP 2 -- drawdown anatomy off the exit-accounted daily-bucketed curve
    # =========================================================================
    print()
    print("=" * 100)
    print("STEP 2 -- top-5 drawdown episodes (exit-accounted daily-bucketed book curve)")
    print("=" * 100)
    episodes = find_drawdowns(eq_x_daily, M_exit.index)
    top5 = sorted(episodes, key=lambda e: -e["depth_pct"])[:5]

    corr_full = M_exit[NEW].corr()
    btc_c_full = M_exit[BTC_LEGS].sum(axis=1)
    gold_c_full = M_exit[GOLD_LEGS].sum(axis=1)
    corr_btc_gold_full = btc_c_full.corr(gold_c_full)

    for rank, ep in enumerate(top5, 1):
        pk_d, tr_d = ep["peak_date"], ep["trough_date"]
        rec_d = ep["recovery_date"]
        underwater_end = rec_d if rec_d is not None else M_exit.index[ep["end_idx"]]
        dur_days = (tr_d - pk_d).days
        uw_days = (underwater_end - pk_d).days
        rec_str = str(rec_d.date()) if rec_d is not None else "NOT RECOVERED by end of sample"
        print(f"\n  #{rank}  depth={ep['depth_pct']:.2f}%   peak={pk_d.date()}  trough={tr_d.date()}  "
              f"recovery={rec_str}")
        print(f"      peak->trough duration = {dur_days}d   days underwater (peak->recovery/end) = {uw_days}d")
        window_slice = M_exit.loc[pk_d:tr_d, NEW]
        contrib = window_slice.sum() * 100
        print("      per-leg weighted contribution during peak->trough (account %):")
        for k in NEW:
            print(f"        {k:<14} {contrib[k]:+.3f}%")
        if len(window_slice) >= 3:
            corr_ep = window_slice.corr()
            btc_c_ep = window_slice[BTC_LEGS].sum(axis=1)
            gold_c_ep = window_slice[GOLD_LEGS].sum(axis=1)
            corr_bg_ep = btc_c_ep.corr(gold_c_ep) if window_slice[BTC_LEGS].sum(axis=1).std() > 0 and \
                window_slice[GOLD_LEGS].sum(axis=1).std() > 0 else np.nan
        else:
            corr_ep, corr_bg_ep = None, np.nan
        print(f"      n days in peak->trough window = {len(window_slice)}")
        print("      leg-pair correlation, episode vs full-period (only pairs with episode n>=3 shown):")
        if corr_ep is not None:
            for i1 in range(len(NEW)):
                for i2 in range(i1 + 1, len(NEW)):
                    a_, b_ = NEW[i1], NEW[i2]
                    print(f"        {a_:<12} x {b_:<12}  episode={corr_ep.loc[a_, b_]:+.2f}   "
                          f"full-period={corr_full.loc[a_, b_]:+.2f}")
            print(f"      BTC-composite x gold-composite  episode={corr_bg_ep:+.2f}   "
                  f"full-period={corr_btc_gold_full:+.2f}")
        else:
            print("        (episode too short (<3 days) for a meaningful correlation -- skipped)")

    # =========================================================================
    # STEP 3 -- who digs, who cushions
    # =========================================================================
    print()
    print("=" * 100)
    print("STEP 3 -- per-leg contribution: book-underwater days vs book-above-water days "
          "(full sample, exit-accounted daily curve)")
    print("=" * 100)
    pk_full = np.maximum.accumulate(eq_x_daily)
    dd_full = (pk_full - eq_x_daily) / pk_full
    underwater_mask = dd_full > 1e-9
    print(f"  n days underwater = {underwater_mask.sum()} / {len(underwater_mask)} "
          f"({100*underwater_mask.mean():.1f}%)")
    print(f"  {'leg':<14}{'contrib underwater (%)':>24}{'contrib above-water (%)':>25}{'total (%)':>12}")
    for k in NEW:
        uw = M_exit[k].values[underwater_mask].sum() * 100
        aw = M_exit[k].values[~underwater_mask].sum() * 100
        print(f"  {k:<14}{uw:>24.3f}{aw:>25.3f}{uw+aw:>12.3f}")

    print("\n  always-negative-in-top5 / always-positive-in-top5 check:")
    for k in NEW:
        signs = []
        for ep in top5:
            v = M_exit.loc[ep["peak_date"]:ep["trough_date"], k].sum()
            signs.append(v)
        n_neg = sum(1 for v in signs if v < 0)
        n_pos = sum(1 for v in signs if v > 0)
        print(f"    {k:<14} negative in {n_neg}/5 episodes, positive in {n_pos}/5   "
              f"values=[{', '.join(f'{v*100:+.2f}%' for v in signs)}]")

    print("\n  BTC-family (4 legs) composite vs gold-family (2 legs) composite:")
    print(f"    full-period daily corr = {corr_btc_gold_full:+.3f}")
    for rank, ep in enumerate(top5, 1):
        w_ = M_exit.loc[ep["peak_date"]:ep["trough_date"]]
        if len(w_) >= 3:
            b_ = w_[BTC_LEGS].sum(axis=1); g_ = w_[GOLD_LEGS].sum(axis=1)
            c_ = b_.corr(g_) if b_.std() > 0 and g_.std() > 0 else np.nan
            print(f"    episode #{rank} (peak {ep['peak_date'].date()}) corr = {c_:+.3f}   "
                  f"BTC-composite contrib={b_.sum()*100:+.3f}%   gold-composite contrib={g_.sum()*100:+.3f}%")
        else:
            print(f"    episode #{rank}: too short for correlation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
