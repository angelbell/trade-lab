"""book_arbiter_v2.py -- re-judge today's book decisions on a TRADE-RESOLUTION equity curve.

Frozen premise (not re-derived here): book()/book_gen() in
experiments/book_hh4h_weight_sweep.py / experiments/btc_family_ext_throttle.py -- the arbiter used
for every book verdict made today (RR4.5 adopted, HH4H sizing rejected, the 4-week-extension
throttle "improving" 12.03 -> 12.70) -- collapsed each leg's R into MONTHLY sums before building
the equity curve (same convention as research/portfolio_alloc.py: cagr_dd_monthly), then measured
maxDD off that MONTHLY curve. That gave book maxDD = 3.62% (a single month, 2019-06 -> 2019-07)
at book CAGR/DD = 12.03 -- not a believable drawdown for a 6-leg book at 3% total risk, and every
"12.03 vs 12.70" comparison was really a comparison of one month's composition.

This script rebuilds every leg EXACTLY as before (reused, not reimplemented) but merges ALL legs'
individual trades into one chronological sequence and compounds trade-by-trade:
    eq = cumprod(1 + w_leg * R_trade)
maxDD/CAGR are read off THAT curve. The inv-vol / 3%-total-risk WEIGHTING formula is UNCHANGED
from today (not the thing in question) -- only the curve construction (monthly-bucket vs.
trade-by-trade) changes.

Reused (imported, not reimplemented):
  - experiments/btc_family_ext_throttle.py: build_base() (6 canonical legs), throttle(),
    ret4w_daily(), book_gen() (tie-back target)
  - experiments/book_hh4h_weight_sweep.py: ROOT, OLD/NEW basket lists
  - breakout_wave.run/resample/swings_zigzag, trend_leg_aging.atr -- for the RR ladder + HH4H
    rebuild (same construction as book_hh4h_weight_sweep.py's main())
  - research/portfolio_alloc.py -- independent second opinion on the 3-leg reference number

Run (full):  .venv/bin/python experiments/book_arbiter_v2.py
Run (smoke, fewer bootstrap draws): .venv/bin/python experiments/book_arbiter_v2.py --smoke
"""
import sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")

from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, swings_zigzag
from trend_leg_aging import atr as atr_fn
from book_hh4h_weight_sweep import ROOT, OLD, NEW
from btc_family_ext_throttle import build_base, throttle, book_gen

BTC15M_CSV = f"{ROOT}/data/vantage_btcusd_m15.csv"


# ---------------------------------------------------------------------------
# OLD arbiter (today's convention): identical formula to book()/book_gen(), generalized to an
# arbitrary leg `basket` (book_gen()/book() hardcode basket=NEW). `window`, if given, forces the
# (start-period, end-period) instead of deriving it from this call's own legs -- used to pin every
# NEW-basket arm to the SAME calendar span as A0 for fair/paired comparison.
# ---------------------------------------------------------------------------
def book_monthly_gen(legs, basket, overrides=None, window=None):
    L = dict(legs)
    if overrides:
        L.update(overrides)
    mon = {k: L[k].groupby(L[k].index.to_period("M")).sum() for k in basket}
    if window is None:
        st = max(mon[k].index.min() for k in basket)
        en = min(mon[k].index.max() for k in basket)
    else:
        st, en = window
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: mon[k].reindex(midx, fill_value=0.0) for k in basket})
    sig = M.std(); w = (1.0 / sig[basket]); w = w / w.sum() * 0.03
    port = (M[basket] * w).sum(axis=1)
    eq = np.cumprod(1 + port.values)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (12 / len(port)) - 1) * 100
    return cagr, dd, cagr / dd, port, w, midx


# ---------------------------------------------------------------------------
# NEW arbiter: SAME leg construction + SAME inv-vol/3%-budget weights (reuses book_monthly_gen's
# weight calc so the two arbiters are apples-to-apples on weighting), but the equity curve is all
# legs' individual trades merged into one chronological sequence, compounded trade-by-trade.
# ---------------------------------------------------------------------------
def trade_book(legs, basket, overrides=None, window=None):
    L = dict(legs)
    if overrides:
        L.update(overrides)
    _, _, _, _, w, midx = book_monthly_gen(legs, basket, overrides, window)
    st_ts = midx[0].to_timestamp().tz_localize("UTC")
    en_ts = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    parts = []
    for k in basket:
        s = L[k]
        s = s[(s.index >= st_ts) & (s.index <= en_ts)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    allr = pd.concat(parts).sort_index(kind="mergesort")   # stable: same-timestamp ties keep leg order
    eq = np.cumprod(1 + allr.values)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    days = (allr.index[-1] - allr.index[0]).total_seconds() / 86400.0
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd, allr, midx


# ---------------------------------------------------------------------------
# RR ladder for btc15m_L -- identical construction to btc_family_ext_throttle.build_base()'s
# btc15m_L block, rr overridable (build_base()/adopted leg = rr 4.5).
# ---------------------------------------------------------------------------
def build_btc15m_L_raw(d15, rr):
    tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                     "pullback_frac": 0.3, "rr": rr}))
    Rn = tL["R"].values - 15.0 / tL["risk"].values
    ei = d15.index.get_indexer(tL["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    base_w = np.where(tL["e_px"].values > pdh[ei], 1.0, 0.5)
    return pd.Series(Rn * base_w, index=pd.DatetimeIndex(tL["time"]))


# ---------------------------------------------------------------------------
# HH4H+PDH ladder sizing -- identical construction to book_hh4h_weight_sweep.py's main(): above
# 4H swing high AND above PDH = 1.0, one of the two = 0.5, neither = weak. 4H swing high shifted
# one 4H bar before being mapped to 15m (no lookahead).
# ---------------------------------------------------------------------------
def build_hh4h_arms(d15, weak_list):
    tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                     "pullback_frac": 0.3, "rr": 4.5}))
    Rn = tL["R"].values - 15.0 / tL["risk"].values
    ei = d15.index.get_indexer(tL["time"])
    idx = pd.DatetimeIndex(tL["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    h4 = d15.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)
    sh = pd.Series(np.nan, index=h4.index)
    for (ci, pi, px, kind) in swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0):
        if kind == +1:
            sh.iloc[ci] = px
    hh = sh.ffill().shift(1).reindex(d15.index, method="ffill").values   # no lookahead
    e = tL["e_px"].values
    above_pdh = e > pdh[ei]
    above_hh = np.isfinite(hh[ei]) & (e > hh[ei])
    out = {}
    for wk in weak_list:
        w_arr = np.where(above_hh & above_pdh, 1.0, np.where(above_hh | above_pdh, 0.5, wk))
        out[wk] = pd.Series(Rn * w_arr, index=idx)
    return out


# ---------------------------------------------------------------------------
# trade-resolution circular block bootstrap: months are the resampling unit (paired across arms,
# same resampled months), but the trades WITHIN a month keep their own chronological order; the
# resampled month sequence's trades are concatenated and compounded trade-by-trade.
# CAGR annualization uses month-count (12/months), the same convention as the existing monthly
# bootstraps (book_bootstrap_arbiter.cdd, btc_family_ext_throttle.bootstrap_table) -- a resampled
# path has no real calendar span, so it is annualized by nominal month count, not elapsed days.
# ---------------------------------------------------------------------------
def month_buckets(allr, midx):
    per = allr.index.to_period("M")
    vals = allr.values
    return [vals[per == p] for p in midx]


def cdd_trade(merged, months):
    if len(merged) == 0:
        return np.nan
    eq = np.cumprod(1 + merged)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max()
    if dd <= 0:
        return np.nan
    cagr = eq[-1] ** (12 / months) - 1
    return cagr / dd


def bootstrap_trade(arm_buckets, months, ndraw, base_name, seed=20260713):
    rng = np.random.default_rng(seed)
    names = list(arm_buckets.keys())
    out = {}
    for blk in (1, 3, 6, 12):
        nb = int(np.ceil(months / blk))
        D = {k: [] for k in names}
        for _ in range(ndraw):
            st = rng.integers(0, months, nb)
            pos = np.concatenate([(np.arange(s, s + blk) % months) for s in st])[:months]
            for k in names:
                buckets = arm_buckets[k]
                merged = np.concatenate([buckets[p] for p in pos]) if len(pos) else np.array([])
                D[k].append(cdd_trade(merged, months))
        base_arr = np.array(D[base_name])
        row = {}
        for k in names:
            a = np.array(D[k])
            row[k] = (np.nanmedian(a), np.nanmean(a > base_arr) * 100)
        out[blk] = row
    return out


def main(smoke=False):
    ndraw = 200 if smoke else 2000

    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()
        d15 = resample(load_mt5_csv(BTC15M_CSV).loc["2018-10-01":], "15min")

    # =========================================================================
    # SANITY: A0 under both arbiters. New DD must exceed the old monthly DD (3.62%), else stop.
    # =========================================================================
    cGorig, dGorig, cdGorig, _ = book_gen(legs)
    cOld, dOld, cdOld, _, wA0, midxA0 = book_monthly_gen(legs, NEW)
    cNew, dNew, cdNew, allrA0, midxNew = trade_book(legs, NEW)

    print("=" * 100)
    print("SANITY CHECK -- A0 (current 6-leg book, btc15m_L = RR4.5/4h-gate/PDH-soft-0.5)")
    print("=" * 100)
    print(f"  book_gen() [ORIGINAL, imported, monthly-DD arbiter]:  CAGR/DD={cdGorig:.2f}  DD={dGorig:.2f}%  (target 12.03)")
    print(f"  book_monthly_gen(NEW) [generalized, same formula]:    CAGR/DD={cdOld:.2f}  DD={dOld:.2f}%  (must match line above)")
    print(f"  trade_book(NEW) [NEW trade-resolution arbiter]:       CAGR/DD={cdNew:.2f}  CAGR={cNew:.2f}%  DD={dNew:.2f}%")
    print(f"  n trades merged (A0, all 6 legs) = {len(allrA0)}   calendar span = "
          f"{allrA0.index[0].date()} -> {allrA0.index[-1].date()}")

    if abs(cdGorig - 12.03) > 0.05 or abs(cdOld - cdGorig) > 1e-6:
        print("\n*** TIE-BACK MISMATCH (old arbiter) -- stopping. Report this. ***")
        return
    if dNew <= dOld:
        print(f"\n*** BUG: new trade-resolution DD ({dNew:.2f}%) is NOT larger than the old "
              f"monthly DD ({dOld:.2f}%). Stopping -- report this before proceeding. ***")
        return
    print("\n  sanity PASSED: new DD > old monthly DD, as expected.\n")

    # =========================================================================
    # Build all arm overrides (NEW-basket family: RR ladder, HH4H, ext-veto)
    # =========================================================================
    rr_grid = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    arm_overrides = {}
    arm_overrides["A0 base (RR4.5, adopted)"] = None
    for rr in rr_grid:
        arm_overrides[f"RR{rr}"] = {"btc15m_L": build_btc15m_L_raw(d15, rr)}
    hh4h = build_hh4h_arms(d15, [0.25, 0.5])
    arm_overrides["HH4H+PDH ladder, weak=0.25"] = {"btc15m_L": hh4h[0.25]}
    arm_overrides["HH4H+PDH ladder, weak=0.5"] = {"btc15m_L": hh4h[0.5]}
    for q, w in ((0.75, 0.5), (0.90, 0.0), (0.60, 0.0)):
        ov = throttle(legs, ["btc15m_L"], q, w)
        arm_overrides[f"ext-veto q{q} w={w}"] = ov

    # =========================================================================
    # TABLE A -- old (monthly) vs new (trade) arbiter, forced to A0's own calendar window
    # =========================================================================
    window = (midxNew[0], midxNew[-1])
    results = {}
    print("=" * 100)
    print("TABLE A -- arm x (old monthly-arbiter CAGR/DD, new trade-arbiter CAGR / maxDD / CAGR-DD)")
    print("=" * 100)
    for name, ov in arm_overrides.items():
        c_old, d_old, cd_old, _, _, _ = book_monthly_gen(legs, NEW, ov, window=window)
        c_new, d_new, cd_new, allr, midx_used = trade_book(legs, NEW, ov, window=window)
        own_c_old, own_d_old, own_cd_old, _, _, own_midx = book_monthly_gen(legs, NEW, ov)  # arm's OWN natural window (diagnostic only)
        results[name] = dict(cd_old=cd_old, d_old=d_old, cd_new=cd_new, c_new=c_new, d_new=d_new,
                              allr=allr, midx=midx_used, n=len(allr),
                              own_months=len(own_midx))
        flag = " (own natural window differs)" if len(own_midx) != len(midxNew) else ""
        print(f"  {name:<32} old CAGR/DD={cd_old:6.2f}  old DD={d_old:5.2f}%   |   "
              f"new CAGR={c_new:7.2f}%  new DD={d_new:6.2f}%  new CAGR/DD={cd_new:6.2f}"
              f"   n={len(allr):>5}{flag}")

    print("\n  rank by NEW arbiter (CAGR/DD, NEW-basket family only):")
    for i, (name, r) in enumerate(sorted(results.items(), key=lambda kv: -kv[1]["cd_new"]), 1):
        print(f"    {i:>2}. {name:<32} new CAGR/DD={r['cd_new']:.2f}   (old was {r['cd_old']:.2f})")

    # --- 3-leg reference (item 4): its OWN natural basket/window, not forced to the 6-leg span ---
    print()
    print("-" * 100)
    print("REFERENCE (item 4) -- current 3-leg book (gold_bo, btc_bo_kama, btc_pull), inv-vol, own natural window")
    print("-" * 100)
    c3_old, d3_old, cd3_old, _, _, midx3 = book_monthly_gen(legs, OLD)
    c3_new, d3_new, cd3_new, allr3, _ = trade_book(legs, OLD)
    print(f"  book_monthly_gen(OLD) [old, own natural window {midx3[0]}..{midx3[-1]}, "
          f"{len(midx3)} months]: CAGR/DD={cd3_old:.2f}  DD={d3_old:.2f}%")
    print(f"  trade_book(OLD)       [new, trade-resolution]:          CAGR/DD={cd3_new:.2f}  "
          f"CAGR={c3_new:.2f}%  DD={d3_new:.2f}%   n={len(allr3)}")
    print("  (CLAUDE.md's logged reference: 3-leg inv-vol CAGR/DD ~2.91 -- compare order of magnitude only;"
          " that figure's own provenance/window is not re-derived here.)")

    # =========================================================================
    # TABLE B -- trade-resolution circular block bootstrap (NEW-basket family only, paired,
    # same resampled months across arms, window forced to A0's window)
    # =========================================================================
    print()
    print("=" * 100)
    print(f"TABLE B -- trade-resolution circular block bootstrap of the BOOK's trade sequence "
          f"({ndraw} draws/block-length, paired, window={window[0]}..{window[1]}, "
          f"{len(midxNew)} months)")
    print("=" * 100)
    arm_buckets = {name: month_buckets(r["allr"], midxNew) for name, r in results.items()}
    bt = bootstrap_trade(arm_buckets, len(midxNew), ndraw, base_name="A0 base (RR4.5, adopted)")
    names = list(arm_overrides.keys())
    print(f"{'block':<8}" + "".join(f"{nm[:26]:>30}" for nm in names))
    for blk in (1, 3, 6, 12):
        row = bt[blk]
        print(f"{f'{blk}mo':<8}" + "".join(f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(30) for nm in names))
    print("\n  P = P(this arm's new-arbiter CAGR/DD > A0 base's new-arbiter CAGR/DD) on the SAME")
    print("  resampled months (paired). ~50% = indistinguishable. A real change stays consistent")
    print("  (P moves away from 50%) as block length grows; noise flips or reverts toward 50%.")

    # =========================================================================
    # explicit reversal check
    # =========================================================================
    print()
    print("=" * 100)
    print("REVERSAL CHECK")
    print("=" * 100)
    rr45 = results["RR4.5"]["cd_new"]
    rr40 = results["RR4.0"]["cd_new"]
    a0 = results["A0 base (RR4.5, adopted)"]["cd_new"]
    print(f"  A0 (adopted RR4.5, via build_base) new CAGR/DD = {a0:.2f}")
    print(f"  RR4.5 (rebuilt via ladder)          new CAGR/DD = {rr45:.2f}   (tie-back to A0, should match closely)")
    print(f"  RR4.0 (previous)                     new CAGR/DD = {rr40:.2f}")
    best_rr = max(rr_grid, key=lambda r: results[f"RR{r}"]["cd_new"])
    print(f"  best RR under the NEW arbiter: RR{best_rr}  (new CAGR/DD={results[f'RR{best_rr}']['cd_new']:.2f})")
    for k in ("HH4H+PDH ladder, weak=0.25", "HH4H+PDH ladder, weak=0.5"):
        print(f"  {k:<34} new CAGR/DD={results[k]['cd_new']:.2f}  (old was {results[k]['cd_old']:.2f}, "
              f"{'BEATS' if results[k]['cd_new'] > a0 else 'below'} A0 new)")
    for k in ("ext-veto q0.75 w=0.5", "ext-veto q0.9 w=0.0", "ext-veto q0.6 w=0.0"):
        print(f"  {k:<34} new CAGR/DD={results[k]['cd_new']:.2f}  (old was {results[k]['cd_old']:.2f}, "
              f"{'BEATS' if results[k]['cd_new'] > a0 else 'below'} A0 new)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
