"""book_rebuild_correct.py -- rebuild the book arbiter with BOTH known bugs fixed at once.

Two bugs were found in the book machinery this session, both rooted in "collapsing to
monthly" (only misfires when a high-frequency leg, the 15m family, is mixed with low-
frequency legs):
  1. DD ARBITER measured maxDD on the MONTHLY equity curve -> hides intra-month drawdowns.
     Fix = TRADE-resolution DD.
  2. WEIGHTS: inv-vol was computed on monthly-R-SUM sigma -> over-weights low-frequency legs.
     Fix = inv-vol on TRADE-R sigma.

experiments/book_leave_one_out.py's own leave-one-out only fixed bug 1 -- it still calls its
local weights() (monthly-sigma). So its leg ranking (btc_bo_kama hurts, gold15m ~50/50,
btc15m_L is load-bearing) was derived on a still-broken weight scheme. This script redoes the
leg-composition sweep, leave-one-out, gold15m verdict and a leverage reality-check with BOTH
fixes applied together, and prints the OLD (still-buggy-weight) leave-one-out numbers next to
the new ones so it's visible what changes and what doesn't.

Reused, not reimplemented:
  - btc_family_ext_throttle.build_base()            -- canonical 6-leg R series (tie-back target)
  - btc_family_ext_throttle.{run, resample, invert, BASE, ROOT}
  - book_leave_one_out.cdd()                        -- trade-resolution CAGR/maxDD/CAGR-DD (bug-1 fix)
  - book_weighting_scheme.monthly_matrix()          -- common-window month index (window bounds only)
  - book_weighting_scheme.scheme_invvol_trade()     -- inv-vol on trade-R sigma (bug-2 fix)
  - book_weighting_scheme.block_bootstrap()         -- paired circular month-block bootstrap
  - research.portfolio_kama.{run_bo, run_pb, kama_gate_btc, cycle_gate_pull, CFG, PB}
        (used ONLY in R4 to recover each core leg's "hold" column, which get_legs() discards;
         same function calls, same params, as get_legs() -- nothing about the signal changes)

New code in this file (the actual experiments, not a re-derivation of the arbiter):
  - stat_fixed_window() / stat_natural(): thin glue combining the two reused fixes; tie-back
    checked against book_weighting_scheme.py's own invvol_trade numbers before anything else runs.
  - R1: 2^3 leg-composition sweep (natural window per basket AND a common 6-leg window).
  - R2: leave-one-out, weights re-derived over the surviving 5 legs each time; bootstrap;
    OLD (monthly-sigma) LOO numbers hardcoded from a fresh run of book_leave_one_out.py this
    session (experiments/out_book_leave_one_out_REFERENCE_prevrun.txt) for comparison.
  - R3: gold15m in/out head-to-head (reuses R2's own arms) + daily-return correlation vs the
    other 5 legs.
  - R4: hourly concurrent-risk-exposure reality check for the best R1 config, built from each
    leg's (entry_time, hold-days, weight) -- a sweep-line over +w at entry / -w at entry+hold.

No lookahead introduced: everything here re-weights/re-aggregates already-priced trade series
from build_base() / build_hold_frames() (next-bar-open fill, intrabar SL/TP priority,
confirmed-close HTF gates all enforced upstream in breakout_wave.py / ema_pullback.py).

Run (smoke, fewer bootstrap draws): .venv/bin/python experiments/book_rebuild_correct.py --smoke
Run (full):                          .venv/bin/python experiments/book_rebuild_correct.py
"""
import sys, io, contextlib, warnings, argparse, itertools
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")

from btc_family_ext_throttle import build_base, run, resample, invert, BASE, ROOT
from book_leave_one_out import cdd, NEW, OLD
from book_weighting_scheme import monthly_matrix, scheme_invvol_trade, block_bootstrap
from src.data_loader import load_mt5_csv
from research.portfolio_kama import run_bo, run_pb, kama_gate_btc, cycle_gate_pull, CFG, PB

BUDGET = 0.03
CORE = OLD                      # ["gold_bo", "btc_bo_kama", "btc_pull"]
CANDS = ["gold15m", "btc15m_L", "btc15m_S"]
assert set(CORE) | set(CANDS) == set(NEW)

# tie-back targets, transcribed from a fresh run of book_weighting_scheme.py --smoke this
# session (experiments/out_book_weighting_scheme_smoke_CHECK.txt), E2's invvol_trade row.
# (bootstrap draw count doesn't change point CAGR/DD, so the --smoke numbers are exact.)
TIEBACK_NEW = dict(n=1445, cagr=62.8, dd=7.67, cd=8.19)
TIEBACK_OLD = dict(n=364, cagr=26.7, dd=9.87, cd=2.71)


# ===========================================================================
# correct arbiter: trade-resolution DD (cdd, reused) x trade-sigma inv-vol weights
# (scheme_invvol_trade, reused). midx supplies only the window boundary (first/last
# period) -- scheme_invvol_trade ignores its `M` argument entirely, so passing None
# is safe and avoids recomputing a monthly matrix we don't need.
# ===========================================================================
def stat_fixed_window(legs, basket, midx, budget=BUDGET):
    w = scheme_invvol_trade(legs, basket, None, midx)
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    parts = []
    for k in basket:
        s = legs[k]
        s = s[(s.index >= st) & (s.index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    port = pd.concat(parts).sort_index()
    days = (port.index[-1] - port.index[0]).days
    c, d, x = cdd(port.values, days)
    n = len(port)
    years = days / 365.25
    npy = n / years if years > 0 else np.nan
    return dict(cagr=c, dd=d, cd=x, n=n, npy=npy, w=w, port=port, days=days, midx=midx)


def stat_natural(legs, basket, budget=BUDGET):
    _, midx = monthly_matrix(legs, basket)
    return stat_fixed_window(legs, basket, midx, budget)


# ===========================================================================
# R1 -- 2^3 leg-composition sweep
# ===========================================================================
def r1(legs):
    combos = list(itertools.product([0, 1], repeat=3))  # order = CANDS
    midx_common = monthly_matrix(legs, NEW)[1]
    nat, com = {}, {}
    for combo in combos:
        cands_on = [CANDS[i] for i, b in enumerate(combo) if b]
        basket = CORE + cands_on
        label = "".join(str(b) for b in combo)
        nat[label] = (basket, stat_natural(legs, basket))
        com[label] = (basket, stat_fixed_window(legs, basket, midx_common))
    return nat, com, midx_common


def print_r1_table(rows, title, all_legs):
    print(f"\n{title}")
    hdr = (f"{'gold15m':>8}{'btc15m_L':>9}{'btc15m_S':>9}{'n':>7}{'N/yr':>8}{'CAGR':>9}"
           f"{'maxDD':>8}{'CAGR/DD':>9}  " + "  ".join(f"{k:>12}" for k in all_legs))
    print(hdr)
    for label, (basket, r) in rows.items():
        g15 = "ON" if label[0] == "1" else "."
        bL = "ON" if label[1] == "1" else "."
        bS = "ON" if label[2] == "1" else "."
        wcells = []
        for k in all_legs:
            wcells.append(f"{r['w'][k]*100:>12.3f}" if k in basket else f"{'--':>12}")
        print(f"{g15:>8}{bL:>9}{bS:>9}{r['n']:>7}{r['npy']:>8.1f}{r['cagr']:>8.1f}%"
              f"{r['dd']:>7.2f}%{r['cd']:>9.2f}  " + "  ".join(wcells))


# ===========================================================================
# R2 -- leave-one-out, correct weights re-derived over the remaining 5 legs each time
# ===========================================================================
# OLD (still-buggy monthly-sigma weights) LOO numbers, transcribed verbatim from a fresh
# unmodified run of book_leave_one_out.py this session
# (experiments/out_book_leave_one_out_REFERENCE_prevrun.txt). Not recomputed here.
OLD_LOO_CD = {
    "6-leg (all)": 6.84, "minus gold_bo": 6.64, "minus btc_bo_kama": 7.77,
    "minus btc_pull": 5.99, "minus gold15m": 6.04, "minus btc15m_L": 4.11,
    "minus btc15m_S": 5.86, "3-leg incumbent": 3.11,
}
OLD_LOO_BT = {  # {arm: {block: (median, P%)}}
    "6-leg (all)":       {1: (6.00, 0), 3: (6.56, 0), 6: (6.87, 0), 12: (6.90, 0)},
    "minus gold_bo":     {1: (5.93, 49), 3: (6.34, 38), 6: (6.76, 41), 12: (6.76, 41)},
    "minus btc_bo_kama": {1: (6.27, 62), 3: (6.97, 68), 6: (7.49, 77), 12: (7.66, 86)},
    "minus btc_pull":    {1: (5.26, 18), 3: (5.66, 9), 6: (5.97, 5), 12: (5.92, 3)},
    "minus gold15m":     {1: (5.83, 48), 3: (6.52, 51), 6: (6.84, 51), 12: (6.89, 50)},
    "minus btc15m_L":    {1: (4.06, 1), 3: (4.17, 0), 6: (4.10, 0), 12: (4.13, 0)},
    "minus btc15m_S":    {1: (5.02, 19), 3: (5.55, 12), 6: (5.83, 9), 12: (5.93, 6)},
    "3-leg incumbent":   {1: (2.88, 1), 3: (2.95, 0), 6: (3.06, 0), 12: (3.18, 1)},
}


def r2(legs, ndraw):
    arms = {"6-leg (all)": NEW}
    for k in NEW:
        arms[f"minus {k}"] = [x for x in NEW if x != k]
    arms["3-leg incumbent"] = OLD

    stats, ports = {}, {}
    for name, basket in arms.items():
        r = stat_natural(legs, basket)
        stats[name] = r
        ports[name] = r["port"]

    bt = block_bootstrap(ports, "6-leg (all)", ndraw)
    return arms, stats, bt


def print_r2(arms, stats, bt):
    base_cd = stats["6-leg (all)"]["cd"]
    print(f"\n{'book (correct weights)':<24}{'n':>6}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>9}{'vs all':>9}"
          f"   {'OLD CAGR/DD':>12}{'OLD vs all':>11}")
    for name in arms:
        r = stats[name]
        delta = "" if name == "6-leg (all)" else f"{r['cd'] - base_cd:+.2f}"
        old_cd = OLD_LOO_CD[name]
        old_delta = "" if name == "6-leg (all)" else f"{old_cd - OLD_LOO_CD['6-leg (all)']:+.2f}"
        print(f"{name:<24}{r['n']:>6}{r['cagr']:>8.1f}%{r['dd']:>7.2f}%{r['cd']:>9.2f}{delta:>9}"
              f"   {old_cd:>12.2f}{old_delta:>11}")

    print(f"\nNEW (trade-sigma weights) paired circular block bootstrap -- P(beats 6-leg all)")
    print(f"  {'block':<7}" + "".join(f"{k:>20}" for k in arms))
    for blk in (1, 3, 6, 12):
        row = bt[blk]
        cells = [f"{row[k][0]:.2f}(P{row[k][1]:.0f}%)" for k in arms]
        print(f"  {f'{blk}mo':<7}" + "".join(f"{c:>20}" for c in cells))

    print(f"\nOLD (monthly-sigma weights, book_leave_one_out.py as-is) same bootstrap, for comparison")
    print(f"  {'block':<7}" + "".join(f"{k:>20}" for k in arms))
    for blk in (1, 3, 6, 12):
        cells = [f"{OLD_LOO_BT[k][blk][0]:.2f}(P{OLD_LOO_BT[k][blk][1]:.0f}%)" for k in arms]
        print(f"  {f'{blk}mo':<7}" + "".join(f"{c:>20}" for c in cells))


# ===========================================================================
# R3 -- gold15m verdict + daily-return correlation
# ===========================================================================
def r3(legs, arms, stats):
    print(f"\n6-leg (all) vs minus-gold15m, correct weights (from R2 above):")
    a, b = stats["6-leg (all)"], stats["minus gold15m"]
    print(f"  6-leg (all)      CAGR/DD={a['cd']:.2f}  maxDD={a['dd']:.2f}%  n={a['n']}")
    print(f"  minus gold15m    CAGR/DD={b['cd']:.2f}  maxDD={b['dd']:.2f}%  n={b['n']}")

    midx = monthly_matrix(legs, NEW)[1]
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    daily = {}
    for k in NEW:
        s = legs[k]
        s = s[(s.index >= st) & (s.index <= en)]
        daily[k] = s.groupby(s.index.normalize()).sum()
    didx = pd.date_range(st.normalize(), en.normalize(), freq="D", tz="UTC")
    D = pd.DataFrame({k: v.reindex(didx, fill_value=0.0) for k, v in daily.items()})
    corr = D.corr()["gold15m"]
    trade_days = (D["gold15m"] != 0).sum()
    print(f"\n  gold15m daily-R correlation vs other 5 legs (common window {st.date()}..{en.date()}, "
          f"{len(D)} cal-days, gold15m traded on {trade_days} of them):")
    for k in NEW:
        if k == "gold15m":
            continue
        print(f"    corr(gold15m, {k:<12}) = {corr[k]:+.3f}")
    print("  (most days both legs are 0/no-trade, which mechanically shrinks |corr| toward 0 --")
    print("   read this as 'no detectable co-movement', not as proof of independence.)")


# ===========================================================================
# R4 -- concurrent-risk reality check
# ===========================================================================
def build_hold_frames():
    """Same construction as get_legs() (portfolio_kama) + build_base() (btc_family_ext_throttle),
    but keeping the 'hold' (days) column both drop. Tie-back checked against build_base()'s R
    series before use."""
    gold_raw = run_bo(resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv"), "1h"),
                       SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                          "daily_sma": 150, "daily_slope_k": 10}))
    gold = gold_raw[["time", "R", "hold"]].reset_index(drop=True)

    btc_raw = run_bo(resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h"),
                      SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))
    btc_k = kama_gate_btc(btc_raw)[["time", "R", "hold"]].reset_index(drop=True)

    dbtc = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
    pb_raw = run_pb(dbtc, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)
    pb = cycle_gate_pull(pb_raw)[["time", "R", "hold"]].reset_index(drop=True)

    g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                  "ext_cap": 8.0, "pullback_frac": 0.25}))
    gold15m = pd.DataFrame({"time": t["time"].values, "R": t["R"].values - 0.3 / t["risk"].values,
                             "hold": t["hold"].values})

    full = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv")
    d15 = resample(full.loc["2018-10-01":], "15min")
    inv = invert(d15); C = 2 * d15["high"].max()
    ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
    Rs = ts_["R"].values - 15.0 / ts_["risk"].values
    pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
    mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
    btc15m_S = pd.DataFrame({"time": ts_["time"].values[mS], "R": Rs[mS], "hold": ts_["hold"].values[mS]})

    tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                     "pullback_frac": 0.3, "rr": 4.5}))
    Rn = tL["R"].values - 15.0 / tL["risk"].values
    ei = d15.index.get_indexer(tL["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    base_w = np.where(tL["e_px"].values > pdh[ei], 1.0, 0.5)
    btc15m_L = pd.DataFrame({"time": tL["time"].values, "R": Rn * base_w, "hold": tL["hold"].values})

    return {"gold_bo": gold, "btc_bo_kama": btc_k, "btc_pull": pb,
            "gold15m": gold15m, "btc15m_S": btc15m_S, "btc15m_L": btc15m_L}


def risk_exposure(frames, w, midx):
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    events = []
    for k, df in frames.items():
        if k not in w.index:
            continue
        tcol = pd.DatetimeIndex(df["time"])
        tcol = tcol.tz_localize("UTC") if tcol.tz is None else tcol.tz_convert("UTC")
        d = df.assign(_t=tcol)
        d = d[(d["_t"] >= st) & (d["_t"] <= en)]
        entries = pd.DatetimeIndex(d["_t"])
        exits = entries + pd.to_timedelta(d["hold"].values, unit="D")
        events.append(pd.Series(w[k], index=entries))
        events.append(pd.Series(-w[k], index=exits))
    step = pd.concat(events).groupby(level=0).sum().sort_index()
    cum = step.cumsum()
    hourly_idx = pd.date_range(st, en, freq="1h", tz="UTC")
    combined = cum.reindex(cum.index.union(hourly_idx)).sort_index().ffill().fillna(0.0)
    risk_h = combined.reindex(hourly_idx).ffill().fillna(0.0)
    return risk_h


def r4(legs, best_basket, best_stat):
    with contextlib.redirect_stderr(io.StringIO()):
        frames = build_hold_frames()
    # tie-back: frames[k]["R"] must equal legs[k].values exactly (same construction, hold added)
    print("\n[tie-back] build_hold_frames() R-values vs build_base() legs (must be exact):")
    ok = True
    for k in NEW:
        a = frames[k]["R"].values
        b = legs[k].values
        same = (len(a) == len(b)) and np.allclose(a, b, rtol=0, atol=1e-12)
        ok = ok and same
        print(f"  {k:<14} n_frame={len(a):<6} n_legs={len(b):<6} exact_match={same}")
    if not ok:
        print("\n*** TIE-BACK MISMATCH in R4 build_hold_frames() -- stopping. Report this. ***")
        sys.exit(1)

    w = best_stat["w"]
    midx = best_stat["midx"]
    risk_h = risk_exposure(frames, w, midx)
    budget = w.sum()
    print(f"\nR4 -- hourly concurrent weighted-risk exposure, best config = {'+'.join(best_basket)}")
    print(f"  nominal risk budget = {budget*100:.3f}% of account   (window {midx[0]}..{midx[-1]}, "
          f"{len(risk_h)} hours)")
    print(f"  median={risk_h.median()*100:.3f}%  mean={risk_h.mean()*100:.3f}%  "
          f"p95={risk_h.quantile(0.95)*100:.3f}%  max={risk_h.max()*100:.3f}%")
    over = (risk_h > budget + 1e-9).mean() * 100
    print(f"  hours with total risk > nominal budget ({budget*100:.3f}%): {over:.2f}% of hours")
    idle_frac = 1 - risk_h.mean() / budget
    print(f"  fraction of nominal 3% budget sitting idle on average: {idle_frac*100:.1f}%")
    live_dd_lo, live_dd_hi = best_stat["dd"] * 1.5, best_stat["dd"] * 2.0
    print(f"  backtest maxDD={best_stat['dd']:.2f}%  x1.5..2.0 => plausible live maxDD "
          f"{live_dd_lo:.2f}%..{live_dd_hi:.2f}%")
    return risk_h


# ===========================================================================
def main(smoke=False):
    ndraw = 200 if smoke else 2000
    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()

    # ---- top-level tie-back before anything else runs ----
    print("=" * 100)
    print("TIE-BACK (must match before proceeding)")
    print("=" * 100)
    chk_new = stat_natural(legs, NEW)
    chk_old = stat_natural(legs, OLD)
    print(f"  stat_natural(legs, NEW/6-leg): n={chk_new['n']} CAGR={chk_new['cagr']:.1f}% "
          f"maxDD={chk_new['dd']:.2f}% CAGR/DD={chk_new['cd']:.2f}   "
          f"(target n={TIEBACK_NEW['n']} CAGR={TIEBACK_NEW['cagr']} maxDD={TIEBACK_NEW['dd']} "
          f"CAGR/DD={TIEBACK_NEW['cd']})")
    print(f"  stat_natural(legs, OLD/3-leg): n={chk_old['n']} CAGR={chk_old['cagr']:.1f}% "
          f"maxDD={chk_old['dd']:.2f}% CAGR/DD={chk_old['cd']:.2f}   "
          f"(target n={TIEBACK_OLD['n']} CAGR={TIEBACK_OLD['cagr']} maxDD={TIEBACK_OLD['dd']} "
          f"CAGR/DD={TIEBACK_OLD['cd']})")
    bad = (chk_new['n'] != TIEBACK_NEW['n'] or abs(chk_new['cd'] - TIEBACK_NEW['cd']) > 0.02
           or chk_old['n'] != TIEBACK_OLD['n'] or abs(chk_old['cd'] - TIEBACK_OLD['cd']) > 0.02)
    if bad:
        print("\n*** TIE-BACK MISMATCH -- stopping before proceeding further. Report this. ***")
        sys.exit(1)

    # ---- R1 ----
    print()
    print("=" * 100)
    print("R1 -- 2^3 leg-composition sweep (core = gold_bo/btc_bo_kama/btc_pull always in;")
    print("       candidates = gold15m / btc15m_L / btc15m_S on/off), correct weights throughout")
    print("=" * 100)
    nat, com, midx_common = r1(legs)
    print_r1_table(nat, "TABLE R1a -- NATURAL window per basket (each basket's own common-data window)", NEW)
    print_r1_table(com, f"TABLE R1b -- COMMON window fixed to the full 6-leg intersection "
                        f"({midx_common[0]}..{midx_common[-1]}, {len(midx_common)}mo)", NEW)
    print("\n  judgment: read off TABLE R1b (common window) for composition comparisons -- R1a lets")
    print("  baskets missing gold15m/btc15m_* run over MORE calendar time (those legs' data starts")
    print("  later), which inflates or deflates CAGR/DD by how much history is included, not by")
    print("  composition. R1a is only useful to see each config's OWN deployable CAGR/DD today.")

    # ---- R2 ----
    print()
    print("=" * 100)
    print("R2 -- leave-one-out, weights re-derived over the surviving 5 legs (correct scheme)")
    print("=" * 100)
    arms, stats, bt = r2(legs, ndraw)
    print_r2(arms, stats, bt)

    # ---- R3 ----
    print()
    print("=" * 100)
    print("R3 -- is gold15m earning its seat?")
    print("=" * 100)
    r3(legs, arms, stats)

    # ---- R4 ----
    print()
    print("=" * 100)
    print("R4 -- leverage reality check")
    print("=" * 100)
    best_label = max(nat, key=lambda lb: com[lb][1]["cd"])  # best by COMMON-window CAGR/DD
    best_basket, best_stat_common = com[best_label]
    print(f"  best config by common-window CAGR/DD: {'+'.join(best_basket)} "
          f"(CAGR/DD={best_stat_common['cd']:.2f}) -- multiple-comparisons caveat: this is 1 of 8")
    print(f"  points swept in R1, not a confirmed winner; using it here only to instantiate R4's")
    print(f"  math on a concrete, real basket/weight vector (its own natural-window weights, not")
    print(f"  the common-window ones, since that's what would actually be deployed).")
    _, best_stat_natural = nat[best_label]
    r4(legs, best_basket, best_stat_natural)

    print()
    print("=" * 100)
    print("done. multiple-comparisons note: R1 swept 8 compositions, R2 swept 6 leave-one-out arms")
    print("(+3-leg incumbent) -- treat point CAGR/DD as a screen; the bootstrap P columns are the")
    print("load-bearing numbers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
