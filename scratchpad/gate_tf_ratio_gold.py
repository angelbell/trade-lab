"""gate_tf_ratio_gold.py -- does the "gate-TF/leg-TF ratio" mechanism found today for
btc15m_L (daily gate=96x too far -> 4h gate=16x hits harder & de-concentrates) TRANSFER to the
two gold legs suspected of the same disease:

  gold15m : 15m leg, daily-SMA150+slope gate  = 96x ratio (IDENTICAL ratio to un-fixed btc15m_L).
            Worst concentration of the 6 legs (top-3-yr 79.1%, yr-Gini 0.60); leave-one-out ~ P=50%.
  gold_bo : 1h leg,  daily-SMA150+slope gate   = 24x ratio. Worst yr-Gini of the 6 legs (0.82).

Experiment (spec-frozen): for each leg, vary ONLY the gate TF (gate content held fixed as far as
possible), across a TF ladder, in three gate-content variants:
  (a) SMA "bars-fixed"  -- daily_sma=150, daily_slope_k=10 UNCHANGED (just resampled at the new
      TF via --gate-tf; breakout_wave already supports this -- the SMA window shrinks in real
      time as the TF speeds up).
  (b) SMA "realtime"    -- daily_sma / daily_slope_k SCALED so the window stays ~150 real days /
      ~10 real days (e.g. 1D->4h needs SMA*6 = 900, slope_k*6 = 60).
  (c) KAMA(14)-rising   -- replace the SMA+slope gate entirely with breakout_wave's existing
      --gate-kama 14 --gate-kama-tf <TF> (the mechanism that worked for btc15m_L), at the same
      TF ladder.

All legs/machinery are IMPORTED, not reimplemented:
  - btc_family_ext_throttle.build_base()          -- the 6 canonical adopted leg series (tie-back)
  - breakout_wave.run / resample                  -- leg re-generation with the gate override
  - book_leave_one_out.cdd / NEW / OLD             -- trade-resolution CAGR/maxDD/CAGR-DD, baskets
  - book_weighting_scheme.monthly_matrix / scheme_invvol_trade / series_from_weights /
    block_bootstrap                                -- the book weighting + paired block bootstrap
  - btc15mL_regime_concentration.book_cdd / concentration / year_table / leg_pf_meanR_n_totR
    -- the exact book arbiter (scheme_invvol_trade + trade-res cdd) whose tie-back values (6-leg
    8.19, 3-leg 2.71) are this script's mandatory tie-back, and the concentration measures.
  - book_hh4h_weight_sweep.leg_stats               -- standardized 1%-risk leg CAGR/DD

No lookahead introduced: only re-runs breakout_wave.run() with different --gate-tf / --gate-kama
args (all HTF gates inside run() are already shift(1)+ffill'd before being read by an entry bar);
this script adds no new signal logic.

Two flagged DEVIATIONS from the literal spec card (implemented as specified, flagged here per
the measure-agent contract -- ask/flag rather than silently "fix"):
  1. The spec text describes "現行" gold15m as ext_cap 8.0 / pullback_frac 0.25 / RR4 / "9-15UTC
     skip". The reusable canonical builder this script is told to tie back to (build_base(), and
     book_hh4h_weight_sweep.py's identical construction) has NO UTC-hour skip -- there is no such
     parameter in breakout_wave.run() at all. Implemented as build_base() actually is (no UTC
     skip) since that is what the mandatory tie-back numbers (8.19 / 2.71) are computed from;
     flagging the mismatch rather than inventing new hour-skip logic.
  2. The KAMA-substitute arms (c) necessarily also drop --ext-cap: in breakout_wave.run(), ext_cap
     is computed INSIDE the `if args.daily_sma > 0` block (it reads off the same daily SMA used by
     the SMA gate), so it has no independent implementation once daily_sma=0. This means the KAMA
     arms lack gold15m's extension-cap safety valve that the SMA arms keep -- a confound between
     "gate content" and "extension cap presence" in the KAMA cells specifically. Flagged, not
     patched (patching would mean inventing a new ext-cap-off-KAMA formula = new logic).

Run (full):  .venv/bin/python scratchpad/gate_tf_ratio_gold.py 2>/dev/null | tee scratchpad/out_gate_tf_ratio_gold.txt
Run (smoke): .venv/bin/python scratchpad/gate_tf_ratio_gold.py --smoke 2>/dev/null | tee scratchpad/out_gate_tf_ratio_gold_smoke.txt
"""
import os, sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")

from radar_gate_race import BASE
from research.regime_gate_lab import CFG as GOLD_BO_CFG
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample

from btc_family_ext_throttle import build_base
from book_leave_one_out import cdd, NEW, OLD
from book_weighting_scheme import monthly_matrix, scheme_invvol_trade, series_from_weights, block_bootstrap
from book_hh4h_weight_sweep import leg_stats
from btc15mL_regime_concentration import concentration, year_table, leg_pf_meanR_n_totR, book_cdd

ROOT = "/home/angelbell/dev/auto-trade"
HOURS = {"1D": 24.0, "8h": 8.0, "4h": 4.0, "2h": 2.0, "1h": 1.0}
TFS_GOLD15M = ["1D", "8h", "4h", "2h", "1h"]     # ratios 96/32/16/8/4 (leg tf=15min)
TFS_GOLD_BO = ["1D", "8h", "4h"]                  # ratios 24/8/4     (leg tf=1h)
RATIO_15M = {"1D": 96, "8h": 32, "4h": 16, "2h": 8, "1h": 4}
RATIO_BO = {"1D": 24, "8h": 8, "4h": 4}


# ---------------------------------------------------------------------------
# leg builders -- IDENTICAL recipe to btc_family_ext_throttle.build_base(), except the
# gate sub-dict is overridden per-arm. Reuses run()/resample() straight from breakout_wave.
# ---------------------------------------------------------------------------
def build_gold15m(gate_over):
    g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    params = {**BASE, "ext_cap": 8.0, "pullback_frac": 0.25, "daily_sma": 0, "daily_slope_k": 0,
              "gate_kama": 0, **gate_over}
    t = run(g, SimpleNamespace(**params))
    return pd.Series(t["R"].values - 0.3 / t["risk"].values, index=pd.DatetimeIndex(t["time"]))


def build_gold_bo(gate_over):
    d = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv"), "1h")
    params = {**GOLD_BO_CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
              "daily_sma": 0, "daily_slope_k": 0, "gate_kama": 0, **gate_over}
    t = run(d, SimpleNamespace(**params))
    return pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))


def gate_arms(tfs, ratio_map, base_sma=150, base_k=10):
    """Ordered dict of arm-name -> gate_over kwargs, per the spec's (a)/(b)/(c) x TF ladder.
    tf=='1D' SMA bars-fixed is SKIPPED here -- it is bit-identical to the current adopted leg,
    which the caller seeds separately as the explicit BASELINE row (avoids a duplicate row)."""
    arms = {}
    for tf in tfs:
        scale = 24.0 / HOURS[tf]
        tag = f"{tf}(x{ratio_map[tf]})"
        if tf != "1D":
            arms[f"SMA bars-fixed(150/10) @ {tag}"] = dict(daily_sma=base_sma, daily_slope_k=base_k,
                                                            gate_tf=tf, gate_kama=0)
            sma_rt, k_rt = round(base_sma * scale), round(base_k * scale)
            arms[f"SMA realtime({sma_rt}/{k_rt}) @ {tag}"] = dict(daily_sma=sma_rt, daily_slope_k=k_rt,
                                                                   gate_tf=tf, gate_kama=0)
        arms[f"KAMA14 @ {tag}"] = dict(daily_sma=0, daily_slope_k=0, gate_kama=14, gate_kama_tf=tf)
    return arms


# ---------------------------------------------------------------------------
# per-leg descriptive stats (n, N/yr, win%, PF, meanR, totR, leg CAGR/DD) + concentration
# + ex-top3-year n/PF/meanR -- all via imported functions, no re-derivation of the formulas.
# ---------------------------------------------------------------------------
def leg_row(s):
    R = s.values
    n = len(R)
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    win = (R > 0).mean() * 100
    pos, neg = R[R > 0].sum(), abs(R[R <= 0].sum())
    pf_raw = pos / neg if neg > 0 else np.nan
    meanR = R.mean()
    totR = R.sum()
    medR = np.median(R)
    stdR = R.std(ddof=1) if n > 1 else np.nan
    pf_chk, dd, cdd = leg_stats(s)
    c = concentration(s, "")
    top_years = c["top_years"]
    rest = s[~s.index.year.isin(top_years)]
    n_r, win_r, pf_r, meanR_r, totR_r = leg_pf_meanR_n_totR(rest.values)
    # IS/OOS split -- SAME convention as breakout_wave.run()'s own console line: split at the
    # median year of THIS series' own history (not a fixed calendar year).
    yrs_u = sorted(s.index.year.unique())
    half = yrs_u[len(yrs_u) // 2] if len(yrs_u) > 1 else None
    isr = s[s.index.year < half].values.mean() if half else np.nan
    oosr = s[s.index.year >= half].values.mean() if half else np.nan
    return dict(n=n, n_yr=n / years, win=win, pf=pf_raw, meanR=meanR, medR=medR, stdR=stdR,
                totR=totR, cagr_dd=cdd, dd=dd, isr=isr, oosr=oosr,
                top3yr=c["top3_year_share"], top_years=top_years, gini=c["gini_year"],
                n_rest=n_r, pf_rest=pf_r, meanR_rest=meanR_r)


def print_leg_table(title, rows, order):
    print("=" * 100); print(title); print("=" * 100)
    hdr = (f"{'arm':<34}{'n':>5}{'N/yr':>6}{'win%':>6}{'PF':>6}{'meanR':>7}{'medR':>7}{'stdR':>7}"
           f"{'totR':>8}{'lgC/DD':>7}{'IS':>7}{'OOS':>7}"
           f"{'top3yr%':>9}{'yrGini':>7}{'exT3 n':>7}{'exT3PF':>7}{'exT3mR':>7}")
    print(hdr)
    for name in order:
        r = rows[name]
        print(f"{name:<34}{r['n']:>5}{r['n_yr']:>6.1f}{r['win']:>6.1f}{r['pf']:>6.2f}{r['meanR']:>+7.3f}"
              f"{r['medR']:>+7.3f}{r['stdR']:>7.3f}"
              f"{r['totR']:>+8.1f}{r['cagr_dd']:>7.2f}{r['isr']:>+7.3f}{r['oosr']:>+7.3f}"
              f"{r['top3yr']:>8.1f}%{r['gini']:>7.2f}"
              f"{r['n_rest']:>7}{r['pf_rest']:>7.2f}{r['meanR_rest']:>+7.3f}")


def main(smoke=False):
    ndraw = 100 if smoke else 2000
    tfs15m = ["4h"] if smoke else TFS_GOLD15M
    tfsbo = ["4h"] if smoke else TFS_GOLD_BO

    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()

    # =========================================================================
    print("=" * 100); print("MANDATORY TIE-BACK"); print("=" * 100)
    c6, d6, x6, _ = book_cdd(legs, NEW)
    c3, d3, x3, _ = book_cdd(legs, OLD)
    print(f"6-leg book (scheme_invvol_trade + cdd): CAGR/DD={x6:.2f}  DD={d6:.2f}%   (target 8.19)")
    print(f"3-leg book (scheme_invvol_trade + cdd): CAGR/DD={x3:.2f}  DD={d3:.2f}%   (target 2.71)")
    if abs(x6 - 8.19) > 0.02 or abs(x3 - 2.71) > 0.02:
        print("\n*** TIE-BACK MISMATCH -- stopping before proceeding further. Report this. ***")
        return

    # baseline-arm regen must reproduce build_base()'s own gold15m / gold_bo bit-for-bit
    base15_over = dict(daily_sma=150, daily_slope_k=10, gate_tf="1D", gate_kama=0)
    baseBO_over = dict(daily_sma=150, daily_slope_k=10, gate_tf="1D", gate_kama=0)
    with contextlib.redirect_stderr(io.StringIO()):
        g15_base = build_gold15m(base15_over)
        gbo_base = build_gold_bo(baseBO_over)
    match15 = np.allclose(g15_base.values, legs["gold15m"].values) and len(g15_base) == len(legs["gold15m"])
    matchBO = np.allclose(gbo_base.values, legs["gold_bo"].values) and len(gbo_base) == len(legs["gold_bo"])
    print(f"\n[tie-back] build_gold15m(SMA150/10,gate_tf=1D) reproduces build_base()['gold15m'] "
          f"bit-for-bit: {match15}  (n={len(g15_base)} vs {len(legs['gold15m'])})")
    print(f"[tie-back] build_gold_bo(SMA150/10,gate_tf=1D) reproduces build_base()['gold_bo'] "
          f"bit-for-bit: {matchBO}  (n={len(gbo_base)} vs {len(legs['gold_bo'])})")
    if not (match15 and matchBO):
        print("\n*** LEG-BUILDER TIE-BACK MISMATCH -- stopping before proceeding further. ***")
        return

    print("\n[data-span flag] gold H1 file has only ~250-300 bars/year 2007-2017 (vs ~5900/year "
          "2018+) -- effectively daily bars mislabeled H1. build_gold_bo()/get_legs() load the "
          "FULL h1 file with NO --start filter, so this sparse pre-2018 stretch IS inside the "
          "current adopted gold_bo leg (and therefore inside the 2.71/8.19 tie-back targets too). "
          "This predates this experiment (build_base() already does this) -- not fixed here, "
          "flagged for the record per CLAUDE.md's own '--start 2018-01-01' warning for gold H1.")

    # =========================================================================
    print()
    print("=" * 100); print("PART 1 -- gold15m gate-TF ladder"); print("=" * 100)
    BASE15 = "BASELINE (adopted, SMA150/10 bars-fixed @ 1D, x96)"
    arms15 = gate_arms(tfs15m, RATIO_15M)
    order15 = [BASE15] + list(arms15.keys())
    rows15, series15 = {BASE15: leg_row(legs["gold15m"])}, {BASE15: legs["gold15m"]}
    with contextlib.redirect_stderr(io.StringIO()):
        for name, over in arms15.items():
            s = build_gold15m(over)
            series15[name] = s
            rows15[name] = leg_row(s)
    print_leg_table("gold15m leg metrics + concentration (n / concentration measures per spec)",
                     rows15, order15)

    print()
    print("gold15m -- 6-leg BOOK judge (book_cdd, scheme_invvol_trade weighting; replace gold15m only)")
    book15 = {BASE15: dict(cagr=c6, dd=d6, cd=x6, port=book_cdd(legs, NEW)[3])}
    with contextlib.redirect_stderr(io.StringIO()):
        for name in arms15:
            L = dict(legs); L["gold15m"] = series15[name]
            c, d, x, s = book_cdd(L, NEW)
            book15[name] = dict(cagr=c, dd=d, cd=x, port=s)
    print(f"{'arm':<34}{'book6 CAGR/DD':>14}{'book6 DD%':>10}")
    for name in order15:
        r = book15[name]
        flag = "  <-- PASS (beats 8.19)" if (name != BASE15 and r["cd"] > 8.19) else ""
        print(f"{name:<34}{r['cd']:>14.2f}{r['dd']:>9.2f}%{flag}")

    # =========================================================================
    print()
    print("=" * 100); print("PART 2 -- gold_bo gate-TF ladder"); print("=" * 100)
    BASEBO = "BASELINE (adopted, SMA150/10 bars-fixed @ 1D, x24)"
    armsBO = gate_arms(tfsbo, RATIO_BO)
    orderBO = [BASEBO] + list(armsBO.keys())
    rowsBO, seriesBO = {BASEBO: leg_row(legs["gold_bo"])}, {BASEBO: legs["gold_bo"]}
    with contextlib.redirect_stderr(io.StringIO()):
        for name, over in armsBO.items():
            s = build_gold_bo(over)
            seriesBO[name] = s
            rowsBO[name] = leg_row(s)
    print_leg_table("gold_bo leg metrics + concentration", rowsBO, orderBO)

    print()
    print("gold_bo -- 3-leg BOOK (OLD, replace gold_bo) and 6-leg BOOK (NEW, replace gold_bo) judge")
    bookBO3 = {BASEBO: dict(cagr=c3, dd=d3, cd=x3, port=book_cdd(legs, OLD)[3])}
    bookBO6 = {BASEBO: dict(cagr=c6, dd=d6, cd=x6, port=book_cdd(legs, NEW)[3])}
    with contextlib.redirect_stderr(io.StringIO()):
        for name in armsBO:
            L = dict(legs); L["gold_bo"] = seriesBO[name]
            c3_, d3_, x3_, s3_ = book_cdd(L, OLD)
            c6_, d6_, x6_, s6_ = book_cdd(L, NEW)
            bookBO3[name] = dict(cagr=c3_, dd=d3_, cd=x3_, port=s3_)
            bookBO6[name] = dict(cagr=c6_, dd=d6_, cd=x6_, port=s6_)
    print(f"{'arm':<34}{'book3 CAGR/DD':>14}{'book3 DD%':>10}{'book6 CAGR/DD':>14}{'book6 DD%':>10}")
    for name in orderBO:
        r3, r6 = bookBO3[name], bookBO6[name]
        flag = "  <-- PASS (book3 beats 2.71)" if (name != BASEBO and r3["cd"] > 2.71) else ""
        print(f"{name:<34}{r3['cd']:>14.2f}{r3['dd']:>9.2f}%{r6['cd']:>14.2f}{r6['dd']:>9.2f}%{flag}")

    # =========================================================================
    print()
    print("=" * 100); print("PART 3 -- annual totR: current (baseline) vs best-by-book6 CAGR/DD"); print("=" * 100)
    base15_name = BASE15
    best15_name = max(order15, key=lambda n: book15[n]["cd"])
    baseBO_name = BASEBO
    bestBO_name = max(orderBO, key=lambda n: bookBO6[n]["cd"])

    def print_year_compare(tag, base_name, best_name, series_map):
        tb = year_table(series_map[base_name]); tbest = year_table(series_map[best_name])
        yrs = sorted(set(tb["year"]) | set(tbest["year"]))
        print(f"\n{tag}: baseline = '{base_name}'   best-by-book6 = '{best_name}'")
        print(f"{'year':<6}{'base totR':>11}{'base n':>8}{'best totR':>11}{'best n':>8}")
        for y in yrs:
            rb = tb[tb["year"] == y]; rk = tbest[tbest["year"] == y]
            bt = rb["totR"].iloc[0] if len(rb) else 0.0
            bn = int(rb["n"].iloc[0]) if len(rb) else 0
            kt = rk["totR"].iloc[0] if len(rk) else 0.0
            kn = int(rk["n"].iloc[0]) if len(rk) else 0
            print(f"{int(y):<6}{bt:>+11.1f}{bn:>8}{kt:>+11.1f}{kn:>8}")

    print_year_compare("gold15m", base15_name, best15_name, series15)
    print_year_compare("gold_bo", baseBO_name, bestBO_name, seriesBO)

    # =========================================================================
    print()
    print("=" * 100)
    print("PART 4 -- paired circular block bootstrap of the BOOK (book6 for both legs; book3 also "
          f"for gold_bo), baseline vs best-by-book6 arm ({ndraw} draws/block-length)")
    print("=" * 100)

    named15 = {base15_name: book15[base15_name]["port"], best15_name: book15[best15_name]["port"]}
    bt15 = block_bootstrap(named15, base15_name, ndraw)
    print(f"\ngold15m -- 6-leg book bootstrap (median CAGR/DD, P(beats baseline))")
    print(f"{'block':<8}{base15_name[:30]:>32}{best15_name[:30]:>32}")
    for blk in (1, 3, 6, 12):
        row = bt15[blk]
        print(f"{f'{blk}mo':<8}" + "".join(
            f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(32) for nm in (base15_name, best15_name)))

    namedBO6 = {baseBO_name: bookBO6[baseBO_name]["port"], bestBO_name: bookBO6[bestBO_name]["port"]}
    btBO6 = block_bootstrap(namedBO6, baseBO_name, ndraw)
    print(f"\ngold_bo -- 6-leg book bootstrap (median CAGR/DD, P(beats baseline))")
    print(f"{'block':<8}{baseBO_name[:30]:>32}{bestBO_name[:30]:>32}")
    for blk in (1, 3, 6, 12):
        row = btBO6[blk]
        print(f"{f'{blk}mo':<8}" + "".join(
            f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(32) for nm in (baseBO_name, bestBO_name)))

    namedBO3 = {baseBO_name: bookBO3[baseBO_name]["port"], bestBO_name: bookBO3[bestBO_name]["port"]}
    btBO3 = block_bootstrap(namedBO3, baseBO_name, ndraw)
    print(f"\ngold_bo -- 3-leg book bootstrap (median CAGR/DD, P(beats baseline))")
    print(f"{'block':<8}{baseBO_name[:30]:>32}{bestBO_name[:30]:>32}")
    for blk in (1, 3, 6, 12):
        row = btBO3[blk]
        print(f"{f'{blk}mo':<8}" + "".join(
            f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(32) for nm in (baseBO_name, bestBO_name)))

    print()
    print("=" * 100)
    print(f"done. multiple-comparisons note: {len(order15)} gold15m arms x {len(orderBO)} gold_bo "
          "arms screened by point book CAGR/DD before the bootstrap was run on the single "
          "best-of-each -- treat the point numbers in PART 1/2 as a screen, the bootstrap P columns "
          "in PART 4 as the load-bearing check.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
