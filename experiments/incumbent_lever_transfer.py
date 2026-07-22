"""incumbent_lever_transfer.py -- do today's btc15m_L levers (RR ladder / PDH soft-size /
4h-KAMA gate) transfer to the 3 LIVE-FORWARD legs (gold_bo, btc_bo_kama, btc_pull), when
judged by the CORRECT arbiter (trade-resolution DD, book_leave_one_out.cdd()) instead of the
old monthly-DD arbiter these 3 legs were originally accepted under?

Reused (imported, not reimplemented):
  - research/portfolio_kama.py: PB (btc_pull config), kama_gate_btc(), cycle_gate_pull(),
    get_legs() (the 3 incumbent legs as-is)
  - experiments/btc_family_ext_throttle.py: build_base() (the 6-leg canonical series)
  - experiments/book_leave_one_out.py: cdd() (trade-resolution DD arbiter), NEW/OLD leg-name lists
  - experiments/book_weighting_scheme.py: monthly_matrix(), scheme_invvol_trade() (the weighting
    scheme), stat_custom(), series_from_weights(), block_bootstrap() (circular block bootstrap)
  - research/regime_gate_lab.py: CFG (breakout_wave base config), at() (causal gate reindex)
  - breakout_wave.py: run(), resample(), kama_adaptive()
  - ema_pullback.py: run()
  - experiments/radar_gate_race.py: kama_up() (KAMA gate on an arbitrary confirm-TF, used for the
    L3 gate-TF sweep -- identical KAMA formula to research.regime_adaptive.kama, tie-back checked)

New code here (the experiment itself, not a re-derivation of the arbiter):
  - book_stat(): glue that calls monthly_matrix + scheme_invvol_trade + stat_custom for an
    arbitrary override of one leg in an arbitrary basket (OLD 3-leg / NEW 6-leg)
  - leg builders that mirror get_legs()'s exact construction but override ONE parameter
    (--rr for L1, a post-hoc PDH weight for L2, the gate confirm-TF for L3)
  - leg_metrics(): descriptive stats (n, N/yr, win%, PF, meanR, median, std, IS/OOS) from a
    trades frame -- plain pandas aggregates, not a re-derivation of any signal/exit logic

No new signal/exit logic anywhere; every trade is produced by the existing run()/run_pb()
via next-bar-open... no: these engines fill AT THE CONFIRMED BREAK/RECLAIM BAR's close
(breakout_wave / ema_pullback conventions, unchanged here) with intrabar SL/TP priority
already enforced inside those engines. PDH/gate context variables are shift(1)+ffill from
already-CLOSED daily/4h/weekly bars -- no lookahead introduced.

Run (smoke):  .venv/bin/python experiments/incumbent_lever_transfer.py --smoke 2>&1 | tee experiments/out_incumbent_lever_transfer_smoke.txt
Run (full):   .venv/bin/python experiments/incumbent_lever_transfer.py 2>&1 | tee experiments/out_incumbent_lever_transfer.txt
"""
import os, sys, io, contextlib, warnings, argparse
from functools import lru_cache
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd

ROOT = "/home/angelbell/dev/auto-trade"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/experiments")

from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG, at
from research.portfolio_kama import PB, kama_gate_btc, cycle_gate_pull, get_legs
from btc_family_ext_throttle import build_base
from book_leave_one_out import cdd, NEW, OLD
from book_weighting_scheme import monthly_matrix, scheme_invvol_trade, stat_custom, block_bootstrap
from radar_gate_race import kama_up

GOLD_H1 = f"{ROOT}/data/vantage_xauusd_h1.csv"
BTC_H1 = f"{ROOT}/data/vantage_btcusd_h1.csv"
REF_3LEG, REF_6LEG = 2.71, 8.19


# ---------------------------------------------------------------------------
# cached raw-data + resample (pure caching of existing loaders -- no new logic)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _d(path, tf):
    return resample(load_mt5_csv(path), tf)


def gold_d():
    return _d(GOLD_H1, "1h")


def btc_d4():
    return _d(BTC_H1, "4h")


# ---------------------------------------------------------------------------
# book arbiter: trade-resolution DD (cdd) + inv-vol-on-trade-sigma weights
# (this IS the "corrected judge" from the spec -- reused verbatim via imports)
# ---------------------------------------------------------------------------
def book_stat(legs, basket):
    M, midx = monthly_matrix(legs, basket)
    w = scheme_invvol_trade(legs, basket, M, midx)
    c, d, x, s, n = stat_custom(legs, basket, w, midx)
    return c, d, x, s, n


def with_override(legs, name, series):
    L = dict(legs)
    L[name] = series
    return L


# ---------------------------------------------------------------------------
# leg standalone descriptive metrics (plain pandas aggregates -- not signal logic)
# ---------------------------------------------------------------------------
def leg_metrics(t_time, t_R, risk=0.01):
    df = pd.DataFrame({"time": pd.DatetimeIndex(t_time), "R": np.asarray(t_R)}).sort_values("time")
    R = df.R.values
    n = len(df)
    span = max((df.time.iloc[-1] - df.time.iloc[0]).days / 365.25, 0.5)
    win = (R > 0).mean() * 100
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else np.inf
    eq = (1 + risk * R).cumprod()
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (1 / span) - 1) * 100
    yrs = sorted(df.time.dt.year.unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    y = df.time.dt.year.values
    isr = R[y < half].mean() if half else R.mean()
    oos = R[y >= half].mean() if half else R.mean()
    return dict(n=n, npy=n / span, win=win, pf=pf, meanR=R.mean(), medR=np.median(R),
                sdR=R.std(), cagr=cagr, dd=dd, cdd=cagr / max(dd, 1e-9), isr=isr, oos=oos, span=span)


def fmt_leg(m):
    return (f"n={m['n']:>4} N/yr={m['npy']:>5.1f} win={m['win']:4.1f}% PF={m['pf']:4.2f} "
            f"meanR={m['meanR']:+.3f} medR={m['medR']:+.3f} sdR={m['sdR']:.3f} "
            f"CAGR/DD={m['cdd']:5.2f} IS/OOS={m['isr']:+.2f}/{m['oos']:+.2f}")


def to_series(df):
    return pd.Series(df["R"].values, index=pd.DatetimeIndex(df["time"]))


# ---------------------------------------------------------------------------
# L1: RR-ladder leg builders (mirror get_legs()'s exact construction, override rr only)
# ---------------------------------------------------------------------------
def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def build_gold_rr(rr):
    t = _quiet(run_bo, gold_d(), SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": rr, "fwd": 500,
                                                      "daily_sma": 150, "daily_slope_k": 10}))
    return t[["time", "R"]]


def build_btc_bo_rr(rr):
    t = _quiet(run_bo, btc_d4(), SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": rr, "fwd": 300}))[["time", "R"]]
    return kama_gate_btc(t)


def build_pull_rr(rr):
    t = _quiet(run_pb, btc_d4(), "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h", "rr": rr}), 0.0)[["time", "R"]]
    return cycle_gate_pull(t)


L1_BUILDERS = {"gold_bo": build_gold_rr, "btc_bo_kama": build_btc_bo_rr, "btc_pull": build_pull_rr}
L1_CURRENT_RR = {"gold_bo": 3.0, "btc_bo_kama": 2.0, "btc_pull": 3.0}


def run_l1(legs0, rr_grid, ndraw):
    print("=" * 110)
    print("L1 -- RR ladder, one leg at a time (other legs held at CURRENT config)")
    print("=" * 110)

    per_leg_series = {}   # leg -> {rr: (df, series)}
    per_leg_book = {}      # leg -> {rr: (c3,d3,x3,s3,n3, c6,d6,x6,s6,n6)}
    for legname, builder in L1_BUILDERS.items():
        cur_rr0 = L1_CURRENT_RR[legname]
        leg_rr_grid = sorted(set(rr_grid) | {cur_rr0})
        print(f"\n--- leg = {legname}  (current RR={cur_rr0}) ---")
        print(f"  {'RR':<6}{'n':>5}{'N/yr':>7}{'win%':>7}{'PF':>6}{'meanR':>8}{'medR':>8}{'sdR':>7}"
              f"{'leg C/DD':>9}{'IS/OOS':>14}{'3-leg C/DD':>12}{'6-leg C/DD':>12}")
        per_leg_series[legname] = {}
        per_leg_book[legname] = {}
        for rr in leg_rr_grid:
            df = builder(rr)
            m = leg_metrics(df.time, df.R)
            s = to_series(df)
            per_leg_series[legname][rr] = (df, s)
            L3 = with_override(legs0, legname, s)
            c3, d3, x3, s3, n3 = book_stat(L3, OLD)
            c6, d6, x6, s6, n6 = book_stat(L3, NEW)
            per_leg_book[legname][rr] = (c3, d3, x3, s3, n3, c6, d6, x6, s6, n6)
            tag = "  <= current" if abs(rr - L1_CURRENT_RR[legname]) < 1e-9 else ""
            print(f"  {rr:<6.1f}{m['n']:>5}{m['npy']:>7.1f}{m['win']:>6.1f}%{m['pf']:>6.2f}"
                  f"{m['meanR']:>+8.3f}{m['medR']:>+8.3f}{m['sdR']:>7.3f}{m['cdd']:>9.2f}"
                  f"{m['isr']:>+6.2f}/{m['oos']:<+6.2f}{x3:>12.2f}{x6:>12.2f}{tag}")

        best_rr = max(leg_rr_grid, key=lambda rr: per_leg_book[legname][rr][2])   # best by 3-leg book CAGR/DD
        cur_rr = cur_rr0
        print(f"\n  best RR by 3-leg BOOK CAGR/DD: {best_rr} (book={per_leg_book[legname][best_rr][2]:.2f}) "
              f"vs current RR={cur_rr} (book={per_leg_book[legname][cur_rr][2]:.2f})")
        best_leg_rr = max(leg_rr_grid, key=lambda rr: leg_metrics(per_leg_series[legname][rr][0].time,
                                                                    per_leg_series[legname][rr][0].R)["cdd"])
        print(f"  best RR by LEG-standalone CAGR/DD: {best_leg_rr} "
              f"(leg CAGR/DD={leg_metrics(per_leg_series[legname][best_leg_rr][0].time, per_leg_series[legname][best_leg_rr][0].R)['cdd']:.2f})")

        # bootstrap: best (by 3-leg book) vs current, on the 3-leg BOOK's trade-resolution series
        s3_cur = per_leg_book[legname][cur_rr][3]
        s3_best = per_leg_book[legname][best_rr][3]
        named = {f"current RR={cur_rr}": s3_cur, f"best RR={best_rr}": s3_best}
        bt = block_bootstrap(named, f"current RR={cur_rr}", ndraw)
        print(f"\n  3-leg BOOK bootstrap ({ndraw} draws/block) -- P(beats current RR)")
        names = list(named.keys())
        print(f"  {'block':<7}" + "".join(f"{nm[:22]:>24}" for nm in names))
        for blk in (1, 3, 6, 12):
            row = bt[blk]
            print(f"  {f'{blk}mo':<7}" + "".join(f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(24) for nm in names))

    # combined: all 3 legs simultaneously at their own best (by 3-leg book) RR
    print("\n--- combined: all 3 legs at their OWN best-by-3-leg-book RR simultaneously ---")
    L_all = dict(legs0)
    chosen = {}
    for legname in L1_BUILDERS:
        best_rr = max(per_leg_book[legname].keys(), key=lambda rr: per_leg_book[legname][rr][2])
        chosen[legname] = best_rr
        L_all[legname] = per_leg_series[legname][best_rr][1]
    print(f"  chosen RR: {chosen}")
    c3, d3, x3, s3, n3 = book_stat(L_all, OLD)
    c6, d6, x6, s6, n6 = book_stat(L_all, NEW)
    print(f"  combined 3-leg BOOK CAGR/DD={x3:.2f} (n={n3})  vs current 3-leg book={REF_3LEG:.2f}")
    print(f"  combined 6-leg BOOK CAGR/DD={x6:.2f} (n={n6})  vs current 6-leg book={REF_6LEG:.2f}")
    print("  [multiple comparisons] this combined arm is the argmax over 3 legs x "
          f"{len(rr_grid)} RR values each -- a screen, not a confirmed result.")
    return per_leg_series, per_leg_book


# ---------------------------------------------------------------------------
# L2: PDH soft-size lever (transplant from btc15m_L)
# ---------------------------------------------------------------------------
def pdh_at(d, times):
    """previous COMPLETE day's high, ffilled to intraday index, looked up at exact trade times."""
    pdh = d["high"].resample("1D").max().dropna().shift(1).reindex(d.index, method="ffill")
    return pdh.reindex(pd.DatetimeIndex(times)).values


def build_gold_full():
    return _quiet(run_bo, gold_d(), SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                                         "daily_sma": 150, "daily_slope_k": 10}))


def build_btc_bo_full():
    t = _quiet(run_bo, btc_d4(), SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))
    return kama_gate_btc(t)


def build_pull_full():
    t = _quiet(run_pb, btc_d4(), "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)
    t = cycle_gate_pull(t)
    # ema_pullback fill_at_close=True -> entry price IS the close of the "time" bar
    e_px = btc_d4()["close"].reindex(pd.DatetimeIndex(t.time)).values
    t = t.copy()
    t["e_px"] = e_px
    return t


L2_LEGS = {
    "gold_bo": (build_gold_full, gold_d),
    "btc_bo_kama": (build_btc_bo_full, btc_d4),
    "btc_pull": (build_pull_full, btc_d4),
}


def run_l2(legs0, weak_grid, ndraw):
    print("\n" + "=" * 110)
    print("L2 -- PDH soft-size (full weight if entry close > prior-day high, else weak weight)")
    print("=" * 110)

    for legname, (builder, dfn) in L2_LEGS.items():
        note = "  [off-mechanism: btc_pull is a pullback, not a breakout -- shown for completeness]" \
            if legname == "btc_pull" else ""
        t = builder()
        d = dfn()
        pdh = pdh_at(d, t.time)
        hot = t["e_px"].values > pdh          # NaN comparisons -> False (weak) automatically
        pct_hot = np.nanmean(hot) * 100
        print(f"\n--- leg = {legname}{note} ---")
        print(f"  entries above prior-day-high: {pct_hot:.1f}%  (n={len(t)})")
        base_R = t["R"].values
        m0 = leg_metrics(t.time, base_R)
        c3_0, d3_0, x3_0, s3_0, n3_0 = book_stat(with_override(legs0, legname, to_series(t)), OLD)
        c6_0, d6_0, x6_0, s6_0, n6_0 = book_stat(with_override(legs0, legname, to_series(t)), NEW)
        print(f"  {'weak_w':<8}{'n':>5}{'win%':>7}{'PF':>6}{'meanR':>8}{'leg C/DD':>9}"
              f"{'3-leg C/DD':>12}{'6-leg C/DD':>12}")
        print(f"  {'1.0(base)':<8}{m0['n']:>5}{m0['win']:>6.1f}%{m0['pf']:>6.2f}{m0['meanR']:>+8.3f}"
              f"{m0['cdd']:>9.2f}{x3_0:>12.2f}{x6_0:>12.2f}")

        results = {"1.0(base)": (s3_0, x3_0)}    # bootstrap on the 3-leg BOOK series, not the raw leg
        for wk in weak_grid:
            Rw = base_R * np.where(hot, 1.0, wk)
            dfw = t.assign(R=Rw)
            m = leg_metrics(dfw.time, Rw)
            sw = to_series(dfw)
            c3, d3, x3, s3, n3 = book_stat(with_override(legs0, legname, sw), OLD)
            c6, d6, x6, s6, n6 = book_stat(with_override(legs0, legname, sw), NEW)
            results[f"{wk}"] = (s3, x3)   # bootstrap on the 3-leg BOOK series, not the raw leg
            print(f"  {wk:<8}{m['n']:>5}{m['win']:>6.1f}%{m['pf']:>6.2f}{m['meanR']:>+8.3f}"
                  f"{m['cdd']:>9.2f}{x3:>12.2f}{x6:>12.2f}")

        best_key = max(results, key=lambda k: results[k][1])
        if best_key != "1.0(base)":
            named = {"1.0(base)": results["1.0(base)"][0], f"weak={best_key}": results[best_key][0]}
            bt = block_bootstrap(named, "1.0(base)", ndraw)
            print(f"\n  3-leg BOOK bootstrap ({ndraw} draws/block) -- best weak={best_key} vs base(1.0)")
            names = list(named.keys())
            print(f"  {'block':<7}" + "".join(f"{nm[:20]:>22}" for nm in names))
            for blk in (1, 3, 6, 12):
                row = bt[blk]
                print(f"  {f'{blk}mo':<7}" + "".join(f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(22) for nm in names))
        else:
            print("  best weak-weight = 1.0 (base, i.e. no throttle helps) -- no bootstrap needed.")


# ---------------------------------------------------------------------------
# L3: btc_bo_kama gate confirm-TF sweep (daily current / 4h / weekly)
# ---------------------------------------------------------------------------
def run_l3(legs0, ndraw):
    print("\n" + "=" * 110)
    print("L3 -- btc_bo_kama gate confirm-TF sweep (daily=current / 4h / weekly)")
    print("=" * 110)

    d4 = btc_d4()
    t_full = _quiet(run_bo, d4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))
    pos = d4.index.get_indexer(pd.DatetimeIndex(t_full.time))

    # current (daily) via the PRODUCTION function
    t_daily_current = kama_gate_btc(t_full[["time", "R"]])

    # tie-back: daily gate reproduced via kama_up (radar_gate_race.py), same KAMA formula
    mask_daily_via_kamaup = kama_up(d4, "1D")[pos]
    t_daily_via_kamaup = t_full[["time", "R"]][mask_daily_via_kamaup]
    print(f"[tie-back] daily gate: kama_gate_btc() n={len(t_daily_current)} R_sum={t_daily_current.R.sum():.2f}  "
          f"vs kama_up(rule='1D') n={len(t_daily_via_kamaup)} R_sum={t_daily_via_kamaup.R.sum():.2f}")
    if len(t_daily_current) != len(t_daily_via_kamaup) or abs(t_daily_current.R.sum() - t_daily_via_kamaup.R.sum()) > 1e-6:
        print("  *** WARNING: daily-gate tie-back mismatch between kama_gate_btc() and kama_up(). "
              "Proceeding but flagging for review. ***")

    mask_4h = kama_up(d4, "240min")[pos]
    mask_w = kama_up(d4, "1W")[pos]
    t_4h = t_full[["time", "R"]][mask_4h]
    t_w = t_full[["time", "R"]][mask_w]

    arms = {"daily (current)": t_daily_current, "4h": t_4h, "weekly": t_w}
    print(f"\n  {'gate':<20}{'n':>5}{'N/yr':>7}{'win%':>7}{'PF':>6}{'meanR':>8}{'leg C/DD':>9}"
          f"{'IS/OOS':>14}{'3-leg C/DD':>12}{'6-leg C/DD':>12}")
    series = {}
    book3 = {}
    for name, df in arms.items():
        m = leg_metrics(df.time, df.R)
        s = to_series(df)
        series[name] = s
        c3, d3, x3, s3, n3 = book_stat(with_override(legs0, "btc_bo_kama", s), OLD)
        c6, d6, x6, s6, n6 = book_stat(with_override(legs0, "btc_bo_kama", s), NEW)
        book3[name] = (x3, s3)
        print(f"  {name:<20}{m['n']:>5}{m['npy']:>7.1f}{m['win']:>6.1f}%{m['pf']:>6.2f}"
              f"{m['meanR']:>+8.3f}{m['cdd']:>9.2f}{m['isr']:>+6.2f}/{m['oos']:<+6.2f}{x3:>12.2f}{x6:>12.2f}")

    named = {name: book3[name][1] for name in arms}
    bt = block_bootstrap(named, "daily (current)", ndraw)
    print(f"\n  3-leg BOOK bootstrap ({ndraw} draws/block) -- P(beats daily/current)")
    names = list(named.keys())
    print(f"  {'block':<7}" + "".join(f"{nm[:18]:>20}" for nm in names))
    for blk in (1, 3, 6, 12):
        row = bt[blk]
        print(f"  {f'{blk}mo':<7}" + "".join(f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(20) for nm in names))


def main(smoke=False):
    ndraw = 100 if smoke else 2000
    rr_grid = [2.0, 3.5, 5.0] if smoke else [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    weak_grid = [0.5, 0.75, 0.0]

    with contextlib.redirect_stderr(io.StringIO()):
        legs0 = build_base()   # 6-leg canonical series; also contains the 3 OLD legs as-is

    print("=" * 110)
    print("TIE-BACK -- correct arbiter (trade-resolution DD, invvol-on-trade-sigma weights) on the "
          "UNCHANGED incumbent legs")
    print("=" * 110)
    c3, d3, x3, s3, n3 = book_stat(legs0, OLD)
    c6, d6, x6, s6, n6 = book_stat(legs0, NEW)
    print(f"  3-leg incumbent book: CAGR/DD={x3:.2f}  (n={n3}, maxDD={d3:.2f}%)   target {REF_3LEG}")
    print(f"  6-leg book:           CAGR/DD={x6:.2f}  (n={n6}, maxDD={d6:.2f}%)   target {REF_6LEG}")
    if abs(x3 - REF_3LEG) > 0.02 or abs(x6 - REF_6LEG) > 0.02:
        print("\n*** TIE-BACK MISMATCH -- stopping before proceeding further. Report this. ***")
        sys.exit(1)

    run_l1(legs0, rr_grid, ndraw)
    run_l2(legs0, weak_grid, ndraw)
    run_l3(legs0, ndraw)

    print("\n" + "=" * 110)
    print("done. multiple-comparisons note: L1 swept 3 legs x 9 RR points (+1 combined-argmax arm), "
          "L2 swept 3 legs x 3 weak-weights, L3 swept 3 gate-TFs on 1 leg -- treat any single-point "
          "'best' as a screen. The bootstrap P columns (paired, same resampled months) are the "
          "load-bearing numbers, not the point CAGR/DD.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
