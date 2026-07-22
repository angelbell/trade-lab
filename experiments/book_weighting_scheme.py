"""book_weighting_scheme.py -- is btc_bo_kama's LOO improvement (6.84->7.77) a WEIGHT problem
(inv-vol-on-monthly-sigma over-weights a low-frequency leg) or a LEG problem (it drags the book
down regardless of its weight)?

Reuses (imports, does not reimplement):
  - experiments/btc_family_ext_throttle.py: build_base()          -- the 6 canonical leg series
  - experiments/book_leave_one_out.py: weights() / trade_series() / cdd() / stat()
      (weights = inv-vol on monthly-R-sum sigma, normalized to a fixed risk budget;
       trade_series = time-ordered weighted-R concatenation over the legs' common window;
       cdd = trade-resolution CAGR/maxDD/CAGR-DD from a compounding return array;
       stat = cdd() + the weighted series, for a given leg basket)

New code (this IS the experiment, not a re-derivation of the existing arbiter):
  - E1: hold the 6-leg basket fixed, scale ONLY btc_bo_kama's weight by a multiplier, and
    redistribute the freed/absorbed budget across the other 5 legs in their CURRENT relative
    proportion (mathematically: this is identical to re-deriving inv-vol weights over the
    remaining basket only, when multiplier=0 -- checked as a tie-back).
  - E2: five alternative weighting FORMULAS applied to the same fixed 6-leg (then 3-leg) basket:
    invvol_monthly (current), invvol_trade, equal, invvol_monthly_freqadj, inv_n.
  - E3: same E1/E2 questions on the 3-leg incumbent book (gold_bo, btc_bo_kama, btc_pull).

Lookahead: none introduced here -- this script only re-weights already-built, already-priced
trade series (build_base() and its underlying breakout_wave/ema_pullback runs already enforce
next-bar-open fill / intrabar SL-TP-priority / confirmed-close HTF gates). No new signal logic.

Run (full):  .venv/bin/python experiments/book_weighting_scheme.py 2>/dev/null | tee experiments/out_book_weighting_scheme.txt
Run (smoke): .venv/bin/python experiments/book_weighting_scheme.py --smoke 2>/dev/null | tee experiments/out_book_weighting_scheme_smoke.txt
"""
import sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")

from btc_family_ext_throttle import build_base
from book_leave_one_out import weights, trade_series, cdd, stat, NEW, OLD

BUDGET = 0.03
KAMA = "btc_bo_kama"


# ---------------------------------------------------------------------------
# plumbing shared by E1/E2/E3: build a trade-resolution series from an ARBITRARY
# weight vector over a FIXED basket/window (mirrors trade_series()'s concatenation
# step exactly, but takes custom weights instead of deriving them internally).
# ---------------------------------------------------------------------------
def series_from_weights(legs, basket, w, midx):
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    parts = []
    for k in basket:
        s = legs[k]
        s = s[(s.index >= st) & (s.index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    return pd.concat(parts).sort_index()


def stat_custom(legs, basket, w, midx):
    s = series_from_weights(legs, basket, w, midx)
    days = (s.index[-1] - s.index[0]).days
    c, d, x = cdd(s.values, days)
    return c, d, x, s, len(s)


def monthly_matrix(legs, basket):
    """Same construction as weights()'s internals -- needed here because E2/E4 need the
    raw monthly matrix (sigma, N-per-month), not just the final inv-vol weight vector."""
    mon = {k: legs[k].groupby(legs[k].index.to_period("M")).sum() for k in basket}
    st = max(s.index.min() for s in mon.values()); en = min(s.index.max() for s in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    return M, midx


# ---------------------------------------------------------------------------
# E2 weighting schemes -- all normalized to sum(w) == BUDGET over `basket`
# ---------------------------------------------------------------------------
def scheme_invvol_monthly(legs, basket, M, midx):
    w = 1.0 / M.std()
    return w / w.sum() * BUDGET


def scheme_invvol_trade(legs, basket, M, midx):
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    sigma = {}
    for k in basket:
        s = legs[k]; s = s[(s.index >= st) & (s.index <= en)]
        sigma[k] = s.std()
    sigma = pd.Series(sigma)
    w = 1.0 / sigma
    return w / w.sum() * BUDGET


def scheme_equal(legs, basket, M, midx):
    return pd.Series({k: BUDGET / len(basket) for k in basket})


def scheme_freqadj(legs, basket, M, midx):
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    n_trades = {}
    for k in basket:
        s = legs[k]; s = s[(s.index >= st) & (s.index <= en)]
        n_trades[k] = len(s)
    n_trades = pd.Series(n_trades)
    avg_per_month = n_trades / len(M)
    adj = M.std() / np.sqrt(avg_per_month)
    w = 1.0 / adj
    return w / w.sum() * BUDGET


def scheme_inv_n(legs, basket, M, midx):
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    n_trades = {}
    for k in basket:
        s = legs[k]; s = s[(s.index >= st) & (s.index <= en)]
        n_trades[k] = len(s)
    n_trades = pd.Series(n_trades)
    w = 1.0 / n_trades
    return w / w.sum() * BUDGET


SCHEMES = {
    "invvol_monthly (current)": scheme_invvol_monthly,
    "invvol_trade": scheme_invvol_trade,
    "equal": scheme_equal,
    "invvol_monthly_freqadj": scheme_freqadj,
    "inv_n": scheme_inv_n,
}


# ---------------------------------------------------------------------------
# generic paired circular block bootstrap over months -- same design as
# book_leave_one_out.py's main() (block 1/3/6/12mo, same seed), factored out so
# E1/E2/E3 share one implementation instead of three hand-copies.
# ---------------------------------------------------------------------------
def block_bootstrap(named_series, base_name, ndraw, seed=20260713, blocks=(1, 3, 6, 12)):
    months = sorted(set(named_series[base_name].index.to_period("M")))
    m = len(months)
    G = {name: {p: g.values for p, g in s.groupby(s.index.to_period("M"))} for name, s in named_series.items()}
    rng = np.random.default_rng(seed)
    out = {}
    for blk in blocks:
        nb = int(np.ceil(m / blk))
        D = {k: [] for k in named_series}
        for _ in range(ndraw):
            st0 = rng.integers(0, m, nb)
            order = [months[(s + j) % m] for s in st0 for j in range(blk)][:m]
            for k in named_series:
                v = np.concatenate([G[k][p] for p in order if p in G[k]])
                D[k].append(cdd(v, 365.25 * m / 12)[2])
        base_arr = np.array(D[base_name])
        row = {}
        for k in named_series:
            a = np.array(D[k])
            row[k] = (np.nanmedian(a), np.nanmean(a > base_arr) * 100)
        out[blk] = row
    return out


def print_bt(bt, names, base_name, title):
    print(f"\n{title}  (median CAGR/DD, P(beats {base_name}))")
    print(f"{'block':<8}" + "".join(f"{nm[:24]:>26}" for nm in names))
    for blk in (1, 3, 6, 12):
        row = bt[blk]
        print(f"{f'{blk}mo':<8}" + "".join(f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(26) for nm in names))


# ===========================================================================
def run_e1(legs, ndraw):
    print("=" * 100)
    print("E1 -- btc_bo_kama weight sweep (6-leg basket held fixed; other 5 rescaled to current ratio)")
    print("=" * 100)

    w0, midx = weights(legs, NEW, budget=BUDGET)
    wk0 = w0[KAMA]
    others = [k for k in NEW if k != KAMA]

    mults = [1.0, 0.75, 0.5, 0.35, 0.25, 0.1, 0.0]
    rows = {}
    for mlt in mults:
        w_new = w0.copy()
        w_new[KAMA] = wk0 * mlt
        scale = (BUDGET - wk0 * mlt) / (BUDGET - wk0)
        for k in others:
            w_new[k] = w0[k] * scale
        assert abs(w_new.sum() - BUDGET) < 1e-9, "budget leak in E1 sweep"
        c, d, x, s, n = stat_custom(legs, NEW, w_new, midx)
        rows[mlt] = dict(cagr=c, dd=d, cd=x, n=n, port=s, w=w_new)

    # tie-back: multiplier 0.0 must reproduce the LOO "minus btc_bo_kama" result (7.77).
    # n differs by construction (this sweep keeps kama's trades in the concatenated series
    # at weight 0 -- they count as n but contribute a (1+0) factor to the compounding
    # product, so CAGR/DD is unaffected; stat(legs, others) drops them from n entirely).
    # Checked: both windows are identical (2019-05..2026-05, 85mo); the n gap is exactly
    # kama's 58 trades that fall inside that window (70 total, 58 in-window).
    c_loo, d_loo, x_loo, s_loo, n_loo = stat(legs, others)
    print(f"[tie-back] mult=0.0 via redistribution formula: CAGR/DD={rows[0.0]['cd']:.2f}  "
          f"n={rows[0.0]['n']} (incl. 58 zero-weighted kama trades)   vs LOO stat(legs, minus kama) "
          f"directly: CAGR/DD={x_loo:.2f}  n={n_loo}  (target 7.77, CAGR/DD must match; n gap of 58 expected)")
    if abs(rows[0.0]["cd"] - x_loo) > 0.02:
        print("\n*** TIE-BACK MISMATCH in E1 -- stopping before proceeding further. Report this. ***")
        sys.exit(1)
    print()

    print(f"{'mult':<8}{'w_kama%':>9}{'n':>6}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>9}")
    for mlt in mults:
        r = rows[mlt]
        print(f"{mlt:<8.2f}{r['w'][KAMA]*100:>9.3f}{r['n']:>6}{r['cagr']:>8.1f}%{r['dd']:>7.2f}%{r['cd']:>9.2f}")

    peak_mlt = max(mults, key=lambda mlt: rows[mlt]["cd"])
    print(f"\npeak by CAGR/DD: mult={peak_mlt} (CAGR/DD={rows[peak_mlt]['cd']:.2f}) "
          f"vs current mult=1.0 (CAGR/DD={rows[1.0]['cd']:.2f})")

    named = {"current (mult=1.0)": rows[1.0]["port"], f"peak (mult={peak_mlt})": rows[peak_mlt]["port"]}
    bt = block_bootstrap(named, "current (mult=1.0)", ndraw)
    print_bt(bt, list(named.keys()), "current (mult=1.0)", "E1 bootstrap -- peak vs current")

    return rows, peak_mlt


def run_e2(legs, basket, basket_name, ndraw, base_scheme_name="invvol_monthly (current)"):
    print()
    print("=" * 100)
    print(f"E2 -- weighting-scheme shootout on the {basket_name} basket ({', '.join(basket)})")
    print("=" * 100)

    M, midx = monthly_matrix(legs, basket)

    # tie-back: scheme_invvol_monthly must reproduce weights() exactly
    w_ref, midx_ref = weights(legs, basket, budget=BUDGET)
    w_chk = scheme_invvol_monthly(legs, basket, M, midx)
    mism = (w_chk.reindex(basket) - w_ref.reindex(basket)).abs().max()
    print(f"[tie-back] scheme_invvol_monthly vs imported weights(): max|diff|={mism:.6f} "
          f"(months {len(midx)} vs {len(midx_ref)})")
    if mism > 1e-9 or len(midx) != len(midx_ref):
        print("\n*** TIE-BACK MISMATCH in E2 -- stopping before proceeding further. Report this. ***")
        sys.exit(1)

    results = {}
    for name, fn in SCHEMES.items():
        w = fn(legs, basket, M, midx)
        c, d, x, s, n = stat_custom(legs, basket, w, midx)
        results[name] = dict(cagr=c, dd=d, cd=x, n=n, port=s, w=w)

    print(f"\nweights (account % per 1R):")
    hdr = f"{'scheme':<28}" + "".join(f"{k:>14}" for k in basket)
    print(hdr)
    for name in SCHEMES:
        r = results[name]
        print(f"{name:<28}" + "".join(f"{r['w'][k]*100:>14.3f}" for k in basket))

    print(f"\n{'scheme':<28}{'n':>6}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>9}")
    for name in SCHEMES:
        r = results[name]
        print(f"{name:<28}{r['n']:>6}{r['cagr']:>8.1f}%{r['dd']:>7.2f}%{r['cd']:>9.2f}")

    best_name = max(SCHEMES, key=lambda k: results[k]["cd"])
    print(f"\nbest by CAGR/DD: {best_name} (CAGR/DD={results[best_name]['cd']:.2f}) "
          f"vs current {base_scheme_name} (CAGR/DD={results[base_scheme_name]['cd']:.2f})")

    named = {base_scheme_name: results[base_scheme_name]["port"]}
    if best_name != base_scheme_name:
        named[best_name] = results[best_name]["port"]
    bt = block_bootstrap(named, base_scheme_name, ndraw)
    print_bt(bt, list(named.keys()), base_scheme_name, f"E2 bootstrap ({basket_name}) -- best scheme vs current")

    return results, best_name


def run_e3(legs, ndraw, e2_results_old, e2_best_old):
    print()
    print("=" * 100)
    print("E3 -- 3-leg incumbent book (gold_bo, btc_bo_kama, btc_pull): drop-kama / scheme replay")
    print("=" * 100)

    c3, d3, x3, s3, n3 = stat(legs, OLD)
    print(f"[tie-back] 3-leg incumbent via stat(legs, OLD): CAGR/DD={x3:.2f}  maxDD={d3:.2f}%  n={n3}  "
          f"(target 3.11)")
    if abs(x3 - 3.11) > 0.02:
        print("\n*** TIE-BACK MISMATCH in E3 -- stopping before proceeding further. Report this. ***")
        sys.exit(1)

    others_old = [k for k in OLD if k != KAMA]
    c2, d2, x2, s2, n2 = stat(legs, others_old)
    print(f"2-leg (OLD minus kama, {', '.join(others_old)}): CAGR/DD={x2:.2f}  maxDD={d2:.2f}%  n={n2}")

    print(f"\n{'book':<40}{'n':>6}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>9}")
    print(f"{'3-leg incumbent (invvol_monthly)':<40}{n3:>6}{c3:>8.1f}%{d3:>7.2f}%{x3:>9.2f}")
    print(f"{'2-leg (drop btc_bo_kama)':<40}{n2:>6}{c2:>8.1f}%{d2:>7.2f}%{x2:>9.2f}")
    for name in SCHEMES:
        r = e2_results_old[name]
        tag = f"3-leg, scheme={name}"
        print(f"{tag:<40}{r['n']:>6}{r['cagr']:>8.1f}%{r['dd']:>7.2f}%{r['cd']:>9.2f}")

    named = {"3-leg incumbent": s3, "2-leg (drop kama)": s2}
    if e2_best_old != "invvol_monthly (current)":
        named[f"3-leg, scheme={e2_best_old}"] = e2_results_old[e2_best_old]["port"]
    bt = block_bootstrap(named, "3-leg incumbent", ndraw)
    print_bt(bt, list(named.keys()), "3-leg incumbent", "E3 bootstrap")


def main(smoke=False):
    ndraw = 100 if smoke else 2000
    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()

    e1_rows, peak_mlt = run_e1(legs, ndraw)
    e2_new_results, e2_new_best = run_e2(legs, NEW, "6-leg (NEW)", ndraw)
    e2_old_results, e2_old_best = run_e2(legs, OLD, "3-leg (OLD/incumbent)", ndraw)
    run_e3(legs, ndraw, e2_old_results, e2_old_best)

    print()
    print("=" * 100)
    print("done. multiple-comparisons note: E1 swept 7 multiplier points, E2 swept 5 weighting "
          "schemes x2 baskets -- treat any single-point 'best' as a screen, not a confirmed result; "
          "the bootstrap P columns above are the load-bearing numbers, not the point CAGR/DD.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
