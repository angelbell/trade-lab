"""btc15m_ext_throttle_verify.py -- two falsification tests on the 4-week-extension throttle
finding from book_healing_veto.py / btc_family_ext_throttle.py (book 12.03 -> 12.70 by halving
btc15m_L trades entered when BTC's trailing-4-week return is >= its own IS-75th pct).

Reuses (no reimplementation):
    btc_family_ext_throttle.build_base   -- the 6-leg book construction (tie-back target 12.03)
    btc_family_ext_throttle.book_gen     -- inv-vol / 3%-risk book math
    btc_family_ext_throttle.flag_hot     -- per-leg IS-quantile "hot" trade flag (no lookahead)
    btc_family_ext_throttle.ret4w_daily  -- the ret4w context series (prior completed daily bar)
    book_bootstrap_arbiter.cdd           -- CAGR/DD of a monthly-return array

Q1  HILL vs EDGE?  btc15m_L-ONLY throttle grid: quantile in {.50,.60,.70,.75,.80,.85,.90,.95,.98}
    x weight in {0,.25,.5,.75,1.0} (w=1.0 must reproduce A0=12.03 exactly, sanity check). Read the
    book CAGR/DD surface: a monotonic climb back to 12.03 as quantile->1.0 is a FALSE plateau (an
    edge of "throttle nothing"); a true interior peak that falls off on BOTH sides is a real hill.
    Block-bootstrap (12mo block, 2000 draws) the best cell and the pre-registered cell (q=.75,
    w=.5) against A0.

Q2  DOES "RAN HARD -> WEAK LONG" SURVIVE MULTIPLE-COMPARISON CORRECTION?  Pool trades (their own R,
    unscaled) from the 3 long BTC legs (btc_bo_kama, btc_pull, btc15m_L), cut by ret4w into
    quintiles (n/meanR/medianR/PF per bin), and test observed_stat = meanR(top quintile) -
    meanR(rest) against a circular-shift null of the DAILY ret4w series (shift amount random,
    >=30 days, 1000 draws): (a) naive one-sided p from the fixed quintile split, (b) a
    multiplicity-corrected p using the SAME 9-quantile grid as Q1 (best-of-9 statistic, real vs
    null-best-of-9). Repeated for btc15m_L alone.

Run:        .venv/bin/python scratchpad/btc15m_ext_throttle_verify.py
Run(smoke): .venv/bin/python scratchpad/btc15m_ext_throttle_verify.py --smoke
"""
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from btc_family_ext_throttle import build_base, book_gen, flag_hot, ret4w_daily, BTC_LEGS
from book_bootstrap_arbiter import cdd

QS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
WS = [0.0, 0.25, 0.5, 0.75, 1.0]
LONG_LEGS = ["btc_bo_kama", "btc_pull", "btc15m_L"]


def throttle_fast(legs, which, q, w, ctx_daily):
    """Same logic as btc_family_ext_throttle.throttle(), but takes a precomputed ctx_daily so we
    don't re-load/re-derive the ret4w series from the CSV on every one of the ~45 grid cells."""
    out = {}
    for name in which:
        s = legs[name]
        hot, thr = flag_hot(s.index, ctx_daily, q)
        out[name] = pd.Series(s.values * np.where(hot, w, 1.0), index=s.index)
    return out


def n_hot_for(legs, leg_name, q, ctx_daily):
    s = legs[leg_name]
    hot, thr = flag_hot(s.index, ctx_daily, q)
    return int(hot.sum()), thr


def pf(x):
    pos, neg = x[x > 0].sum(), abs(x[x <= 0].sum())
    return pos / neg if neg > 0 else np.nan


def hot_mask_thr(ctx, q):
    """top-(1-q) fraction of ctx by a fixed full-sample quantile threshold (not per-leg IS-split
    -- this is a symmetric statistical test on the pooled/solo trade set, not a trading rule)."""
    thr = np.nanquantile(ctx, q)
    return np.isfinite(ctx) & (ctx >= thr)


def stat_at_q(R, ctx, q):
    hot = hot_mask_thr(ctx, q)
    rest = np.isfinite(ctx) & ~hot
    if hot.sum() < 2 or rest.sum() < 2:
        return np.nan
    return R[hot].mean() - R[rest].mean()


def grid_min_stat(R, ctx, qs=QS):
    vals = [stat_at_q(R, ctx, q) for q in qs]
    if all(np.isnan(v) for v in vals):
        return np.nan
    return np.nanmin(vals)


def qcut_top_vs_rest(R, ctx, nbins=5):
    """fixed-count quintile split (pd.qcut): returns (stat, bin_labels, fin_mask)."""
    fin = np.isfinite(ctx)
    bins = pd.qcut(ctx[fin], nbins, labels=False, duplicates="drop")
    top = bins == np.nanmax(bins)
    Rf = R[fin]
    stat = Rf[top].mean() - Rf[~top].mean()
    return stat, bins, fin


def quintile_table(R, ctx, nbins=5):
    fin = np.isfinite(ctx)
    bins = pd.qcut(ctx[fin], nbins, labels=False, duplicates="drop")
    Rf = R[fin]
    rows = []
    for k in range(int(np.nanmax(bins)) + 1):
        sub = Rf[bins == k]
        rows.append(dict(bin=k, n=len(sub), meanR=sub.mean(), medianR=np.median(sub), pf=pf(sub)))
    return rows


def circular_shift(ctx_daily, shift):
    vals = np.roll(ctx_daily.values, shift)
    return pd.Series(vals, index=ctx_daily.index)


def run_shift_null(R_pool, idx_pool, R_solo, idx_solo, ctx_daily, n_shift, seed=20260713):
    N = len(ctx_daily)
    rng = np.random.default_rng(seed)
    lo, hi = 30, N - 30
    shifts = rng.integers(lo, hi, n_shift)
    null_naive_pool, null_grid_pool = [], []
    null_naive_solo, null_grid_solo = [], []
    for sh in shifts:
        shifted = circular_shift(ctx_daily, sh)
        ctx_p = shifted.reindex(idx_pool, method="ffill").values
        ctx_s = shifted.reindex(idx_solo, method="ffill").values
        stat_p, _, _ = qcut_top_vs_rest(R_pool, ctx_p)
        stat_s, _, _ = qcut_top_vs_rest(R_solo, ctx_s)
        null_naive_pool.append(stat_p)
        null_naive_solo.append(stat_s)
        null_grid_pool.append(grid_min_stat(R_pool, ctx_p))
        null_grid_solo.append(grid_min_stat(R_solo, ctx_s))
    return (np.array(null_naive_pool), np.array(null_grid_pool),
            np.array(null_naive_solo), np.array(null_grid_solo))


def one_sided_p(obs, null_arr):
    a = null_arr[np.isfinite(null_arr)]
    if len(a) == 0 or not np.isfinite(obs):
        return np.nan
    return float(np.mean(a <= obs))


def bootstrap_arm(port_arrays, months, ndraw, blocks=(1, 3, 6, 12), seed=20260713):
    rng = np.random.default_rng(seed)
    names = list(port_arrays.keys())
    base_name = names[0]
    out = {}
    for blk in blocks:
        nb = int(np.ceil(months / blk))
        D = {k: [] for k in names}
        for _ in range(ndraw):
            st = rng.integers(0, months, nb)
            k_ = np.concatenate([(np.arange(s, s + blk) % months) for s in st])[:months]
            for k in names:
                D[k].append(cdd(port_arrays[k][k_], months))
        base_arr = np.array(D[base_name])
        row = {}
        for k in names:
            a = np.array(D[k])
            row[k] = (np.nanmedian(a), np.nanmean(a > base_arr) * 100)
        out[blk] = row
    return out


def main(smoke=False):
    ndraw = 200 if smoke else 2000
    n_shift = 100 if smoke else 1000

    legs = build_base()
    ctx_daily = ret4w_daily()

    # --- tie-back ---
    cagr0, dd0, cd0, port0 = book_gen(legs)
    print(f"[tie-back] book_gen(legs), no override: book CAGR/DD={cd0:.2f}  DD={dd0:.1f}%  (target 12.03)")
    if abs(cd0 - 12.03) > 0.05:
        print("\n*** TIE-BACK MISMATCH -- stopping. Report this. ***")
        return
    print()

    # =========================================================================================
    # Q1 -- btc15m_L-only A4 grid: hill or edge?
    # =========================================================================================
    print("=" * 100)
    print("Q1 -- btc15m_L-ONLY throttle grid (quantile x weight): book CAGR/DD  (base A0 = 12.03)")
    print("=" * 100)
    grid_cd = {}
    grid_port = {}
    n_hot_row = {}
    mismatches = []
    for q in QS:
        nh, thr = n_hot_for(legs, "btc15m_L", q, ctx_daily)
        n_hot_row[q] = nh
        for w in WS:
            ov = throttle_fast(legs, ["btc15m_L"], q, w, ctx_daily)
            cagr, dd, cd, port = book_gen(legs, ov)
            grid_cd[(q, w)] = cd
            grid_port[(q, w)] = port
            if w == 1.0 and abs(cd - 12.03) > 0.005:
                mismatches.append((q, w, cd))

    if mismatches:
        print("\n*** w=1.0 sanity check FAILED (should equal 12.03 exactly for every quantile) ***")
        for q, w, cd in mismatches:
            print(f"    q={q} w={w} -> book CAGR/DD={cd:.4f}")
        print("*** stopping -- report this as a bug. ***")
        return

    print(f"{'quantile':<10}{'n_hot':<8}" + "".join(f"w={w:<9}" for w in WS))
    for q in QS:
        row = "".join(f"{grid_cd[(q, w)]:<12.2f}" for w in WS)
        print(f"{q:<10.2f}{n_hot_row[q]:<8}{row}")
    print("\n  (w=1.0 column = sanity check, all cells must be 12.03 -- confirmed above)")
    print("  reading: fix a weight column (e.g. w=0.5) and read down the quantile rows. Monotonic")
    print("  climb toward 12.03 as quantile->0.98 = EDGE (climbing toward 'throttle nothing').")
    print("  An interior peak that falls on both sides = HILL (real plateau).")

    # best cell among meaningful weights (exclude the trivial w=1.0 identity column)
    candidates = {(q, w): grid_cd[(q, w)] for q in QS for w in WS if w != 1.0}
    best_cell = max(candidates, key=candidates.get)
    bq, bw = best_cell
    print(f"\n  best cell (w != 1.0): quantile={bq}, weight={bw}, book CAGR/DD={grid_cd[best_cell]:.2f}"
          f"  (n_hot={n_hot_row[bq]})")
    prereg_cell = (0.75, 0.5)
    print(f"  pre-registered cell: quantile={prereg_cell[0]}, weight={prereg_cell[1]}, "
          f"book CAGR/DD={grid_cd[prereg_cell]:.2f}  (n_hot={n_hot_row[prereg_cell[0]]})")

    # block bootstrap: A0 vs best cell vs prereg cell
    print()
    print("-" * 100)
    print(f"Q1 block bootstrap (book monthly returns, {ndraw} draws/block-length, paired)")
    print("-" * 100)
    common_idx = port0.index
    arm_ports = {"A0 base": port0.values}
    if best_cell != prereg_cell:
        arm_ports[f"best q{bq}_w{bw}"] = grid_port[best_cell].reindex(common_idx).values
    arm_ports[f"prereg q0.75_w0.5"] = grid_port[prereg_cell].reindex(common_idx).values
    months = len(common_idx)
    bt = bootstrap_arm(arm_ports, months, ndraw)
    names = list(arm_ports.keys())
    print(f"book months = {months}")
    print(f"{'block':<8}" + "".join(f"{nm:>26}" for nm in names))
    for blk in (1, 3, 6, 12):
        row = bt[blk]
        print(f"{f'{blk}mo':<8}" + "".join(f"{row[nm][0]:.2f} (P{row[nm][1]:.0f}%)".rjust(26) for nm in names))
    print("  P = P(arm's book CAGR/DD > A0's) on the same resampled months (paired). ~50% = indistinguishable.")

    # =========================================================================================
    # Q2 -- pooled long-leg trades: does "ran hard -> weak long" survive circular-shift nulls?
    # =========================================================================================
    print()
    print("=" * 100)
    print("Q2 -- pooled long BTC legs (btc_bo_kama, btc_pull, btc15m_L), cut by ret4w into quintiles")
    print("=" * 100)
    R_pool = np.concatenate([legs[n].values for n in LONG_LEGS])
    idx_pool = legs[LONG_LEGS[0]].index
    for n in LONG_LEGS[1:]:
        idx_pool = idx_pool.append(legs[n].index)
    ctx_pool = ctx_daily.reindex(idx_pool, method="ffill").values

    rows = quintile_table(R_pool, ctx_pool)
    print(f"{'quintile(1=low..5=high ret4w)':<32}{'n':>6}{'meanR':>10}{'medianR':>10}{'PF':>8}")
    for r in rows:
        print(f"{r['bin']+1:<32}{r['n']:>6}{r['meanR']:>+10.3f}{r['medianR']:>+10.3f}{r['pf']:>8.2f}")
    means = [r['meanR'] for r in rows]
    monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1)) or \
                all(means[i] >= means[i+1] for i in range(len(means)-1))
    print(f"\n  monotonic across quintiles: {monotonic}")

    obs_naive_pool, bins_pool, fin_pool = qcut_top_vs_rest(R_pool, ctx_pool)
    obs_grid_pool = grid_min_stat(R_pool, ctx_pool)
    print(f"\n  observed stat (top quintile meanR - rest meanR), POOLED  = {obs_naive_pool:+.4f}")
    print(f"  observed BEST-OF-9-GRID stat (min over quantile grid), POOLED = {obs_grid_pool:+.4f}")

    idx_solo = pd.DatetimeIndex(legs["btc15m_L"].index)
    R_solo = legs["btc15m_L"].values
    ctx_solo = ctx_daily.reindex(idx_solo, method="ffill").values
    obs_naive_solo, _, _ = qcut_top_vs_rest(R_solo, ctx_solo)
    obs_grid_solo = grid_min_stat(R_solo, ctx_solo)
    print(f"\n  observed stat (top quintile meanR - rest meanR), btc15m_L ALONE = {obs_naive_solo:+.4f}")
    print(f"  observed BEST-OF-9-GRID stat, btc15m_L ALONE = {obs_grid_solo:+.4f}")

    print(f"\nrunning circular-shift null ({n_shift} shifts, daily ret4w series rolled, min |shift|=30d)...")
    nn_pool, ng_pool, nn_solo, ng_solo = run_shift_null(R_pool, idx_pool, R_solo, idx_solo, ctx_daily, n_shift)

    p_naive_pool = one_sided_p(obs_naive_pool, nn_pool)
    p_grid_pool = one_sided_p(obs_grid_pool, ng_pool)
    p_naive_solo = one_sided_p(obs_naive_solo, nn_solo)
    p_grid_solo = one_sided_p(obs_grid_solo, ng_solo)

    print()
    print("-" * 100)
    print("Q2 circular-shift null p-values (one-sided: P(null stat <= observed stat))")
    print("-" * 100)
    print(f"{'':<28}{'naive (fixed quintile)':>26}{'multiplicity-corrected (best-of-9)':>38}")
    print(f"{'POOLED (3 long legs)':<28}{f'p={p_naive_pool:.3f}':>26}{f'p={p_grid_pool:.3f}':>38}")
    print(f"{'btc15m_L ALONE':<28}{f'p={p_naive_solo:.3f}':>26}{f'p={p_grid_solo:.3f}':>38}")
    print(f"\n  null distribution summary (mean, std, min, max):")
    for nm, arr in (("naive pool", nn_pool), ("grid pool", ng_pool),
                    ("naive solo", nn_solo), ("grid solo", ng_solo)):
        a = arr[np.isfinite(arr)]
        print(f"    {nm:<12} mean={a.mean():+.4f}  std={a.std():.4f}  min={a.min():+.4f}  max={a.max():+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
