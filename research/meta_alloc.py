"""meta_alloc.py -- DYNAMIC cross-leg risk allocation: tilt a FIXED total risk budget toward
whichever leg is currently "working", recomputed monthly from trailing data (no lookahead).

This is the CROSS-leg, monthly-granularity cousin of the (dead) within-leg equity gate, and the
SAME FAMILY as the (rejected) vol-targeting overlay -- so the bar is pre-registered and high:
a dynamic scheme must beat STATIC inv-vol OUT-OF-SAMPLE *without* degrading IS (OOS-up-via-IS-down
= regime-luck = the exact failure that killed vol-targeting), survive a lookback PLATEAU, and clear
a turnover cost + a month-order reshuffle null. Thin sample caveat: only 2-3 legs, ~96 sparse months.

Gate-keeper (Part 1): cross-leg momentum needs monthly relative-return PERSISTENCE; if monthly leg
returns are iid and the trailing-winner doesn't out-return the trailing-loser beyond a reshuffle
null, the idea is dead on arrival and we STOP (no fishing a lucky lookback on a thin grid).

NB: monthly-grid absolute CAGR/DD runs HIGHER than trade-level -- compare schemes WITHIN this frame
(the static inv_vol line here is the benchmark), never against the 1.83 trade-level number.

  .venv/bin/python research/meta_alloc.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.portfolio_kama import get_legs
from research.portfolio_alloc import (monthly_matrix, cagr_dd_monthly, port_ret,
                                      w_equal, w_inv_vol, SPLIT, RISK)

RNG = np.random.default_rng(7)


# ============================== Part 1: monthly persistence pre-test ==============================
def runs_z(sign):
    s = sign[sign != 0]
    n1 = int((s > 0).sum()); n2 = int((s < 0).sum()); n = n1 + n2
    if n1 == 0 or n2 == 0:
        return np.nan
    runs = 1 + int((s[1:] != s[:-1]).sum())
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    return (runs - mu) / np.sqrt(var) if var > 0 else np.nan


def xs_spread(M, L):
    """next-month return of trailing-L-month WINNER leg minus LOSER leg, averaged. >0 = momentum."""
    trail = M.rolling(L).sum().shift(1)                 # trailing cum return, no lookahead
    diffs = []
    for i in range(len(M)):
        row = trail.iloc[i]
        if row.isna().any():
            continue
        win, los = row.idxmax(), row.idxmin()
        if win == los:
            continue
        diffs.append(M.iloc[i][win] - M.iloc[i][los])
    return np.mean(diffs) if diffs else np.nan


def part1(M, B=2000):
    print("\n" + "=" * 82)
    print("1. MONTHLY persistence pre-test (gate-keeper). Cross-leg momentum needs monthly relative")
    print("   persistence; iid monthly leg-returns => dynamic tilt is worthless. acf>0 / spread>0.")
    print(f"\n  per-leg monthly-R autocorr (n_months={len(M)}):")
    for c in M.columns:
        r = M[c].values
        acf = [np.corrcoef(r[:-k], r[k:])[0, 1] for k in (1, 2, 3)]
        print(f"    {c:<14} acf1={acf[0]:+.2f} acf2={acf[1]:+.2f} acf3={acf[2]:+.2f}  "
              f"runsZ={runs_z(np.sign(r)):+.2f}")
    print(f"\n  cross-sectional momentum (winner-minus-loser next-month R; reshuffle null B={B}):")
    persistent = False
    for L in (3, 6, 12):
        obs = xs_spread(M, L)
        null = np.empty(B)
        for b in range(B):
            Msh = M.sample(frac=1.0, replace=False, random_state=int(RNG.integers(1e9))).reset_index(drop=True)
            Msh.index = M.index
            null[b] = xs_spread(Msh, L)
        p = np.nanmean(null >= obs)
        flag = "PERSISTENT" if p < 0.10 else ""
        persistent |= p < 0.10
        print(f"    L={L:<3} spread={obs:+.4f}  p(shuffle)={p:.3f}  {flag}")
    # sanity: synthetic iid matrix -> p ~ 0.5
    iidM = pd.DataFrame(RNG.standard_normal((len(M), M.shape[1])), index=M.index, columns=M.columns)
    obs = xs_spread(iidM, 6)
    null = np.array([xs_spread(iidM.sample(frac=1.0).set_axis(iidM.index), 6) for _ in range(B)])
    print(f"    [iid sanity L=6] spread={obs:+.4f} p={np.nanmean(null>=obs):.3f} (must be ~0.5)")
    return persistent


# ============================== Part 2: dynamic schemes (no lookahead) ==============================
def dyn_weights(M, L, mode, budget):
    """weights[t] from trailing window ending t-1, renormalized to budget. returns DataFrame."""
    n = M.shape[1]
    W = pd.DataFrame(0.0, index=M.index, columns=M.columns)
    for i in range(len(M)):
        if i < L:
            W.iloc[i] = budget / n; continue              # warmup = equal
        win = M.iloc[i - L:i]                              # strictly past (i-L .. i-1)
        if mode == "relstr":
            cum = win.sum().values
            raw = np.clip(cum - cum.min() + 1e-9, 0, None)  # shift to non-neg, tilt to recent winners
            w = raw / raw.sum() if raw.sum() > 0 else np.full(n, 1 / n)
        elif mode == "riskparity":
            v = win.std().values
            raw = 1.0 / np.where(v > 0, v, np.inf)
            w = raw / raw.sum()
        else:  # eq_ma: keep legs whose trailing equity > its own MA, reallocate to survivors
            on = np.array([win[c].sum() > 0 for c in M.columns], float)
            w = on / on.sum() if on.sum() > 0 else np.full(n, 1 / n)
        W.iloc[i] = w * budget
    return W


def dyn_ret(M, W):
    return (M * W).sum(axis=1)


def turnover_cost(W, bps=5.0):
    """monthly sum|dw| * bps -> account-% drag series (static book ~ 0 turnover)."""
    dW = W.diff().abs().sum(axis=1).fillna(0.0)
    return dW * (bps / 1e4)


def split_cdd(ret):
    isr = ret[ret.index.year < SPLIT]; oos = ret[ret.index.year >= SPLIT]
    return (cagr_dd_monthly(ret)[2], cagr_dd_monthly(isr)[2], cagr_dd_monthly(oos)[2])


def report(tag, ret, bench_oos):
    f, i, o = split_cdd(ret)
    mark = "PASS?" if (o > bench_oos[0] and i >= bench_oos[1] - 0.10) else ""
    print(f"  {tag:<22} FULL={f:4.2f}  IS={i:4.2f}  OOS={o:4.2f}  {mark}")
    return f, i, o


def main():
    legs = get_legs()
    books = {"2leg(G.bo+B.K)": {k: legs[k] for k in ("gold_bo", "btc_bo_kama")},
             "3leg(all)": legs}
    for bname, blegs in books.items():
        M = monthly_matrix(blegs)
        Mis = M[M.index.year < SPLIT]
        n = M.shape[1]; budget = n * RISK
        print(f"\n{'='*82}\n=== {bname}  legs={list(M.columns)}  months={len(M)} (IS={len(Mis)}) "
              f"budget={budget*100:.0f}% ===")

        # benchmarks (static)
        eq_i, eq_o = split_cdd(port_ret(M, w_equal(Mis, budget)))[1:]
        iv = port_ret(M, w_inv_vol(Mis, budget))
        iv_f, iv_i, iv_o = split_cdd(iv)
        print("  -- STATIC benchmarks (the bar to beat = inv_vol) --")
        print(f"  {'equal(book)':<22} IS={eq_i:4.2f} OOS={eq_o:4.2f}")
        print(f"  {'inv_vol(adopted)':<22} FULL={iv_f:4.2f}  IS={iv_i:4.2f}  OOS={iv_o:4.2f}  <-- BENCHMARK")
        bench = (iv_o, iv_i)

        if not part1(M):
            print(f"\n  >>> [{bname}] NO monthly persistence -> dynamic cross-leg tilt DEAD on arrival. STOP.")
            continue

        print("\n  -- LEVER: dynamic schemes (no-lookahead), lookback L plateau, +turnover cost --")
        for mode in ("relstr", "riskparity", "eq_ma"):
            print(f"\n    mode={mode}:")
            for L in (3, 6, 9, 12):
                W = dyn_weights(M, L, mode, budget)
                gross = dyn_ret(M, W)
                net = gross - turnover_cost(W, bps=5.0)
                _, i_, o_ = split_cdd(net)
                f_g, _, o_g = split_cdd(gross)
                mark = "PASS?" if (o_ > bench[0] and i_ >= bench[1] - 0.10) else ""
                print(f"      L={L:<3} net FULL={split_cdd(net)[0]:4.2f} IS={i_:4.2f} OOS={o_:4.2f} "
                      f"(gross OOS={o_g:4.2f})  {mark}")
    print(f"\n{'='*82}")
    print("PASS rule: net OOS > static inv_vol OOS AND IS not materially below it (OOS-up-via-IS-down")
    print("= regime-luck = REJECT). Thin sample; in-sample only -> live-forward arbitrates regime.")


if __name__ == "__main__":
    main()
