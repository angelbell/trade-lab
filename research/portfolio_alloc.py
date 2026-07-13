"""portfolio_alloc.py -- two DD-side levers on the validated book (no new entries):
  (1) LEG WEIGHTING   -- redistribute a FIXED total risk budget across legs (equal vs
                         inverse-vol vs inverse-DD vs min-variance vs IS-best-CAGR/DD).
  (2) VOL TARGETING   -- scale each leg's monthly exposure by target/trailing-realized-vol
                         (constant ex-ante vol; trailing window = no lookahead).

Discipline (this is the overfit-trap zone):
  - Work on a MONTHLY return grid (covariance/vol well-defined).
  - Total risk budget held CONSTANT (sum of weights = n_legs * 1%) so weighting comparisons
    are apples-to-apples (diversification cuts DD only at constant total risk).
  - Weights / vol-windows chosen on IS (<2022) ONLY, judged on OOS (>=2022) + plateau.
  - Equal-weight = the current book (1% per leg) = the bar to beat.

  .venv/bin/python research/portfolio_alloc.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.portfolio_kama import get_legs

SPLIT = 2022
RISK = 0.01            # per-leg risk in the current book
TARGET_VOL = None      # set from equal-weight IS vol so vol-target is risk-matched


def monthly_matrix(legs):
    """legs: {name: df(time,R)} -> monthly R-sum DataFrame (cols=legs, index=month)."""
    cols = {}
    for name, t in legs.items():
        s = t.copy()
        s["m"] = s.time.dt.to_period("M")
        cols[name] = s.groupby("m").R.sum()
    M = pd.concat(cols, axis=1).fillna(0.0)
    M.index = M.index.to_timestamp()
    return M


def cagr_dd_monthly(ret):
    """ret: monthly portfolio return series (already in account-% terms). -> CAGR%, DD%, ratio.

    ⚠️ The DD here is measured on the MONTHLY equity curve, so every drawdown that opens AND
    closes inside a month is invisible. That is fine for weight SELECTION (below) but it is NOT
    a drawdown a live account would feel: with a high-frequency leg in the book it collapses
    maxDD onto 'the worst single month' and CAGR/DD stops meaning anything. Report with
    cagr_dd_trades() instead. Kept only for w_is_best's IS objective, where the two agree.
    """
    ret = ret.dropna()
    if len(ret) < 6:
        return 0.0, 0.0, 0.0
    eq = (1 + ret).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = len(ret) / 12.0
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def trade_series(legs, w_map, scale=None):
    """All legs' trades merged into one account-return series, in time order.

    w_map: {leg: account fraction risked per 1R}.  scale: optional monthly Series (vol-target
    leverage), applied to whichever month each trade falls in.
    """
    parts = []
    for name, t in legs.items():
        s = pd.Series(t.R.values * w_map[name], index=pd.DatetimeIndex(t.time))
        if scale is not None:
            k = scale.reindex(s.index.to_period("M").to_timestamp()).values
            s = s * np.nan_to_num(k, nan=1.0)
        parts.append(s)
    return pd.concat(parts).sort_index()


def cagr_dd_trades(s):
    """CAGR%, maxDD%, ratio measured on the TRADE-resolution equity curve.

    This is the arbiter. Monthly aggregation hides intra-month drawdowns entirely; on a book
    containing a 15m leg that understated maxDD by ~2x and reordered every arm we compared
    (2026-07-13). Legs trade at wildly different frequencies, so only a trade-ordered curve
    puts them on the same footing.
    """
    s = s.dropna()
    if len(s) < 10:
        return 0.0, 0.0, 0.0
    eq = np.cumprod(1 + s.values)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100
    days = max((s.index[-1] - s.index[0]).days, 1)
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def port_ret(M, w):
    """portfolio monthly return given per-leg risk weights w (account fraction per unit R)."""
    return (M * w).sum(axis=1)


def report(tag, M, w, legs):
    s = trade_series(legs, dict(zip(M.columns, w)))
    isr = s[s.index.year < SPLIT]
    oos = s[s.index.year >= SPLIT]
    cf, df_, rf = cagr_dd_trades(s)
    _, _, ri = cagr_dd_trades(isr)
    _, _, ro = cagr_dd_trades(oos)
    wt = " ".join(f"{x*100:.2f}" for x in w)
    print(f"  {tag:<16} w[{wt}]  FULL CAGR/DD={rf:4.2f} (C{cf:+.0f}/DD{df_:.0f})  "
          f"IS={ri:4.2f}  OOS={ro:4.2f}")
    return rf, ro


# ---------- weighting schemes (all derived from IS stats, budget held constant) ----------
def w_equal(Mis, budget):
    n = Mis.shape[1]
    return np.full(n, budget / n)


def w_inv_vol(Mis, budget):
    v = Mis.std().values
    raw = 1.0 / np.where(v > 0, v, np.inf)
    return raw / raw.sum() * budget


def w_inv_vol_trade(Mis, budget, legs=None):
    """inv-vol on TRADE-level R sigma, not monthly sigma.

    Monthly sigma is contaminated by FREQUENCY: a leg that trades 8x/yr has mostly-zero months,
    so its monthly sigma is small and inv-vol hands it the biggest per-trade size -- 'low vol'
    that is really 'rarely trades' (2026-07-13: btc_bo_kama got 1.01%/trade vs btc15m_L's 0.23%,
    a 4.4x gap on legs of similar per-trade risk). Use this whenever the book mixes timeframes.
    """
    sig = pd.Series({c: legs[c].R.std() for c in Mis.columns})
    w = 1.0 / sig
    return (w / w.sum() * budget).values


def w_inv_dd(Mis, budget):
    dds = []
    for c in Mis.columns:
        eq = (1 + Mis[c] * RISK).cumprod()
        dds.append(((eq.cummax() - eq) / eq.cummax()).max())
    raw = 1.0 / np.where(np.array(dds) > 0, dds, np.inf)
    return raw / raw.sum() * budget


def w_min_var(Mis, budget):
    """min-variance over the simplex (coarse grid -- robust for 2-3 legs)."""
    cov = Mis.cov().values
    n = Mis.shape[1]
    best, bw = np.inf, None
    for w in simplex_grid(n, step=0.05):
        var = w @ cov @ w
        if var < best:
            best, bw = var, w
    return bw * budget


def w_is_best(Mis, budget):
    """grid the simplex for best IS CAGR/DD (most fit-prone -- the OOS check is the judge)."""
    best, bw = -np.inf, None
    for w in simplex_grid(Mis.shape[1], step=0.05):
        _, _, r = cagr_dd_monthly(port_ret(Mis, w * budget))
        if r > best:
            best, bw = r, w
    return bw * budget


def simplex_grid(n, step=0.05):
    """all weight vectors on the n-simplex with given step (sum=1, w>=0)."""
    k = int(round(1 / step))
    def rec(rem, slots):
        if slots == 1:
            yield [rem]; return
        for i in range(rem + 1):
            for tail in rec(rem - i, slots - 1):
                yield [i] + tail
    for combo in rec(k, n):
        yield np.array(combo, dtype=float) / k


# ---------- vol targeting (trailing-vol scaling, no lookahead) ----------
def vol_target_lev(M, w, window, target, cap=3.0):
    """per-leg monthly leverage = target / trailing-vol(window), shifted (uses past only)."""
    lev = pd.DataFrame(index=M.index)
    for j, c in enumerate(M.columns):
        tv = (M[c] * w[j]).rolling(window).std().shift(1)   # trailing, no lookahead
        lev[c] = (target / tv).clip(upper=cap).fillna(1.0)
    return lev


def report_vt(tag, M, w, window, target, legs):
    lev = vol_target_lev(M, w, window, target)
    wm = dict(zip(M.columns, w))
    parts = []
    for name, t in legs.items():                            # apply each month's leverage per leg
        s = pd.Series(t.R.values * wm[name], index=pd.DatetimeIndex(t.time))
        k = lev[name].reindex(s.index.to_period("M").to_timestamp()).values
        parts.append(s * np.nan_to_num(k, nan=1.0))
    full = pd.concat(parts).sort_index()
    isr = full[full.index.year < SPLIT]; oos = full[full.index.year >= SPLIT]
    _, _, rf = cagr_dd_trades(full)
    _, _, ri = cagr_dd_trades(isr); _, _, ro = cagr_dd_trades(oos)
    print(f"  {tag:<16} FULL CAGR/DD={rf:4.2f}  IS={ri:4.2f}  OOS={ro:4.2f}")
    return rf, ro


def main():
    legs = get_legs()
    # focus on the best 2-leg book (gold bo + BTC bo+KAMA) AND the 3-leg, both tested
    books = {
        "2leg(G.bo+B.K)": {k: legs[k] for k in ("gold_bo", "btc_bo_kama")},
        "3leg(all)":      legs,
    }
    for bname, blegs in books.items():
        M = monthly_matrix(blegs)
        Mis = M[M.index.year < SPLIT]
        n = M.shape[1]
        budget = n * RISK
        target = (Mis * (budget / n)).sum(axis=1).std()    # equal-weight IS vol = risk match
        print(f"\n=== {bname}  legs={list(M.columns)}  months={len(M)} "
              f"(IS<{SPLIT}={len(Mis)})  budget={budget*100:.0f}% ===")
        print("  -- LEVER 1: weighting (budget constant; weights from IS, judged OOS) --")
        schemes = [("equal(book)", w_equal), ("inv_vol", w_inv_vol),
                   ("inv_vol_TRADE", lambda Mi, b: w_inv_vol_trade(Mi, b, blegs)),
                   ("inv_dd", w_inv_dd), ("min_var", w_min_var), ("IS_best_CDD", w_is_best)]
        for sname, fn in schemes:
            report(sname, M, fn(Mis, budget), blegs)
        print("  -- LEVER 2: vol targeting on equal weights (window plateau; trailing=no leak) --")
        we = w_equal(Mis, budget)
        for win in (6, 9, 12, 18):
            report_vt(f"voltgt w{win}", M, we, win, target, blegs)


if __name__ == "__main__":
    main()
