"""LEAVE-ONE-OUT: does every leg earn its seat in the book?  (trade-resolution arbiter)

The 6-leg book was assembled under the MONTHLY-DD arbiter, which we now know compresses away every
drawdown that opens and closes inside a month -- and it was that arbiter that ranked the legs. So the
composition itself has never been judged on a real drawdown.

The DD anatomy adds a concrete suspicion: gold15m is negative in ALL FIVE of the book's deepest
drawdowns (it has never once cushioned), while btc15m_L is positive in 4 of 5.

Test: drop one leg at a time, re-derive the inv-vol weights over the REMAINING legs at the same 3%
total risk (so each book is a fair, fully-invested alternative), and read the trade-resolution
CAGR/DD.  A leg earns its seat iff removing it makes the book WORSE.
Paired circular block bootstrap (month blocks, trade order preserved inside each month) decides.

Also reported: the 3-leg incumbent book, and the risk actually at work (the "3% budget" is nominal --
peak concurrent risk is 2.6%, median 0.42%).
Run: .venv/bin/python scratchpad/book_leave_one_out.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from btc_family_ext_throttle import build_base

NEW = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
OLD = ["gold_bo", "btc_bo_kama", "btc_pull"]
NDRAW = 2000


def weights(legs, basket, budget=0.03):
    mon = {k: legs[k].groupby(legs[k].index.to_period("M")).sum() for k in basket}
    st = max(s.index.min() for s in mon.values()); en = min(s.index.max() for s in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    w = 1.0 / M.std(); w = w / w.sum() * budget
    return w, midx


def trade_series(legs, basket):
    """one weighted-R series over all legs' trades, in time order, clipped to the common window."""
    w, midx = weights(legs, basket)
    st = midx[0].to_timestamp().tz_localize("UTC")
    en = midx[-1].to_timestamp(how="end").tz_localize("UTC")
    parts = []
    for k in basket:
        s = legs[k]
        s = s[(s.index >= st) & (s.index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    return pd.concat(parts).sort_index(), w, midx


def cdd(vals, days):
    eq = np.cumprod(1 + vals); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    if dd <= 0: return np.nan, np.nan, np.nan
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd


def stat(legs, basket):
    s, w, midx = trade_series(legs, basket)
    days = (s.index[-1] - s.index[0]).days
    return cdd(s.values, days) + (s, len(s))


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()

    arms = {"6-leg book (all)": NEW}
    for k in NEW:
        arms[f"  minus {k}"] = [x for x in NEW if x != k]
    arms["3-leg incumbent"] = OLD

    print(f"{'book':<24}{'n':>6}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>9}{'vs all':>9}")
    S, base = {}, None
    for name, basket in arms.items():
        c, d, x, s, n = stat(legs, basket)
        S[name] = s
        if base is None: base = x
        delta = "" if name == "6-leg book (all)" else f"{x - base:+.2f}"
        print(f"{name:<24}{n:>6}{c:>8.1f}%{d:>7.2f}%{x:>9.2f}{delta:>9}")

    # paired circular block bootstrap over MONTHS (trade order preserved inside each month)
    print(f"\npaired circular block bootstrap ({NDRAW} draws) -- P(this book beats the full 6-leg book)")
    months = sorted(set(S["6-leg book (all)"].index.to_period("M")))
    m = len(months)
    G = {name: {p: g.values for p, g in s.groupby(s.index.to_period("M"))} for name, s in S.items()}
    rng = np.random.default_rng(20260713)
    print(f"  {'block':<7}" + "".join(f"{k.strip():>20}" for k in arms))
    for blk in (1, 3, 6, 12):
        nb = int(np.ceil(m / blk))
        D = {k: [] for k in arms}
        for _ in range(NDRAW):
            st = rng.integers(0, m, nb)
            order = [months[(s + j) % m] for s in st for j in range(blk)][:m]
            for k in arms:
                v = np.concatenate([G[k][p] for p in order if p in G[k]])
                D[k].append(cdd(v, 365.25 * m / 12)[2])
        b = np.array(D["6-leg book (all)"])
        row = []
        for k in arms:
            a = np.array(D[k])
            row.append(f"{np.nanmedian(a):.2f}(P{np.nanmean(a > b)*100:.0f}%)")
        print(f"  {f'{blk}mo':<7}" + "".join(f"{r:>20}" for r in row))
    print("\n  a leg EARNS its seat iff removing it LOWERS the book (P well under 50%).")


if __name__ == "__main__":
    main()
