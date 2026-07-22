"""gold15m's pullback-limit depth rises MONOTONICALLY with the book: 0.25 -> 8.28, 0.40 -> 9.09.
Monotone is the signature of a real lever -- but 0.40 is the EDGE of the grid I swept, and a grid
that stops at its own maximum is the classic way to walk off a cliff without seeing it.

So: (1) sweep past the edge, in 0.05 steps, until it breaks. A real lever has a HILL with a
    far side; a fit has a spike or a wall at the boundary.
(2) the book's circular block bootstrap over months (1/3/6/12) -- a real improvement's P RISES with
    block length; a path-fit's falls (today's three kills all fell to the 20s).
(3) per-year, because a lever that only fires in one era is beta.
(4) the mechanism check: does n stay flat? If the deeper limit is silently DROPPING trades
    (breaks that run away and never pull back), it is a selection rule, not an execution lever --
    and selection rules must beat a random-drop null.
Run: .venv/bin/python experiments/gold15m_pullback_stress.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from gold15m_levers import gold15m
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import book, w_trade

RNG = np.random.default_rng(20260714)
NDRAW = 2000


def series(legs):
    w = w_trade(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    return pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                                index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                      for k in SIX]).sort_index()


def cdd(v, days):
    eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max()
    return (eq[-1] ** (365.25 / max(days, 1)) - 1) / max(dd, 1e-9)


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    r0 = book(legs0, SIX)[2]

    print(f"1. 掃引の端の外を見る（現行 0.25 = {r0:.2f}）。本物なら丘、当てはめなら壁か棘。\n")
    print(f"  {'押し目':>7}{'n':>6}{'本/年':>7}{'約定率':>8}{'PF実':>7}{'meanR':>9}"
          f"{'totR/年':>9}{'ブック':>9}{'差':>8}")
    n0 = None
    keep = {}
    for fr in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        s, g = gold15m(pullback_frac=fr)
        if n0 is None:
            n0 = len(s)
        legs = dict(legs0); legs["gold15m"] = s
        rb = book(legs, SIX)[2]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        pf = s[s > 0].sum() / abs(s[s <= 0].sum())
        print(f"  {fr:>7.2f}{len(s):>6}{len(s)/yrs:>7.0f}{100*len(s)/n0:>7.0f}%{pf:>7.2f}"
              f"{s.mean():>+9.3f}{s.sum()/yrs:>+9.1f}{rb:>9.2f}{rb-r0:>+8.2f}"
              + ("  ← 現行" if fr == 0.25 else ""))
        if fr in (0.30, 0.35, 0.40, 0.45):
            keep[fr] = s

    print("\n\n2. ブックの巡回ブロック・ブートストラップ（本物なら P はブロックとともに上がる。")
    print("   今日殺した3案はすべて 20%台へ落ちた）\n")
    base = series(legs0)
    bm = base.index.to_period("M"); months = sorted(set(bm)); M = len(months)
    b_by = {m: base[bm == m].values for m in months}
    print(f"  {'押し目':>7}{'ブック':>8}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    for fr, s in keep.items():
        legs = dict(legs0); legs["gold15m"] = s
        arm = series(legs); am = arm.index.to_period("M")
        a_by = {m: arm[am == m].values for m in months}
        row = f"  {fr:>7.2f}{book(legs, SIX)[2]:>8.2f}"
        for Lb in (1, 3, 6, 12):
            nb = int(np.ceil(M / Lb)); wins = 0
            for _ in range(NDRAW):
                st = RNG.integers(0, M, nb)
                order = np.concatenate([(np.arange(s2, s2 + Lb) % M) for s2 in st])[:M]
                b = np.concatenate([b_by[months[i]] for i in order if len(b_by[months[i]])])
                a = np.concatenate([a_by[months[i]] for i in order if len(a_by[months[i]])])
                if len(b) < 20 or len(a) < 20:
                    continue
                wins += cdd(a, 365 * M / 12) > cdd(b, 365 * M / 12)
            row += f"{100*wins/NDRAW:>7.0f}%"
        print(row)

    print("\n\n3. 年別 meanR（一era集中の検出）\n")
    s25, _ = gold15m(pullback_frac=0.25)
    s40, _ = gold15m(pullback_frac=0.40)
    yrs = sorted(set(s25.index.year) | set(s40.index.year))
    print("  " + " " * 12 + "".join(f"{y:>9}" for y in yrs))
    for tag, s in (("押し目 0.25", s25), ("押し目 0.40", s40)):
        row = f"  {tag:<12}"
        for y in yrs:
            g = s[s.index.year == y]
            row += f"{g.mean():>+9.2f}" if len(g) >= 5 else f"{'·':>9}"
        print(row)
    print("  " + " " * 12 + "".join(f"{y:>9}" for y in yrs))
    for tag, s in (("  (本数)0.25", s25), ("  (本数)0.40", s40)):
        row = f"  {tag:<12}"
        for y in yrs:
            row += f"{(s.index.year == y).sum():>9}"
        print(row)


if __name__ == "__main__":
    main()
