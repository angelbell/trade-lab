"""The wide-stop filter's LEG numbers looked real (block-bootstrap P rose 75->94%). But the leg is
not the arbiter (structural law 10). At the BOOK level the gain is +0.19 at best (8.28 -> 8.47) and
the threshold surface wobbles (8.47 / 8.31 / 8.38 / 8.14) -- which is what noise looks like.

Settle it the only way that settles it: paired circular block bootstrap over months on the BOOK's
trade-resolution CAGR/DD. A real improvement's P RISES with block length; a path-fit's P falls.
Run: .venv/bin/python scratchpad/wide_stop_bookboot.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import w_trade

RNG = np.random.default_rng(20260714)
NDRAW = 2000


def series(legs):
    w = w_trade(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    parts = [pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                       index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
             for k in SIX]
    return pd.concat(parts).sort_index()


def cdd(v, days):
    eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max()
    return (eq[-1] ** (365.25 / max(days, 1)) - 1) / max(dd, 1e-9)


def main():
    L = raw_legs()
    base = series({k: v[0] for k, v in L.items()})
    print("ブックの月次トレードをブロックで並べ替え、CAGR/DD で現行と直接対決させる")
    print("（本物なら P はブロックとともに上がる。今日殺した3案はすべて下がった）\n")
    print(f"  {'条件':<28}{'CAGR/DD':>9}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    bm = base.index.to_period("M")
    months = sorted(set(bm))
    M = len(months)
    for thr in (3.0, 2.5, 2.0):
        legs = {k: v[0] for k, v in L.items()}
        legs["btc15m_L"] = L["btc15m_L"][0][L["btc15m_L"][1] <= thr]
        arm = series(legs)
        am = arm.index.to_period("M")
        b_by = {m: base[bm == m].values for m in months}
        a_by = {m: arm[am == m].values for m in months}
        r_arm = cdd(arm.values, (arm.index[-1] - arm.index[0]).days)
        row = f"  {'btc15m_L に 損切り<='+str(thr)+'%':<28}{r_arm:>9.2f}"
        for Lb in (1, 3, 6, 12):
            nb = int(np.ceil(M / Lb)); wins = 0
            for _ in range(NDRAW):
                st = RNG.integers(0, M, nb)
                order = np.concatenate([(np.arange(s, s + Lb) % M) for s in st])[:M]
                b = np.concatenate([b_by[months[i]] for i in order if len(b_by[months[i]])])
                a = np.concatenate([a_by[months[i]] for i in order if len(a_by[months[i]])])
                if len(b) < 20 or len(a) < 20:
                    continue
                wins += cdd(a, 365 * M / 12) > cdd(b, 365 * M / 12)
            row += f"{100*wins/NDRAW:>7.0f}%"
        print(row)
    print(f"\n  現行の CAGR/DD = {cdd(base.values, (base.index[-1]-base.index[0]).days):.2f}")


if __name__ == "__main__":
    main()
