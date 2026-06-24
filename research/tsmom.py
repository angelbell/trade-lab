"""tsmom.py -- time-series momentum (Moskowitz-Ooi-Pedersen 2012) as a candidate LEG.

TSMOM: long if past-L-month return > 0, short if < 0, monthly rebalance. It's the one proven
trend-canon primitive not yet a leg. Adding it only helps the book if it's LOW-correlated with
gold_bo (a correlated positive leg dilutes -- 3-leg 1.57 < 2-leg 1.71). TSMOM is a different
mechanism (slow monthly, holds through trends, can go SHORT) than our intraday breakouts, so the
DECISIVE test is one number: corr(TSMOM, gold_bo / book). Low (<~0.4) => real diversifier; ~0.8 =>
just gold/BTC trend beta.

Causal: month-t signal uses only returns ending t-1 (shift). In-sample only; live-forward arbitrates.

  .venv/bin/python research/tsmom.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.portfolio_kama import get_legs
from research.portfolio_alloc import monthly_matrix, cagr_dd_monthly, SPLIT

COST_M = 0.0005   # ~5bp monthly rebalance cost


def monthly_rets(csv, tf="4h"):
    d = resample(load_mt5_csv(csv), tf)
    m = d["close"].resample("1ME").last().dropna()
    r = m.pct_change().dropna()
    r.index = r.index.to_period("M")          # align with monthly_matrix (period) for correlation
    return r


def tsmom(rm, L, side):
    """causal: s_t = sign(sum past-L returns ending t-1); leg ret_t = s_t * rm_t - cost*|dpos|."""
    sig = np.sign(rm.rolling(L).sum().shift(1))
    if side == "long":
        sig = sig.clip(lower=0)
    pos = sig.fillna(0)
    ret = pos * rm - COST_M * pos.diff().abs().fillna(0)
    return ret.dropna()


def bh(rm):
    return rm.copy()


def stat(tag, ret):
    if len(ret) < 12:
        print(f"  {tag:<26} n={len(ret)} (too few)"); return
    c, dd, r = cagr_dd_monthly(ret)
    isr = ret[ret.index.year < SPLIT]; oos = ret[ret.index.year >= SPLIT]
    _, _, ri = cagr_dd_monthly(isr); _, _, ro = cagr_dd_monthly(oos)
    print(f"  {tag:<26} mo={len(ret):>3} win%={(ret>0).mean()*100:>3.0f} CAGR={c:+5.1f}% DD={dd:4.1f}% "
          f"CAGR/DD={r:5.2f} | IS={ri:4.2f} OOS={ro:4.2f}")


def main():
    rg = monthly_rets("data/vantage_xauusd_h1.csv", "4h")
    rb = monthly_rets("data/vantage_btcusd_h1.csv", "4h")
    print(f"tsmom -- gold {rg.index.min()}->{rg.index.max()} ({len(rg)}mo), "
          f"BTC {rb.index.min()}->{rb.index.max()} ({len(rb)}mo)")

    # book legs (monthly) for the decisive correlation
    legs = get_legs()
    Mbook = monthly_matrix({k: legs[k] for k in ("gold_bo", "btc_bo_kama")})
    Mbook.index = Mbook.index.to_period("M")          # period index to match TSMOM streams
    gb_m = Mbook["gold_bo"]; bk_m = Mbook["btc_bo_kama"]               # raw R-sums (fine for corr)
    book_m = Mbook["gold_bo"] * 0.0079 + Mbook["btc_bo_kama"] * 0.0121  # actual account-% returns (0.79%/1.21% risk)

    print("\n  == 1. STANDALONE (monthly) -- long-only must beat buy&hold (else just beta) ==")
    stat("gold buy&hold", bh(rg)); stat("BTC buy&hold", bh(rb))
    for inst, rm in [("gold", rg), ("BTC", rb)]:
        for side in ("longshort", "long"):
            for L in (3, 6, 9, 12):
                stat(f"{inst} TSMOM L{L} {side}", tsmom(rm, L, side))

    print("\n  == 2. CORRELATION vs book legs (DECISIVE: low=diversifier, ~0.8=just trend beta) ==")
    print(f"  {'TSMOM stream':<22}{'corr gold_bo':>13}{'corr btc_K':>11}{'corr book':>11}")
    def corr(a, b):
        x = pd.concat([a, b], axis=1).dropna()
        return x.iloc[:, 0].corr(x.iloc[:, 1]) if len(x) >= 12 else np.nan
    for inst, rm, ownleg in [("gold", rg, gb_m), ("BTC", rb, bk_m)]:
        for side in ("longshort", "long"):
            for L in (6, 12):
                t = tsmom(rm, L, side)
                print(f"  {inst+' L'+str(L)+' '+side:<22}{corr(t, gb_m):>+13.2f}{corr(t, bk_m):>+11.2f}"
                      f"{corr(t, book_m):>+11.2f}")

    print("\n  == 3. DIVERSIFICATION: add gold+BTC TSMOM (L12 longshort) to book at ~constant risk ==")
    tg = tsmom(rg, 12, "longshort"); tb = tsmom(rb, 12, "longshort")
    tsm = (tg.reindex(book_m.index).fillna(0) + tb.reindex(book_m.index).fillna(0)) / 2
    # scale TSMOM to the book's monthly vol so the add is risk-comparable, then blend at constant total
    sc = book_m.std() / tsm.std() if tsm.std() > 0 else 1.0
    _, _, r0 = cagr_dd_monthly(book_m)
    for w in (0.0, 0.2, 0.35):
        blend = (book_m + w * sc * tsm) / (1 + w)
        c, dd, r = cagr_dd_monthly(blend)
        print(f"    book + {w:.2f}*TSMOM(vol-matched): CAGR/DD={r:.2f} (CAGR{c:+.0f}/DD{dd:.0f})"
              f"{'  <= book alone' if w == 0 else ''}")
    print(f"  (corr(gold+BTC TSMOM L12 ls, book) = {corr(tsm, book_m):+.2f})")
    print("\n  verdict: corr<~0.4 + positive standalone + plateau => diversifier; else trend beta.")


if __name__ == "__main__":
    main()
