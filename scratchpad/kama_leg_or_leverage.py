"""Is btc_bo_kama a DIVERSIFIER, or just a place to park risk budget?

Removing it takes the book from CAGR 63.4%/DD 7.66% to CAGR 81.7%/DD 9.95%. Same CAGR/DD
(8.27 vs 8.21). Those two ratios -- 63.4/81.7 = 0.776 and 7.66/9.95 = 0.770 -- are the same
number. That is the signature of PURE DE-LEVERAGING, not diversification: the 6-leg book may
be nothing more than the 5-leg book scaled down to ~77% size, and btc_bo_kama's only job is
absorbing 0.82% of the 3% budget into a leg that barely moves.

The decisive test is the equal-DD comparison the lab already uses for sizing:
    take the 5-leg book (no btc_bo_kama), shrink its TOTAL RISK until its maxDD equals the
    6-leg book's 7.66%, and read the CAGR.
  - If CAGR lands near 63.4%  -> btc_bo_kama adds NOTHING. A risk dial does its whole job.
  - If CAGR lands clearly BELOW 63.4% -> the leg genuinely buys return at that drawdown, i.e.
    it really is diversifying and deserves its seat.

Also reported, to separate "high win rate" from "helps when the book is down":
  - each leg's win%, longest losing streak, per-trade R sigma
  - each leg's mean contribution on the book's WORST days vs all days
Run: .venv/bin/python scratchpad/kama_leg_or_leverage.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from book_spec_fix import build, w_trade

SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
FIVE = [k for k in SIX if k != "btc_bo_kama"]


def curve(legs, basket, budget):
    w = w_trade(legs, basket, budget)
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = [pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                       index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
             for k in basket]
    return pd.concat(parts).sort_index()


def cdd(s):
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    days = (s.index[-1] - s.index[0]).days
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd


def streak(s):
    best = cur = 0
    for r in s.values:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def main():
    legs = build("2018-01-01", False)

    print("Q1 -- 勝率と負けの連続（btc_bo_kama だけが中央値で勝っている、は本当か）")
    print(f"  {'leg':<14}{'n':>5}{'win%':>7}{'中央値R':>9}{'最長連敗':>9}{'RのSD':>8}{'meanR':>8}")
    for k in SIX:
        s = legs[k]
        print(f"  {k:<14}{len(s):>5}{100*(s>0).mean():>6.1f}%{np.median(s.values):>9.2f}"
              f"{streak(s):>9}{s.std():>8.2f}{s.mean():>+8.3f}")

    c6 = curve(legs, SIX, 0.03)
    a6, d6, r6 = cdd(c6)
    c5 = curve(legs, FIVE, 0.03)
    a5, d5, r5 = cdd(c5)
    print(f"\nQ2 -- 素の比較（総リスク3%）")
    print(f"  6レッグ            CAGR {a6:6.1f}%   maxDD {d6:5.2f}%   CAGR/DD {r6:5.2f}")
    print(f"  5レッグ(kama抜き)  CAGR {a5:6.1f}%   maxDD {d5:5.2f}%   CAGR/DD {r5:5.2f}")
    print(f"  比率: CAGR {a6/a5:.3f}倍   DD {d6/d5:.3f}倍   "
          f"{'← ほぼ同じ = 単なる縮小の疑い' if abs(a6/a5 - d6/d5) < 0.05 else ''}")

    print(f"\nQ3 -- 決定的な検定: 5レッグのリスクを絞って、6レッグと同じ maxDD ({d6:.2f}%) に揃える")
    print(f"  {'5レッグの総リスク':<20}{'CAGR':>9}{'maxDD':>9}{'CAGR/DD':>9}")
    lo, hi = 0.005, 0.03
    for _ in range(40):                       # bisect on total risk so that maxDD == d6
        mid = (lo + hi) / 2
        if cdd(curve(legs, FIVE, mid))[1] > d6: hi = mid
        else: lo = mid
    for b in (0.03, 0.025, mid, 0.02):
        a, d, r = cdd(curve(legs, FIVE, b))
        tag = "  ← 6レッグと同じDD" if abs(b - mid) < 1e-9 else ""
        print(f"  {b*100:>6.2f}%{'':<13}{a:>8.1f}%{d:>8.2f}%{r:>9.2f}{tag}")
    a, d, r = cdd(curve(legs, FIVE, mid))
    print(f"\n  同じ maxDD {d:.2f}% のとき:  5レッグ CAGR {a:.1f}%   vs   6レッグ CAGR {a6:.1f}%")
    gap = a - a6
    print(f"  差 = {gap:+.1f} ポイント  ->  "
          + ("btc_bo_kama は何も足していない（リスク・ダイヤルで代替可能）"
             if gap > -2 else
             "btc_bo_kama は同じDDで CAGR を押し上げている＝本物の分散"))

    print(f"\nQ4 -- ブックが沈んだ日、各レッグは何をしていたか（6レッグ、日次）")
    d = c6.groupby(c6.index.floor("D")).sum()
    worst = d.nsmallest(max(int(len(d) * 0.05), 10)).index          # book's worst 5% of days
    print(f"  {'leg':<14}{'最悪5%の日の平均寄与':>22}{'全日平均':>12}{'その日に建てた率':>16}")
    for k in SIX:
        s = legs[k]
        w = w_trade(legs, SIX, 0.03)[k]
        dd_ = (s * w).groupby(s.index.floor("D")).sum()
        on_worst = dd_.reindex(worst).fillna(0.0)
        print(f"  {k:<14}{100*on_worst.mean():>21.3f}%{100*dd_.reindex(d.index).fillna(0).mean():>11.3f}%"
              f"{100*(on_worst != 0).mean():>15.0f}%")


if __name__ == "__main__":
    main()
