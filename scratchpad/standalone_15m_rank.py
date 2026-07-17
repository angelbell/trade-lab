"""User pivot: the BOOK (inv-vol weighting, inter-leg allocation) is shelved until the account grows.
The near-term goal is a good STANDALONE strategy to grow a small JPY account -- one leg, one account,
compounding at low TF/high frequency.

Standalone is a DIFFERENT arena than the book (structural law 10b: the same rule has a different
optimum solo vs in the book). And a small account grown by compounding needs FREQUENCY -- a leg that
trades 8 times a year cannot compound a small account no matter how good each trade is. So the real
candidate set is the three 15m legs; the swing legs are shown only to make that exclusion honest
(measured, not asserted).

For each leg, ALONE, swap included, at trade resolution:
  rank      standalone CAGR/DD -- leverage-free (a ratio, ~invariant to the risk fraction f).
            maxDD is the bootstrapped MEDIAN (circular block, never a single path; CLAUDE 8).
  growth    the account MULTIPLE over the whole span at f = 1% / 2% / 3% risk, as a DISTRIBUTION
            (median + 5/95%), with the live-expected DD = median x1.75 printed next to it.
  falsify   raw PF / meanR / win% (swap-in) -> IS vs OOS -> per-year spread.

No lookahead: R series come straight from the frozen canonical legs (rr_with_swap.leg).
Run: .venv/bin/python scratchpad/standalone_15m_rank.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from arb_common import Boot, cd
from rr_with_swap import leg, SIX

INTRADAY = ["btc15m_L", "gold15m", "btc15m_S"]


def paths(bt, s):
    """全ブートストラップ経路での (CAGR%, maxDD%, 口座倍率)。s は既に f を掛けた列。"""
    mk = s.index.to_period("M")
    by = {m: s.values[mk == m] for m in bt.months}
    days = max((s.index[-1] - s.index[0]).days, 1)
    n = len(s)
    c = np.empty(len(bt.layout)); d = np.empty(len(bt.layout)); mult = np.empty(len(bt.layout))
    for i, seq in enumerate(bt.layout):
        v = np.concatenate([by[bt.months[j]] for j in seq])[:n]
        c[i], d[i] = cd(v, days)
        mult[i] = np.prod(1.0 + v)
    return c, d, mult


def main():
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else np.inf
    R = {k: leg(k)[0] for k in SIX}
    HOLD = {k: leg(k)[1] for k in SIX}

    # ---- 1. 素の性格（スワップ込み・f=1 名目）＋ 単独 CAGR/DD --------------------------------
    print("1. レッグ単独の性格（スワップ込み）と、単独運用の CAGR/DD（f不変・中央値DD）\n")
    print(f"  {'leg':<13}{'期間':>18}{'n':>6}{'本/年':>7}{'勝率':>7}{'PF':>7}{'meanR':>8}"
          f"{'保有(日)中央':>13}{'CAGR/DD':>9}")
    rows = {}
    for k in SIX:
        s = R[k]; days = max((s.index[-1] - s.index[0]).days, 1); yr = days / 365.25
        bt = Boot(sorted(set(s.index.to_period("M"))), nb=1000, k=3)
        f = 0.01
        rr = np.median(bt.ratios(s * f))
        holdd = np.median(HOLD[k]) / 96.0   # 15m足→日（gold/BTC 15m は 96本/日）
        rows[k] = (bt, s, rr)
        span = f"{s.index[0].date()}→{s.index[-1].date()}"
        mark = "  ◎" if k in INTRADAY else ""
        print(f"  {k:<13}{span:>18}{len(s):>6}{len(s)/yr:>7.0f}{(s>0).mean()*100:>6.0f}%"
              f"{pf(s.values):>7.2f}{s.mean():>+8.3f}{holdd:>13.2f}{rr:>9.2f}{mark}")
    print("\n  ◎ = 15分レッグ（単独で小口座を回す候補）。スイング3本は本/年が一桁＝複利が効かない。")

    # ---- 2. 資金倍率の分布（f=1/2/3%）--------------------------------------------------------
    print("\n2. 全期間の口座倍率（単独運用・複利）— f を上げた時の成長と想定実DD\n")
    for k in INTRADAY:
        bt, s, _ = rows[k]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        print(f"  {k}  （{yrs:.1f}年）")
        print(f"    {'リスク/本':>9}{'CAGR中央':>10}{'中央値DD':>10}{'想定実DD(×1.75)':>16}"
              f"{'倍率 中央':>11}{'倍率 5%':>10}{'倍率 95%':>10}")
        for f in (0.01, 0.02, 0.03):
            c, d, mult = paths(bt, s * f)
            print(f"    {f*100:>7.0f}%{np.median(c):>+9.0f}%{np.median(d):>9.1f}%"
                  f"{np.median(d)*1.75:>15.1f}%{np.median(mult):>10.1f}x"
                  f"{np.quantile(mult,0.05):>9.1f}x{np.quantile(mult,0.95):>9.1f}x")
        print()

    # ---- 3. IS/OOS と年別（一つの時代のベータでないか）--------------------------------------
    print("3. IS/OOS（時系列で前半・後半）と年別R（f=1 名目・スワップ込み）\n")
    for k in INTRADAY:
        s = R[k]; mid = s.index[0] + (s.index[-1] - s.index[0]) / 2
        isr, oos = s[s.index <= mid], s[s.index > mid]
        print(f"  {k}: IS meanR {isr.mean():+.3f} (n{len(isr)}, PF{pf(isr.values):.2f}) | "
              f"OOS meanR {oos.mean():+.3f} (n{len(oos)}, PF{pf(oos.values):.2f})")
        yr = s.groupby(s.index.year)
        cells = "  ".join(f"{y}:{g.sum():+.0f}R" for y, g in yr)
        print(f"       年別  {cells}\n")


if __name__ == "__main__":
    main()
