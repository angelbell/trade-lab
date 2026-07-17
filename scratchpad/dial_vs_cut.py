"""Two implementations of "just bet more on btc15m_L" disagree, and the verdict on the 1-day cut hangs
on which is right:

  dial-by-R       multiply the leg's R series by 1.5, then let inv-vol re-weight   -> +2.8pt
  dial-by-weight  multiply the leg's WEIGHT by 1.5, then renormalise to 3%         -> -0.2pt

They are not the same operation. Scaling R makes sigma(L) 1.5x bigger, so inv-vol hands the leg 1/1.5
of the weight -- and the leg then trades at 1.5x that weight, so the book's REAL total risk is no
longer 3% (the normalisation only constrains the sum of the weights, not the sizes actually traded).
That is a leverage leak dressed up as an allocation. dial-by-weight has no leak: it moves budget from
the other five legs into btc15m_L and the total stays 3%.

So dial-by-weight is the honest null, and the question it answers is the one that matters:
**does the book under-allocate to btc15m_L, or not?** Sweep its weight multiplier and look at the
shape (CLAUDE checklist 4 -- a real optimum is a HILL, a fit is a spike):

  if the curve RISES through 1.0, the book underfeeds the leg, and the 1-day cut is just a clumsy way
  of buying more of it (= a leverage dial, not an exit finding -- reject it).
  if the curve PEAKS at ~1.0, then "more btc15m_L" is NOT available by dialling, and the cut's +3.7pt
  is doing something a dial cannot do (= a real finding, adopt-worthy).

Same paired arbiter throughout: walk-forward weights, one fixed leverage per arm set so its
bootstrapped-median maxDD equals D0, CAGR compared on 1000 identical resampled histories.
Run: .venv/bin/python scratchpad/dial_vs_cut.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from arb_common import Boot, months_union, cd, BUDGET
from rr_with_swap import leg, SIX


def cvar(v, q=0.10):
    return abs(np.mean(np.sort(v)[:max(1, int(q * len(v)))]))


class Pair(Boot):
    def cagrs(self, s):
        mk = s.index.to_period("M")
        by = {m: s.values[mk == m] for m in self.months}
        days = max((s.index[-1] - s.index[0]).days, 1)
        n = len(s)
        c = np.empty(len(self.layout))
        for i, seq in enumerate(self.layout):
            c[i] = cd(np.concatenate([by[self.months[j]] for j in seq])[:n], days)[0]
        return c


def main():
    B = {k: leg(k)[0] for k in SIX}
    A = {k: (leg(k, fwd=96)[0] if k == "btc15m_L" else B[k]) for k in SIX}
    st = max(B[k].index.min() for k in SIX); en = min(B[k].index.max() for k in SIX)
    cp = lambda D: {k: D[k][(D[k].index >= st) & (D[k].index <= en)] for k in SIX}
    B, A = cp(B), cp(A)
    yrs = sorted({y for k in SIX for y in B[k].index.year}); first = yrs[0] + 2
    EQ = pd.Series(BUDGET / len(SIX), index=SIX)

    def wts(L, how, dial=1.0):
        o = {}
        for k in SIX:
            v = L[k].values
            o[k] = (1.0 if len(v) < 5 else
                    1.0 / max(v.std(), 1e-9) if how == "sigma" else 1.0 / max(cvar(v), 1e-9))
        r = pd.Series(o)
        r["btc15m_L"] *= dial              # 予算を他の5本から移す（合計は 3% のまま＝漏れ無し）
        return r / r.sum() * BUDGET

    def mix(L, how, dial=1.0):
        by = {}
        for y in yrs:
            past = {k: L[k][L[k].index.year < y] for k in SIX}
            by[y] = wts(past, how, dial) if (y >= first and min(len(past[k]) for k in SIX) >= 5) else EQ
        return (pd.concat([pd.Series(L[k].values * np.array([by[y][k] for y in L[k].index.year]),
                                     index=L[k].index) for k in SIX]).sort_index(),
                by[yrs[-1]]["btc15m_L"])

    DIALS = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
    ARMS = {}
    for how in ("sigma", "cvar"):
        for d in DIALS:
            ARMS[(how, "base", d)] = mix(B, how, d)
        ARMS[(how, "cut1d", 1.0)] = mix(A, how, 1.0)

    bt = Pair(months_union(*[s for s, _ in ARMS.values()]), nb=1000, k=3)
    D0 = bt.dd_median(ARMS[("sigma", "base", 1.0)][0])
    print(f"基準 maxDD = {D0:.2f}%。各アームの倍率を『ブートストラップ中央値DD = {D0:.2f}%』に固定し、")
    print("同じ1000本の並べ替え履歴の上で CAGR を対比較。重みはウォークフォワード、総リスクは常に 3%。\n")

    for how in ("sigma", "cvar"):
        base = None
        print(f"  【{'σ逆数（現行）' if how=='sigma' else 'CVaR10 逆数'}】")
        print(f"    {'btc15m_L の予算ダイヤル':<26}{'重み':>9}{'CAGR中央値':>12}{'差':>9}"
              f"{'  P(現行に勝つ)':>15}")
        for d in DIALS:
            s, wl = ARMS[(how, "base", d)]
            sc = bt.equal_dd_cagr(s, D0)[1]
            c = bt.cagrs(s * sc)
            if base is None:
                base = c
            print(f"    {f'×{d}':<26}{100*wl:>8.3f}%{np.median(c):>+11.1f}%"
                  f"{np.median(c-base):>+8.1f}pt{100*np.mean(c>base):>13.0f}%"
                  + ("  ← 現行" if d == 1.0 else ""))
        s, wl = ARMS[(how, "cut1d", 1.0)]
        sc = bt.equal_dd_cagr(s, D0)[1]
        c = bt.cagrs(s * sc)
        print(f"    {'1日カット（ダイヤル無し）':<26}{100*wl:>8.3f}%{np.median(c):>+11.1f}%"
              f"{np.median(c-base):>+8.1f}pt{100*np.mean(c>base):>13.0f}%\n")


if __name__ == "__main__":
    main()
