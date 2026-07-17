"""Last methodological fix of the day, then the verdict.

The P columns so far asked P(CAGR/DD of arm > CAGR/DD of sigma) across resampled histories. That
question is contaminated: CAGR/DD rises when DD merely SHRINKS, so an arm that is smaller and safer
and earns less can still "win" it. The proof is in the table -- the reversed dummy (weights
PROPORTIONAL to CVaR), which is -2.4 CAGR pt at equal drawdown and obviously wrong, scored P = 80%.

So the ratio is not the arbiter. CLAUDE's rule is: equalise the drawdown, then compare CAGR. Do that
per bootstrap path:

  1. fix each arm's leverage ONCE, deterministically, so its bootstrapped-MEDIAN maxDD equals D0
  2. on each of the 1000 identical resampled histories, read both arms' CAGR at that fixed leverage
  3. report the paired difference: median, and P(arm > sigma), by block length

Now a "win" means MORE MONEY AT THE SAME PAIN, which is the only claim worth adopting. The reversed
dummy must fail this, and if it does, the test has teeth. Median DD of each arm is printed as the
check that the drawdown really was equalised.
Run: .venv/bin/python scratchpad/cvar_final_test.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from arb_common import Boot, months_union, cd, BUDGET
from rr_with_swap import leg, SIX


def cvar(v, q):
    return abs(np.mean(np.sort(v)[:max(1, int(q * len(v)))]))


class Pair(Boot):
    def cagrs(self, s):
        mk = s.index.to_period("M")
        by = {m: s.values[mk == m] for m in self.months}
        days = max((s.index[-1] - s.index[0]).days, 1)
        n = len(s)
        c = np.empty(len(self.layout)); d = np.empty(len(self.layout))
        for i, seq in enumerate(self.layout):
            v = np.concatenate([by[self.months[j]] for j in seq])[:n]
            c[i], d[i] = cd(v, days)
        return c, d


def main():
    B = {k: leg(k)[0] for k in SIX}
    A = {k: (leg(k, fwd=96)[0] if k == "btc15m_L" else B[k]) for k in SIX}
    st = max(B[k].index.min() for k in SIX); en = min(B[k].index.max() for k in SIX)
    cl = lambda D: {k: D[k][(D[k].index >= st) & (D[k].index <= en)] for k in SIX}
    B, A = cl(B), cl(A)
    yrs = sorted({y for k in SIX for y in B[k].index.year}); first = yrs[0] + 2
    EQ = pd.Series(BUDGET / len(SIX), index=SIX)

    def wts(L, how, q=0.10):
        o = {}
        for k in SIX:
            v = L[k].values
            o[k] = (1.0 if len(v) < 5 else
                    1.0 / max(v.std(), 1e-9) if how == "sigma" else
                    1.0 if how == "equal" else
                    max(cvar(v, q), 1e-9) if how == "cvar_p" else
                    1.0 / max(cvar(v, q), 1e-9))
        r = pd.Series(o)
        return r / r.sum() * BUDGET

    def mix(L, how, q=0.10):
        by = {}
        for y in yrs:
            past = {k: L[k][L[k].index.year < y] for k in SIX}
            by[y] = wts(past, how, q) if (y >= first and min(len(past[k]) for k in SIX) >= 5) else EQ
        return pd.concat([pd.Series(L[k].values * np.array([by[y][k] for y in L[k].index.year]),
                                    index=L[k].index) for k in SIX]).sort_index()

    ARMS = {
        "σ逆数（現行）":                 mix(B, "sigma"),
        "CVaR worst 5%":              mix(B, "cvar", 0.05),
        "CVaR worst 10%":             mix(B, "cvar", 0.10),
        "CVaR worst 20%":             mix(B, "cvar", 0.20),
        "等分（0.5%ずつ）":              mix(B, "equal"),
        "CVaR 比例（逆向きダミー）":       mix(B, "cvar_p"),
        "σ逆数 + 1日カット":             mix(A, "sigma"),
        "CVaR10 + 1日カット":           mix(A, "cvar", 0.10),
    }
    bt = Pair(months_union(*ARMS.values()), nb=1000, k=3)
    D0 = bt.dd_median(ARMS["σ逆数（現行）"])
    print(f"基準 maxDD = {D0:.2f}%。各アームの倍率を『ブートストラップ中央値DD = {D0:.2f}%』に一度だけ固定し、")
    print("その倍率のまま、同じ1000本の並べ替え履歴の上で CAGR を対比較する（＝同じ痛みで、いくら稼ぐか）。\n")
    scale = {nm: bt.equal_dd_cagr(s, D0)[1] for nm, s in ARMS.items()}
    C, D = {}, {}
    for nm, s in ARMS.items():
        C[nm], D[nm] = bt.cagrs(s * scale[nm])
    print(f"  {'アーム':<22}{'総リスク':>9}{'DD中央値':>10}{'CAGR中央値':>12}{'差(中央値)':>12}"
          f"{'  P(現行に勝つ):':<16}{'1か月':>7}{'3か月':>7}{'6か月':>7}{'12か月':>8}")
    b = "σ逆数（現行）"
    for nm in ARMS:
        row = (f"  {nm:<22}{3*scale[nm]:>8.2f}%{np.median(D[nm]):>9.2f}%"
               f"{np.median(C[nm]):>+11.1f}%{np.median(C[nm]-C[b]):>+11.1f}pt{'':<16}")
        for k in (1, 3, 6, 12):
            pk = Pair(bt.months, nb=800, k=k)
            ca = pk.cagrs(ARMS[b] * scale[b])[0]
            cb = pk.cagrs(ARMS[nm] * scale[nm])[0]
            row += f"{100*np.mean(cb > ca):>6.0f}%"
        print(row + ("  ← 現行" if nm == b else ""))
    print("\n  ※ 差は経路ごとの対差の中央値（同じ履歴の上で引き算しているので、経路の運は相殺される）。")


if __name__ == "__main__":
    main()
