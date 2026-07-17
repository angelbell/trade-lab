"""Everything today converges on one place: the book's sizing rule under-feeds btc15m_L. Two separate
"improvements" (the 1-day max hold, +3.5pt) and (btc15m_L bet x1.5, +2.8pt) turn out to be the same
lever wearing different clothes -- both just buy more exposure to the leg that sigma-inv-vol starves.
sigma starves it because its R is skewed +1.84 (loss floored at -1R, wins run to +6.8R) and standard
deviation charges that right tail as risk.

Hand-picking a multiplier for the best leg is pure hindsight, so the only honest fix is a RULE that
(a) is computed walk-forward from prior trades only, and (b) does not charge the right tail.
CVaR-inverse is that rule -- but before it goes anywhere near the book it has to survive the same
falsifiers everything else did today:

  tail sweep     the "worst 10%" is a free parameter. Real measure -> PLATEAU across 5-33%.
  reversed dummy weights PROPORTIONAL to CVaR must be clearly worse (mechanism check).
  block bootstrap P(beat sigma) must RISE with block length, paired on identical resampled histories.
  vs the dial    does the rule capture what the hand-dial captured, or is it doing something else?
  per-year       a smoothness gain that lives in one era is beta.

All measured on the paired bootstrap (arb_common), equal bootstrapped-median maxDD, budget 3%.
Run: .venv/bin/python scratchpad/cvar_weight_final.py
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


def main():
    B = {k: leg(k)[0] for k in SIX}
    st = max(B[k].index.min() for k in SIX); en = min(B[k].index.max() for k in SIX)
    B = {k: B[k][(B[k].index >= st) & (B[k].index <= en)] for k in SIX}
    yrs = sorted({y for k in SIX for y in B[k].index.year}); first = yrs[0] + 2
    EQ = pd.Series(BUDGET / len(SIX), index=SIX)

    def wts(L, how, q=0.10, dial=1.0):
        o = {}
        for k in SIX:
            v = L[k].values
            if len(v) < 5:
                o[k] = 1.0
            elif how == "sigma":
                o[k] = 1.0 / max(v.std(), 1e-9)
            elif how == "cvar":
                o[k] = 1.0 / max(cvar(v, q), 1e-9)
            elif how == "cvar_p":                       # 逆向きダミー
                o[k] = max(cvar(v, q), 1e-9)
        r = pd.Series(o)
        r["btc15m_L"] *= dial
        return r / r.sum() * BUDGET

    def mix(how, q=0.10, dial=1.0):
        by = {}
        for y in yrs:
            past = {k: B[k][B[k].index.year < y] for k in SIX}
            by[y] = wts(past, how, q, dial) if (y >= first and min(len(past[k]) for k in SIX) >= 5) \
                else EQ
        return pd.concat([pd.Series(B[k].values * np.array([by[y][k] for y in B[k].index.year]),
                                    index=B[k].index) for k in SIX]).sort_index()

    arms = {("sigma", 0.10, 1.0): "σ逆数（現行）"}
    for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33):
        arms[("cvar", q, 1.0)] = f"CVaR worst {int(100*q)}%"
    arms[("cvar_p", 0.10, 1.0)] = "CVaR 比例（逆向きダミー）"
    arms[("sigma", 0.10, 1.5)] = "σ逆数 + btc15m_L を手で×1.5"
    arms[("cvar", 0.10, 1.5)] = "CVaR10 + btc15m_L を手で×1.5"

    S = {k: mix(*k) for k in arms}
    bt = Boot(months_union(*S.values()), nb=1000, k=3)
    D0 = bt.dd_median(S[("sigma", 0.10, 1.0)])
    base = bt.equal_dd_cagr(S[("sigma", 0.10, 1.0)], D0)[0]
    print(f"基準 maxDD = {D0:.2f}%（σ重み・WF・巡回ブロック3か月・中央値、1000経路を全アーム共通）")
    print("重みは毎年1月1日に、その年より前のトレードだけで再計算（先読みなし）。同DDに揃えて CAGR。\n")
    print(f"  {'重みの決め方':<26}{'btc15m_L の重み':>16}{'CAGR':>9}{'差':>8}"
          f"{'  P(σに勝つ):':<12}{'1か月':>7}{'3か月':>7}{'6か月':>7}{'12か月':>8}")
    r_sig = {}
    for k in (1, 3, 6, 12):
        r_sig[k] = Boot(bt.months, nb=800, k=k).ratios(S[("sigma", 0.10, 1.0)])
    for key, lab in arms.items():
        c = bt.equal_dd_cagr(S[key], D0)[0]
        wlast = wts({kk: B[kk][B[kk].index.year < yrs[-1]] for kk in SIX}, key[0], key[1], key[2])
        ps = []
        for k in (1, 3, 6, 12):
            r = Boot(bt.months, nb=800, k=k).ratios(S[key])
            ps.append(100 * np.mean(r > r_sig[k]))
        print(f"  {lab:<26}{100*wlast['btc15m_L']:>15.3f}%{c:>+9.1f}%{c-base:>+7.1f}pt{'':<12}"
              + "".join(f"{p:>6.0f}%" for p in ps)
              + ("  ← 現行" if key == ("sigma", 0.10, 1.0) else ""))

    print("\n年別の R（重み無し・レッグ素の合計）と、ブックの年別リターン（σ重み vs CVaR10重み）")
    s0, s1 = S[("sigma", 0.10, 1.0)], S[("cvar", 0.10, 1.0)]
    print(f"  {'年':<7}{'σ重み':>10}{'CVaR10':>10}{'差':>9}")
    for y in yrs:
        a = s0[s0.index.year == y].sum() * 100
        b = s1[s1.index.year == y].sum() * 100
        print(f"  {y:<7}{a:>+9.1f}%{b:>+9.1f}%{b-a:>+8.1f}pt")


if __name__ == "__main__":
    main()
