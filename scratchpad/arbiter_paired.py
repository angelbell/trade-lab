"""Re-run of today's two open verdicts on the PAIRED bootstrap (scratchpad/arb_common.py), because the
first pass measured them with an unpaired one and the noise band (+/-2-3 CAGR pt) was wider than the
effects (1-3 pt). Symptom that gave it away: the same equal-weight portfolio scored -1.1pt in one
column and +1.6pt in another, and the sigma baseline moved 41.9% -> 45.1% between runs.

Now every arm is de-levered on the SAME 1000 resampled histories, so the equal-maxDD comparison is
paired and the leftover Monte-Carlo noise mostly cancels.

  Q1  the book's weighting rule. sigma(R) charges btc15m_L's +6.8R right tail as "risk" (skew +1.84)
      and starves btc15m_S (best meanR +0.84, but sigma 3.19). Downside-only measures don't. But the
      weights must be WALK-FORWARD (recomputed each Jan 1 from prior trades) or it is lookahead --
      and full-sample ulcer already proved that point by scoring +9.9pt IS and -5.3pt WF.
  Q2  btc15m_L's 1-day max hold. It earns 33R LESS and only scores better through sigma. If the
      sigma channel is the whole story, a downside-only weighting should erase it.

Reported: paired equal-maxDD CAGR, and P(arm beats sigma) on the SAME resampled histories, by block
length -- a real effect's P rises with the block, a path-fit's collapses.
Run: .venv/bin/python scratchpad/arbiter_paired.py
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


def raw(L, how):
    o = {}
    for k in SIX:
        v = L[k].values
        if len(v) < 5:
            o[k] = 1.0; continue
        o[k] = {"sigma":  1.0 / max(v.std(), 1e-9),
                "cvar":   1.0 / max(cvar(v), 1e-9),
                "dnrisk": 1.0 / max(np.sqrt((v[v < 0] ** 2).mean()) if (v < 0).any() else 1e-9, 1e-9),
                "semi":   1.0 / max(np.sqrt(((v[v < v.mean()] - v.mean()) ** 2).mean()), 1e-9),
                "equal":  1.0}[how]
    r = pd.Series(o)
    return r / r.sum() * BUDGET


def main():
    B = {k: leg(k)[0] for k in SIX}
    A = {k: (leg(k, fwd=96)[0] if k == "btc15m_L" else B[k]) for k in SIX}
    st = max(B[k].index.min() for k in SIX); en = min(B[k].index.max() for k in SIX)
    clip = lambda D: {k: D[k][(D[k].index >= st) & (D[k].index <= en)] for k in SIX}
    B, A = clip(B), clip(A)
    yrs = sorted({y for k in SIX for y in B[k].index.year}); first = yrs[0] + 2

    def mix(L, how, walkfwd):
        by = {}
        for y in yrs:
            if not walkfwd:
                by[y] = raw(L, how)
            else:
                past = {k: L[k][L[k].index.year < y] for k in SIX}
                by[y] = raw(past, how) if (y >= first and min(len(past[k]) for k in SIX) >= 5) \
                    else raw(L, "equal")
        return pd.concat([pd.Series(L[k].values * np.array([by[y][k] for y in L[k].index.year]),
                                    index=L[k].index) for k in SIX]).sort_index()

    HOW = ["sigma", "cvar", "dnrisk", "semi", "equal"]
    streams = {(h, m, w): mix(L, h, w) for h in HOW for m, L in (("base", B), ("cut1d", A))
               for w in (False, True)}
    bt = Boot(months_union(*streams.values()), nb=1000, k=3)
    D0 = bt.dd_median(streams[("sigma", "base", True)])
    print(f"基準 maxDD = {D0:.2f}%（σ重み・WF・巡回ブロック3か月・中央値、1000経路を全アーム共通）")
    print("全ての数字を、この maxDD にそろえた上での CAGR。対で比較（同じ並べ替えの上で）。\n")

    print("Q1  重みの決め方（btc15m_L は現行のまま＝打ち切り無し）")
    print(f"  {'重み':<9}{'先読み(全期間) CAGR':>20}{'差':>8}{'ウォークフォワード CAGR':>24}{'差':>8}")
    b_is = b_wf = None
    for h in HOW:
        c_is = bt.equal_dd_cagr(streams[(h, "base", False)], D0)[0]
        c_wf = bt.equal_dd_cagr(streams[(h, "base", True)], D0)[0]
        if b_is is None:
            b_is, b_wf = c_is, c_wf
        print(f"  {h:<9}{c_is:>+19.1f}%{c_is-b_is:>+7.1f}pt{c_wf:>+23.1f}%{c_wf-b_wf:>+7.1f}pt"
              + ("  ← 現行" if h == "sigma" else ""))

    print("\n  対ブートストラップ: σ重み(WF)に勝つ確率（本物ならブロックを長くするほど上がる）")
    print(f"  {'重み':<9}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    for h in HOW[1:]:
        row = []
        for k in (1, 3, 6, 12):
            bk = Boot(bt.months, nb=800, k=k)
            r0 = bk.ratios(streams[("sigma", "base", True)])
            r1 = bk.ratios(streams[(h, "base", True)])
            row.append(100 * np.mean(r1 > r0))
        print(f"  {h:<9}" + "".join(f"{v:>7.0f}%" for v in row))

    print("\n\nQ2  btc15m_L を1日で強制決済（『σ が右の裾を罰する』のが正体なら、下方尺度で消えるはず）")
    print(f"  {'重み':<9}{'現行 CAGR':>12}{'1日で切る CAGR':>17}{'差':>9}"
          f"{'12か月ブロックでの P':>22}")
    for h in HOW:
        c0 = bt.equal_dd_cagr(streams[(h, "base", True)], D0)[0]
        c1 = bt.equal_dd_cagr(streams[(h, "cut1d", True)], D0)[0]
        b12 = Boot(bt.months, nb=800, k=12)
        p = 100 * np.mean(b12.ratios(streams[(h, "cut1d", True)])
                          > b12.ratios(streams[(h, "base", True)]))
        print(f"  {h:<9}{c0:>+11.1f}%{c1:>+16.1f}%{c1-c0:>+8.1f}pt{p:>21.0f}%"
              + ("  ← 現行" if h == "sigma" else ""))


if __name__ == "__main__":
    main()
