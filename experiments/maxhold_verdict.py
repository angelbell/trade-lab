"""The 1-day max hold (`fwd=96`) lifts the swap-included book 4.84 -> 5.57. The decomposition says it
is NOT the time stop (same entries, force-closed = 4.84 -> 4.04, counterfactual -72.4R) and NOT the
extra slots (max_pos=2 = 3.27). It only works as the COMBINATION: cut the trade at 1 day, and the
single slot then refills with the next signal (n 763 -> 1002).

That leaves one uncomfortable fact: **the arm EARNS LESS**. 252R becomes 219R. It scores better only
because sigma(R) drops (the right tail is amputated), and a smoother leg buys leverage -- exactly the
trap that cost five retractions today. Equal-maxDD is supposed to be the antidote, and the arm passed
it (+7.8 CAGR pt). But that verdict was read off ONE path's maxDD, which is the other trap.

So settle it the way the checklist demands:
  1. circular block bootstrap over months (1/3/6/12). A real mechanism's P(better) RISES with the
     block length; a path-fit collapses toward a coin flip.
  2. bootstrapped maxDD (median, not the realized path) for both arms, then CAGR/DD off the MEDIAN DD.
  3. the equal-DD CAGR gap, recomputed on every bootstrap path -- report the DISTRIBUTION, not one number.
Run: .venv/bin/python experiments/maxhold_verdict.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from rr_with_swap import leg, SIX

RNG = np.random.default_rng(20260714)
NB = 2000


def w_inv(L, budget=0.03):
    s = pd.Series({k: L[k].std() for k in SIX})
    return (1 / s) / (1 / s).sum() * budget


def stream(L, w, scale=1.0):
    st = max(L[k].index.min() for k in SIX); en = min(L[k].index.max() for k in SIX)
    return pd.concat([pd.Series(L[k][(L[k].index >= st) & (L[k].index <= en)].values * w[k] * scale,
                                index=L[k][(L[k].index >= st) & (L[k].index <= en)].index)
                      for k in SIX]).sort_index()


def cd(vals, days):
    eq = np.cumprod(1 + vals); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100
    return cagr, dd


def blocks(s, k):
    """巡回ブロック・ブートストラップ: 月をブロック単位で並べ直し、トレード列を作る。"""
    mk = s.index.to_period("M")
    months = sorted(mk.unique())
    by = {m: s.values[mk == m] for m in months}
    nm = len(months)
    nblk = int(np.ceil(nm / k))
    out = []
    for _ in range(NB):
        v = []
        for b in RNG.integers(0, nm, nblk):
            for j in range(k):
                v.append(by[months[(b + j) % nm]])
        out.append(np.concatenate(v)[:len(s)])
    return out, (s.index[-1] - s.index[0]).days


def main():
    B0 = {k: leg(k)[0] for k in SIX}
    A = {k: (leg(k, fwd=96)[0] if k == "btc15m_L" else B0[k]) for k in SIX}
    lab = {"現行（打切り無し）": B0, "1日で強制決済(fwd=96)": A}

    print("1. 実測（1本の経路）")
    print(f"   {'':<24}{'n(L)':>6}{'totR(L)':>9}{'σ(L)':>7}{'重み(L)':>9}"
          f"{'CAGR':>8}{'maxDD':>8}{'CAGR/DD':>9}")
    S, W = {}, {}
    for nm, L in lab.items():
        w = w_inv(L); s = stream(L, w); W[nm] = w; S[nm] = s
        c, d = cd(s.values, (s.index[-1] - s.index[0]).days)
        print(f"   {nm:<24}{len(L['btc15m_L']):>6}{L['btc15m_L'].sum():>+9.0f}"
              f"{L['btc15m_L'].std():>7.3f}{100*w['btc15m_L']:>8.3f}%{c:>+8.1f}%{d:>7.2f}%{c/d:>9.2f}")

    print("\n2. maxDD を1本の経路から読まない（巡回ブロック=3か月, 2000回）")
    print(f"   {'':<24}{'実測DD':>8}{'DD中央値':>10}{'DD 95%点':>10}{'実測の位置':>11}"
          f"{'中央値DDでの CAGR/DD':>22}")
    boot = {}
    for nm in lab:
        bs, days = blocks(S[nm], 3)
        dds = np.array([cd(v, days)[1] for v in bs])
        cgs = np.array([cd(v, days)[0] for v in bs])
        boot[nm] = (bs, days, cgs, dds)
        c0, d0 = cd(S[nm].values, days)
        pct = 100 * np.mean(d0 > dds)
        print(f"   {nm:<24}{d0:>7.2f}%{np.median(dds):>9.2f}%{np.percentile(dds,95):>9.2f}%"
              f"{pct:>10.0f}%{c0/np.median(dds):>22.2f}")

    print("\n3. ブロック長を変える（本物なら、長くするほど P が上がる）")
    print(f"   {'ブロック':>8}{'P(1日で切る方が CAGR/DD 高い)':>32}{'CAGR/DD 差 中央値':>20}")
    for k in (1, 3, 6, 12):
        bb, days = blocks(S["現行（打切り無し）"], k)
        ba, _ = blocks(S["1日で強制決済(fwd=96)"], k)
        rb = np.array([(lambda c, d: c / max(d, 1e-9))(*cd(v, days)) for v in bb])
        ra = np.array([(lambda c, d: c / max(d, 1e-9))(*cd(v, days)) for v in ba])
        print(f"   {k:>6}か月{100*np.mean(ra > rb):>28.0f}%{np.median(ra - rb):>+20.2f}")
    print("   ※ 2つの腕を独立にリサンプルしている（同じ月の並びで対にはしていない）")

    print("\n4. 同じ maxDD にそろえて CAGR（中央値DD基準・レバレッジを排除）")
    D0 = np.median(boot["現行（打切り無し）"][3])
    print(f"   基準 maxDD = {D0:.2f}%（現行のブートストラップ中央値）\n")
    print(f"   {'':<24}{'倍率':>7}{'総リスク':>9}{'CAGR':>9}{'現行比':>9}")
    base_c = None
    for nm, L in lab.items():
        w = W[nm]
        lo, hi = 0.2, 4.0
        for _ in range(50):
            m = (lo + hi) / 2
            s = stream(L, w, m)
            bs, days = blocks(s, 3)
            if np.median([cd(v, days)[1] for v in bs[:300]]) > D0:
                hi = m
            else:
                lo = m
        s = stream(L, w, lo)
        c = cd(s.values, (s.index[-1] - s.index[0]).days)[0]
        if base_c is None:
            base_c = c
        print(f"   {nm:<24}{lo:>7.2f}{3*lo:>8.2f}%{c:>+9.1f}%{c-base_c:>+8.1f}pt")


if __name__ == "__main__":
    main()
