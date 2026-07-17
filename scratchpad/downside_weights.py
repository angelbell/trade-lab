"""The 1-day max hold "improves" the book while EARNING LESS (252R -> 219R). The whole gain runs
through sigma: cutting the hold amputates the RIGHT tail, sigma(R) falls 2.21 -> 1.79, and the book's
inv-vol rule reads that as "safer" and hands the leg 19% more money.

That is a defect in the ARBITER, not a discovery about exits. A trend leg's R distribution is heavily
right-skewed (loss floored at -1R, wins run to +6.9R). Standard deviation charges the +6.9R tail as
"risk" identically to the -1R tail. So ANY rule that cuts winners short is rewarded by inv-vol, and
any rule that lets them run is punished. Structural law 4 says cutting winners is the single most
expensive thing you can do -- so the sizing rule is actively fighting the book's own edge.

Downside-only risk measures don't have this hole:
  sigma       std(R)                      -- current
  semi        std of R below the mean      -- right tail free
  dnrisk      RMS of the losses only       -- pure loss magnitude
  cvar        mean of the worst 10% of R
  ulcer       RMS drawdown of the leg's own equity (a path measure, not a moment)

Two questions, one script:
  Q1  Which weighting gives the best BOOK (bootstrapped median maxDD, not one path)?
  Q2  Under a downside weighting, does the 1-day max hold's advantage SURVIVE or evaporate?
      If it evaporates, the arm was never an exit finding -- it was gaming std().
Run: .venv/bin/python scratchpad/downside_weights.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from rr_with_swap import leg, SIX

RNG = np.random.default_rng(20260714)
NB = 600


def risk_of(x, how):
    v = x.values
    if how == "sigma":
        return v.std()
    if how == "semi":
        m = v.mean()
        d = v[v < m] - m
        return np.sqrt((d ** 2).mean())
    if how == "dnrisk":
        d = v[v < 0]
        return np.sqrt((d ** 2).mean()) if len(d) else 1e-9
    if how == "cvar":
        return abs(np.mean(np.sort(v)[:max(1, int(0.10 * len(v)))]))
    if how == "ulcer":
        eq = np.cumprod(1 + v * 0.01)          # 1%/本の名目でレッグ単体の資産曲線
        pk = np.maximum.accumulate(eq)
        return np.sqrt((((pk - eq) / pk) ** 2).mean()) * 100
    raise ValueError(how)


def w_of(L, how, budget=0.03):
    r = pd.Series({k: risk_of(L[k], how) for k in SIX})
    return (1 / r) / (1 / r).sum() * budget


def stream(L, w, scale=1.0):
    st = max(L[k].index.min() for k in SIX); en = min(L[k].index.max() for k in SIX)
    return pd.concat([pd.Series(L[k][(L[k].index >= st) & (L[k].index <= en)].values * w[k] * scale,
                                index=L[k][(L[k].index >= st) & (L[k].index <= en)].index)
                      for k in SIX]).sort_index()


def cd(v, days):
    eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
    return (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100, ((pk - eq) / pk).max() * 100


def boot_dd(s, k=3, nb=NB):
    mk = s.index.to_period("M"); months = sorted(mk.unique())
    by = {m: s.values[mk == m] for m in months}
    nm = len(months); nblk = int(np.ceil(nm / k)); days = (s.index[-1] - s.index[0]).days
    out = []
    for _ in range(nb):
        v = np.concatenate([by[months[(b + j) % nm]] for b in RNG.integers(0, nm, nblk)
                            for j in range(k)])[:len(s)]
        out.append(cd(v, days)[1])
    return float(np.median(out))


def eq_dd_cagr(L, how, D0):
    """重みを how で決め、ブートストラップ中央値maxDD が D0 になるまで倍率を振り、その時の CAGR。"""
    w = w_of(L, how)
    lo, hi = 0.2, 4.0
    for _ in range(22):
        m = (lo + hi) / 2
        if boot_dd(stream(L, w, m), nb=200) > D0:
            hi = m
        else:
            lo = m
    s = stream(L, w, lo)
    return cd(s.values, (s.index[-1] - s.index[0]).days)[0], lo, w


def main():
    B0 = {k: leg(k)[0] for k in SIX}
    A = {k: (leg(k, fwd=96)[0] if k == "btc15m_L" else B0[k]) for k in SIX}

    print("レッグの R 分布は右に長い（損失は −1R で床、勝ちは +6.9R まで走る）")
    print(f"  {'leg':<14}{'n':>5}{'meanR':>8}{'σ':>7}{'半偏差':>8}{'損失RMS':>9}{'歪度':>7}"
          f"{'最大の勝ち':>9}")
    for k in SIX:
        v = B0[k].values
        sk = ((v - v.mean()) ** 3).mean() / max(v.std() ** 3, 1e-9)
        print(f"  {k:<14}{len(v):>5}{v.mean():>+8.3f}{risk_of(B0[k],'sigma'):>7.2f}"
              f"{risk_of(B0[k],'semi'):>8.2f}{risk_of(B0[k],'dnrisk'):>9.2f}{sk:>+7.2f}{v.max():>+9.2f}")

    D0 = boot_dd(stream(B0, w_of(B0, "sigma")))
    print(f"\n基準 maxDD = {D0:.2f}%（現行ブック・巡回ブロック3か月の中央値）")
    print("全ての行を、この maxDD にそろえてから CAGR で比べる（レバレッジを完全に排除）\n")

    print("Q1  重みの決め方を替える（ブックの現行 = σ の逆数）")
    print(f"  {'リスク尺度':<10}{'gold_bo':>9}{'btc_bo':>8}{'btc_pull':>10}{'gold15m':>9}"
          f"{'btc15m_L':>10}{'btc15m_S':>10}{'CAGR':>9}{'現行比':>9}")
    base = None
    for how in ("sigma", "semi", "dnrisk", "cvar", "ulcer"):
        c, sc, w = eq_dd_cagr(B0, how, D0)
        if base is None:
            base = c
        print(f"  {how:<10}" + "".join(f"{100*w[k]:>9.3f}%" for k in SIX)
              + f"{c:>+9.1f}%{c-base:>+8.1f}pt" + ("  ← 現行" if how == "sigma" else ""))

    print("\nQ2  その重みの下で、『1日で強制決済』はまだ勝つか")
    print(f"  {'リスク尺度':<10}{'現行のCAGR':>12}{'1日で切るCAGR':>15}{'差':>9}"
          f"{'btc15m_L の重み 現行→1日':>26}")
    for how in ("sigma", "semi", "dnrisk", "cvar", "ulcer"):
        c0, _, w0 = eq_dd_cagr(B0, how, D0)
        c1, _, w1 = eq_dd_cagr(A, how, D0)
        print(f"  {how:<10}{c0:>+12.1f}%{c1:>+14.1f}%{c1-c0:>+8.1f}pt"
              f"{100*w0['btc15m_L']:>19.3f}% →{100*w1['btc15m_L']:>6.3f}%"
              + ("  ← 現行" if how == "sigma" else ""))


if __name__ == "__main__":
    main()
