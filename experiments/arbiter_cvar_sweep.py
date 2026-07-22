"""Walk-forward, CVaR-inverse weighting beat the book's sigma-inverse rule by +3.0 CAGR pt at equal
bootstrapped-median drawdown, with block-bootstrap P rising 53 -> 63% as the block lengthens. That is
the shape of a real effect, but +3.0pt is small and "the worst 10%" was a number I picked. Two ways it
could still be a fit:

  1. the tail fraction is a free parameter. A real risk measure gives a PLATEAU across 5-33%; an
     overfit one spikes at exactly the value I happened to choose.
  2. equal-weight already beats sigma walk-forward (+1.6pt), so most of CVaR's gain might just be
     "stop trusting the estimated weights". Shrinking CVaR toward equal separates the two: if the
     best blend is 100% CVaR, the measure is doing the work; if it is ~50/50 or ~0%, the gain is
     mostly shrinkage and CVaR is barely adding anything.

Also tested, because the obvious next thought is "if sigma ignores edge, put edge IN":
  mean/cvar      w proportional to meanR / CVaR      -- full edge tilt (estimated on past trades only)
  sqrtmean/cvar  w proportional to sqrt(meanR)/CVaR  -- shrunk edge tilt
Structural law 8 says nothing has beaten static inv-vol; an estimated-edge tilt is the classic way to
lose that bet, so it is here as a falsifier, not a hope.

Weights are recomputed every Jan 1 from PRIOR trades only. Every row is de-levered to the same
bootstrapped-median maxDD before its CAGR is read. Budget 3%.
Run: .venv/bin/python experiments/arbiter_cvar_sweep.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from rr_with_swap import leg, SIX

RNG = np.random.default_rng(20260714)
BUDGET = 0.03


def cvar(v, q):
    return abs(np.mean(np.sort(v)[:max(1, int(q * len(v)))]))


def raw_w(L, how, q=0.10):
    out = {}
    for k in SIX:
        v = L[k].values
        if len(v) < 5:
            out[k] = 1.0; continue
        if how == "sigma":
            out[k] = 1.0 / v.std()
        elif how == "equal":
            out[k] = 1.0
        elif how == "cvar":
            out[k] = 1.0 / max(cvar(v, q), 1e-9)
        elif how == "mean_cvar":
            out[k] = max(v.mean(), 0.01) / max(cvar(v, q), 1e-9)
        elif how == "sqrtmean_cvar":
            out[k] = np.sqrt(max(v.mean(), 0.01)) / max(cvar(v, q), 1e-9)
        else:
            raise ValueError(how)
    return pd.Series(out)


def weights(L, how, q=0.10, shrink=0.0):
    """shrink = 等分に寄せる割合（0 = 尺度そのまま, 1 = 完全に等分）。"""
    r = raw_w(L, how, q); r = r / r.sum()
    e = pd.Series(1.0 / len(SIX), index=SIX)
    return ((1 - shrink) * r + shrink * e) * BUDGET


def cd(v, days):
    eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
    return (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100, ((pk - eq) / pk).max() * 100


def boot_dd(s, nb=200, k=3):
    mk = s.index.to_period("M"); months = sorted(mk.unique())
    by = {m: s.values[mk == m] for m in months}
    nm = len(months); nblk = int(np.ceil(nm / k)); days = (s.index[-1] - s.index[0]).days
    return float(np.median([cd(np.concatenate([by[months[(b + j) % nm]]
                                               for b in RNG.integers(0, nm, nblk)
                                               for j in range(k)])[:len(s)], days)[1]
                            for _ in range(nb)]))


def eq_cagr(s, D0):
    lo, hi = 0.10, 5.0
    for _ in range(16):
        m = (lo + hi) / 2
        if boot_dd(s * m) > D0:
            hi = m
        else:
            lo = m
    return cd((s * lo).values, (s.index[-1] - s.index[0]).days)[0]


def main():
    B = {k: leg(k)[0] for k in SIX}
    st = max(B[k].index.min() for k in SIX); en = min(B[k].index.max() for k in SIX)
    B = {k: B[k][(B[k].index >= st) & (B[k].index <= en)] for k in SIX}
    yrs = sorted({y for k in SIX for y in B[k].index.year})
    first = yrs[0] + 2

    def wf(how, q=0.10, shrink=0.0):
        by = {}
        for y in yrs:
            past = {k: B[k][B[k].index.year < y] for k in SIX}
            by[y] = weights(past, how, q, shrink) if y >= first and min(len(past[k]) for k in SIX) >= 5 \
                else weights(B, "equal")
        return pd.concat([pd.Series(B[k].values * np.array([by[y][k] for y in B[k].index.year]),
                                    index=B[k].index) for k in SIX]).sort_index()

    D0 = boot_dd(wf("sigma"))
    c_sig, c_eq = eq_cagr(wf("sigma"), D0), eq_cagr(wf("equal"), D0)
    print(f"基準 maxDD = {D0:.2f}%（現行σ重み・WF・巡回ブロック3か月の中央値）。全行を同DDに揃えて CAGR。")
    print(f"  σ逆数（現行） = {c_sig:+.1f}%      等分 = {c_eq:+.1f}% ({c_eq-c_sig:+.1f}pt)\n")

    print("1. CVaR のテール割合を振る（本物ならプラトー、過剰適合なら 10% だけ尖る）")
    print(f"  {'worst q%':>9}{'CAGR':>9}{'σ比':>9}")
    for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50):
        c = eq_cagr(wf("cvar", q=q), D0)
        print(f"  {100*q:>8.0f}%{c:>+9.1f}%{c-c_sig:>+8.1f}pt" + ("  ← 既定" if q == 0.10 else ""))

    print("\n2. 等分への縮小（gain が『尺度』なのか『重みを信じないこと』なのかを分ける）")
    print(f"  {'縮小':>7}{'CVaR10':>10}{'σ比':>9}   |{'':>3}{'σ逆数':>9}{'σ比':>9}")
    for sh in (0.0, 0.25, 0.50, 0.75, 1.0):
        cc = eq_cagr(wf("cvar", shrink=sh), D0)
        cs = eq_cagr(wf("sigma", shrink=sh), D0)
        print(f"  {100*sh:>6.0f}%{cc:>+10.1f}%{cc-c_sig:>+8.1f}pt   |   {cs:>+9.1f}%{cs-c_sig:>+8.1f}pt")

    print("\n3. 『σ はエッジを見ない』なら、エッジを入れたらどうなるか（法則8の反証）")
    print(f"  {'重み':<18}{'CAGR':>9}{'σ比':>9}")
    for how, lab in (("mean_cvar", "meanR / CVaR10"), ("sqrtmean_cvar", "√meanR / CVaR10")):
        c = eq_cagr(wf(how), D0)
        print(f"  {lab:<18}{c:>+9.1f}%{c-c_sig:>+8.1f}pt")


if __name__ == "__main__":
    main()
