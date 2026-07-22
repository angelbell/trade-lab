"""btc15m_L with a 1-day max hold lifts the swap-included book 4.84 -> 5.57. Every other "gain" today
that looked like this turned out to be inv-vol quietly buying leverage, so assume it is until proven
otherwise. Two red flags are already visible:

  - n goes 763 -> 1002. A shorter max hold frees the position slot sooner, so MORE signals are taken.
    More trades -> lower sigma(R) -> inv-vol hands the leg a bigger weight.
  - total R FALLS: 763 x 0.331 = 253R becomes 1002 x 0.219 = 219R. Earning LESS while the book
    improves is the signature of a bet-size effect, not an edge.

Settle it the only way that cannot be gamed: de-lever every arm to the SAME book maxDD and compare
CAGR. Plus the pinned-weight control (weights frozen at today's values), and the reversed dummy
(a LONGER max hold, which must come out worse if the mechanism is real).
Run: .venv/bin/python experiments/maxhold_equal_dd.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from rr_with_swap import leg, SIX


def w_inv(legs, budget=0.03):
    sig = pd.Series({k: legs[k].std() for k in SIX})
    return (1 / sig) / (1 / sig).sum() * budget


def curve(legs, w, scale=1.0):
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    return pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values
                                * w[k] * scale,
                                index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                      for k in SIX]).sort_index()


def cd(s):
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
    return cagr, dd


def main():
    B0 = {k: leg(k)[0] for k in SIX}
    w0 = w_inv(B0)
    C0, D0 = cd(curve(B0, w0))
    print(f"スワップ込みの正典: CAGR {C0:+.1f}%  maxDD {D0:.2f}%  CAGR/DD {C0/max(D0,1e-9):.2f}")
    print(f"→ **全ての腕を maxDD {D0:.2f}% にそろえて CAGR で比べる**（レバレッジを完全に排除）\n")
    print(f"  {'btc15m_L の最大保有':<24}{'n':>6}{'totR':>8}{'σ(R)':>7}{'重み':>8}"
          f"{'総リスク':>9}{'CAGR':>9}{'現行比':>10}{'重み固定のブック':>17}")

    def eq_dd(legs):
        w = w_inv(legs)
        lo, hi = 0.2, 4.0
        for _ in range(60):
            m = (lo + hi) / 2
            if cd(curve(legs, w, m))[1] > D0:
                hi = m
            else:
                lo = m
        c, d = cd(curve(legs, w, lo))
        return lo, c, w

    base_c = None
    for fwd, lab in ((96, "96本（1日）"), (192, "192本（2日）"), (300, "300本（3日）"),
                     (500, "500本（現行・5日）"), (1000, "1000本（10日）＝逆向きダミー")):
        s = leg("btc15m_L", fwd=fwd)[0]
        L = dict(B0); L["btc15m_L"] = s
        sc, c, w = eq_dd(L)
        if fwd == 500:
            base_c = c
        cf, df = cd(curve(L, w0))                       # 重みを現行に固定した版
        print(f"  {lab:<24}{len(s):>6}{s.sum():>+8.0f}{s.std():>7.3f}"
              f"{100*w['btc15m_L']:>7.3f}%{3*sc:>8.2f}%{c:>8.1f}%"
              f"{(c - base_c) if base_c is not None else 0:>+9.1f}pt{cf/max(df,1e-9):>17.2f}"
              + ("  ← 現行" if fwd == 500 else ""))
    print(f"\n  ※ 「重み固定のブック」= 現行の重み(btc15m_L {100*w0['btc15m_L']:.3f}%)のまま。")
    print(f"     現行のブック（重み固定 = 同じもの）= {C0/max(D0,1e-9):.2f}")


if __name__ == "__main__":
    main()
