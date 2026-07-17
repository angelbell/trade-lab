"""T1 (expanding-window percentile) says the book gain dies without lookahead.
T3 (walk-forward: threshold from the first half) says the LEG improves out of sample (PF 1.77 -> 1.96).
They cannot both be the last word. Settle it on the only question a trader actually faces:

    Freeze a threshold using the FIRST HALF only. Trade the SECOND HALF with it.
    Does the BOOK -- the arbiter -- improve on that second half?

Nothing else counts. The expanding-window version re-fits the cut at every trade and is noisy in the
burn-in; the walk-forward version is what a person would really do (calibrate once, then freeze).
Measure the book on the OOS half alone, with every leg truncated to that half, weights recomputed
inside it, and report the random-drop null on the OOS half too.
Run: .venv/bin/python scratchpad/wide_stop_walkforward.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import w_trade

RNG = np.random.default_rng(20260714)
NDRAW = 2000


def book_of(legs):
    w = w_trade(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    s = pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                             index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                   for k in SIX]).sort_index()
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def main():
    L = raw_legs()
    R, sp = L["btc15m_L"]
    half_t = R.index[len(R) // 2]                       # btc15m_L の中央時点で全レッグを切る
    print(f"分割点: {half_t.date()}   （前半で閾値を決め、後半で裁定する）\n")

    def legs_on(period, thr=None):
        out = {}
        for k, (s, _) in L.items():
            x = s[s.index < half_t] if period == "IS" else s[s.index >= half_t]
            out[k] = x
        if thr is not None:
            m = sp <= thr
            x = R[m]
            out["btc15m_L"] = x[x.index < half_t] if period == "IS" else x[x.index >= half_t]
        return out

    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    for period in ("IS", "OOS"):
        legs = legs_on(period)
        c0, d0, r0 = book_of(legs)
        b = legs["btc15m_L"]
        print(f"  === {period}（{'前半' if period=='IS' else '後半'}）  "
              f"フィルタ無し: ブック CAGR {c0:+.1f}% / maxDD {d0:.2f}% / **CAGR/DD {r0:.2f}**"
              f"   btc15m_L n={len(b)} PF {pf(b.values):.2f}")
        print(f"      {'前半で決めた閾値':<22}{'n':>6}{'PF':>7}{'meanR':>9}"
              f"{'ブックCAGR':>11}{'ブックDD':>10}{'ブックCAGR/DD':>14}{'差':>8}")
        for p in (60, 70, 80):
            thr = np.percentile(sp[:len(R) // 2], p)     # ★ 前半だけから決める
            legs = legs_on(period, thr)
            c, d, r = book_of(legs)
            b = legs["btc15m_L"]
            print(f"      {str(p)+'%点 = '+f'{thr:.2f}%':<22}{len(b):>6}{pf(b.values):>7.2f}"
                  f"{b.mean():>+9.3f}{c:>10.1f}%{d:>9.2f}%{r:>14.2f}{r-r0:>+8.2f}"
                  + ("  ★" if r > r0 + 0.05 else ""))
        print()

    print("\n後半（OOS）でのランダム除去null: 同じ本数を無作為に捨てたら（2000回）\n")
    legs = legs_on("OOS")
    r0 = book_of(legs)[2]
    for p in (60, 70, 80):
        thr = np.percentile(sp[:len(R) // 2], p)
        m = sp <= thr
        x = R[m]; x = x[x.index >= half_t]
        lg = legs_on("OOS", thr)
        r_obs = book_of(lg)[2]
        full = R[R.index >= half_t]
        nulls = []
        for _ in range(NDRAW):
            k = np.sort(RNG.choice(len(full), len(x), replace=False))
            l2 = dict(legs); l2["btc15m_L"] = full.iloc[k]
            nulls.append(book_of(l2)[2])
        nulls = np.array(nulls)
        print(f"  {p}%点 (閾値 {thr:.2f}%):  観測 {r_obs:.2f}  vs ランダム除去 中央値 {np.median(nulls):.2f} "
              f"[5% {np.percentile(nulls,5):.2f}, 95% {np.percentile(nulls,95):.2f}]  "
              f"→ **{100*np.mean(r_obs > nulls):.0f}%ile**   （フィルタ無しの後半 = {r0:.2f}）")


if __name__ == "__main__":
    main()
