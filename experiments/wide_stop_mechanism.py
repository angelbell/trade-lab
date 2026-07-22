"""Why does "skip breakouts whose stop exceeds 2% of price" work -- and only as a PRODUCT
(stop/ATR alone hurts, ATR/price alone hurts)?

Hypothesis, registered BEFORE running: a 15m wave larger than ~2% of BTC's price is not an orderly
consolidation -- it is a liquidation cascade or a news spike. Breakouts out of those do not follow
through. The threshold is ABSOLUTE (a % of price) because BTC's liquidation ladders are set in
percentage terms by leverage, not in ATR terms; dividing by ATR destroys exactly the information.

The hypothesis makes a falsifiable PREDICTION:
    the rule should work on BTC legs and FAIL on gold legs.
Gold has no 100x liquidation cascade. If the rule helps gold too, the mechanism story is wrong and
this is just "skip trades with big stops", a generic variance cut wearing a costume.

Test it on all five breakout legs, each judged on the BOOK, at each leg's own natural threshold
scale (sweep the % and report the best AND the whole curve, so a lone spike is visible).
Run: .venv/bin/python experiments/wide_stop_mechanism.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import w_trade


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
    return cagr / max(dd, 1e-9)


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    r0 = book_of(legs0)
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    print("事前登録の予言: **BTC の脚には効き、gold の脚には効かない**（gold に清算カスケードは無い）。")
    print(f"gold にも効いたら機構の読みは誤り＝ただの分散カット。   現行ブック = {r0:.2f}\n")

    for k in ["btc15m_L", "btc15m_S", "btc_bo_kama", "gold15m", "gold_bo"]:
        s, sp = L[k]
        q = np.percentile(sp, [10, 50, 90])
        print(f"  === {k}   損切り/価格(%): 中央値 {q[1]:.2f}%  10/90%点 {q[0]:.2f}% / {q[2]:.2f}%")
        print(f"      {'閾値':>9}{'n':>6}{'残す率':>8}{'PF':>7}{'totR/年':>9}{'ブック':>9}{'差':>8}")
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        print(f"      {'全部':>9}{len(s):>6}{'100%':>8}{pf(s.values):>7.2f}"
              f"{s.sum()/yrs:>+9.1f}{r0:>9.2f}{0.0:>+8.2f}")
        # 各レッグの分布に合わせて、上位 10/20/30/40/50% を切る（同じ「切る割合」で比較する）
        for cutpct in (10, 20, 30, 40, 50):
            thr = np.percentile(sp, 100 - cutpct)
            m = sp <= thr
            legs = dict(legs0); legs[k] = s[m]
            rb = book_of(legs)
            print(f"      {f'<= {thr:.2f}%':>9}{m.sum():>6}{100-cutpct:>7}%{pf(s.values[m]):>7.2f}"
                  f"{s.values[m].sum()/yrs:>+9.1f}{rb:>9.2f}{rb-r0:>+8.2f}"
                  + ("  ★" if rb > r0 + 0.05 else ""))
        print()

    print("\n判定: ★ が BTC の脚に集中し、gold の脚に出なければ、清算カスケード説を支持する。")
    print("      gold にも同じだけ出たら、機構の読みは棄却（＝ただの分散カット）。")


if __name__ == "__main__":
    main()
