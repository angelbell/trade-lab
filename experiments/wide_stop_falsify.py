"""The stop/price filter looks too good: cut 30% of btc15m_L's trades, keep 100% of the return,
PF 1.74 -> 2.00, book 7.88 -> 8.33, block-bootstrap P rising to 94%. Two ways it can still be fake:

T1  LOOKAHEAD IN THE THRESHOLD. "1.71%" is the 70th percentile of the WHOLE sample. Nobody trading
    in 2019 could know it. The only honest version computes the cut from PAST trades only.
    Re-run with an EXPANDING-WINDOW percentile (burn-in 100 trades, then the p-th percentile of
    everything seen so far). If the gain dies, the filter was reading the future.

T2  IT IS JUST LEVERAGE. The book weights legs by 1/sigma(trade R). Cutting the widest-stop trades
    removes high-variance outcomes -> sigma falls -> btc15m_L's weight RISES -> a positive-expectancy
    leg gets a bigger bet. That alone lifts CAGR/DD, with no "filter" doing any work.
    Test: apply the filter but PIN every weight at its baseline value. If the gain vanishes, the
    filter is a leverage dial and the same lift is available by just betting more.

Plus: T3 walk-forward (threshold chosen on the first half only), T4 book-level random-drop null
(drop the same NUMBER of trades at random), T5 per-year.
Run: .venv/bin/python experiments/wide_stop_falsify.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import w_trade

RNG = np.random.default_rng(20260714)
NDRAW = 2000


def book_from(legs, w=None):
    """CAGR/DD on the trade-resolution curve. w=None -> recompute inv-vol weights (the live rule);
    w=<Series> -> pin the weights (isolates the filter from the leverage it silently buys)."""
    ww = w_trade(legs, SIX) if w is None else w
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    s = pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * ww[k],
                             index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                   for k in SIX]).sort_index()
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    R, sp = L["btc15m_L"]
    w0 = w_trade(legs0, SIX)
    c0, d0, r0 = book_from(legs0)
    yrs = (R.index[-1] - R.index[0]).days / 365.25
    print(f"現行: ブック CAGR {c0:+.1f}% / maxDD {d0:.2f}% / **CAGR/DD {r0:.2f}**   "
          f"btc15m_L の重み {100*w0['btc15m_L']:.3f}%  σ(R) {R.std():.3f}\n")

    print("T2  これは『フィルタ』か、それとも『レバレッジ』か")
    print("    重みを現行のまま固定して、フィルタだけを当てる。差が消えたらレバレッジ。\n")
    print(f"  {'しきい値':<20}{'n':>6}{'σ(R)':>8}{'重み(再計算)':>13}"
          f"{'ブック(重み再計算)':>18}{'ブック(重み固定)':>17}")
    for thr in (3.0, 2.5, 2.0, 1.71, 1.5):
        m = sp <= thr
        legs = dict(legs0); legs["btc15m_L"] = R[m]
        wn = w_trade(legs, SIX)
        _, _, r_re = book_from(legs)               # 重み再計算（＝現行の運用ルール）
        _, _, r_fx = book_from(legs, w0)           # 重み固定（＝フィルタ単独の効果）
        print(f"  {'損切り<=価格の '+str(thr)+'%':<20}{m.sum():>6}{R[m].std():>8.3f}"
              f"{100*wn['btc15m_L']:>12.3f}%{r_re:>18.2f}{r_fx:>17.2f}")
    print(f"\n  参考: 何もフィルタせずに btc15m_L の重みだけ上げたら（＝純粋なレバレッジ）")
    for mult in (1.05, 1.10, 1.15, 1.20):
        w = w0.copy(); w["btc15m_L"] *= mult
        _, _, r = book_from(legs0, w)
        print(f"    重み ×{mult:.2f}  ({100*w['btc15m_L']:.3f}%)  →  ブック {r:.2f}")

    print("\n\nT1  しきい値の先読み: 『過去だけ』で閾値を決めたらどうなるか")
    print("    各トレード時点で、それまでに見たトレードの p パーセンタイルを閾値にする（助走100本）\n")
    print(f"  {'方式':<34}{'n':>6}{'年本数':>8}{'PF':>7}{'totR/年':>10}{'ブック':>9}{'差':>8}")
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    print(f"  {'全部（現行・フィルタ無し）':<34}{len(R):>6}{len(R)/yrs:>8.0f}{pf(R.values):>7.2f}"
          f"{R.sum()/yrs:>+10.1f}{r0:>9.2f}{0.0:>+8.2f}")
    m = sp <= 1.71
    legs = dict(legs0); legs["btc15m_L"] = R[m]
    print(f"  {'固定1.71%（先読みあり・元の案）':<34}{m.sum():>6}{m.sum()/yrs:>8.0f}"
          f"{pf(R.values[m]):>7.2f}{R.values[m].sum()/yrs:>+10.1f}{book_from(legs)[2]:>9.2f}"
          f"{book_from(legs)[2]-r0:>+8.2f}")
    for p in (60, 70, 80):
        keep = np.ones(len(sp), bool)
        for i in range(len(sp)):
            if i < 100:
                continue                                   # 助走: 最初の100本は全部取る
            keep[i] = sp[i] <= np.percentile(sp[:i], p)    # ★ 過去だけ
        legs = dict(legs0); legs["btc15m_L"] = R[keep]
        _, _, rb = book_from(legs)
        thr_end = np.percentile(sp[:-1], p)
        print(f"  {'拡大窓 '+str(p)+'%点（先読み無し, 最終閾値 '+f'{thr_end:.2f}%'+'）':<34}"
              f"{keep.sum():>6}{keep.sum()/yrs:>8.0f}{pf(R.values[keep]):>7.2f}"
              f"{R.values[keep].sum()/yrs:>+10.1f}{rb:>9.2f}{rb-r0:>+8.2f}"
              + ("  ★" if rb > r0 + 0.05 else ""))

    print("\n\nT3  ウォークフォワード: 前半だけで閾値を決めて、後半に当てる\n")
    half = len(R) // 2
    sp_is = sp[:half]
    print(f"  {'前半で選んだ閾値':<26}{'後半 n':>8}{'後半 PF':>9}{'後半 meanR':>12}"
          f"{'(フィルタ無しの後半)':>20}")
    base_oos = R.values[half:]
    for p in (60, 70, 80):
        thr = np.percentile(sp_is, p)
        m2 = sp[half:] <= thr
        print(f"  {str(p)+'%点 = '+f'{thr:.2f}%':<26}{m2.sum():>8}{pf(base_oos[m2]):>9.2f}"
              f"{base_oos[m2].mean():>+12.3f}"
              f"{'PF '+f'{pf(base_oos):.2f}'+' / meanR '+f'{base_oos.mean():+.3f}':>20}")

    print("\n\nT4  ブックのランダム除去null: 同じ本数を無作為に捨てたら（2000回）\n")
    for thr in (3.0, 1.71):
        m = sp <= thr
        legs = dict(legs0); legs["btc15m_L"] = R[m]
        r_obs = book_from(legs)[2]
        nulls = []
        for _ in range(NDRAW):
            k = np.sort(RNG.choice(len(R), m.sum(), replace=False))
            lg = dict(legs0); lg["btc15m_L"] = R.iloc[k]
            nulls.append(book_from(lg)[2])
        nulls = np.array(nulls)
        print(f"  損切り<=価格の {thr}%:  観測 {r_obs:.2f}  vs  ランダム除去 中央値 {np.median(nulls):.2f} "
              f"[5%点 {np.percentile(nulls,5):.2f}, 95%点 {np.percentile(nulls,95):.2f}]"
              f"   → **{100*np.mean(r_obs > nulls):.0f}%ile**")

    print("\n\nT5  年別のブック寄与（一era集中の検出）\n")
    m = sp <= 1.71
    print(f"  {'年':<7}{'捨てた本数':>10}{'捨てた分の totR':>16}{'残した分の meanR':>18}"
          f"{'全部の meanR':>15}")
    for y in sorted(set(R.index.year)):
        yy = R.index.year == y
        cut = yy & ~m
        kep = yy & m
        print(f"  {y:<7}{cut.sum():>10}{R.values[cut].sum():>+16.1f}"
              f"{R.values[kep].mean() if kep.sum() else np.nan:>+18.3f}"
              f"{R.values[yy].mean():>+15.3f}")


if __name__ == "__main__":
    main()
