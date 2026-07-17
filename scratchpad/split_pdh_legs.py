"""Split btc15m_L into the two legs the autopsy found, and judge both questions the user asked:

  Q1  Can the high-PF half be traded STANDALONE?  (break ABOVE the previous day's high, full size:
      n=208, PF 2.67, meanR +1.23R.)  Give it the numbers a person needs to actually run it alone:
      trades/yr, win%, longest losing streak, maxDD and CAGR at 1% risk, per-year.
  Q2  Does splitting HELP THE BOOK?  Today the leg is one line with one weight and a hard-coded 0.5
      size for the weak half. As two legs, inv-vol sizes each by its own sigma -- which is what
      inv-vol is FOR.

Q2 has a trap that must be closed before any number is believed: splitting one leg into two hands
the allocator an extra degree of freedom. A book can improve from that alone, with the PDH rule
contributing nothing. So the null is not "no split" -- it is a RANDOM split of the same two sizes
(208 / 534), re-weighted the same way. If the random split lifts the book as much, PDH is doing no
work and this is just allocation overfitting.
Also pin the weights (today's 3 retractions were all inv-vol silently buying leverage).
Run: .venv/bin/python scratchpad/split_pdh_legs.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from wide_stop_stress import raw_legs, SIX

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
NDRAW = 2000


def w_inv(legs, basket, budget=0.03):
    sig = pd.Series({k: legs[k].std() for k in basket})
    w = 1.0 / sig
    return w / w.sum() * budget


def book(legs, basket, w=None):
    ww = w_inv(legs, basket) if w is None else w
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    s = pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * ww[k],
                             index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                   for k in basket]).sort_index()
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9), s


def streak(v):
    b = c = 0
    for x in v:
        c = c + 1 if x <= 0 else 0
        b = max(b, c)
    return b


def solo(tag, s, risk=0.01):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    pf = s[s > 0].sum() / abs(s[s <= 0].sum())
    eq = np.cumprod(1 + risk * s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    h = len(s) // 2
    print(f"  {tag:<32}{len(s):>5}{len(s)/yrs:>7.0f}{100*(s>0).mean():>7.1f}%{pf:>7.2f}"
          f"{s.mean():>+9.3f}{s[:h].mean():>+8.3f}{s[h:].mean():>+8.3f}"
          f"{streak(s.values):>7}{cagr:>8.1f}%{dd:>8.1f}%{cagr/max(dd,1e-9):>9.2f}")


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    c0, d0, r0, _ = book(legs0, SIX)
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    above = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    Rraw = t["R"].values - 15.0 / t["risk"].values      # コスト込み。サイズ倍率は掛けない（重みが決める）
    idx = pd.DatetimeIndex(t["time"])
    A = pd.Series(Rraw[above], index=idx[above])        # 前日高値の「上」でブレイク
    B = pd.Series(Rraw[~above], index=idx[~above])      # 前日高値の「中」でブレイク

    print("Q1  PF が高い組は、単独で回せるか（賭け率 1%/トレード）\n")
    print(f"  {'':<32}{'n':>5}{'本/年':>7}{'勝率':>8}{'PF':>7}{'meanR':>9}{'IS':>8}{'OOS':>8}"
          f"{'最長連敗':>7}{'CAGR':>8}{'maxDD':>8}{'CAGR/DD':>9}")
    solo("btc15m_L 全部（現行・混ぜたまま）", legs0["btc15m_L"])
    solo("A: 前日高値の【上】でブレイク", A)
    solo("B: 前日高値の【中】でブレイク", B)
    print(f"\n  参考: 現行ブック（6レッグ）の CAGR {c0:+.1f}% / maxDD {d0:.2f}% / CAGR/DD {r0:.2f}")

    print("\n\nQ2  分割はブックを改善するか。**帰無 = 同じ大きさのランダム分割**\n")
    SEVEN = [k for k in SIX if k != "btc15m_L"] + ["btc15m_A", "btc15m_B"]
    legs7 = {k: v for k, v in legs0.items() if k != "btc15m_L"}
    legs7["btc15m_A"], legs7["btc15m_B"] = A, B
    c7, d7, r7, _ = book(legs7, SEVEN)
    w7 = w_inv(legs7, SEVEN)
    print(f"  {'構成':<30}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>10}{'差':>8}")
    print(f"  {'現行（6レッグ・混ぜたまま）':<30}{c0:>8.1f}%{d0:>7.2f}%{r0:>10.2f}{0.0:>+8.2f}")
    print(f"  {'分割（7レッグ・PDHで割る）':<30}{c7:>8.1f}%{d7:>7.2f}%{r7:>10.2f}{r7-r0:>+8.2f}"
          + ("  ★" if r7 > r0 + 0.05 else ""))
    print(f"\n  分割後の重み: A(上)={100*w7['btc15m_A']:.3f}%  B(中)={100*w7['btc15m_B']:.3f}%  "
          f"合計={100*(w7['btc15m_A']+w7['btc15m_B']):.3f}%   "
          f"（現行の btc15m_L 単独の重み = {100*w_inv(legs0, SIX)['btc15m_L']:.3f}%）")
    print(f"  σ(R):  A={A.std():.3f}   B={B.std():.3f}   （混ぜたまま={legs0['btc15m_L'].std():.3f}）")

    print("\n  ★ 帰無: **ランダムに** 208本 / 534本 に割って、同じように別々の重みを与える（2000回）")
    Rall = pd.Series(Rraw, index=idx)
    nulls = []
    for _ in range(NDRAW):
        k = RNG.permutation(len(Rall))
        a = Rall.iloc[np.sort(k[:above.sum()])]
        b = Rall.iloc[np.sort(k[above.sum():])]
        lg = {kk: v for kk, v in legs0.items() if kk != "btc15m_L"}
        lg["btc15m_A"], lg["btc15m_B"] = a, b
        nulls.append(book(lg, SEVEN)[2])
    nulls = np.array(nulls)
    print(f"     ランダム分割: 中央値 {np.median(nulls):.2f}  [5%点 {np.percentile(nulls,5):.2f}, "
          f"95%点 {np.percentile(nulls,95):.2f}]")
    print(f"     **PDH 分割 {r7:.2f} は、この分布の {100*np.mean(r7 > nulls):.0f} パーセンタイル**")
    print(f"     → ランダムに割っても中央値が {np.median(nulls):.2f} なら、**分割そのものが "
          f"{np.median(nulls)-r0:+.2f} を生んでいる**（＝重みの自由度が1つ増えただけ）")


if __name__ == "__main__":
    main()
