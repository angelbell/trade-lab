"""The naive split broke the book because inv-vol normalises PER LEG: split one leg into two and its
family's share of the 3% budget roughly DOUBLES (0.475% -> 0.647%). A random split costs -1.41 for
exactly that reason -- nothing to do with PDH.

Fix the accounting, then ask the question again:
  keep the BTC-15m-long family's total weight at what it has today (0.475%), and split it INTERNALLY.
Three internal splits, so the allocator's pathology is visible rather than hidden:
  S1  inv-vol inside the family (1/sigma)          -- what the book's own rule would do
  S2  today's de-facto split (full size / half size) -- the current book, restated
  S3  all of it on A, none on B                     -- "trade only the good half"
and a 4th arm the user actually asked about:
  S4  A alone as a STANDALONE strategy, next to every other leg standalone, so "which single leg
      would I trade first" has an answer.
Random-split null again, on the fixed accounting.
Run: .venv/bin/python experiments/split_pdh_fixed.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from wide_stop_stress import raw_legs, SIX

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
NDRAW = 1000


def w_inv(legs, basket, budget=0.03):
    sig = pd.Series({k: legs[k].std() for k in basket})
    return (1.0 / sig) / (1.0 / sig).sum() * budget


def curve(parts):
    """parts = list of (Series, weight). Trade-resolution account-return series."""
    s = pd.concat([pd.Series(x.values * w, index=x.index) for x, w in parts]).sort_index()
    return s.sort_index()


def cdd(s):
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def streak(v):
    b = c = 0
    for x in v:
        c = c + 1 if x <= 0 else 0
        b = max(b, c)
    return b


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    w0 = w_inv(legs0, SIX)
    WL = w0["btc15m_L"]                                   # 一家の総重量（これを固定する）
    st = max(legs0[k].index.min() for k in SIX)
    en = min(legs0[k].index.max() for k in SIX)
    others = [(legs0[k][(legs0[k].index >= st) & (legs0[k].index <= en)], w0[k])
              for k in SIX if k != "btc15m_L"]
    c0, d0, r0 = cdd(curve(others + [(legs0["btc15m_L"][(legs0["btc15m_L"].index >= st) &
                                                        (legs0["btc15m_L"].index <= en)], WL)]))

    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    above = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    Rraw = t["R"].values - 15.0 / t["risk"].values
    idx = pd.DatetimeIndex(t["time"])
    A = pd.Series(Rraw[above], index=idx[above]); A = A[(A.index >= st) & (A.index <= en)]
    B = pd.Series(Rraw[~above], index=idx[~above]); B = B[(B.index >= st) & (B.index <= en)]

    print(f"一家（BTC15分ロング）の総重量を **{100*WL:.3f}% に固定**して、内部の配分だけを変える。")
    print(f"現行ブック = {r0:.2f}   σ(A)={A.std():.3f}  σ(B)={B.std():.3f}\n")
    print(f"  {'内部の配分':<40}{'A の重み':>10}{'B の重み':>10}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>10}{'差':>8}")
    ia = (1 / A.std()) / (1 / A.std() + 1 / B.std())
    arms = {
        "S2. 現行（フル / 半分）＝いまのブック": (1.0, 0.5),
        "S1. 一家の中で inv-vol（1/σ）": (ia / max(ia, 1 - ia), (1 - ia) / max(ia, 1 - ia)),
        "S3. A だけ（B は建てない）": (1.0, 0.0),
        "S4. A に厚く（フル / 0.25）": (1.0, 0.25),
        "S5. B に厚く（0.5 / 1.0）＝逆向きダミー": (0.5, 1.0),
    }
    for tag, (fa, fb) in arms.items():
        # 一家の総重量を WL に保つ: 露出の加重平均が現行(=フル/半分の混合)と同じスケールになるよう正規化
        base_scale = (len(A) * 1.0 + len(B) * 0.5) / (len(A) + len(B))
        sc = base_scale / ((len(A) * fa + len(B) * fb) / (len(A) + len(B))) if (fa + fb) > 0 else 0
        wa, wb = WL * fa * sc, WL * fb * sc
        c, d, r = cdd(curve(others + [(A, wa), (B, wb)]))
        print(f"  {tag:<40}{100*wa:>9.3f}%{100*wb:>9.3f}%{c:>8.1f}%{d:>7.2f}%{r:>10.2f}{r-r0:>+8.2f}"
              + ("  ★" if r > r0 + 0.05 else ""))

    print("\n  帰無: ランダムに 229/534 に割って、S1 と同じ内部 inv-vol を当てる（1000回）")
    Rall = pd.Series(Rraw, index=idx); Rall = Rall[(Rall.index >= st) & (Rall.index <= en)]
    nulls = []
    for _ in range(NDRAW):
        k = RNG.permutation(len(Rall))
        a = Rall.iloc[np.sort(k[:len(A)])]; b = Rall.iloc[np.sort(k[len(A):])]
        i2 = (1 / a.std()) / (1 / a.std() + 1 / b.std())
        fa, fb = i2 / max(i2, 1 - i2), (1 - i2) / max(i2, 1 - i2)
        base_scale = (len(A) * 1.0 + len(B) * 0.5) / (len(A) + len(B))
        sc = base_scale / ((len(a) * fa + len(b) * fb) / (len(a) + len(b)))
        nulls.append(cdd(curve(others + [(a, WL * fa * sc), (b, WL * fb * sc)]))[2])
    nulls = np.array(nulls)
    ia_s = (1 / A.std()) / (1 / A.std() + 1 / B.std())
    fa, fb = ia_s / max(ia_s, 1 - ia_s), (1 - ia_s) / max(ia_s, 1 - ia_s)
    base_scale = (len(A) * 1.0 + len(B) * 0.5) / (len(A) + len(B))
    sc = base_scale / ((len(A) * fa + len(B) * fb) / (len(A) + len(B)))
    obs = cdd(curve(others + [(A, WL * fa * sc), (B, WL * fb * sc)]))[2]
    print(f"    ランダム分割(S1と同じ配分則): 中央値 {np.median(nulls):.2f} "
          f"[5% {np.percentile(nulls,5):.2f}, 95% {np.percentile(nulls,95):.2f}]")
    print(f"    **PDH 分割 = {obs:.2f} → {100*np.mean(obs > nulls):.0f} パーセンタイル**")

    print("\n\n単独運用したときの、全レッグの比較（賭け率 1%/トレード）")
    print(f"  {'':<30}{'n':>5}{'本/年':>7}{'勝率':>7}{'PF':>7}{'meanR':>9}"
          f"{'最長連敗':>7}{'CAGR':>8}{'maxDD':>8}{'CAGR/DD':>9}")
    cand = {"★ A: 前日高値の【上】ブレイク": A, "B: 前日高値の【中】ブレイク": B}
    cand.update({k: legs0[k] for k in SIX})
    rows = []
    for k, s in cand.items():
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        pf = s[s > 0].sum() / abs(s[s <= 0].sum())
        eq = np.cumprod(1 + 0.01 * s.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        cagr = (eq[-1] ** (1 / yrs) - 1) * 100
        rows.append((cagr / max(dd, 1e-9), k, len(s), len(s) / yrs, 100 * (s > 0).mean(), pf,
                     s.mean(), streak(s.values), cagr, dd))
    for r, k, n, py, wn, pf, mr, sk, cg, dd in sorted(rows, reverse=True):
        print(f"  {k:<30}{n:>5}{py:>7.0f}{wn:>6.1f}%{pf:>7.2f}{mr:>+9.3f}{sk:>7}{cg:>7.1f}%"
              f"{dd:>7.1f}%{r:>9.2f}")


if __name__ == "__main__":
    main()
