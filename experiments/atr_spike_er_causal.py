"""ERゲートのしきい値から先読みを取り除く。

前段の評価は `t["er"].quantile(0.50)` ＝【全期間】の中央値をしきい値にしていた。
これは未来を使っている。実運用でできるのは次のどちらか:
  (a) 拡張窓の分位: その時点までに観測した ER の分位（最初の1年は待つ）
  (b) 固定の絶対値: 事前に決め打つ（台地であることが条件）
両方を出し、先読み版と並べて、どれだけ目減りするかを見る。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_er_gate import build, er_series          # noqa: E402
from experiments.atr_spike_barspread import spikes                  # noqa: E402

USDJPY, LOT = 150.0, 0.01
WARM = pd.Timedelta(days=365)          # 拡張窓の助走


def prep_er(d, t, win=120):
    e = er_series(d["close"], win)
    t = t.copy()
    t["er"] = e.reindex(t["time"]).to_numpy()
    t = t.dropna(subset=["er"]).sort_values("time").reset_index(drop=True)
    t["yen"] = t["pct"].to_numpy() * t["e_px"].to_numpy() * LOT * USDJPY
    # 拡張窓の分位（自分自身を含めない＝shift）
    for q in (0.33, 0.50):
        t[f"exp{int(q*100)}"] = t["er"].expanding(min_periods=20).quantile(q).shift(1)
    t["warm"] = t["time"] >= (t["time"].iloc[0] + WARM)
    return t


def rep(lab, ys, span):
    if len(ys) < 10:
        print(f"  {lab:<30} 本数不足")
        return
    eq = np.cumsum(ys)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = ys[ys > 0].sum(), -ys[ys < 0].sum()
    print(f"  {lab:<30} {len(ys):>5} {len(ys)/span:>7.0f} {(ys>0).mean()*100:>6.1f}% "
          f"{w/ls if ls>0 else np.nan:>6.2f} {ys.mean():>+8,.0f}円 "
          f"{ys.sum()/span:>+10,.0f}円 {dd:>9,.0f}円")


if __name__ == "__main__":
    dB, tB = build("btcusd")
    sB = spikes(dB)
    lead = (sB | sB.shift(1)).fillna(False)
    dE, tE = build("ethusd", lead=lead)
    TB, TE = prep_er(dB, tB), prep_er(dE, tE)

    print("=== ER(窓120)の水準（先読み版のしきい値の参考）")
    for nm, t in (("BTC", TB), ("ETH", TE)):
        print(f"  {nm}: 33%点={t['er'].quantile(0.33):.4f} 中央={t['er'].quantile(0.50):.4f} "
              f"67%点={t['er'].quantile(0.67):.4f}  範囲 {t['er'].min():.4f}〜{t['er'].max():.4f}")

    # 助走後の期間だけで、3つの流儀を並べる
    print("\n=== 助走1年を除いた期間で比較（0.01ロット固定）")
    hdr = (f"  {'構成':<30} {'N':>5} {'年本数':>7} {'勝率':>7} {'PF':>6} "
           f"{'1本':>9} {'年間':>11} {'maxDD':>10}")
    for nm, t in (("BTC", TB), ("ETH", TE)):
        w = t[t["warm"]]
        span = (w["time"].max() - w["time"].min()).days / 365.25
        print(f"\n  -- {nm}（助走後 {span:.1f}年）")
        print(hdr)
        rep("全部取る", w["yen"].to_numpy(), span)
        rep("【先読み】全期間中央値で切る", w.loc[w["er"] >= t["er"].quantile(0.50), "yen"].to_numpy(), span)
        rep("拡張窓の中央値で切る", w.loc[w["er"] >= w["exp50"], "yen"].to_numpy(), span)
        rep("拡張窓の33%点で切る", w.loc[w["er"] >= w["exp33"], "yen"].to_numpy(), span)
        for thr in (0.04, 0.05, 0.06, 0.08, 0.10):
            rep(f"固定しきい ER>={thr:.2f}", w.loc[w["er"] >= thr, "yen"].to_numpy(), span)

    print("\n=== 合成（BTC＋ETH・助走後）")
    parts = {}
    for lab in ("全部取る", "拡張窓の中央値", "固定 ER>=0.06", "固定 ER>=0.08"):
        acc = []
        for nm, t in (("BTC", TB), ("ETH", TE)):
            w = t[t["warm"]]
            if lab == "全部取る":
                g = w
            elif lab == "拡張窓の中央値":
                g = w[w["er"] >= w["exp50"]]
            elif lab == "固定 ER>=0.06":
                g = w[w["er"] >= 0.06]
            else:
                g = w[w["er"] >= 0.08]
            acc.append(pd.DataFrame({"time": g["time"].values, "yen": g["yen"].to_numpy()}))
        parts[lab] = pd.concat(acc, ignore_index=True).sort_values("time")
    span = (parts["全部取る"]["time"].max() - parts["全部取る"]["time"].min()).days / 365.25
    print(hdr)
    for lab, P in parts.items():
        rep(lab, P["yen"].to_numpy(), span)

    print("\n  -- 年別（合成・円）")
    for lab, P in parts.items():
        yy = P.assign(y=P["time"].dt.year).groupby("y")["yen"].agg(["sum", "count"])
        print(f"    {lab:<16}" + " ".join(f"{y}:{r['sum']:+,.0f}/{int(r['count'])}"
                                          for y, r in yy.iterrows()))

    assert len(TB) > 150, len(TB)
    print(f"\nOK: BTC N={len(TB)} ETH N={len(TE)}")
