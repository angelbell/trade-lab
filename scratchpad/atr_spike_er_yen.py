"""ERゲートを固定ロットの実額で評価する。比が上がっても総額が減るなら、固定ロットでは損。

ERゲートは間引き帰無もブロック・ブートストラップも通った（比 9.88→16.87 / 10.03→12.58）。
だが本数がほぼ半減する。**0.01ロット固定では『比』は使えない**（賭け率を上げられないので、
滑らかさを利益に変換できない）。判断すべきは 年間の円と maxDD の円。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_er_gate import build, er_series          # noqa: E402
from scratchpad.atr_spike_barspread import spikes                  # noqa: E402

USDJPY, LOT = 150.0, 0.01


def yen(t):
    return t["pct"].to_numpy() * t["e_px"].to_numpy() * LOT * USDJPY


if __name__ == "__main__":
    dB, tB = build("btcusd")
    sB = spikes(dB)
    lead = (sB | sB.shift(1)).fillna(False)
    dE, tE = build("ethusd", lead=lead)
    span = (tB["time"].max() - tB["time"].min()).days / 365.25

    rows = []
    for nm, d, t in (("BTC", dB, tB), ("ETH", dE, tE)):
        e = er_series(d["close"], 120)
        t = t.copy()
        t["er"] = e.reindex(t["time"]).to_numpy()
        t = t.dropna(subset=["er"])
        t["yen"] = yen(t)
        rows.append((nm, t))

    print(f"=== 0.01ロット固定・実額（{span:.1f}年・1ドル{USDJPY:.0f}円）")
    print(f"  {'構成':<28} {'N':>5} {'年本数':>7} {'勝率':>7} {'PF':>6} "
          f"{'1本':>9} {'年間':>11} {'maxDD':>10}")

    def rep(lab, ys):
        eq = np.cumsum(ys)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        w, ls = ys[ys > 0].sum(), -ys[ys < 0].sum()
        print(f"  {lab:<28} {len(ys):>5} {len(ys)/span:>7.0f} {(ys>0).mean()*100:>6.1f}% "
              f"{w/ls if ls>0 else np.nan:>6.2f} {ys.mean():>+8,.0f}円 "
              f"{ys.sum()/span:>+10,.0f}円 {dd:>9,.0f}円")

    for nm, t in rows:
        thr50 = t["er"].quantile(0.50)
        thr67 = t["er"].quantile(0.33)
        rep(f"{nm} 全部取る", t["yen"].to_numpy())
        rep(f"{nm} ER上位67%のみ", t.loc[t["er"] >= thr67, "yen"].to_numpy())
        rep(f"{nm} ER上位50%のみ", t.loc[t["er"] >= thr50, "yen"].to_numpy())

    print("\n  -- 合成（BTC＋ETH）")
    for lab, q in (("全部取る", None), ("ER上位67%のみ", 0.33), ("ER上位50%のみ", 0.50)):
        parts = []
        for nm, t in rows:
            g = t if q is None else t[t["er"] >= t["er"].quantile(q)]
            parts.append(pd.DataFrame({"time": g["time"].values, "yen": g["yen"].to_numpy()}))
        P = pd.concat(parts, ignore_index=True).sort_values("time")
        rep(lab, P["yen"].to_numpy())

    print("\n  -- 年別（合成・円）")
    for lab, q in (("全部取る", None), ("ER上位50%", 0.50)):
        parts = []
        for nm, t in rows:
            g = t if q is None else t[t["er"] >= t["er"].quantile(q)]
            parts.append(pd.DataFrame({"y": g["time"].dt.year.values, "yen": g["yen"].to_numpy()}))
        P = pd.concat(parts, ignore_index=True)
        yy = P.groupby("y")["yen"].agg(["sum", "count"])
        print(f"    {lab:<12}" + " ".join(f"{y}:{r['sum']:+,.0f}円/{int(r['count'])}本"
                                          for y, r in yy.iterrows()))

    assert len(rows) == 2
    print("\nOK")
