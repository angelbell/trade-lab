"""ERが効くときだけ 0.02 ロット、それ以外は 0.01 ロット。刻みで表現できる2段サイズ。

🚨 対照が要る（法則7.5）: 「良い部分集合で倍にする」は本質的にレバレッジを買う操作なので、
   【単に全部を0.02にしただけ】と区別しないと意味がない。
   全部0.01 と 全部0.02 は円もDDもちょうど2倍になる＝原点を通る直線。
   その直線より上（年間/DD が高い）に来て初めて、ERが情報を足したことになる。

並べる構成（BTC 1時間・実スプレッド課金・Vantage 2022-・助走1年）:
   全部0.01 / 全部0.02 / ER高0.02・ER低0.01 / ER高0.02・ER低0（ゲート×2倍）
   / ER高0.01・ER低0（素のゲート） / ER高0.03・ER低0.01
k は 1.5 と 2.0 の両方で見る（ゲートの価値が k で変わるため）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_frequency import load, leg, gate            # noqa: E402
from scratchpad.atr_spike_barspread import spread_series              # noqa: E402

USDJPY = 150.0


def stat(lots, T, span):
    """lots: トレードごとのロット（0 は見送り）。yen は 0.01 ロット基準なので 100倍して掛ける。"""
    y = T["yen"].to_numpy() * (lots / 0.01)
    y = y[lots > 0]
    if len(y) < 10:
        return None
    eq = np.cumsum(y)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = y[y > 0].sum(), -y[y < 0].sum()
    ann = y.sum() / span
    return dict(n=len(y), ny=len(y) / span, win=(y > 0).mean() * 100,
                pf=w / ls if ls > 0 else np.nan, ann=ann, dd=dd,
                ratio=ann / dd if dd > 0 else np.nan, mean=y.mean())


if __name__ == "__main__":
    b1 = load("btcusd", "h1")
    sp1 = spread_series("BTCUSD|h1")
    for k in (1.5, 2.0):
        T = gate(leg(b1, k, 20, sp1), b1, 120)
        span = (T["time"].max() - T["time"].min()).days / 365.25
        on = (T["er"] >= T["x50"]).to_numpy()
        print(f"\n===== BTC 1時間 k={k}（助走後 {span:.1f}年・ER高の割合 {on.mean()*100:.0f}%）")
        print(f"  {'構成':<28} {'年本数':>6} {'勝率':>7} {'PF':>6} {'1本':>8} "
              f"{'年間':>11} {'maxDD':>10} {'年間/DD':>8}")
        CASES = [
            ("全部 0.01", np.where(True, 0.01, 0.0) * np.ones(len(T))),
            ("全部 0.02", np.full(len(T), 0.02)),
            ("ER高0.02 / ER低0.01", np.where(on, 0.02, 0.01)),
            ("ER高0.03 / ER低0.01", np.where(on, 0.03, 0.01)),
            ("ER高0.02 / ER低0（ゲート×2）", np.where(on, 0.02, 0.0)),
            ("ER高0.01 / ER低0（素のゲート）", np.where(on, 0.01, 0.0)),
            ("ER高0.03 / ER低0", np.where(on, 0.03, 0.0)),
        ]
        base = None
        for lab, lots in CASES:
            s = stat(lots, T, span)
            if s is None:
                continue
            if lab == "全部 0.01":
                base = s
            mark = ""
            if base and lab != "全部 0.01":
                mark = " ★直線超え" if s["ratio"] > base["ratio"] + 1e-9 else ""
            print(f"  {lab:<28} {s['ny']:>6.0f} {s['win']:>6.1f}% {s['pf']:>6.2f} "
                  f"{s['mean']:>+7,.0f}円 {s['ann']:>+10,.0f}円 {s['dd']:>9,.0f}円 "
                  f"{s['ratio']:>8.2f}{mark}")

        print(f"  （全部0.01 の年間/DD = {base['ratio']:.2f} が基準線。"
              f"単なる倍がけでは動かないので、これを超えたものだけが情報を足している）")

        print("  -- 同じ maxDD にそろえた比較（各構成を全部0.01のDDに合わせて縮尺）")
        for lab, lots in CASES:
            s = stat(lots, T, span)
            if s is None or s["dd"] <= 0:
                continue
            scale = base["dd"] / s["dd"]
            print(f"     {lab:<26} DDを{base['dd']:,.0f}円に合わせると 年間 "
                  f"{s['ann']*scale:>+10,.0f}円（実効ロット×{scale:.2f}）")

    print("\n（判定: 『同じDDに揃えたときの年間の円』が全部0.01を上回れば、ERは本当に情報を足している）")
