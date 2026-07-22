"""1時間足 vs 15分足 × ERゲートあり/なし を、0.01ロット固定の実額で並べる。

前提（このセッションで確定したもの）:
  k は 1.25-1.75 が台地。k=2.0 は既に3割低い → 主表は k=1.5、比較用に k=2.0 も出す
  保有上限は【時間】の変数 → 1時間足 fwd20 / 15分足 fwd80（ともに壁時計20時間）
  コストは MT5 のバー別実スプレッドを約定バーで課金
  ERゲートのしきい値は拡張窓の中央値（先読みなし・助走1年）

ERの窓は2通り出す:
  本数そろえ  1時間120本 / 15分120本（=30時間）
  壁時計そろえ 1時間120本 / 15分480本（=120時間）
fwd が時間の変数だったので、ER も時間の変数である可能性が高い。両方見る。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_frequency import load, leg, gate            # noqa: E402
from scratchpad.atr_spike_barspread import spread_series              # noqa: E402

HDR = (f"  {'構成':<38} {'年本数':>6} {'勝率':>7} {'PF':>6} {'1本':>8} "
       f"{'年間':>11} {'maxDD':>10} {'年間/DD':>8}")


def show(lab, y, span):
    if len(y) < 10:
        print(f"  {lab:<38} 本数不足")
        return None
    eq = np.cumsum(y)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = y[y > 0].sum(), -y[y < 0].sum()
    ann = y.sum() / span
    print(f"  {lab:<38} {len(y)/span:>6.0f} {(y>0).mean()*100:>6.1f}% "
          f"{w/ls if ls>0 else np.nan:>6.2f} {y.mean():>+7,.0f}円 "
          f"{ann:>+10,.0f}円 {dd:>9,.0f}円 {ann/dd if dd>0 else np.nan:>8.2f}")
    return dict(n=len(y), ann=ann, dd=dd)


if __name__ == "__main__":
    b1, b15 = load("btcusd", "h1"), load("btcusd", "m15")
    sp1, sp15 = spread_series("BTCUSD|h1"), spread_series("BTCUSD|m15")

    keep = {}
    for k in (1.5, 2.0):
        print(f"\n{'='*112}\n===== k = {k}   BTC・0.01ロット固定・実スプレッド課金")
        print(HDR)
        # --- 1時間足
        t1 = leg(b1, k, 20, sp1)
        T1 = gate(t1, b1, 120)
        s1 = (T1["time"].max() - T1["time"].min()).days / 365.25
        keep[("1h", k, "off")] = (T1, s1, None)
        show("1時間 ゲート無し", T1["yen"].to_numpy(), s1)
        m = (T1["er"] >= T1["x50"]).to_numpy()
        keep[("1h", k, "on")] = (T1, s1, m)
        show("1時間 ERゲート有り（窓120本＝120時間）", T1.loc[m, "yen"].to_numpy(), s1)

        # --- 15分足（ERの窓を2通り）
        t15 = leg(b15, k, 80, sp15)
        for win, lab in ((120, "窓120本＝30時間"), (480, "窓480本＝120時間")):
            T15 = gate(t15, b15, win)
            s15 = (T15["time"].max() - T15["time"].min()).days / 365.25
            if win == 120:
                keep[("15m", k, "off")] = (T15, s15, None)
                show("15分 ゲート無し", T15["yen"].to_numpy(), s15)
            m15 = (T15["er"] >= T15["x50"]).to_numpy()
            keep[("15m", k, f"on{win}")] = (T15, s15, m15)
            show(f"15分 ERゲート有り（{lab}）", T15.loc[m15, "yen"].to_numpy(), s15)

    print(f"\n{'='*112}\n===== 併走（1時間＋15分・k=1.5）")
    print(HDR)
    for lab, g1key, g15key in (("両方 ゲート無し", ("1h", 1.5, "off"), ("15m", 1.5, "off")),
                               ("両方 ERゲート有り（15分は窓480）",
                                ("1h", 1.5, "on"), ("15m", 1.5, "on480")),
                               ("1時間だけゲート・15分は素",
                                ("1h", 1.5, "on"), ("15m", 1.5, "off"))):
        parts, span = [], 0
        for key in (g1key, g15key):
            T, s, m = keep[key]
            g = T if m is None else T[m]
            parts.append(pd.DataFrame({"time": g["time"].values, "yen": g["yen"].to_numpy()}))
            span = max(span, s)
        P = pd.concat(parts, ignore_index=True).sort_values("time")
        show(lab, P["yen"].to_numpy(), span)

    print(f"\n{'='*112}\n===== 年別の円（k=1.5）")
    for lab, key in (("1時間 ゲート無し", ("1h", 1.5, "off")),
                     ("1時間 ゲート有り", ("1h", 1.5, "on")),
                     ("15分 ゲート無し", ("15m", 1.5, "off")),
                     ("15分 ゲート有り(窓480)", ("15m", 1.5, "on480"))):
        T, s, m = keep[key]
        g = T if m is None else T[m]
        yy = g.groupby(g["time"].dt.year)["yen"].agg(["sum", "count"])
        print(f"  {lab:<22}" + " ".join(f"{y}:{r['sum']:+,.0f}/{int(r['count'])}"
                                        for y, r in yy.iterrows()))

    assert len(keep) >= 8, len(keep)
    print(f"\nOK: {len(keep)} 構成")
