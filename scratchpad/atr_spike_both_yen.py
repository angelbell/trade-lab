"""「両方採用」のサイズ表を Vantage の実額（円）で出す。

Binance 8.5年で確認された最良: 実体>2.0ATR かつ ER高 → 0.03 ／ それ以外(実体>1.5ATR) → 0.01
（同じ maxDD に揃えた通算R で +12%。実体だけ −8%・ER だけ +1% なので、積にしか情報がない）
ここでは実際に建てる Vantage の板・実スプレッド課金で、円と maxDD を出す。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_frequency import load, leg, gate            # noqa: E402
from scratchpad.atr_spike_barspread import spread_series, wilder_atr  # noqa: E402


def stat(lots, T, span):
    y = T["yen"].to_numpy() * (lots / 0.01)
    y = y[lots > 0]
    if len(y) < 10:
        return None
    eq = np.cumsum(y)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = y[y > 0].sum(), -y[y < 0].sum()
    return dict(n=len(y), ny=len(y) / span, win=(y > 0).mean() * 100,
                pf=w / ls if ls > 0 else np.nan, mean=y.mean(),
                ann=y.sum() / span, dd=dd)


if __name__ == "__main__":
    b1 = load("btcusd", "h1")
    sp1 = spread_series("BTCUSD|h1")
    T = gate(leg(b1, 1.5, 20, sp1), b1, 120)
    ap = wilder_atr(b1).shift(1)
    body = ((b1["close"] - b1["open"]) / ap)
    T["body"] = body.reindex(T["time"]).to_numpy()
    T = T.dropna(subset=["body"]).reset_index(drop=True)
    span = (T["time"].max() - T["time"].min()).days / 365.25
    on = (T["er"] >= T["x50"]).to_numpy()
    strong = (T["body"] >= 2.0).to_numpy()
    hit = strong & on

    print(f"=== BTC 1時間・Vantage 2022-（助走後 {span:.1f}年・N={len(T)}・"
          f"厚く張る玉 {hit.mean()*100:.0f}%＝年{hit.sum()/span:.0f}本）")
    print(f"  {'構成':<36} {'年本数':>6} {'勝率':>7} {'PF':>6} {'1本':>8} "
          f"{'年間':>11} {'maxDD':>10} {'同DD揃え':>11}")
    base = stat(np.full(len(T), 0.01), T, span)
    CASES = [("全部 0.01", np.full(len(T), 0.01)),
             ("全部 0.02（対照）", np.full(len(T), 0.02)),
             ("実体>2.0かつER高→0.02 / 他0.01", np.where(hit, 0.02, 0.01)),
             ("【両方採用】同→0.03 / 他0.01", np.where(hit, 0.03, 0.01)),
             ("同→0.05 / 他0.01", np.where(hit, 0.05, 0.01)),
             ("同→0.03 / 他0.02", np.where(hit, 0.03, 0.02))]
    for lab, lots in CASES:
        s = stat(lots, T, span)
        if s is None:
            continue
        sc = s["ann"] * (base["dd"] / s["dd"]) if s["dd"] > 0 else np.nan
        print(f"  {lab:<36} {s['ny']:>6.0f} {s['win']:>6.1f}% {s['pf']:>6.2f} "
              f"{s['mean']:>+7,.0f}円 {s['ann']:>+10,.0f}円 {s['dd']:>9,.0f}円 "
              f"{sc:>+10,.0f}円")

    print("\n  -- 年別の円（【両方採用】0.03/0.01）")
    lots = np.where(hit, 0.03, 0.01)
    T2 = T.copy()
    T2["y2"] = T2["yen"].to_numpy() * (lots / 0.01)
    for lab, col in (("全部0.01", "yen"), ("両方採用", "y2")):
        yy = T2.groupby(T2["time"].dt.year)[col].agg(["sum", "count"])
        print(f"    {lab:<10}" + " ".join(f"{y}:{r['sum']:+,.0f}/{int(r['count'])}"
                                          for y, r in yy.iterrows()))

    print(f"\n  -- 1トレードの最大損失（損切り幅×ロット）の中央値")
    risk = (T["rf"] if "rf" in T else None)
    if risk is None:
        # rf は leg() が持っていないので損切り幅を再計算せず、yen とRから逆算
        pass
    for lot, lab in ((0.01, "0.01ロット"), (0.03, "0.03ロット")):
        med = np.median(np.abs(T.loc[T["yen"] < 0, "yen"])) * (lot / 0.01)
        print(f"     {lab}: 負けトレードの中央値 {med:,.0f}円")

    assert len(T) > 150, len(T)
    print(f"\nOK: N={len(T)}")
