"""Pine で実装できる形（長い移動窓の中央値）が、検証した形（拡張窓の中央値）を再現するか。

拡張窓＝その時点までに観測した ER 全部の中央値。Pine ではバーごとに累積配列を持つことになり重すぎる。
実装できるのは `ta.percentile_linear_interpolation(er, N, 50)` ＝ 直近N本の中央値。
∴ N を振って、拡張窓版と同じ判定（厚い/薄い）になるか、成績が保たれるかを確かめる。
一致率と、同じ maxDD に揃えた通算R の両方を見る。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_er_binance import load, build, attach, COST   # noqa: E402
from scratchpad.atr_spike_er_gate import er_series                      # noqa: E402
from scratchpad.atr_spike_barspread import wilder_atr                   # noqa: E402
import scratchpad.atr_spike_er_binance as EB                            # noqa: E402


def ratio(r):
    if len(r) < 5:
        return np.nan
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return r.sum() / dd if dd > 0 else np.nan


if __name__ == "__main__":
    B = load("btcusdt")
    e = er_series(B["close"], 120)
    orig = EB.K
    EB.K = 1.5
    T = attach(build(B, "long", COST["btcusdt"]), e)
    EB.K = orig
    ap = wilder_atr(B).shift(1)
    T["body"] = ((B["close"] - B["open"]) / ap).reindex(T["time"]).to_numpy()
    T = T.dropna(subset=["body"]).reset_index(drop=True)
    r = T["R_net"].to_numpy()
    strong = (T["body"] >= 2.0).to_numpy()
    span = (T["time"].max() - T["time"].min()).days / 365.25

    # 基準＝拡張窓（トレード列の上での拡張窓。検証で使ったもの）
    ref = (T["er"] >= T["x50"]).to_numpy()
    base = ratio(r)
    print(f"=== BTC 1時間・Binance（{span:.1f}年・N={len(T)}）  段は 0.02/0.01")
    print(f"  基準: 全部0.01 の 通算/DD = {base:.2f}")
    w = np.where(strong & ref, 2.0, 1.0)
    print(f"  拡張窓（検証で使った形）: 厚い玉 {(strong&ref).mean()*100:.0f}% "
          f"通算/DD={ratio(r*w):.2f}（対 基準 {(ratio(r*w)/base-1)*100:+.0f}%）")

    # 実装できる形＝【足の系列】の上での移動窓中央値
    print(f"\n  {'移動窓(本)':>10} {'厚い玉%':>8} {'拡張窓との一致率':>14} {'通算/DD':>9} {'対 基準':>9}")
    er_bar = e                                    # 足ごとの ER（確定値）
    for N in (250, 500, 1000, 2000, 4000, 8000):
        med = er_bar.rolling(N, min_periods=N // 2).median()
        hi = (er_bar >= med).reindex(T["time"]).fillna(False).to_numpy()
        agree = (hi == ref).mean() * 100
        w2 = np.where(strong & hi, 2.0, 1.0)
        print(f"  {N:>10} {(strong&hi).mean()*100:>7.0f}% {agree:>13.0f}% "
              f"{ratio(r*w2):>9.2f} {(ratio(r*w2)/base-1)*100:>+8.0f}%")

    print("\n  （足の系列で測る中央値はトレード列の中央値と母集団が違うので、"
          "\n   一致率100%にはならない。見るべきは 通算/DD が保たれるか）")

    # Vantage でも同じことを確認
    from scratchpad.atr_spike_frequency import load as vload, leg as vleg   # noqa: E402
    from scratchpad.atr_spike_barspread import spread_series               # noqa: E402
    b1 = vload("btcusd", "h1")
    t = vleg(b1, 1.5, 20, spread_series("BTCUSD|h1"))
    ev = er_series(b1["close"], 120)
    t["er"] = ev.reindex(t["time"]).to_numpy()
    t["body"] = ((b1["close"] - b1["open"]) / wilder_atr(b1).shift(1)).reindex(t["time"]).to_numpy()
    t = t.dropna(subset=["er", "body"]).sort_values("time").reset_index(drop=True)
    t["x50"] = t["er"].expanding(min_periods=20).quantile(0.50).shift(1)
    t = t[t["time"] >= t["time"].iloc[0] + pd.Timedelta(days=365)]
    spanv = (t["time"].max() - t["time"].min()).days / 365.25
    st_v = (t["body"] >= 2.0).to_numpy()
    ref_v = (t["er"] >= t["x50"]).to_numpy()
    print(f"\n=== Vantage 2022-（{spanv:.1f}年・N={len(t)}）  年間の円で確認")
    print(f"  {'方式':<22} {'厚い玉%':>8} {'年間':>11} {'maxDD':>10}")

    def yen(mask2):
        y = t["yen"].to_numpy() * np.where(mask2, 2.0, 1.0)
        eq = np.cumsum(y)
        return y.sum() / spanv, float((np.maximum.accumulate(eq) - eq).max())

    a, d = yen(np.zeros(len(t), dtype=bool))
    print(f"  {'全部0.01':<22} {'—':>8} {a:>+10,.0f}円 {d:>9,.0f}円")
    a, d = yen(st_v & ref_v)
    print(f"  {'拡張窓（検証形）':<22} {(st_v&ref_v).mean()*100:>7.0f}% {a:>+10,.0f}円 {d:>9,.0f}円")
    for N in (500, 1000, 2000, 4000):
        med = ev.rolling(N, min_periods=N // 2).median()
        hi = (ev >= med).reindex(t["time"]).fillna(False).to_numpy()
        a, d = yen(st_v & hi)
        print(f"  {'移動窓 '+str(N)+'本':<22} {(st_v&hi).mean()*100:>7.0f}% {a:>+10,.0f}円 {d:>9,.0f}円")

    assert len(T) > 300
    print(f"\nOK: Binance N={len(T)} / Vantage N={len(t)}")
