"""最終形: BTC 1時間 ＋ ETH 1時間（BTC同時確認）を、実スプレッド課金で1本の運用として見る。

これまでの絞り込み:
  横断面（BTC同時確認）は全関門を通過したが、暗号資産の外へは出ない
  実スプレッドを課金すると 9銘柄中 7つが死に、BTC と ETH だけが残る
  時間足ラダー（実スプレッド）では 1時間足が 15分足・5分足に勝つ
∴ 残るのは BTC 1時間 ＋ ETH 1時間（BTC確認つき）の2本だけ。合成して素性を出す。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from experiments.atr_spike_barspread import spread_series, leg, spikes   # noqa: E402
from experiments.atr_spike_leader_cap import apply_cap                   # noqa: E402


def desc(r, span, lab):
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = r[r > 0].sum(), -r[r < 0].sum()
    print(f"  {lab:<26} N={len(r):4d} 年{len(r)/span:3.0f}本 勝率={(r>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本R={r.mean():+.3f} totR={r.sum():+7.1f} "
          f"DD={dd:5.1f}R 比={r.sum()/dd if dd>0 else np.nan:6.2f}")
    return dd


if __name__ == "__main__":
    btc = load_mt5_csv("data/vantage_btcusd_h1.csv").loc["2022-01-01":]
    btc = btc[~btc.index.duplicated(keep="first")].sort_index()
    eth = load_mt5_csv("data/vantage_ethusd_h1.csv").loc["2022-01-01":]
    eth = eth[~eth.index.duplicated(keep="first")].sort_index()
    span = (btc.index[-1] - btc.index[0]).days / 365.25
    sB = spikes(btc)
    lead = (sB | sB.shift(1)).fillna(False)

    tb = leg(btc, 20, cost_series=spread_series("BTCUSD|h1"))
    te = leg(eth, 20, cost_series=spread_series("ETHUSD|h1"), lead=lead)

    print(f"=== 最終形（Vantage 2022-・{span:.1f}年・実スプレッド課金）")
    desc(tb["R_net"].to_numpy(), span, "BTC 1時間（単独）")
    desc(te["R_net"].to_numpy(), span, "ETH 1時間（BTC確認）")

    P = pd.concat([
        pd.DataFrame({"sym": "BTC", "time": tb["time"].values,
                      "hold": tb["hold"].values, "R": tb["R_net"].to_numpy()}),
        pd.DataFrame({"sym": "ETH", "time": te["time"].values,
                      "hold": te["hold"].values, "R": te["R_net"].to_numpy()}),
    ], ignore_index=True).sort_values("time").reset_index(drop=True)
    desc(P["R"].to_numpy(), span, "合成（上限なし）")
    for cap in (1, 2):
        k = apply_cap(P, cap)
        desc(P.loc[k, "R"].to_numpy(), span, f"合成（同時建玉 上限{cap}本）")

    ov = P.groupby("sym").size()
    print(f"\n  内訳: " + " ".join(f"{s}={n}" for s, n in ov.items()))
    print("  年別の1本R / 本数:")
    for s, g in P.groupby("sym"):
        yy = g.groupby(g["time"].dt.year)["R"].agg(["mean", "count"])
        print(f"    {s}: " + " ".join(f"{y}:{r['mean']:+.3f}/{int(r['count'])}"
                                      for y, r in yy.iterrows()))
    yy = P.groupby(P["time"].dt.year)["R"].agg(["mean", "count", "sum"])
    print("    合成: " + " ".join(f"{y}:{r['sum']:+.1f}R/{int(r['count'])}"
                                  for y, r in yy.iterrows()))
    # 同時刻の重なり
    both = 0
    tb_i = pd.DatetimeIndex(tb["time"])
    for t in te["time"]:
        if ((tb_i >= t - pd.Timedelta(hours=1)) & (tb_i <= t + pd.Timedelta(hours=1))).any():
            both += 1
    print(f"  ETH のうち BTC と同時刻(±1h)に建つもの: {both}/{len(te)} "
          f"({both/len(te)*100:.0f}%)")

    assert len(P) > 250, len(P)
    print(f"\nOK: 合成 N={len(P)}")
