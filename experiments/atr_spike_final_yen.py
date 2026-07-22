"""最終形を【0.01ロット固定】の実額（USD / 円）で出す。

R 基準と価格% 基準で PF が変わる（R はストップ幅で割るので固定比率サイズを暗黙に含む）。
ユーザーは 0.01 ロット固定なので、判断に使うべきは **1トレードあたりの実額**。
暗号資産CFD は 1ロット = 1コイン（Vantage）→ 0.01ロット = 0.01コイン。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from experiments.atr_spike_barspread import spread_series, leg, spikes   # noqa: E402

LOT = 0.01          # 0.01ロット = 0.01コイン
USDJPY = 150.0      # 円換算の目安


def stat_usd(t, lab, span):
    usd = (t["pct"].to_numpy() * t["e_px"].to_numpy()) * LOT
    w, ls = usd[usd > 0].sum(), -usd[usd < 0].sum()
    eq = np.cumsum(usd)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    print(f"  {lab:<22} N={len(usd):4d} 年{len(usd)/span:3.0f}本 勝率={(usd>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本={usd.mean():+7.2f}$ "
          f"({usd.mean()*USDJPY:+8,.0f}円) 通算={usd.sum():+9,.0f}$ DD={dd:8,.0f}$")
    return usd


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

    print(f"=== 0.01ロット固定での実額（Vantage 2022-・{span:.1f}年・実スプレッド課金・"
          f"1ドル{USDJPY:.0f}円）")
    ub = stat_usd(tb, "BTC 1時間", span)
    ue = stat_usd(te, "ETH 1時間(BTC確認)", span)

    P = pd.concat([
        pd.DataFrame({"sym": "BTC", "time": tb["time"].values, "usd": ub}),
        pd.DataFrame({"sym": "ETH", "time": te["time"].values, "usd": ue}),
    ], ignore_index=True).sort_values("time").reset_index(drop=True)
    u = P["usd"].to_numpy()
    eq = np.cumsum(u)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = u[u > 0].sum(), -u[u < 0].sum()
    print(f"  {'合成':<22} N={len(u):4d} 年{len(u)/span:3.0f}本 勝率={(u>0).mean()*100:5.1f}% "
          f"PF={w/ls:5.2f} 1本={u.mean():+7.2f}$ ({u.mean()*USDJPY:+8,.0f}円) "
          f"通算={u.sum():+9,.0f}$ DD={dd:8,.0f}$")
    print(f"\n  年別（合成・0.01ロット）:")
    yy = P.groupby(P["time"].dt.year)["usd"].agg(["sum", "count"])
    for y, r in yy.iterrows():
        print(f"    {y}: {r['sum']:+8,.0f}$ ({r['sum']*USDJPY:+10,.0f}円) / {int(r['count'])}本")
    print(f"\n  ⚠️ 1トレードあたりの最大損失（＝損切り幅×0.01コイン）の中央値: "
          f"BTC {np.median(tb['rf']*tb['e_px'])*LOT*USDJPY:,.0f}円 / "
          f"ETH {np.median(te['rf']*te['e_px'])*LOT*USDJPY:,.0f}円")
    print(f"  ⚠️ 建玉の名目（0.01コイン）: BTC {btc['close'].iloc[-1]*LOT*USDJPY:,.0f}円 / "
          f"ETH {eth['close'].iloc[-1]*LOT*USDJPY:,.0f}円")

    assert len(u) > 250, len(u)
    print(f"\nOK: 合成 N={len(u)} 1本={u.mean():+.2f}$")
