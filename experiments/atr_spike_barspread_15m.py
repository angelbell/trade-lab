"""15分足を実スプレッドで決着させる。壁時計20時間＝fwd80。

1時間足では実スプレッド課金で BTC(比9.88)・ETH(比10.03) だけが生き残った。
15分足は損切り幅が半分なので、同じスプレッドが1Rの倍を食う。仮定コストでの試算では
0.05%→比15.06 / 0.10%→9.21 / 0.20%→2.92 と急に崩れた。実測でどこに着地するかを見る。

価格は Vantage（実際に建てる板）。BTC は m15 の CSV がある。ETH は m15 の CSV が無いので
Binance の m15 を価格に使い、コストだけ Vantage の実スプレッドを当てる（近似であることを明記）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from experiments.atr_spike_barspread import (spread_series, leg, show, spikes)   # noqa: E402

FWD_15M, FWD_1H = 80, 20


def brokerize(d):
    idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    d = d.set_index(idx)
    return d[~d.index.duplicated(keep="first")].sort_index()


if __name__ == "__main__":
    print("=== 15分足・実スプレッド課金（2022-・壁時計20時間＝fwd80）")

    # --- BTC: 価格もスプレッドも Vantage
    d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2022-01-01":]
    d = d[~d.index.duplicated(keep="first")].sort_index()
    span = (d.index[-1] - d.index[0]).days / 365.25
    sp = spread_series("BTCUSD|m15")
    print(f"  -- BTCUSD 15分（Vantage 価格・Vantage スプレッド・{span:.1f}年）")
    for lab, kw in (("一律0.05%", dict(flat=0.0005)), ("一律0.20%", dict(flat=0.0020))):
        t = leg(d, FWD_15M, **kw)
        if t is not None:
            show(t, span, lab)
    tb = leg(d, FWD_15M, cost_series=sp)
    rb = show(tb, span, "★実スプレッド")
    print(f"       課金スプレッド 中央値={tb['cost'].median()*100:.4f}% "
          f"（1Rの{tb['cost'].median()/tb['rf'].median()*100:.0f}%）"
          f" 90%点={tb['cost'].quantile(0.9)*100:.4f}%  損切幅中央={tb['rf'].median()*100:.2f}%")
    print("       年別の1本R: " + " ".join(
        f"{y}:{g:+.3f}" for y, g in tb.groupby(tb['time'].dt.year)["R_net"].mean().items()))

    # 1時間足と本数をそろえた比較のため、同じ期間の1時間足も出す
    d1 = load_mt5_csv("data/vantage_btcusd_h1.csv").loc["2022-01-01":]
    d1 = d1[~d1.index.duplicated(keep="first")].sort_index()
    s1 = (d1.index[-1] - d1.index[0]).days / 365.25
    t1 = leg(d1, FWD_1H, cost_series=spread_series("BTCUSD|h1"))
    print("  -- 参考: BTCUSD 1時間（同じ期間・実スプレッド）")
    r1 = show(t1, s1, "★実スプレッド")

    # --- ETH: 価格は Binance（Vantage の m15 が無い）・コストは Vantage 実スプレッド
    de = brokerize(load_mt5_csv("data/binance_ethusdt_m15.csv")).loc["2022-01-01":]
    se = (de.index[-1] - de.index[0]).days / 365.25
    db = brokerize(load_mt5_csv("data/binance_btcusdt_m15.csv")).loc["2022-01-01":]
    sBm = spikes(db)
    lead = (sBm | sBm.shift(1)).fillna(False)
    spe = spread_series("ETHUSD|m15")
    print(f"  -- ETHUSD 15分（⚠️価格は Binance・コストは Vantage 実スプレッド・{se:.1f}年）")
    te = leg(de, FWD_15M, cost_series=spe, lead=lead)
    if te is not None:
        re = show(te, se, "★実スプレッド（BTC確認）")
        print(f"       課金スプレッド 中央値={te['cost'].median()*100:.3f}% "
              f"（1Rの{te['cost'].median()/te['rf'].median()*100:.0f}%）"
              f" 90%点={te['cost'].quantile(0.9)*100:.3f}%  損切幅中央={te['rf'].median()*100:.2f}%")

    assert rb is not None and r1 is not None
    print(f"\nOK: BTC 15分 比={rb['ratio']:.2f}（年{rb['n']/span:.0f}本） / "
          f"1時間 比={r1['ratio']:.2f}（年{r1['n']/s1:.0f}本）")
