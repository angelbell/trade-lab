"""壁時計をそろえた上で、BTC 同時確認の効き方を時間足別に測り直す。

先の測定は fwd=20本のまま全時間足に当てていたが、fwd は【時間】の変数と分かった
（15分・1時間とも壁時計20時間で最良）。確認条件の効き方もその下で測り直す必要がある。

各足で「その足の最良に近い壁時計20時間」の fwd を使い、確認あり/なしを比べる。
間引き帰無（同じ本数を無作為に残す×2000）も併記する。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_tf_ladder import get, leg, spikes, stats   # noqa: E402
from scratchpad.atr_spike_leader_binance import dropnull             # noqa: E402

COST = 0.0005
# (足, ラベル, 壁時計20時間の fwd)
PLAN = [("m5", "5分", 240), ("m15", "15分", 80), ("h1", "1時間", 20), ("h4", "4時間", 5)]

if __name__ == "__main__":
    keep = {}
    print(f"=== ETHUSDT に BTC 同時確認（同じ足の BTC 拡大足・同足or1本前）"
          f"  壁時計20時間にそろえた fwd・コスト{COST*100:.2f}%")
    print(f"  {'足':>6} {'fwd':>5} | {'確認あり: N/本年/勝率/PF/1本R/比':>44} | "
          f"{'確認なし: N/PF/1本R/比':>34} | {'差(1本R)':>9} {'帰無%ile':>10}")
    for tf, lab, fwd in PLAN:
        db = get("btcusdt", tf)
        de = get("ethusdt", tf)
        if db is None or de is None:
            print(f"  {lab:>6} データ無し")
            continue
        span = (de.index[-1] - de.index[0]).days / 365.25
        sB = spikes(db)
        lead = (sB | sB.shift(1)).fillna(False)
        t_all = leg(de, fwd)
        t_led = leg(de, fwd, lead=lead)
        if t_all is None or t_led is None:
            print(f"  {lab:>6} 本数不足")
            continue
        A = stats(t_led, COST, span)
        B = stats(t_all, COST, span)
        # 帰無は「確認なしの母集団から同じ本数を無作為に残す」
        pa = (t_led["gross"].to_numpy() - COST) / t_led["rf"].to_numpy()
        pb = (t_all["gross"].to_numpy() - COST) / t_all["rf"].to_numpy()
        dn = dropnull(pb, pa)
        print(f"  {lab:>6} {fwd:>5} | N={A['n']:4d} {A['ny']:3.0f}/年 {A['win']:5.1f}% "
              f"PF={A['pf']:5.2f} {A['meanR']:+.3f}R 比{A['ratio']:6.2f} | "
              f"N={B['n']:4d} PF={B['pf']:5.2f} {B['meanR']:+.3f}R 比{B['ratio']:6.2f} | "
              f"{A['meanR']-B['meanR']:>+9.3f} "
              f"{'—' if dn is None else f'{dn[0]:5.1f}/{dn[1]:5.1f}':>10}")
        keep[tf] = (A, B)

    print("\n=== BTC 自身も参考（確認条件なし・同じ fwd）")
    print(f"  {'足':>6} {'fwd':>5} {'N':>6} {'本/年':>7} {'勝率':>7} {'PF':>6} "
          f"{'1本R':>8} {'totR':>8} {'DD(R)':>7} {'totR/DD':>8}")
    for tf, lab, fwd in PLAN:
        db = get("btcusdt", tf)
        if db is None:
            continue
        span = (db.index[-1] - db.index[0]).days / 365.25
        t = leg(db, fwd)
        if t is None:
            continue
        g = stats(t, COST, span)
        print(f"  {lab:>6} {fwd:>5} {g['n']:>6} {g['ny']:>7.0f} {g['win']:>6.1f}% "
              f"{g['pf']:>6.2f} {g['meanR']:>+8.3f} {g['totR']:>+8.1f} {g['ddR']:>7.1f} "
              f"{g['ratio']:>8.2f}")

    assert "m15" in keep and "h1" in keep
    a15, a1h = keep["m15"][0], keep["h1"][0]
    print(f"\nOK: 15分の確認あり 比={a15['ratio']:.2f}（年{a15['ny']:.0f}本） / "
          f"1時間 比={a1h['ratio']:.2f}（年{a1h['ny']:.0f}本）")
