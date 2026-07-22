"""時間足ラダーの交絡つぶし: 保有上限 fwd を無調整で20本のまま移したのは正しいか。

fwd は【形】ではなく【時間】の変数。20本は 1時間足で20時間だが、15分足では5時間・5分足では100分。
下の足の劣化が「機構の劣化」なのか「保有時間が足りないだけ」なのかを分ける。

各時間足で fwd を振り、(a) 本数 (b) 素1本R (c) 実勢コストでの1本R と totR/DD を見る。
壁時計をそろえた点（15分なら80本＝20時間、5分なら240本）に印をつける。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_tf_ladder import get, leg, stats   # noqa: E402

COST = 0.0005          # BTC の実勢に近い保守値
# (足, ラベル, 振る fwd, 壁時計20時間に相当する fwd)
PLAN = [("m5", "5分", (20, 60, 120, 240, 480, 960), 240),
        ("m15", "15分", (20, 40, 80, 160, 320, 640), 80),
        ("h1", "1時間", (10, 20, 40, 80, 160), 20),
        ("h4", "4時間", (5, 10, 20, 40), 5)]

if __name__ == "__main__":
    hold = {}
    for sym in ("btcusdt", "ethusdt"):
        print(f"\n{'='*96}\n===== {sym.upper()}  保有上限を振る（コスト往復 {COST*100:.2f}%）")
        for tf, lab, fwds, wall in PLAN:
            d = get(sym, tf)
            if d is None:
                print(f"  {lab}: データ無し")
                continue
            span = (d.index[-1] - d.index[0]).days / 365.25
            print(f"  -- {lab}（★ = 壁時計20時間に相当）")
            print(f"     {'fwd':>6} {'保有時間':>9} {'N':>6} {'勝率':>7} {'素PF':>6} "
                  f"{'素1本R':>8} {'PF':>6} {'1本R':>8} {'totR':>8} {'DD(R)':>7} {'totR/DD':>8}")
            hrs = {"m5": 1 / 12, "m15": 0.25, "h1": 1.0, "h4": 4.0}[tf]
            for fw in fwds:
                t = leg(d, fw)
                if t is None:
                    continue
                g0 = stats(t, 0.0, span)
                g1 = stats(t, COST, span)
                star = "★" if fw == wall else " "
                print(f"    {star}{fw:>5} {fw*hrs:>8.0f}h {g0['n']:>6} {g1['win']:>6.1f}% "
                      f"{g0['pf']:>6.2f} {g0['meanR']:>+8.3f} {g1['pf']:>6.2f} "
                      f"{g1['meanR']:>+8.3f} {g1['totR']:>+8.1f} {g1['ddR']:>7.1f} "
                      f"{g1['ratio']:>8.2f}")
                if sym == "btcusdt" and tf == "m15" and fw == 80:
                    hold["m15_80"] = g1
                if sym == "btcusdt" and tf == "h1" and fw == 20:
                    hold["h1_20"] = g1

    print("\n（判定: 各足の最良 fwd が壁時計でそろうなら fwd は時間の変数＝無調整の移植が誤り。"
          "\n  本数でそろうなら形の変数＝無調整でよく、下の足の劣化は機構そのもの）")
    assert "m15_80" in hold and "h1_20" in hold
    print(f"\nOK: BTC 15分fwd80 の totR/DD={hold['m15_80']['ratio']:.2f} / "
          f"1時間fwd20={hold['h1_20']['ratio']:.2f}")
