"""15分足の生死はスプレッド次第。壁時計をそろえた仕様（fwd=80）でコストを振る。

台帳の「5m/15m はスプレッドで生死」を、この機構について数で出す。
損切り幅の中央値（価格に対する%）を併記し、コストが 1R の何%を食うかを見せる。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_tf_ladder import get, leg, spikes, stats   # noqa: E402

CASES = [("BTC 15分 (fwd80)", "btcusdt", "m15", 80, False),
         ("BTC 1時間 (fwd20)", "btcusdt", "h1", 20, False),
         ("ETH 15分 (fwd80)・BTC確認", "ethusdt", "m15", 80, True),
         ("ETH 1時間 (fwd20)・BTC確認", "ethusdt", "h1", 20, True)]

if __name__ == "__main__":
    keep = None
    for lab, sym, tf, fwd, conf in CASES:
        d = get(sym, tf)
        db = get("btcusdt", tf)
        if d is None or db is None:
            continue
        lead = None
        if conf:
            sB = spikes(db)
            lead = (sB | sB.shift(1)).fillna(False)
        t = leg(d, fwd, lead=lead)
        if t is None:
            continue
        span = (d.index[-1] - d.index[0]).days / 365.25
        rf = float(np.median(t["rf"].to_numpy()))
        print(f"\n=== {lab}   N={len(t)}（年{len(t)/span:.0f}本）"
              f"損切幅の中央値={rf*100:.2f}%")
        print(f"  {'往復コスト':>9} {'1Rの何%':>8} {'勝率':>7} {'PF':>6} {'1本R':>8} "
              f"{'totR':>8} {'DD(R)':>7} {'totR/DD':>8}")
        for cst in (0.0002, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030):
            g = stats(t, cst, span)
            print(f"  {cst*100:>8.2f}% {cst/rf*100:>7.0f}% {g['win']:>6.1f}% {g['pf']:>6.2f} "
                  f"{g['meanR']:>+8.3f} {g['totR']:>+8.1f} {g['ddR']:>7.1f} {g['ratio']:>8.2f}")
        if lab.startswith("BTC 15分"):
            keep = (len(t), stats(t, 0.0005, span))
    assert keep is not None and keep[1]["pf"] > 1.4, keep
    print(f"\nOK: BTC 15分 fwd80 N={keep[0]} PF@0.05%={keep[1]['pf']:.2f} "
          f"比={keep[1]['ratio']:.2f}")
