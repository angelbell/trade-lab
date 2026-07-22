"""一般化の対立仮説つぶし: 「一家の誰か」ではなく「その一家の覇権銘柄」を先導者に据える。

前段（atr_spike_leader_general.py）で、貴金属・指数・FX では横断面の同時性に効果が出なかった
（差 −0.109〜+0.048%、符号一致 1/3・1/2・2/4・2/2）。暗号資産は +0.881/+0.554・8/9。

却下の前に潰す対立仮説: 暗号資産で効いたのは「誰かと同時」ではなく「**BTC** と同時」だった
（BTC は先導者10候補中1位）。金属では金が、指数では NAS100 が、FX ではドル指数が同じ役をするのに、
「一家の他の誰か」で括ったせいで先導と追随を混ぜて薄めたのではないか。
→ 各銘柄を順に**単独の先導者**に据えて測り直す（暗号資産で使ったのと同じ手続き）。
   FX にはドル指数(usdx.r)も候補に入れる。対ドル通貨(XXXUSD)にはドル**下落**が追い風なので、
   ドル指数側は【陰線の拡大足】＝鏡像の引き金で判定する。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_leader_general import (load, spikes, leg, st, dropnull,   # noqa: E402
                                                 wilder_atr, K)

GROUPS = [
    ("貴金属", "2018-01-01",
     [("xauusd", 0.0003), ("xagusd", 0.0010), ("copper-cr", 0.0010)], []),
    ("指数", "2016-01-01",
     [("nas100.r", 0.0002), ("ger40.r", 0.0002)], []),
    ("FX 対ドル", "2020-03-01",
     [("eurusd", 0.0001), ("gbpusd", 0.0001), ("audusd", 0.0001), ("nzdusd", 0.0001)],
     [("usdx.r", "down")]),
    ("FX 対ドル (全期間)", "2000-01-01",
     [("eurusd", 0.0001), ("gbpusd", 0.0001), ("audusd", 0.0001), ("nzdusd", 0.0001)],
     [("usdjpy", "down"), ("usdcad", "down")]),
    ("FX ドル建て", "2000-01-01",
     [("usdjpy", 0.0001), ("usdcad", 0.0001)],
     [("eurusd", "down"), ("gbpusd", "down")]),
]


def spikes_down(d):
    """陰線の拡大足（実体の絶対値 > ATR*K かつ陰線）。"""
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    return pd.Series((o - c > ap * K) & (c < o) & np.isfinite(ap) & (ap > 0), index=d.index)


if __name__ == "__main__":
    for fam, start, members, extra in GROUPS:
        tr, sp = {}, {}
        for sym, cost in members:
            d = load(sym, start)
            sp[sym] = spikes(d)
            t = leg(d, cost)
            if t is not None:
                tr[sym] = t
        for sym, direction in extra:
            d = load(sym, start)
            sp[f"{sym}({'陰線' if direction=='down' else '陽線'})"] = (
                spikes_down(d) if direction == "down" else spikes(d))
        if len(tr) < 2:
            print(f"\n===== {fam}: 追随側の銘柄が足りない ({len(tr)})")
            continue
        print(f"\n===== {fam}  ({start}- ・追随側 {len(tr)}銘柄)")
        print(f"  {'先導者':<18} {'追随N':>6} {'残す率':>7} {'同時 平均%':>11} "
              f"{'静か 平均%':>11} {'差':>8} {'同時PF':>7} {'静かPF':>7} {'帰無%ile 平均/PF':>18}")
        rows = []
        for L, sig in sp.items():
            s = (sig | sig.shift(1)).fillna(False)
            w, o = [], []
            for sym, t in tr.items():
                if sym == L:
                    continue
                m = s.reindex(t["time"]).fillna(False).to_numpy()
                p = t["pct"].to_numpy()
                w.append(p[m])
                o.append(p[~m])
            if not w:
                continue
            W, O = np.concatenate(w), np.concatenate(o)
            if len(W) < 30 or len(O) < 30:
                print(f"  {L:<18} 本数不足 ({len(W)}/{len(O)})")
                continue
            ALLP = np.concatenate([W, O])
            dn = dropnull(ALLP, W)
            dif = W.mean() * 100 - O.mean() * 100
            rows.append((L, dif, dn))
            print(f"  {L:<18} {len(ALLP):>6} {len(W)/len(ALLP)*100:>6.0f}% "
                  f"{W.mean()*100:>+11.4f} {O.mean()*100:>+11.4f} {dif:>+8.4f} "
                  f"{st(W)[2]:>7.2f} {st(O)[2]:>7.2f} {dn[0]:>8.1f} /{dn[1]:>6.1f}")
        if rows:
            best = max(rows, key=lambda r: r[1])
            print(f"  → 最良の先導者: {best[0]}  差={best[1]:+.4f}%  "
                  f"帰無%ile={best[2][0]:.1f}/{best[2][1]:.1f}"
                  f"   （暗号資産の BTC は 差=+0.881% ・%ile 100/100）")

    print("\n（判定: どの一家でも最良の先導者が %ile 95 に届かず差が桁違いに小さいなら、"
          "\n  横断面は暗号資産という一家の内部事情であって、層ではない）")
