"""「先導者」は本当に先導しているのか。BTC の拡大足を前後にずらして、どこで効くかを見る。

いま台帳には「BTCも同時に拡大足か」と書いてあり、条件は【同じ足 or 直前の足】。
これは先導と同時を混ぜている。分けて測る:
   lag=-2,-1 : BTC が【後】に拡大足（アルトが先）
   lag= 0    : 同じ足
   lag=+1,+2,+3 : BTC が【先】に拡大足
+1 側だけが効くなら先導。0 が主なら同時性。−側も同じなら「そのあたりの時間帯が良い」だけ。

🚨 −1/−2（BTCが後）は**建てる時点では未知**なので運用には使えない。ここでは機構の診断にだけ使う。
   使える条件（0 と +）と、使えない条件（−）で効き方が違うかが要点。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_leader_binance import (load, spikes, leg, st, dropnull,   # noqa: E402
                                                 ALTS)

COST = 0.0005

if __name__ == "__main__":
    sB = spikes(load("btcusdt"))
    recs = []
    for sym in ALTS:
        t = leg(load(sym))
        if t is None:
            continue
        row = {"sym": sym, "pct": t["gross"].to_numpy() - COST,
               "y": t["time"].dt.year.values}
        for lag in range(-3, 4):
            row[f"L{lag}"] = sB.shift(lag).reindex(t["time"]).fillna(False).to_numpy()
        recs.append(pd.DataFrame(row))
    R = pd.concat(recs, ignore_index=True)
    allp = R["pct"].to_numpy()
    n, w, pf, mu = st(allp)
    print(f"=== Binance アルト9銘柄 2018-2026  母集団 N={n} 勝率={w:.1f}% PF={pf:.2f} 平均={mu:+.3f}%")
    print("\n  lag の意味: 正=BTC が【先】に拡大足 / 0=同じ足 / 負=BTC が【後】（建てる時点では未知）")
    print(f"  {'lag':>5} {'該当N':>6} {'割合':>6} {'勝率':>7} {'PF':>6} {'平均%':>9} "
          f"{'非該当 PF/平均%':>20} {'帰無%ile 平均/PF':>18} {'運用可':>6}")
    for lag in (3, 2, 1, 0, -1, -2, -3):
        m = R[f"L{lag}"].to_numpy()
        a, b = allp[m], allp[~m]
        if len(a) < 20:
            print(f"  {lag:>5} 本数不足 {len(a)}")
            continue
        dn = dropnull(allp, a)
        ok = "○" if lag >= 0 else "×"
        print(f"  {lag:>+5} {len(a):>6} {m.mean()*100:>5.0f}% {st(a)[1]:>6.1f}% "
              f"{st(a)[2]:>6.2f} {st(a)[3]:>+9.3f} "
              f"{st(b)[2]:>9.2f} /{st(b)[3]:>+8.3f} {dn[0]:>8.1f} /{dn[1]:>6.1f} {ok:>6}")

    print("\n=== 排他的に切る（重なりを除く: その lag だけで BTC が拡大足だった分）")
    print(f"  {'条件':<28} {'N':>6} {'勝率':>7} {'PF':>6} {'平均%':>9} {'帰無%ile':>10}")
    lead = R["L1"].to_numpy() | R["L2"].to_numpy() | R["L3"].to_numpy()
    same = R["L0"].to_numpy()
    after = R["L-1"].to_numpy() | R["L-2"].to_numpy() | R["L-3"].to_numpy()
    for lab, m in (("同じ足のみ（前後に無し）", same & ~lead & ~after),
                   ("BTCが先のみ（1-3本前）", lead & ~same & ~after),
                   ("BTCが後のみ（1-3本後）", after & ~same & ~lead),
                   ("同じ足＋BTCが先", same & lead),
                   ("BTC がまったく静か", ~same & ~lead & ~after)):
        q = allp[m]
        if len(q) < 20:
            print(f"  {lab:<28} 本数不足 {len(q)}")
            continue
        dn = dropnull(allp, q)
        print(f"  {lab:<28} {len(q):>6} {st(q)[1]:>6.1f}% {st(q)[2]:>6.2f} "
              f"{st(q)[3]:>+9.3f} {dn[0]:>5.1f}/{dn[1]:<5.1f}")

    print("\n=== 運用可能な条件だけの比較（現行 = 同じ足 or 直前）")
    cur = R["L0"].to_numpy() | R["L1"].to_numpy()
    for lab, m in (("現行: 同じ足 or 1本前", cur),
                   ("同じ足のみ", R["L0"].to_numpy()),
                   ("同じ足 or 1-2本前", cur | R["L2"].to_numpy()),
                   ("同じ足 or 1-3本前", cur | R["L2"].to_numpy() | R["L3"].to_numpy())):
        q = allp[m]
        dn = dropnull(allp, q)
        print(f"  {lab:<24} N={len(q):4d} ({m.mean()*100:2.0f}%) 勝率={st(q)[1]:5.1f}% "
              f"PF={st(q)[2]:5.2f} 平均={st(q)[3]:+.3f}%  帰無%ile={dn[0]:.1f}/{dn[1]:.1f}")

    m0 = R["L0"].to_numpy()
    assert len(R) > 2000, len(R)
    assert st(allp[m0])[2] > st(allp[~m0])[2], (st(allp[m0])[2], st(allp[~m0])[2])
    print(f"\nOK: 同足 PF={st(allp[m0])[2]:.2f} > 非同足 PF={st(allp[~m0])[2]:.2f} (N={len(R)})")
