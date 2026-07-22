"""同時性ではなく「BTC の噴火期」が本体ではないか。

前段（`atr_spike_leader_lag.py`）で分かったこと:
  排他的に切ると【同じ足のみ・前後にBTCの拡大足が無い】は N=628 PF1.55 帰無%ile 43.3/66.9 ＝ 有意でない。
  効いているのは同時性そのものではなく、BTC がその周辺で繰り返し拡大足を出している状態。
  （lag=−1 が PF8.32 なのは「動きが続いた＝ロングが勝った」とほぼ同義なので使えないし驚きもない）

∴ 使える形に直す: **直近 N 本で BTC が拡大足を出した回数**（現在足を含む・未来は一切見ない）。
   これが「同じ足で BTC も拡大足」より良ければ、変数の正体は同時性でなく BTC の活動度。

比較の相手は常に間引き帰無（同じ本数を無作為に残す×2000）。
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
    btc = load("btcusdt")
    sB = spikes(btc).astype(float)
    # 直近 N 本の拡大足の本数（現在足を含む＝建てる時点で既知）
    counts = {n: sB.rolling(n, min_periods=1).sum() for n in (3, 6, 12, 24, 48, 96)}

    recs = []
    for sym in ALTS:
        t = leg(load(sym))
        if t is None:
            continue
        row = {"sym": sym, "pct": t["gross"].to_numpy() - COST,
               "y": t["time"].dt.year.values,
               "same": spikes(btc).reindex(t["time"]).fillna(False).to_numpy()}
        for n, c in counts.items():
            row[f"c{n}"] = c.reindex(t["time"]).fillna(0).to_numpy()
        recs.append(pd.DataFrame(row))
    R = pd.concat(recs, ignore_index=True)
    allp = R["pct"].to_numpy()
    print(f"=== Binance アルト9銘柄 2018-2026  母集団 N={len(R)} PF={st(allp)[2]:.2f} "
          f"平均={st(allp)[3]:+.3f}%")

    print("\n=== 直近N本のBTC拡大足の本数（現在足を含む）でしきい値を切る")
    print(f"  {'窓':>5} {'しきい':>6} {'N':>6} {'割合':>6} {'勝率':>7} {'PF':>6} {'平均%':>9} "
          f"{'帰無%ile 平均/PF':>18}")
    best = None
    for n in (3, 6, 12, 24, 48, 96):
        for thr in (1, 2, 3, 4):
            m = (R[f"c{n}"] >= thr).to_numpy()
            if m.sum() < 60 or m.mean() > 0.75:
                continue
            q = allp[m]
            dn = dropnull(allp, q)
            flag = ""
            if best is None or dn[1] > best[-1]:
                best = (n, thr, len(q), st(q)[2], st(q)[3], dn[1])
                flag = "  ←最良(PF帰無)"
            print(f"  {n:>5} {thr:>6} {len(q):>6} {m.mean()*100:>5.0f}% {st(q)[1]:>6.1f}% "
                  f"{st(q)[2]:>6.2f} {st(q)[3]:>+9.3f} {dn[0]:>8.1f} /{dn[1]:>6.1f}{flag}")

    print("\n=== 現行（同じ足 or 1本前）との直接対決")
    sB1 = (spikes(btc) | spikes(btc).shift(1)).fillna(False)
    cur = np.concatenate([sB1.reindex(pd.DatetimeIndex(g["time"] if "time" in g else []))
                          .to_numpy() for g in []]) if False else None
    recs2 = []
    for sym in ALTS:
        t = leg(load(sym))
        if t is None:
            continue
        recs2.append(sB1.reindex(t["time"]).fillna(False).to_numpy())
    curm = np.concatenate(recs2)
    cands = [("現行: 同じ足 or 1本前", curm)]
    for n, thr in ((6, 1), (12, 1), (12, 2), (24, 2), (24, 3), (48, 3), (48, 4)):
        cands.append((f"直近{n}本でBTC {thr}回以上", (R[f"c{n}"] >= thr).to_numpy()))
    cands.append(("現行 かつ 直近24本で2回以上", curm & (R["c24"] >= 2).to_numpy()))
    cands.append(("現行 または 直近24本で3回以上", curm | (R["c24"] >= 3).to_numpy()))
    print(f"  {'条件':<28} {'N':>6} {'割合':>6} {'勝率':>7} {'PF':>6} {'平均%':>9} "
          f"{'帰無%ile 平均/PF':>18}")
    for lab, m in cands:
        q = allp[m]
        dn = dropnull(allp, q)
        print(f"  {lab:<28} {len(q):>6} {m.mean()*100:>5.0f}% {st(q)[1]:>6.1f}% "
              f"{st(q)[2]:>6.2f} {st(q)[3]:>+9.3f} {dn[0]:>8.1f} /{dn[1]:>6.1f}")

    print("\n=== 排他: 同時性と活動度のどちらが効いているか（2×2）")
    act = (R["c24"] >= 2).to_numpy()
    print(f"  {'':<18} {'直近24本でBTC2回以上':>26} {'BTC活動なし':>26}")
    for kb, lb in ((True, "同じ足でも拡大足"), (False, "同じ足は静か")):
        line = []
        for ka in (True, False):
            q = allp[(R["same"].to_numpy() == kb) & (act == ka)]
            line.append(f"N={len(q):4d} PF={st(q)[2]:5.2f} {st(q)[3]:+.3f}%" if len(q) >= 25
                        else "         --         ")
        print(f"  {lb:<18} " + " ".join(f"{x:>26}" for x in line))

    assert len(R) > 2000, len(R)
    assert best is not None
    print(f"\nOK: 最良は 直近{best[0]}本で{best[1]}回以上 (N={best[2]} PF={best[3]:.2f} "
          f"平均{best[4]:+.3f}% PF帰無%ile={best[5]:.1f})")
