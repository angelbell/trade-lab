"""「BTC が特別」なのか「同時に何かが動いていれば何でもいいのか」を分ける。

由来（`atr_spike_btc_leader.py`）: アルト9銘柄プールで「BTC も同時に拡大足」が
平均 +0.913% vs +0.032%、間引き帰無 %ile 100.0、符号一致 8/9（二項 P=0.039）、5年とも同符号。

落とし穴: これが「BTC の先導」ではなく「単に同時性がある＝相場全体が動いている」だけなら、
先導者を別の銘柄に差し替えても同じリフトが出るはず。10銘柄すべてを順に先導者にして測る。
偽薬として「BTC が24本前に拡大足」も置く（同じ本数・違う時刻＝何も出ないはず）。

判定: BTC のリフトが他の先導者より明確に大きければ先導者仮説。横並びなら単なる同時性。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_btc_leader import load, spikes, leg, st, dropnull, ALTS   # noqa: E402

ALL_SYM = ["btcusd"] + ALTS


def lift(trades, sig, lag=0):
    """先導者 sig（同足 or 直前）で割ったときの (平均%差, 残す割合, with, without)。"""
    s = sig.shift(lag) if lag else sig
    s = (s | s.shift(1)).fillna(False)
    w, o = [], []
    for sym, t in trades.items():
        m = s.reindex(t["time"]).fillna(False).to_numpy()
        p = t["pct"].to_numpy()
        w.append(p[m])
        o.append(p[~m])
    W, O = np.concatenate(w), np.concatenate(o)
    if len(W) < 30 or len(O) < 30:
        return None
    return (W.mean() * 100 - O.mean() * 100, len(W) / (len(W) + len(O)), W, O)


if __name__ == "__main__":
    sp = {s: spikes(load(s)) for s in ALL_SYM}
    tr = {}
    for s in ALL_SYM:
        t = leg(load(s))
        if t is not None:
            tr[s] = t

    print("=== 先導者を差し替える（その銘柄自身は母集団から外す）  Vantage 2022- ・コスト0.05%")
    print(f"  {'先導者':<10} {'追随側N':>8} {'残す割合':>9} {'同時 平均%':>11} {'静か 平均%':>11} "
          f"{'差':>8} {'同時PF':>7} {'静かPF':>7} {'帰無%ile 平均/PF':>18}")
    rows = []
    for L in ALL_SYM:
        sub = {k: v for k, v in tr.items() if k != L}
        r = lift(sub, sp[L])
        if r is None:
            continue
        dif, frac, W, O = r
        ALLP = np.concatenate([W, O])
        dn = dropnull(ALLP, W)
        rows.append((L, dif, frac, len(ALLP), st(W)[2], st(O)[2], dn))
        print(f"  {L:<10} {len(ALLP):>8} {frac*100:>8.0f}% {W.mean()*100:>+10.3f} "
              f"{O.mean()*100:>+10.3f} {dif:>+8.3f} {st(W)[2]:>7.2f} {st(O)[2]:>7.2f} "
              f"{dn[0]:>8.1f} /{dn[1]:>6.1f}")

    rows.sort(key=lambda x: -x[1])
    print(f"\n  差の順位: " + " > ".join(f"{r[0]}({r[1]:+.2f})" for r in rows))
    btc_rank = [i for i, r in enumerate(rows) if r[0] == "btcusd"][0] + 1
    print(f"  BTC の順位 = {btc_rank}/{len(rows)}")

    print("\n=== 偽薬: BTC の拡大足を n 本ずらす（同じ本数・違う時刻）")
    print(f"  {'ずらし':>8} {'残す割合':>9} {'差(平均%)':>11} {'帰無%ile 平均':>14}")
    sub = {k: v for k, v in tr.items() if k != "btcusd"}
    for lag in (0, 6, 24, 72, 168):
        r = lift(sub, sp["btcusd"], lag)
        if r is None:
            continue
        dif, frac, W, O = r
        dn = dropnull(np.concatenate([W, O]), W)
        print(f"  {lag:>8} {frac*100:>8.0f}% {dif:>+11.3f} {dn[0]:>13.1f}")

    print("\n=== 先導者を「BTC以外のアルトが2銘柄以上」にした場合（同時性そのものの効果）")
    altsp = pd.DataFrame({s: sp[s] for s in ALTS}).fillna(False)
    for thr in (1, 2, 3):
        cnt = altsp.sum(axis=1)
        r = lift(sub, cnt >= thr)
        if r is None:
            continue
        dif, frac, W, O = r
        dn = dropnull(np.concatenate([W, O]), W)
        print(f"  アルト{thr}銘柄以上   残す{frac*100:3.0f}%  差={dif:+.3f}%  "
              f"帰無%ile 平均={dn[0]:.1f} PF={dn[1]:.1f}")

    print("\n=== BTC同時 と アルト同時 を二元で切る（どちらが効いているか）")
    a2 = (altsp.sum(axis=1) >= 2)
    a2 = (a2 | a2.shift(1)).fillna(False)
    b1 = (sp["btcusd"] | sp["btcusd"].shift(1)).fillna(False)
    cells = {(True, True): [], (True, False): [], (False, True): [], (False, False): []}
    for sym, t in sub.items():
        bb = b1.reindex(t["time"]).fillna(False).to_numpy()
        aa = a2.reindex(t["time"]).fillna(False).to_numpy()
        p = t["pct"].to_numpy()
        for kb in (True, False):
            for ka in (True, False):
                cells[(kb, ka)].append(p[(bb == kb) & (aa == ka)])
    print(f"  {'':<16} {'アルト2銘柄以上':>18} {'アルトは静か':>18}")
    for kb in (True, False):
        line = []
        for ka in (True, False):
            q = np.concatenate(cells[(kb, ka)])
            line.append(f"N={len(q):4d} PF={st(q)[2]:4.2f} {st(q)[3]:+.3f}%" if len(q) >= 15
                        else "        --        ")
        print(f"  {'BTC同時' if kb else 'BTCは静か':<16} " + " ".join(f"{x:>18}" for x in line))

    r0 = lift({k: v for k, v in tr.items() if k != "btcusd"}, sp["btcusd"])
    assert r0[0] > 0.5, r0[0]
    assert sps.binomtest(8, 9, 0.5).pvalue < 0.05
    print(f"\nOK: BTC 先導の差 {r0[0]:+.3f}% を再現")
