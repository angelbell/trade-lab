"""因果的なしきい値（拡張窓の中央値）でERゲートのブロック・ブートストラップを取り直す。

先の検定は全期間の中央値をしきい値にしていた＝先読み。実運用の形で回し直す。
指標は **年間の円 と maxDD の円**（0.01ロット固定なので比は使えない）を両方見る。
真の改善はブロックを長くするほど勝率が上がる。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_er_gate import build                     # noqa: E402
from experiments.atr_spike_er_causal import prep_er                 # noqa: E402
from experiments.atr_spike_barspread import spikes                  # noqa: E402

NBOOT = 1000
RNG = np.random.default_rng(555)


def metrics(ys):
    if len(ys) < 5:
        return np.nan, np.nan
    eq = np.cumsum(ys)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return ys.sum(), dd


if __name__ == "__main__":
    dB, tB = build("btcusd")
    sB = spikes(dB)
    lead = (sB | sB.shift(1)).fillna(False)
    dE, tE = build("ethusd", lead=lead)
    acc = []
    for d, t in ((dB, tB), (dE, tE)):
        T = prep_er(d, t)
        T = T[T["warm"]]
        acc.append(pd.DataFrame({"time": T["time"].values, "yen": T["yen"].to_numpy(),
                                 "on": (T["er"] >= T["exp50"]).to_numpy()}))
    P = pd.concat(acc, ignore_index=True).sort_values("time").reset_index(drop=True)
    P["mo"] = P["time"].dt.to_period("M")
    months = sorted(P["mo"].unique())
    bymo = {m: g for m, g in P.groupby("mo")}
    nm = len(months)
    a_tot, a_dd = metrics(P["yen"].to_numpy())
    o_tot, o_dd = metrics(P.loc[P["on"], "yen"].to_numpy())
    print(f"=== 合成・助走後 N={len(P)}（{months[0]}〜{months[-1]}・{nm}か月）")
    print(f"  実測: 全部取る 通算{a_tot:+,.0f}円 / DD {a_dd:,.0f}円  →  "
          f"ERゲート 通算{o_tot:+,.0f}円 / DD {o_dd:,.0f}円")
    print(f"\n  {'ブロック':>9} | {'通算の円が勝つ割合':>20} | {'DDの円が小さい割合':>20} | "
          f"{'両方勝つ割合':>14}")
    for b in (1, 3, 6, 12):
        w1 = w2 = w3 = ok = 0
        for _ in range(NBOOT):
            need = int(np.ceil(nm / b))
            starts = RNG.integers(0, nm, size=need)
            pa, po = [], []
            for st in starts:
                blk = [months[(st + i) % nm] for i in range(b)]
                gs = [bymo[m] for m in blk if m in bymo]
                if not gs:
                    continue
                g = pd.concat(gs, ignore_index=True)
                pa.append(g["yen"].to_numpy())
                po.append(g.loc[g["on"], "yen"].to_numpy())
            if not pa:
                continue
            ta, da = metrics(np.concatenate(pa))
            to, do = metrics(np.concatenate(po))
            if not (np.isfinite(ta) and np.isfinite(to)):
                continue
            ok += 1
            w1 += int(to > ta)
            w2 += int(do < da)
            w3 += int(to > ta and do < da)
        print(f"  {b:>7}か月 | {w1/max(ok,1)*100:>19.1f}% | {w2/max(ok,1)*100:>19.1f}% | "
              f"{w3/max(ok,1)*100:>13.1f}%")

    print("\n（読み方: ブロックを長くするほど勝率が上がれば本物。"
          "\n  0.01ロット固定なので『通算の円が増えて DD が減る』の両立が条件）")
    assert len(P) > 200, len(P)
    print(f"\nOK: N={len(P)}")
