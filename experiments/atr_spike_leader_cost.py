"""「BTC同時」で選別したアルト群が、どこまでのコストに耐えるか。

判定順（記憶の規律）: 素の率×幅 → 偶然性 → **コスト** → 口座寄与。
前2つは通した（帰無%ile 100・符号一致8/9・5年とも同符号・実体/ボラ 6/6層・偽薬ゼロ）。
アルトのスプレッドは未実測なので、往復コストを 0.02%〜0.50% で振って死ぬ点を出す。
Vantage の BTC は約 0.023%（$15/66,000）＝アルトはこれより広いはず。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_btc_leader import load, spikes, leg, st, ALTS   # noqa: E402

if __name__ == "__main__":
    sB = spikes(load("btcusd"))
    sB1 = (sB | sB.shift(1)).fillna(False)
    recs = []
    for sym in ALTS:
        t = leg(load(sym), cost_pct=0.0)          # コスト無しで作り、後から引く
        if t is None:
            continue
        recs.append(pd.DataFrame({"sym": sym, "gross": t["pct"].to_numpy(),
                                  "y": t["time"].dt.year.values,
                                  "btc": sB1.reindex(t["time"]).fillna(False).to_numpy()}))
    R = pd.concat(recs, ignore_index=True)
    print(f"母集団 N={len(R)}（コスト控除前）  BTC同時 {int(R['btc'].sum())}本")

    print("\n=== 往復コストを振る（アルト9銘柄プール）")
    print(f"  {'コスト':>8} {'BTC同時: N/勝率/PF/平均%':>40} {'BTC静か: PF/平均%':>26} "
          f"{'全体: PF/平均%':>22}")
    for cst in (0.0002, 0.0005, 0.0010, 0.0020, 0.0030, 0.0050):
        p = R["gross"].to_numpy() - cst
        a, b = p[R["btc"].to_numpy()], p[~R["btc"].to_numpy()]
        print(f"  {cst*100:7.2f}% "
              f"N={len(a):4d} {st(a)[1]:5.1f}% PF={st(a)[2]:5.2f} {st(a)[3]:+.3f}%".ljust(50)
              + f"PF={st(b)[2]:5.2f} {st(b)[3]:+.3f}%".ljust(26)
              + f"PF={st(p)[2]:5.2f} {st(p)[3]:+.3f}%")

    print("\n=== 銘柄別に「コスト 0.20% でも黒字か」（BTC同時のみ）")
    for cst in (0.0010, 0.0020):
        out = []
        for sym in ALTS:
            q = R[(R["sym"] == sym) & R["btc"]]["gross"].to_numpy() - cst
            if len(q) >= 20:
                out.append(f"{sym[:3]}:{st(q)[2]:.2f}")
        print(f"  コスト {cst*100:.2f}%  " + "  ".join(out))

    print("\n=== 使える銘柄だけに絞る（ETH/SOL/ADA/DOT/XRP/LTC・BTC同時のみ・年別 平均%）")
    USE = ["ethusd", "solusd", "adausd", "dotusd", "xrpusd", "ltcusd"]
    S = R[R["sym"].isin(USE) & R["btc"]]
    for cst in (0.0005, 0.0020):
        p = S["gross"].to_numpy() - cst
        yr = pd.Series(p).groupby(S["y"].to_numpy()).agg(["mean", "count", "sum"])
        print(f"  コスト {cst*100:.2f}%  N={len(p)} 勝率={st(p)[1]:.1f}% PF={st(p)[2]:.2f} "
              f"平均={st(p)[3]:+.3f}%  年別平均: " +
              " ".join(f"{y}:{r['mean']*100:+.2f}%/{int(r['count'])}" for y, r in yr.iterrows()))

    p20 = R.loc[R["btc"], "gross"].to_numpy() - 0.0020
    assert len(R) > 1000, len(R)
    assert st(p20)[2] > 1.0, st(p20)[2]
    print(f"\nOK: コスト0.20%でも BTC同時の PF={st(p20)[2]:.2f}（>1）")
