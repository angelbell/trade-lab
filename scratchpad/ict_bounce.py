"""ICT 再設計 — フェーズ1.5: ショート入口を「反発率→巡行幅」で先に測る（RR を被せる前に）。

規律（バウンス検証の順序）: 戻り売りは反発（フェード）系の入口。いきなり固定 RR を被せて
「勝率＝損益分岐」で殺してはいけない。順序は 反発率 → 巡行幅(MFE分布) → 選別可否 → やっと RR。

目標を一切被せず、約定後の MFE（favorable な最大 R、損切りに当たるまで・同足は損切り優先）を測る:
  反発率  = P(MFE >= k)  … これが「RR=k での勝率」に一致（損益分岐 = 1/(1+k)）
  巡行幅  = MFE の中央値・平均・標準偏差・分位（歪む分布なので分位を併記）
比較対象: ロング（RR4 で生きている参照）/ ショート base / SMT 選別後（選別で分布が上がるか）。

Run: .venv/bin/python scratchpad/ict_bounce.py 2>/dev/null
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, F_CANON, mfe_scan
from ict_population import canonical_setups, load_prepped
from ict_gates import sweep_frame, smt_short_gate

PAIRS = [("eurusd", "gbpusd"), ("gbpusd", "eurusd"), ("audusd", "nzdusd"), ("nzdusd", "audusd")]
KS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5]     # RR 候補（反発率＝この閾値への到達率）


def dist(scan):
    """(date, mfe, stopped, final) のリスト → 反発率・巡行幅の要約。"""
    if len(scan) < 20:
        return None
    mfe = np.array([x[1] for x in scan])
    stopped = np.array([x[2] for x in scan])
    reach = {k: float((mfe >= k).mean()) for k in KS}
    return dict(n=len(mfe), stopped=float(stopped.mean()),
                med=float(np.median(mfe)), mean=float(mfe.mean()), std=float(mfe.std()),
                q25=float(np.quantile(mfe, 0.25)), q75=float(np.quantile(mfe, 0.75)),
                q90=float(np.quantile(mfe, 0.90)), reach=reach)


def line(lab, d):
    if d is None:
        print(f"  {lab:22s} (n<20)"); return
    reach = "  ".join(f"{k}:{100*d['reach'][k]:4.1f}%" for k in KS)
    print(f"  {lab:22s} n={d['n']:5d} stop{100*d['stopped']:4.0f}% | "
          f"MFE中央{d['med']:.2f} 平均{d['mean']:.2f}±{d['std']:.2f} "
          f"q25/75/90={d['q25']:.2f}/{d['q75']:.2f}/{d['q90']:.2f}")
    print(f"  {'':22s} 反発率(=RR別勝率): {reach}")


def main():
    print("反発率(=MFE>=k への到達率)と巡行幅を、RR を被せる前に測る。損益分岐: RR0.5→67% 1→50% "
          "1.5→40% 2→33% 3→25% 4→20% 4.5→18%")
    print("=" * 122)
    cache = {}
    for name in set([p for pair in PAIRS for p in pair]):
        df, tarr, dates, span = load_prepped(name)
        cache[name] = dict(df=df, tarr=tarr, dates=dates, span=span,
                           S=canonical_setups(df, tarr, dates, 0))

    for X, Y in PAIRS:
        cx = cache[X]
        sp, cost = MODEL[X]
        print(f"\n=== {X}  (SMT 相方={Y}) ===  ({cx['span']}年)")
        # 参照: ロング（RR4 で生きている）
        long_scan = mfe_scan(cx["df"], cx["S"], F_CANON, BUF, sp, "long")
        line("[参照] ロング", dist(long_scan))
        # ショート base（目標なしの生の反発）
        short_scan = mfe_scan(cx["df"], cx["S"], F_CANON, BUF, sp, "short")
        line("ショート base", dist(short_scan))
        # SMT 選別後（相方が buyside 掃除せず）— 選別で分布が上がるか
        cy = cache[Y]
        swY = sweep_frame(cy["df"], cy["tarr"], cy["dates"], 0)
        short_map = {x[0]: x for x in short_scan}
        smt_dates = set(d for d, net in
                        smt_short_gate({x[0]: 0.0 for x in short_scan}, swY))
        smt_scan = [short_map[d] for d in smt_dates if d in short_map]
        line("ショート SMT選別後", dist(smt_scan))


if __name__ == "__main__":
    main()
