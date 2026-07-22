"""atr_spike_pullback_btc_h1.py の追補: 実際に帰無を超えた代表セルに対する
巡回ブロック・ブートストラップ（勝率のブロック長依存）。

本体では「A系の最良セル」を k=2.0 に固定して選んでいたが、結果を見るとk=2.0 long は
帰無(%ile>=95)を一度も超えていない(pf>0の225セル中0件)。一方 short 側と k=1.5 long 側は
広く帰無を超えている。判定の核心はこの「超えた側」がブロックを伸ばしても勝率が崩れないか
なので、ここで代表2セルに絞って追加測定する:
  1. short A系 k=2.0 pf=0.786 RR=3.0 (帰無%ile=100, A系shortの主力パターン)
  2. long  A系 k=1.5 pf=0.618 RR=4.5 (帰無%ile=100, long側で唯一広く超えた"尖り"の代表)

Run: .venv/bin/python scratchpad/atr_spike_pullback_blockboot.py 2>&1 | tee scratchpad/out_atr_spike_pullback_blockboot.txt
"""
SCREEN = "atr_spike_btc_h1"   # 本体スクリプトが既に research/screens/ に成果物を作成済み

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atr_spike_pullback_btc_h1 import (  # noqa: E402
    load_mt5_csv, invert, BTC_H1, compute_features, raw_triggers, build_entries,
    run_walk, fixed_bet_metrics, block_bootstrap_winrate, FWD_MAIN
)

df_long = load_mt5_csv(BTC_H1)
df_short = invert(df_long)

targets = [
    ("short A系 k=2.0 pf=0.786 RR=3.0", df_short, 2.0, "A", None, 0.786, 200, 3.0),
    ("long  A系 k=1.5 pf=0.618 RR=4.5", df_long, 1.5, "A", None, 0.618, 200, 4.5),
]

for name, work_df, k, system, stopk, pf, fill_win, rr in targets:
    atr_prev, body = compute_features(work_df)
    s_idx = raw_triggers(work_df, atr_prev, body, k)
    entries = build_entries(work_df, atr_prev, s_idx, system, stopk, rr)
    t = run_walk(work_df, entries, pf, fill_win, FWD_MAIN, cost=0.0)
    m = fixed_bet_metrics(t)
    print(f"\n=== {name} ===")
    print(f"  n={m['n']}  N/年={m['n_per_year']:.1f}  win={m['win']:.1f}%  PF={m['pf']:.2f}"
          f"  meanR={m['meanR']:+.3f}  totR={m['totR']:+.1f}")
    bb = block_bootstrap_winrate(t, [1, 3, 6, 12])
    for kmo, (med, lo, hi, nb) in bb.items():
        print(f"    ブロック{kmo:>2}か月: 勝率中央値={med:.1f}%  95%CI=[{lo:.1f},{hi:.1f}]  (有効draw={nb}/1000)")

print("\n" + "=" * 90)
print("コスト梯子 (short A系 k=2.0 pf=0.786 RR=3.0 -- 帰無を超えた代表セル)")
print("=" * 90)
from atr_spike_pullback_btc_h1 import cost_ladder_report, fixed_dollar_in_R  # noqa: E402

C_mirror = 2 * df_long["high"].max()
atr_prev_s, body_s = compute_features(df_short)
s_idx_s = raw_triggers(df_short, atr_prev_s, body_s, 2.0)
entries_s = build_entries(df_short, atr_prev_s, s_idx_s, "A", None, 3.0)
for pf in [0.0, 0.786]:
    rows = cost_ladder_report(df_short, entries_s, pf, 200, 3.0, FWD_MAIN, "short", 2.0, "A", None,
                               C=C_mirror)
    print(f"\n  pf={pf}:")
    if rows:
        for r in rows:
            print(f"    cost={r['cost']:.5f}  meanR={r['meanR']:+.4f}  PF={r['pf']:.2f}"
                  f"  win={r['win']:.1f}%  totR={r['totR']:+.1f}")

print("\n固定$25が年代別の中央値トレードで何R相当か (short A系 k=2.0, pf=0.786):")
conv = fixed_dollar_in_R(df_short, entries_s, 0.786, 200, 3.0, FWD_MAIN)
for y, v in conv.items():
    if v is None:
        print(f"  {y}: データ無し")
    else:
        print(f"  {y}: n={v['n']}  risk中央値=${v['risk_med']:.1f}  $25={v['cost25_in_R']:.4f}R相当")
