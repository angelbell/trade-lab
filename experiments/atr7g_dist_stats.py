"""最終候補2本の pnl_pct 分布（中央値・標準偏差）を検算目的で追加出力するだけの小スクリプト。"""
SCREEN = "atr_spike_btc_h1"
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, weekly_up_regime, build_pdh_dist_series)  # noqa: E402

df, inv, C = load_frames()
wu = weekly_up_regime(df)

CANDS = [
    ("候補1: k2.0 B系(2.0) TR2 fwd20", dict(k=2.0, system="B", stopk=2.0, val=2.0, fwd=20)),
    ("候補2: k2.5 A系 TR3 fwd20", dict(k=2.5, system="A", stopk=2.0, val=3.0, fwd=20)),
]

for label, cfg in CANDS:
    atr_prev = atr_prev_of(df)
    s_idx = raw_triggers(df, atr_prev, cfg["k"])
    pdh = build_pdh_dist_series(df, atr_prev)
    mask = (pdh[s_idx] > 0.0) & wu[s_idx]
    s_sel = s_idx[mask]
    ent = build_entries(df, atr_prev, s_sel, cfg["system"], 0.0, stopk=cfg["stopk"], trail=True)
    t = run_cell(df, ent, fill_win=200, fwd=cfg["fwd"], trail_atr=cfg["val"])
    p = t["pnl_pct"].to_numpy()
    print(f"{label}: N={len(p)} mean%={p.mean()*100:+.4f} median%={np.median(p)*100:+.4f} "
          f"std%={p.std(ddof=1)*100:.4f} skew={float(((p-p.mean())**3).mean()/p.std(ddof=1)**3):.2f}")

assert True
print("\n実行コマンド: .venv/bin/python experiments/atr7g_dist_stats.py")
