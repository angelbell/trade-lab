"""atr7_common.py の検算: 凍結アンカー再現 + レジーム先読み点検。掃引本番の前に一度だけ走らせる。"""
SCREEN = "atr_spike_btc_h1"
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, daily_up_regime, weekly_up_regime, check_no_lookahead_pdh,
                          build_pdh_dist_series)  # noqa: E402

df, inv, C = load_frames()
years = span_years(df)
atr_prev = atr_prev_of(df)
s_idx = raw_triggers(df, atr_prev, 2.0)
pdh = build_pdh_dist_series(df, atr_prev)
s_pdh = s_idx[pdh[s_idx] > 0.0]

ent = build_entries(df, atr_prev, s_pdh, "A", 3.0, trail=True)
t = run_cell(df, ent, fill_win=200, fwd=20, trail_atr=3.0)
s = stats(t, years)
print(f"アンカー再現: N={s['N']} N/年={s['N_yr']:.1f} 勝率={s['win']:.1f}% PF={s['PF']:.2f} "
      f"平均={s['mean_pct']:+.4f}% maxDD={s['maxDD_pct']:.1f}% 黒字年={s['pos_years']}/{s['n_years']}")
assert s["N"] == 636, s["N"]
assert abs(s["PF"] - 1.30) < 0.005, s["PF"]
assert abs(s["mean_pct"] - 0.0474) < 0.0005, s["mean_pct"]
assert abs(s["maxDD_pct"] - 9.1) < 0.05, s["maxDD_pct"]
print("OK: atr7_common 経由でもアンカー N=636/PF1.30/平均+0.0474%/maxDD9.1% を再現")

ok, n_checked = check_no_lookahead_pdh(df, atr_prev, s_idx)
assert ok and n_checked > 1000
print(f"OK: pdh_dist 先読み無し (検査本数={n_checked})")

du = daily_up_regime(df)
wu = weekly_up_regime(df)
du_trunc = daily_up_regime(df.iloc[:-3000])
wu_trunc = weekly_up_regime(df.iloc[:-3000])
n = len(du_trunc) - 200  # 端の窓効果を避けて手前を切る
match_d = int((du[:n] == du_trunc[:n]).sum())
match_w = int((wu[:n] == wu_trunc[:n]).sum())
print(f"先読み点検(日足SMA200regime, 末尾3000本切断): {match_d}/{n} 一致")
print(f"先読み点検(週足30MAregime, 末尾3000本切断): {match_w}/{n} 一致")
assert match_d == n, "daily_up_regime に先読みあり"
assert match_w == n, "weekly_up_regime に先読みあり"
print("OK: レジーム2種とも先読み無し")

print(f"\nゲート通過率: 日足SMA200上={du[s_idx].mean()*100:.1f}%  週足30MA上={wu[s_idx].mean()*100:.1f}%")
print("\n実行コマンド: .venv/bin/python scratchpad/atr7_sanity.py")
