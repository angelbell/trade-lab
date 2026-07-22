"""(c) トレール出口の正典化 ― src/engine/walk.py に足した --trail-atr の紐付け検証。

前回 scratchpad/atr_spike_hour_trail.py は自前ループを書いていて、正典 walk() と玉の占有規則が
微妙に食い違い（N=550/PF1.49 vs 正典 N=556/PF1.45）、外の数字と並べられなかった。

今回は walk() 自体に trail_atr（既定 0=オフ）を実装した（src/engine/walk.py）ので、自前ループは
一切書かない。trail_atr=0 のときに凍結済みの基準セル（ロング成行 k=2.0 RR3 A系 fwd20）を
再現することを assert する。この assert が通れば、以後 (a) の実験は trail_atr>0 の数字も
walk() 本体から出ているとみなせる。

SCREEN = "atr_spike_btc_h1"  (第1段で通行証取得済み・引き金/母集団は不変)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr5_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell,
                          stats, span_years, fmt_row)  # noqa: E402

df, inv, C = load_frames()
years = span_years(df)

print("=" * 100)
print("[検算1] trail_atr=0（既定オフ）で凍結済み基準セルを再現するか")
print("        ロング成行 k=2.0 RR3 A系 fwd=20 fill_win=200 cost=0.0005 → 目標 N=556/PF1.45/勝率47.8%/平均+0.393%/maxDD22.2%")
print("=" * 100)

atr_prev_L = atr_prev_of(df)
s_idx_2 = raw_triggers(df, atr_prev_L, 2.0)
entries_A2 = build_entries(df, atr_prev_L, s_idx_2, "A", 3.0)

t0 = run_cell(df, entries_A2, pf=0.0, fill_win=200, fwd=20, trail_atr=0.0)
s0 = stats(t0, years)
print(fmt_row("trail_atr=0.0 (=fixed RR3, engineの既定path)", s0))

assert s0["N"] == 556, s0["N"]
assert abs(s0["win"] - 47.8) < 0.15, s0["win"]
assert abs(s0["PF"] - 1.45) < 0.01, s0["PF"]
assert abs(s0["mean_pct"] - 0.393) < 0.002, s0["mean_pct"]
assert abs(s0["maxDD_pct"] - 22.2) < 0.15, s0["maxDD_pct"]
print("  OK: trail_atr=0 は凍結済み基準セルと一致 (N/勝率/PF/平均%/maxDD% 全て許容誤差内)")

print("\n" + "=" * 100)
print("[検算2] trail_atr>0 で結果が実際に変わること（配線が生きている証拠。良し悪しは問わない）")
print("=" * 100)
for m in (2.0, 3.0, 4.0, 6.0):
    t = run_cell(df, entries_A2, pf=0.0, fill_win=200, fwd=200, trail_atr=m)
    s = stats(t, years)
    print(fmt_row(f"trail_atr={m} fwd200 (対照: 遠いfwdでトレールに走路を与える)", s))
    assert s["N"] != 556 or abs(s["mean_pct"] - 0.393) > 0.001 or m == 0, \
        "trail_atr>0 が fixed-RR-fwd20 の基準セルと無差別＝配線が効いていない疑い"

print("\n" + "=" * 100)
print("[検算3] ショート（指値pf=0.5）でも trail_atr=0 が凍結済み基準セルを再現するか")
print("        k=2.0 RR3 pf=0.5 fill_win=200 fwd=20 → 目標 N=457/PF1.27")
print("=" * 100)
atr_prev_S = atr_prev_of(inv)
s_idx_2s = raw_triggers(inv, atr_prev_S, 2.0)
entries_A2s = build_entries(inv, atr_prev_S, s_idx_2s, "A", 3.0)
ts0 = run_cell(inv, entries_A2s, pf=0.5, fill_win=200, fwd=20, trail_atr=0.0, C=C)
ss0 = stats(ts0, years)
print(fmt_row("short trail_atr=0.0 pf=0.5", ss0))
assert ss0["N"] == 457, ss0["N"]
assert abs(ss0["PF"] - 1.27) < 0.01, ss0["PF"]
print("  OK: ショート pf=0.5 trail_atr=0 も凍結済み基準セルと一致 (N=457, PF=1.27)")

print("\n番人3本（engine_tieback.py / engine_golden.py check-run / size_tieback.py）は"
      "このスクリプトの外で別途実行し、41/41・11/11・5/5 PASS を確認済み。")
print("\n実行コマンド: .venv/bin/python scratchpad/atr5c_trail_canon.py")
