"""ICT改善案3・追加: FXメジャー(m15)で同じICTラベル(狩り→reclaim/FVG)の層別スクリーン。
FXには採用済みレッグが無いのでサイズ・オーバーレイはやらない、層別のみ
（「ICT検出器の情報はbtc/goldのブレイクに固有か、FXのブレイクにも乗るか」を見る一枚）。

母集団: M_squeeze_matrix.py と同じ枠 --- breakout_wave の素の canonical Pattern-B ブレイク
（pullback_frac=0・ゲート無し・rr=100/fwd=500/cost=0 ＝ 素の巡行幅）。

流用（車輪の再発明禁止）: M_squeeze_matrix.{load_fx, run_bare_breakout}、
M_squeeze_screen.{mfe_stop_only, layer_stats, random_subset_null}、
ict_size_transplant.compute_labels（ラベルA/B定義をそのまま流用、新規実装ゼロ）。

Run: .venv/bin/python scratchpad/ict_label_fx_screen.py [--smoke] 2>&1 | tee scratchpad/out_ict_label_fx_screen.txt
"""
import sys, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import numpy as np
import pandas as pd

from M_squeeze_matrix import load_fx, run_bare_breakout
from M_squeeze_screen import mfe_stop_only, layer_stats, random_subset_null
from ict_size_transplant import compute_labels

FX6 = ["eurusd", "gbpusd", "usdjpy", "audusd", "nzdusd", "usdcad"]
X_LIST = [48, 96]


def sign_of(delta, pct_med):
    if delta > 0 and pct_med >= 60:
        return "順"
    if delta < 0 and pct_med <= 40:
        return "逆"
    return "無"


def report_cell(sym, X, MFE, valid, flag, label_name):
    m_all = valid
    m_lab = valid & flag
    n = int(m_all.sum()); n_lab = int(m_lab.sum())
    if n_lab < 10 or (n - n_lab) < 10:
        print(f"    {sym:<8}X={X:<4}{label_name:<8} n不足 (n={n}, ラベルあり={n_lab})")
        return
    s_all = layer_stats(MFE[m_all]); s_lab = layer_stats(MFE[m_lab])
    meds, _ = random_subset_null(MFE[m_all], n_lab, reps=1000)
    pct = 100 * np.mean(meds < s_lab["median"])
    delta = s_lab["median"] - s_all["median"]
    sign = sign_of(delta, pct)
    print(f"    {sym:<8}X={X:<4}{label_name:<8} n={n:5d} ラベルあり={n_lab:5d}({100*n_lab/n:.0f}%)  "
          f"MFE中央値base={s_all['median']:.2f}  ラベルあり={s_lab['median']:.2f}  Δ={delta:+.2f}  "
          f"P(<1R)base={s_all['p_lt1']:.1f}%  P(<1R)あり={s_lab['p_lt1']:.1f}%  "
          f"P(>=3R)base={s_all['p3']:.1f}%  P(>=3R)あり={s_lab['p3']:.1f}%  null%ile={pct:.0f}  符号={sign}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("銘柄×ラベル×Xの符号表（MFE中央値ベース、素のcanonical Pattern-Bブレイク母集団）")
    for sym in FX6:
        d = load_fx(sym, "15min", smoke=args.smoke)
        if d is None or len(d) < 500:
            print(f"  {sym}: データ不足でskip"); continue
        t = run_bare_breakout(d)
        if t is None or len(t) < 50:
            print(f"  {sym}: n不足でskip"); continue
        MFE, idx = mfe_stop_only(d, t)
        print(f"\n  --- {sym} (n={len(t)}) ---")
        for X in X_LIST:
            labelA, labelB = compute_labels(d, t, idx, X)
            valid = np.ones(len(idx), dtype=bool)
            report_cell(sym, X, MFE, valid, labelA, "A(狩り)")
            report_cell(sym, X, MFE, valid, labelB, "B(FVG)")
            report_cell(sym, X, MFE, valid, labelA & labelB, "A∧B")


if __name__ == "__main__":
    main()
