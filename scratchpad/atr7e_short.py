"""【第7段・掃引 軸7】ショートの成行再検定（取りこぼし）。`transplant-17` では戻り売り指値0.5（BTC由来の
執行）で崩壊したが、それは執行の不一致であって方向の否定ではないという既報の結論を受け、**成行**で
k梯子・損切りA/B系・出口(固定RR/ATRトレール)・fwd を掃引する。母集団はロングのベータ検定・掃引条件
（週足30MA上）とは非対称なので、ここではまず無条件（原仕様どおり）で掃引し、続けて「日足SMA200下
（下降レジーム）」をロングの鏡像として参考測定する。

執行は mirror.invert() した反転フレームに walk() をそのまま適用。コストは絶対値0.009円を
e_real=(C-e_px) に対して引く（mirror-cost-overcharge を回避、cost は %化しない）。
"""
SCREEN = "atr_spike_btc_h1"
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, fmt_row, build_pdh_dist_series, drop_null, block_bootstrap,
                          load_mt5_csv, USDJPY_D1)  # noqa: E402


def daily_down_regime_on_inv(inv, C):
    """ショート(下降局面)ゲート参考測定: 日足SMA200"下" をロングのミラーとして定義。
    反転フレームでなく実データ(USDJPY_D1)を直接使い、通常の「下」を計算してからinv.indexへ写す。"""
    d1 = load_mt5_csv(USDJPY_D1)
    sma200 = d1["close"].rolling(200).mean().shift(1)
    down = pd.Series(np.where(d1["close"].shift(1) < sma200, 1, -1), index=d1.index)
    down_h1 = down.reindex(inv.index.floor("D")).ffill()
    down_h1.index = inv.index
    return down_h1.to_numpy() > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2015-12-31"]
        inv = inv.loc[:"2015-12-31"]
    years = span_years(inv)

    print("=" * 118)
    print("軸7-1: k梯子（成行・A系・ATR×3トレール・fwd20・PDH系フィルタ無し、無条件母集団）")
    print("=" * 118)
    for k in (1.5, 2.0, 2.5, 3.0):
        atr_prev = atr_prev_of(inv)
        s_idx = raw_triggers(inv, atr_prev, k)
        ent = build_entries(inv, atr_prev, s_idx, "A", 3.0, trail=True)
        t = run_cell(inv, ent, fill_win=200, fwd=20, trail_atr=3.0, C=C)
        if t is None:
            print(f"  k={k:.1f} 約定0件"); continue
        print("  " + fmt_row(f"k={k:.1f} A系 TR3 fwd20", stats(t, years)))

    print("\n" + "=" * 118)
    print("軸7-2: 損切りA/B系（k=2.0固定・出口=ATR×3トレール・fwd20）")
    print("=" * 118)
    atr_prev = atr_prev_of(inv)
    s_idx = raw_triggers(inv, atr_prev, 2.0)
    ent = build_entries(inv, atr_prev, s_idx, "A", 3.0, trail=True)
    t = run_cell(inv, ent, fill_win=200, fwd=20, trail_atr=3.0, C=C)
    if t is not None:
        print("  " + fmt_row("A系", stats(t, years)))
    for stopk in (1.5, 2.0, 2.5):
        ent = build_entries(inv, atr_prev, s_idx, "B", 3.0, stopk=stopk, trail=True)
        t = run_cell(inv, ent, fill_win=200, fwd=20, trail_atr=3.0, C=C)
        if t is None:
            print(f"  B系(stopk={stopk}) 約定0件"); continue
        print("  " + fmt_row(f"B系(stopk={stopk})", stats(t, years)))

    print("\n" + "=" * 118)
    print("軸7-3: 出口(固定RR{2,3,4.5,6} vs トレール{2,3,4,6}) × fwd(10,20,40,60,100)  [A系,k2.0]")
    print("=" * 118)
    RRS, TRAILS, FWDS = (2.0, 3.0, 4.5, 6.0), (2.0, 3.0, 4.0, 6.0), (10, 20, 40, 60, 100)
    for fwd in FWDS:
        cells = []
        for rr in RRS:
            ent = build_entries(inv, atr_prev, s_idx, "A", rr, trail=False)
            t = run_cell(inv, ent, fill_win=200, fwd=fwd, C=C)
            s = stats(t, years) if t is not None else None
            cells.append((f"RR{rr:g}", s))
        for tr in TRAILS:
            ent = build_entries(inv, atr_prev, s_idx, "A", 0.0, trail=True)
            t = run_cell(inv, ent, fill_win=200, fwd=fwd, trail_atr=tr, C=C)
            s = stats(t, years) if t is not None else None
            cells.append((f"TR{tr:g}", s))
        row = " | ".join(f"{lab}:PF{s['PF']:.2f}/DD{s['maxDD_pct']:.1f}" if s else f"{lab}:N/A"
                          for lab, s in cells)
        print(f"  fwd={fwd:>3}: {row}")

    print("\n" + "=" * 118)
    print("軸7-4: 前日安値フィルタ(pdh_distの反転フレーム版=前日安値からの距離)＋帰無400回")
    print("       ゲート無し / 日足SMA200下(参考) の2母集団で比較。k2.0 A系 TR3 fwd20。")
    print("=" * 118)
    pdh_inv = build_pdh_dist_series(inv, atr_prev)  # 反転フレームに同じ式=実物の前日安値鏡像
    down = daily_down_regime_on_inv(inv, C)
    for pop_name, pop_mask in (("無条件", np.ones(len(s_idx), bool)),
                                ("日足SMA200下(参考ゲート)", down[s_idx])):
        s_pop = s_idx[pop_mask]
        ent = build_entries(inv, atr_prev, s_pop, "A", 3.0, trail=True)
        t_base = run_cell(inv, ent, fill_win=200, fwd=20, trail_atr=3.0, C=C)
        if t_base is None:
            print(f"  [{pop_name}] 約定0件"); continue
        pool_pct = t_base["pnl_pct"].to_numpy()
        s_base = stats(t_base, years)
        print(f"\n  -- 母集団: {pop_name} --")
        print("  " + fmt_row("フィルタ無し", s_base))
        for th in (-1.0, -0.5, 0.0):
            m = pdh_inv[s_pop] > th
            s_sel = s_pop[m]
            ent = build_entries(inv, atr_prev, s_sel, "A", 3.0, trail=True)
            t = run_cell(inv, ent, fill_win=200, fwd=20, trail_atr=3.0, C=C)
            if t is None:
                print(f"  前日安値dist>{th:+.1f} 約定不足"); continue
            s = stats(t, years)
            nl = drop_null(pool_pct, s["N"], s["mean_pct"] / 100, s["PF"])
            print("  " + fmt_row(f"前日安値dist>{th:+.1f} (通過{m.mean()*100:.1f}%)", s,
                                  nl["pf_pct"], nl["mean_pct"]))

    print(f"\n実行コマンド: .venv/bin/python scratchpad/atr7e_short.py{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
