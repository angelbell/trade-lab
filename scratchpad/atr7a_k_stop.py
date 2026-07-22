"""【第7段・掃引 軸1,2】USDJPY h1 ATR拡大足ロング。母集団=レジームON側（主=週足30MA上、副=日足SMA200上）
に限定し、k梯子(1.5/2.0/2.5/3.0)と損切りA系/B系(1.5/2.0/2.5×ATR)を掃引する。
各セルで「ゲート無し版」を必ず横に並べる（ゲートがパラメータ変更後も正しいかを見るため）。
出口=ATR×3トレール・fwd=20・PDHフィルタ(>0)・fill_win=200・A系がこの段のデフォルト（次段exit_horizonで動かす）。
"""
SCREEN = "atr_spike_btc_h1"
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, fmt_row, daily_up_regime, weekly_up_regime,
                          build_pdh_dist_series)  # noqa: E402

POPS = {"ゲート無し": None, "週足30MA上(主)": "weekly", "日足SMA200上(副)": "daily"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2015-12-31"]
    years = span_years(df)

    du = daily_up_regime(df)
    wu = weekly_up_regime(df)

    print("=" * 118)
    print("軸1: k梯子 (損切り=A系固定、出口=ATR×3トレール、fwd=20、PDH>0)")
    print("=" * 118)
    for pop_name, pop_key in POPS.items():
        print(f"\n----- 母集団: {pop_name} -----")
        for k in (1.5, 2.0, 2.5, 3.0):
            atr_prev = atr_prev_of(df)
            s_idx = raw_triggers(df, atr_prev, k)
            pdh = build_pdh_dist_series(df, atr_prev)
            mask = pdh[s_idx] > 0.0
            if pop_key == "daily":
                mask &= du[s_idx]
            elif pop_key == "weekly":
                mask &= wu[s_idx]
            s_sel = s_idx[mask]
            ent = build_entries(df, atr_prev, s_sel, "A", 3.0, trail=True)
            t = run_cell(df, ent, fill_win=200, fwd=20, trail_atr=3.0)
            if t is None:
                print(f"  k={k:.1f} 約定0件")
                continue
            s = stats(t, years)
            print("  " + fmt_row(f"k={k:.1f} A系 ATR×3トレール fwd20 PDH>0", s))

    print("\n" + "=" * 118)
    print("軸2: 損切りA系/B系 (k=2.0固定・出口=ATR×3トレール・fwd=20・PDH>0)")
    print("=" * 118)
    for pop_name, pop_key in POPS.items():
        print(f"\n----- 母集団: {pop_name} -----")
        atr_prev = atr_prev_of(df)
        s_idx = raw_triggers(df, atr_prev, 2.0)
        pdh = build_pdh_dist_series(df, atr_prev)
        mask0 = pdh[s_idx] > 0.0
        if pop_key == "daily":
            mask0 &= du[s_idx]
        elif pop_key == "weekly":
            mask0 &= wu[s_idx]
        s_sel = s_idx[mask0]
        # A系
        ent = build_entries(df, atr_prev, s_sel, "A", 3.0, trail=True)
        t = run_cell(df, ent, fill_win=200, fwd=20, trail_atr=3.0)
        if t is not None:
            print("  " + fmt_row("A系(拡大足の反対端)", stats(t, years)))
        for stopk in (1.5, 2.0, 2.5):
            ent = build_entries(df, atr_prev, s_sel, "B", 3.0, stopk=stopk, trail=True)
            t = run_cell(df, ent, fill_win=200, fwd=20, trail_atr=3.0)
            if t is None:
                print(f"  B系(stopk={stopk}) 約定0件")
                continue
            print("  " + fmt_row(f"B系(stopk={stopk})", stats(t, years)))

    print(f"\n実行コマンド: .venv/bin/python scratchpad/atr7a_k_stop.py{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
