"""【第7段・掃引 軸3,4】出口(固定RR{2,3,4.5,6} vs ATRトレール倍率{2,3,4,6}) × 保有上限fwd{10,20,40,60,100}。
k=2.0固定・PDH>0固定・母集団=週足30MA上(主)。損切りはA系とB系(stopk=2.0、前段で僅かに優位だった方)の
両方を通す。台地か単独の尖りかを軸ごとに明示。最後にk=2.5(前段のPF最良セル)でも同じグリッドを確認。
"""
SCREEN = "atr_spike_btc_h1"
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, fmt_row, weekly_up_regime, build_pdh_dist_series)  # noqa: E402

RRS = (2.0, 3.0, 4.5, 6.0)
TRAILS = (2.0, 3.0, 4.0, 6.0)
FWDS = (10, 20, 40, 60, 100)


def grid(df, s_sel, atr_prev, years, system, k, stopk=2.0):
    print(f"\n----- 損切り={system}系{'' if system=='A' else f'(stopk={stopk})'}  k={k} -----")
    print(f"  {'fwd':>4} | " + " | ".join(f"RR{r:g}".rjust(16) for r in RRS) + "  ||  " +
          " | ".join(f"TR{t:g}".rjust(16) for t in TRAILS))
    tot_rr = {r: [] for r in RRS}
    tot_tr = {t: [] for t in TRAILS}
    for fwd in FWDS:
        cells_rr, cells_tr = [], []
        for rr in RRS:
            ent = build_entries(df, atr_prev, s_sel, system, rr, stopk=stopk, trail=False)
            t = run_cell(df, ent, fill_win=200, fwd=fwd)
            s = stats(t, years) if t is not None else None
            cells_rr.append(s)
            if s: tot_rr[rr].append(s["tot_pct"] / max(s["maxDD_pct"], 0.01))
        for tr in TRAILS:
            ent = build_entries(df, atr_prev, s_sel, system, 0.0, stopk=stopk, trail=True)
            t = run_cell(df, ent, fill_win=200, fwd=fwd, trail_atr=tr)
            s = stats(t, years) if t is not None else None
            cells_tr.append(s)
            if s: tot_tr[tr].append(s["tot_pct"] / max(s["maxDD_pct"], 0.01))

        def cell_str(s):
            if s is None:
                return "N/A".rjust(16)
            return f"PF{s['PF']:.2f}/DD{s['maxDD_pct']:.1f}/N{s['N']}".rjust(16)
        print(f"  {fwd:>4} | " + " | ".join(cell_str(s) for s in cells_rr) + "  ||  " +
              " | ".join(cell_str(s) for s in cells_tr))
    print(f"  総%/maxDD (fwd平均):")
    print("    固定RR: " + " ".join(f"RR{r:g}={np.mean(v):.2f}" for r, v in tot_rr.items() if v))
    print("    トレール: " + " ".join(f"TR{t:g}={np.mean(v):.2f}" for t, v in tot_tr.items() if v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=float, default=2.0)
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2015-12-31"]
    years = span_years(df)
    wu = weekly_up_regime(df)

    atr_prev = atr_prev_of(df)
    s_idx = raw_triggers(df, atr_prev, cli.k)
    pdh = build_pdh_dist_series(df, atr_prev)
    mask = (pdh[s_idx] > 0.0) & wu[s_idx]
    s_sel = s_idx[mask]
    print(f"母集団: 週足30MA上・PDH>0・k={cli.k}  N(信号)={len(s_sel)}")

    grid(df, s_sel, atr_prev, years, "A", cli.k)
    grid(df, s_sel, atr_prev, years, "B", cli.k, stopk=2.0)

    print(f"\n実行コマンド: .venv/bin/python scratchpad/atr7b_exit_horizon.py --k {cli.k}"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
