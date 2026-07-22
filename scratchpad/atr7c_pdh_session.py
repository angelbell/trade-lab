"""【第7段・掃引 軸5,6】USDJPY h1 ATR拡大足ロング、母集団=週足30MA上(主)。
軸5: 前日高値フィルタ on/off + 閾値{-1.0,-0.5,0.0}（丸い絶対値、パーセンタイルで決めない）。
軸6: セッション(アジア/ロンドン/NY) + ロンドン/NYオープン前後。
2つの出口候補を両方通す:
  候補1(主): k2.0 B系(stopk2.0) ATR×2トレール fwd20  ―― 前段7bで台地・最良DD比だった設定
  候補2(副): k2.5 A系 ATR×3トレール fwd20            ―― 前段7aでPF最良だった設定(N薄い)
帰無=同じ通過率のランダム間引き400回（ゲート自体は維持したまま、フィルタ層だけを間引く）。
"""
SCREEN = "atr_spike_btc_h1"
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, fmt_row, weekly_up_regime, build_pdh_dist_series,
                          session_hour, drop_null, SESSIONS, OPEN_WINDOWS)  # noqa: E402

CANDS = [
    dict(name="候補1: k2.0 B系(2.0) TR2 fwd20", k=2.0, system="B", stopk=2.0, exit="trail",
         val=2.0, fwd=20),
    dict(name="候補2: k2.5 A系 TR3 fwd20", k=2.5, system="A", stopk=2.0, exit="trail",
         val=3.0, fwd=20),
]


def build_pop(df, atr_prev, k, pdh_thresh=0.0):
    s_idx = raw_triggers(df, atr_prev, k)
    wu = weekly_up_regime(df)
    pdh = build_pdh_dist_series(df, atr_prev)
    mask = (pdh[s_idx] > pdh_thresh) & wu[s_idx]
    return s_idx[mask]


def run_one(df, s_sel, atr_prev, years, system, stopk, exit_kind, val, fwd):
    if exit_kind == "trail":
        ent = build_entries(df, atr_prev, s_sel, system, 0.0, stopk=stopk, trail=True)
        t = run_cell(df, ent, fill_win=200, fwd=fwd, trail_atr=val)
    else:
        ent = build_entries(df, atr_prev, s_sel, system, val, stopk=stopk, trail=False)
        t = run_cell(df, ent, fill_win=200, fwd=fwd)
    return t, (stats(t, years) if t is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2015-12-31"]
    years = span_years(df)
    wu = weekly_up_regime(df)

    for cand in CANDS:
        k, system, stopk, exit_kind, val, fwd = (cand["k"], cand["system"], cand["stopk"],
                                                   cand["exit"], cand["val"], cand["fwd"])
        atr_prev = atr_prev_of(df)
        s_idx_all = raw_triggers(df, atr_prev, k)
        pdh_full = build_pdh_dist_series(df, atr_prev)
        s_gated = s_idx_all[wu[s_idx_all]]        # ゲートのみ(PDHフィルタ前)=帰無母集団のプール

        print("\n" + "=" * 118)
        print(f"##### {cand['name']} #####")
        print("=" * 118)

        # ---- 軸5: PDHフィルタ ----
        print("\n-- 軸5: 前日高値フィルタ --")
        t_nofilter, s_nofilter = run_one(df, s_gated, atr_prev, years, system, stopk, exit_kind, val, fwd)
        if s_nofilter:
            print("  " + fmt_row("フィルタ無し(ゲートのみ)", s_nofilter))
        pool_pct = t_nofilter["pnl_pct"].to_numpy() if t_nofilter is not None else None
        for th in (-1.0, -0.5, 0.0):
            mask = pdh_full[s_gated] > th
            s_sel = s_gated[mask]
            t, s = run_one(df, s_sel, atr_prev, years, system, stopk, exit_kind, val, fwd)
            if s is None:
                print(f"  PDH>{th:+.1f} 約定不足")
                continue
            nl = drop_null(pool_pct, s["N"], s["mean_pct"] / 100, s["PF"]) if pool_pct is not None else {}
            print("  " + fmt_row(f"PDH>{th:+.1f} (通過{mask.mean()*100:.1f}%)", s,
                                  nl.get("pf_pct"), nl.get("mean_pct")))

        # ---- 軸6: セッション ----
        print("\n-- 軸6: セッション（PDH>0固定） --")
        s_base = s_gated[pdh_full[s_gated] > 0.0]
        t_base, s_stats_base = run_one(df, s_base, atr_prev, years, system, stopk, exit_kind, val, fwd)
        pool_sess = t_base["pnl_pct"].to_numpy() if t_base is not None else None
        hours_base = session_hour(df, s_base)
        for lab, (lo, hi) in {**SESSIONS, **OPEN_WINDOWS}.items():
            m = (hours_base >= lo) & (hours_base < hi)
            s_sel = s_base[m]
            t, s = run_one(df, s_sel, atr_prev, years, system, stopk, exit_kind, val, fwd)
            if s is None:
                print(f"  {lab} 約定不足")
                continue
            nl = drop_null(pool_sess, s["N"], s["mean_pct"] / 100, s["PF"]) if pool_sess is not None else {}
            # 年別安定性: プラス年比率
            print("  " + fmt_row(f"{lab} (通過{m.mean()*100:.1f}%)", s,
                                  nl.get("pf_pct"), nl.get("mean_pct")))

    print(f"\n実行コマンド: .venv/bin/python scratchpad/atr7c_pdh_session.py{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
