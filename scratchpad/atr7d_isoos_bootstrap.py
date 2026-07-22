"""【第7段・掃引 締め】IS(2000-2012)/OOS(2013-2026)分割 + 巡回ブロック・ブートストラップ(1/3/6/12か月)。
2候補とも 母集団=週足30MA上・PDH>0固定・fill_win=200・fwd=20 の最終形で評価する。
  候補1: k2.0 B系(stopk2.0) ATR×2トレール
  候補2: k2.5 A系(拡大足の反対端) ATR×3トレール
「ゲート無し版」も同じ設定で並べる（法則7.5点検＝ゲートがなお正しいか）。
"""
SCREEN = "atr_spike_btc_h1"
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr7_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell, stats,
                          span_years, fmt_row, weekly_up_regime, build_pdh_dist_series,
                          block_bootstrap)  # noqa: E402

CANDS = [
    dict(name="候補1: k2.0 B系(2.0) TR2 fwd20 週足30MA上 PDH>0", k=2.0, system="B", stopk=2.0,
         val=3.0 if False else 2.0, fwd=20, gate=True),
    dict(name="候補1・ゲート無し比較", k=2.0, system="B", stopk=2.0, val=2.0, fwd=20, gate=False),
    dict(name="候補2: k2.5 A系 TR3 fwd20 週足30MA上 PDH>0", k=2.5, system="A", stopk=2.0,
         val=3.0, fwd=20, gate=True),
    dict(name="候補2・ゲート無し比較", k=2.5, system="A", stopk=2.0, val=3.0, fwd=20, gate=False),
]


def build_trades(df, k, system, stopk, val, fwd, gate):
    atr_prev = atr_prev_of(df)
    s_idx = raw_triggers(df, atr_prev, k)
    pdh = build_pdh_dist_series(df, atr_prev)
    mask = pdh[s_idx] > 0.0
    if gate:
        wu = weekly_up_regime(df)
        mask &= wu[s_idx]
    s_sel = s_idx[mask]
    ent = build_entries(df, atr_prev, s_sel, system, 0.0, stopk=stopk, trail=True)
    return run_cell(df, ent, fill_win=200, fwd=fwd, trail_atr=val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2015-12-31"]

    for cand in CANDS:
        print("\n" + "=" * 110)
        print(f"##### {cand['name']} #####")
        print("=" * 110)
        t_full = build_trades(df, cand["k"], cand["system"], cand["stopk"], cand["val"],
                               cand["fwd"], cand["gate"])
        if t_full is None:
            print("  約定0件")
            continue
        s_full = stats(t_full, span_years(df))
        print("  " + fmt_row("全期間", s_full))

        for a0, a1, lab in (("2000-01-01", "2012-12-31", "IS 2000-2012"),
                             ("2013-01-01", None, "OOS 2013-2026")):
            dsub = df.loc[a0:a1] if a1 else df.loc[a0:]
            t_sub = build_trades(dsub, cand["k"], cand["system"], cand["stopk"], cand["val"],
                                  cand["fwd"], cand["gate"])
            if t_sub is None:
                print(f"  {lab}: 約定不足")
                continue
            s_sub = stats(t_sub, span_years(dsub))
            print("  " + fmt_row(lab, s_sub))

        if s_full["N"] >= 20:
            for metric in ("mean", "pf"):
                bb = block_bootstrap(t_full, [1, 3, 6, 12], metric=metric)
                label = "平均%" if metric == "mean" else "PF"
                print(f"  ブロックbootstrap({label}): " +
                      " / ".join(f"{k}mo中央値{v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]n={v[3]}"
                                 for k, v in bb.items()))
            # maxDDのブートストラップ中央値も出す(法則8: 実測1本の経路を鵜呑みにしない)
            rng = np.random.default_rng(20260722)
            p = t_full["pnl_pct"].to_numpy()
            n = len(p)
            dds = []
            for _ in range(1000):
                samp = rng.choice(p, size=n, replace=True)
                eq = np.cumsum(samp)
                dds.append(float((np.maximum.accumulate(eq) - eq).max()) * 100)
            dds = np.array(dds)
            print(f"  maxDDブートストラップ(iid resample, 参考): 中央値={np.median(dds):.2f}% "
                  f"[{np.percentile(dds,2.5):.2f},{np.percentile(dds,97.5):.2f}] "
                  f"実測={s_full['maxDD_pct']:.2f}%")

    print(f"\n実行コマンド: .venv/bin/python scratchpad/atr7d_isoos_bootstrap.py{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
