"""STEP3: 生き残った候補（usdjpy long, eurusd long, gbpusd long, BTC）の年別内訳・
ランダム間引き帰無・BTCとの年別R相関。凍結仕様のまま、調整は一切しない。"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from atr_spike_transplant_step2 import (atr_prev_of, raw_triggers, build_pdh_dist,
                                         build_entries, run_cell, stats, per_year_rows,
                                         pf_of, drop_null)
from atr_spike_transplant_step2_run import load_side, cost_frac_for

K = 2.0  # 代表値として k=2.0（凍結カードの主参照値）で年別・帰無・相関を見る


def build_cell(name, direction, k, filt=True):
    d, C = load_side(name, direction)
    atr_prev = atr_prev_of(d)
    s_idx = raw_triggers(d, atr_prev, k)
    if direction == "long":
        pf = 0.0
        if filt:
            pdh = build_pdh_dist(d, atr_prev)
            s_idx = s_idx[pdh[s_idx] > 0.0]
    else:
        pf = 0.5
    entries = build_entries(d, atr_prev, s_idx, rr=1000.0)
    price_ref = (C - d["close"]) if C is not None else d["close"]
    median_price = float(price_ref.median())
    cost_frac, _ = cost_frac_for(name, median_price, 1.0)
    t = run_cell(d, entries, pf, cost_frac, C=C)
    return d, t


def base_population(name, direction):
    """帰無の元＝k=0（フィルタ無し、pfは同じ）の母集団。"""
    return build_cell(name, direction, 0.0, filt=False)


print("=" * 100)
print(f"STEP3: 生存候補の年別内訳・ランダム間引き帰無（k={K}固定・母集団はk=0フィルタ無し）")
print("=" * 100)

CANDIDATES = [("usdjpy", "long"), ("eurusd", "long"), ("gbpusd", "long"), ("btcusd", "long")]
yearly_R = {}

for name, direction in CANDIDATES:
    d, t = build_cell(name, direction, K, filt=True)
    if t is None:
        print(f"\n{name}/{direction}: トレード無し")
        continue
    span = (d.index[-1] - d.index[0]).days / 365.25
    s = stats(t, span)
    print(f"\n### {name} / {direction} k={K} (凍結仕様)")
    print(f"  全体: N={s['N']} N/年={s['N_yr']:.1f} 勝率={s['win']:.1f}% PF={s['PF']:.2f} "
          f"平均%={s['mean_pct']:+.3f} maxDD%={s['maxDD_pct']:.1f}")
    print(f"  {'年':>6} {'N':>5} {'勝率':>6} {'PF':>6} {'平均%':>8} {'総%':>8}")
    for row in per_year_rows(t):
        pf_s = f"{row['PF']:.2f}" if np.isfinite(row["PF"]) else "inf"
        print(f"  {row['year']:6d} {row['N']:5d} {row['win']:5.1f}% {pf_s:>6} "
              f"{row['mean_pct']:+8.3f} {row['tot_pct']:+8.1f}")
    yearly_R[f"{name}_{direction}"] = t.groupby("y")["pnl_pct"].sum()

    # ランダム間引き帰無: k=0(フィルタ無し)の母集団から同数だけランダムに残す
    _, t0 = base_population(name, direction)
    if t0 is not None and len(t0) >= s["N"]:
        nl = drop_null(t0["pnl_pct"].to_numpy(), s["N"], s["mean_pct"] / 100, s["PF"], reps=400)
        print(f"  帰無(k=0母集団からのランダム間引き, N={s['N']}, 400回): "
              f"PF%ile={nl['pf_pctile']:.1f}  平均%ile={nl['mean_pctile']:.1f}")
    else:
        print(f"  帰無: 母集団不足(N0={0 if t0 is None else len(t0)} < N={s['N']})でスキップ")

print("\n" + "=" * 100)
print("BTC との年別R相関（構造法則6＝エッジと独立性のトレードオフの確認）")
print("=" * 100)
btc_yr = yearly_R.get("btcusd_long")
for key, yr in yearly_R.items():
    if key == "btcusd_long" or btc_yr is None:
        continue
    joined = pd.concat([btc_yr.rename("btc"), yr.rename("other")], axis=1, join="inner")
    if len(joined) >= 3:
        corr = joined["btc"].corr(joined["other"])
        print(f"  {key} vs btcusd_long: 共通年={len(joined)} 相関={corr:+.3f}")
    else:
        print(f"  {key} vs btcusd_long: 共通年={len(joined)} (不足)")
