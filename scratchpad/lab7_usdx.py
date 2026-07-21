"""案7: USDX で gold を条件付け — STEP1 単体測定・冗長性スクリーン。

vantage_usdx.r_h1.csv -> 日足KAMA(14)の向き（上向き/下向き、falling判定は
src/engine/gates.exit_flip と同じ因果定義: (km<km.shift(1)).shift(1)）。

STEP1a 冗長性: USDX日足KAMA下向き と gold の「日足SMA150上向きゲート」
（src.engine.gates.gate_sma をそのままimport、gold_bo/gold15mと同じ既存定義:
daily_sma=150, daily_slope_k=10）の日次一致率。
STEP1b: USDXゲート2状態でgold h1の先行方向/量（H=24時間・72時間）を層別。ブートストラップ付き。

USDXデータ: 開始日・本数はスクリプト出力に明記（.venv実行時に印字）。

実装注記:
  gold h1 は .loc["2018-01-01":] で切る（CLAUDE.md規則）。USDXの開始日がそれより後なら、
  実効的な重なり期間はUSDX開始日以降になる（下の出力で明記）。

Run:
  .venv/bin/python scratchpad/lab7_usdx.py --smoke
  .venv/bin/python scratchpad/lab7_usdx.py --full | tee scratchpad/out_lab7_usdx.txt
"""
import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import kama_adaptive
from src.engine.gates import gate_sma
from scratchpad.lab_common import (forward_direction, forward_move, build_table,
                                    month_block_bootstrap, fmt_table)

DATA = "/home/angelbell/dev/auto-trade/data"
H_LIST = [24, 72]
H_NAME = {24: "H=24本(24h)", 72: "H=72本(72h)"}


def run(smoke: bool):
    print(f"\n{'='*70}\n案7 USDXでgoldを条件付け\n{'='*70}")

    dusdx = load_mt5_csv(f"{DATA}/vantage_usdx.r_h1.csv")
    print(f"  USDXデータ: {len(dusdx):,}本(h1)  {dusdx.index[0]} -> {dusdx.index[-1]}")
    if smoke:
        dusdx = dusdx.tail(8000)

    dc = dusdx["close"].resample("1D").last().dropna()
    kmg = kama_adaptive(dc, 14)
    usdx_down = (kmg < kmg.shift(1)).shift(1)  # exit_flipと同じ因果的falling判定

    dgold = load_mt5_csv(f"{DATA}/vantage_xauusd_h1.csv").loc[GOLD_H1_START:]
    if smoke:
        dgold = dgold.tail(8000)
    args = SimpleNamespace(daily_sma=150, daily_slope_k=10, gate_tf="1D", ext_cap=0)
    reg, _ = gate_sma(dgold, args)
    reg_series = pd.Series(reg, index=dgold.index)
    gold_gate_daily = reg_series.resample("1D").last()

    both = pd.DataFrame({"usdx_down": usdx_down, "gold_up": gold_gate_daily}).dropna()
    both["usdx_down"] = both["usdx_down"].astype(bool)
    both["gold_up"] = both["gold_up"].astype(bool)
    agree = (both["usdx_down"] == both["gold_up"]).mean()
    print(f"\n  STEP1a 冗長性: USDX日足KAMA下向き vs gold日足SMA150上向きゲート  "
          f"n_days={len(both)}  一致率(仕様どおり文字通りの一致: 両方True or 両方False)={agree*100:.1f}%")
    print(f"  参考: USDX下向きの割合={both['usdx_down'].mean()*100:.1f}%  "
          f"goldゲートONの割合={both['gold_up'].mean()*100:.1f}%  "
          f"（USDX下向き==goldゲートOFFで見た一致率={(both['usdx_down']==~both['gold_up']).mean()*100:.1f}%, "
          f"「ドル安→金上昇」仮説の向きの参考値）")

    # --- STEP1b: USDXゲート2状態 -> gold h1 先行方向/量 ---
    usdx_down_h1 = usdx_down.reindex(dgold.index, method="ffill")
    close = dgold["close"]
    for H in H_LIST:
        dirH = forward_direction(close, H)
        moveH = forward_move(close, H)
        valid = usdx_down_h1.notna() & dirH.notna() & moveH.notna()
        labels = usdx_down_h1[valid].map({True: "USDX下向き", False: "USDX上向き"})
        labels = labels.reindex(dgold.index)
        table = build_table(labels, dirH, moveH)
        print(f"\n  -- USDXゲート2状態, gold {H_NAME[H]} --")
        print(fmt_table(table, "USDX"))
        boot = month_block_bootstrap(labels, dirH, moveH,
                                      top_label="USDX下向き", bot_label="USDX上向き")
        print(f"  (USDX下向き)-(USDX上向き) 方向差: 月次ブロック・ブートストラップ"
              f"(n_boot={boot['n_boot']}, 月数={boot['n_months']}) 中央値={boot['diff_median']:+.5f} "
              f"95%CI=[{boot['diff_lo']:+.5f}, {boot['diff_hi']:+.5f}]")
        print(f"  動く量比(下向き/上向き):          中央値={boot['ratio_median']:.3f} "
              f"95%CI=[{boot['ratio_lo']:.3f}, {boot['ratio_hi']:.3f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    run(args.smoke or not args.full)


if __name__ == "__main__":
    main()
