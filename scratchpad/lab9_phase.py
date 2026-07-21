"""案9: 4時間足内の位相 — STEP1 単体測定・冗長性スクリーン。

X1_s = その15分バーが属する4時間足内の経過本数（0..15、ブローカー時刻 0,4,8,12,16,20時区切り）。
X2_s = (close[s] - その4時間足の始値) / (その4時間足のここまでの高値-安値)（レンジ0なら欠損）。

対象: BTC 15分（主）、gold 15分（副、.loc["2018-01-01":]）。

実装注記:
  X1は16カテゴリのまま（分位切りしない）で方向/量の平均±seを表にする（仕様どおり）。
  X1にはブロック・ブートストラップの明示的な指定が無いが、報告形式の一般要件
  （案ごとにブートストラップ95%区間を出す）に合わせ、全期間平均が最大のカテゴリと
  最小のカテゴリの「差」を月次ブロック・ブートストラップで区間推定する（仕様を超える
  補足統計として明記）。
  X2は仕様どおり共通ハーネスの五分位に通す。

Run:
  .venv/bin/python scratchpad/lab9_phase.py --smoke
  .venv/bin/python scratchpad/lab9_phase.py --full | tee scratchpad/out_lab9_phase.txt
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.lab_common import (forward_direction, forward_move, quantile_labels,
                                    build_table, month_block_bootstrap, split_tables, fmt_table)

H_LIST = [16, 96]
H_NAME = {16: "H=16本(4h)", 96: "H=96本(24h)"}


def build_X1_X2(d: pd.DataFrame):
    # floor("4h") aligns exactly to 0,4,8,12,16,20 broker-time boundaries (index is
    # already tagged as UTC = broker server time), giving a unique bucket key per block.
    bkt = d.index.floor("4h")
    g = d.groupby(bkt)
    X1 = g.cumcount()
    X1.index = d.index
    open4 = g["open"].transform("first")
    hi_so_far = g["high"].cummax()
    lo_so_far = g["low"].cummin()
    rng = hi_so_far - lo_so_far
    X2 = (d["close"] - open4) / rng
    X2 = X2.where(rng > 0)
    return X1, X2


def month_block_bootstrap_cat(X1, dir_, move_, top_cat, bot_cat, n_boot=1000, seed=0):
    labels = X1.copy()
    return month_block_bootstrap(labels, dir_, move_, top_cat, bot_cat, n_boot=n_boot, seed=seed)


def run_symbol(name: str, path: str, start: str | None, smoke: bool):
    d = load_mt5_csv(path)
    if start:
        d = d.loc[start:]
    if smoke:
        d = d.tail(3000)
    print(f"\n{'='*70}\n案9 4H内位相: {name}  bars={len(d):,}  {d.index[0]} -> {d.index[-1]}\n{'='*70}")

    X1, X2 = build_X1_X2(d)
    close = d["close"]

    for H in H_LIST:
        dirH = forward_direction(close, H)
        moveH = forward_move(close, H)
        move_all_mean = moveH.dropna().mean()

        print(f"\n  -- X1 (0..15本目, {H_NAME[H]}) 16カテゴリ表 --")
        t1 = build_table(X1, dirH, moveH)
        print(fmt_table(t1, "位相"))
        if len(t1) > 1:
            best = t1["mean"].idxmax(); worst = t1["mean"].idxmin()
            boot1 = month_block_bootstrap_cat(X1, dirH, moveH, best, worst)
            print(f"  [補足] 最大カテゴリ({best})-最小カテゴリ({worst}) 方向差: "
                  f"月次ブロック・ブートストラップ(n_boot={boot1['n_boot']}, 月数={boot1['n_months']}) "
                  f"中央値={boot1['diff_median']:+.5f} 95%CI=[{boot1['diff_lo']:+.5f}, {boot1['diff_hi']:+.5f}]")

        valid2 = X2.notna() & dirH.notna() & moveH.notna()
        labels2 = quantile_labels(X2[valid2], 5).reindex(d.index)
        t2 = build_table(labels2, dirH, moveH)
        print(f"\n  -- X2 (バー内位置, {H_NAME[H]}) 五分位表 --")
        print(fmt_table(t2, "Q(pos)"))
        boot2 = month_block_bootstrap(labels2, dirH, moveH, top_label=5, bot_label=1)
        print(f"  Q5-Q1 方向差: 月次ブロック・ブートストラップ(n_boot={boot2['n_boot']}, "
              f"月数={boot2['n_months']}) 中央値={boot2['diff_median']:+.5f} "
              f"95%CI=[{boot2['diff_lo']:+.5f}, {boot2['diff_hi']:+.5f}]")
        print(f"  Q5/Q1 動く量比:                  中央値={boot2['ratio_median']:.3f} "
              f"95%CI=[{boot2['ratio_lo']:.3f}, {boot2['ratio_hi']:.3f}]")

        t2_pre, t2_post = split_tables(labels2, dirH, moveH, "2022-01-01")
        print(f"\n  -- X2 前後分割（2022-01-01, {H_NAME[H]}） --")
        print("  [pre]")
        print(fmt_table(t2_pre, "Q(pos)"))
        print("  [post]")
        print(fmt_table(t2_post, "Q(pos)"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    smoke = args.smoke or not args.full

    run_symbol("BTC15m", "/home/angelbell/dev/auto-trade/data/vantage_btcusd_m15.csv", None, smoke)
    run_symbol("gold15m", "/home/angelbell/dev/auto-trade/data/vantage_xauusd_m15.csv", "2018-01-01", smoke)


if __name__ == "__main__":
    main()
