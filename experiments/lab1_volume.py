"""案1: ブレイク足の出来高（tick_volume）— STEP1 単体測定・冗長性スクリーン。

X_s = tick_volume[s] / 「過去20営業日の同じ時刻(HH:MM)のtick_volumeの中央値」（因果的ローリング）。
対象: BTC 15分（2018-）、gold 15分（.loc["2018-01-01":]）。

実装注記（近似）:
  「過去20営業日」を、そのHH:MMの過去20回の"出現"（groupby(HH:MM)でのローリング）で近似した。
  Vantage feedは週末に自然に休むため、24時間連続取引のBTCでは出現回数≒暦日にほぼ一致するが、
  祝日等をまたぐ銘柄では厳密な「営業日」カウントとは僅かにズレ得る（許容範囲内と判断）。
  中央値windowはmin_periods=10（起動直後の高分散推定を避けるため）。

先行方向/量の共通ハーネス（H=16本=4時間, 96本=24時間）で五分位を測る。
追加: X と (バーレンジ/ATR14) のスピアマン順位相関を1行。

Run:
  .venv/bin/python experiments/lab1_volume.py --smoke
  .venv/bin/python experiments/lab1_volume.py --full | tee experiments/out_lab1_volume.txt
"""
import argparse
import sys

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import spearmanr

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.lab_common import (forward_direction, forward_move, quantile_labels,
                                    build_table, month_block_bootstrap, split_tables, fmt_table)

H_LIST = [16, 96]
H_NAME = {16: "H=16本(4h)", 96: "H=96本(24h)"}


def build_X(d: pd.DataFrame) -> pd.Series:
    hhmm = d.index.strftime("%H:%M")
    vol = d["volume"]
    df = pd.DataFrame({"vol": vol.values, "hhmm": hhmm}, index=d.index)
    med20 = df.groupby("hhmm")["vol"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    X = df["vol"] / med20
    X.index = d.index
    return X.replace([np.inf, -np.inf], np.nan)


def run_symbol(name: str, path: str, start: str | None, smoke: bool):
    d = load_mt5_csv(path)
    if start:
        d = d.loc[start:]
    if smoke:
        d = d.tail(3000)
    print(f"\n{'='*70}\n案1 出来高比: {name}  bars={len(d):,}  {d.index[0]} -> {d.index[-1]}\n{'='*70}")

    X = build_X(d)
    a = ta.atr(d["high"], d["low"], d["close"], length=14)
    rng_atr = (d["high"] - d["low"]) / a
    valid = X.notna() & rng_atr.notna()
    rho, pval = spearmanr(X[valid], rng_atr[valid])
    print(f"  スピアマン順位相関 X vs バーレンジ/ATR14: rho={rho:+.4f}  p={pval:.2e}  n={int(valid.sum())}")

    close = d["close"]
    for H in H_LIST:
        dirH = forward_direction(close, H)
        moveH = forward_move(close, H)
        valid2 = X.notna() & dirH.notna() & moveH.notna()
        labels = quantile_labels(X[valid2], 5)
        labels = labels.reindex(d.index)
        table = build_table(labels, dirH, moveH)
        print(f"\n  -- {H_NAME[H]} 五分位表（全期間） --")
        print(fmt_table(table, "Q(vol)"))

        boot = month_block_bootstrap(labels, dirH, moveH, top_label=5, bot_label=1)
        print(f"  Q5-Q1 方向差: 月次ブロック・ブートストラップ(n_boot={boot['n_boot']}, "
              f"月数={boot['n_months']}) 中央値={boot['diff_median']:+.5f} "
              f"95%CI=[{boot['diff_lo']:+.5f}, {boot['diff_hi']:+.5f}]")
        print(f"  Q5/Q1 動く量比:                  中央値={boot['ratio_median']:.3f} "
              f"95%CI=[{boot['ratio_lo']:.3f}, {boot['ratio_hi']:.3f}]")

        t_pre, t_post = split_tables(labels, dirH, moveH, "2022-01-01")
        print(f"\n  -- {H_NAME[H]} 前後分割（2022-01-01 境界、ラベルは全期間固定） --")
        print("  [pre]")
        print(fmt_table(t_pre, "Q(vol)"))
        print("  [post]")
        print(fmt_table(t_post, "Q(vol)"))


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
