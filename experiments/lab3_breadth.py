"""案3: 横断ブレッドス — STEP1 単体測定・冗長性スクリーン。

対象銘柄(h1、あるものだけ使う): xauusd, xagusd, usousd, nas100.r, ger40.r, us2000.r,
eurusd, gbpusd, audusd, nzdusd, usdcad, usdjpy, btcusd。
各銘柄を日足に落とし KAMA(14) が上向きか（breakout_wave.kama_adaptive、gate_kamaと同じ
因果シフト: 「前日終値までの情報で決まった、前日のKAMA上向き状態」を当日の状態として使う）。
breadth_t = 上向き比率（時点ごとに利用可能な銘柄が8未満ならその日は欠損）。

STEP1a 冗長性: 「breadth > 全期間中央値」と「BTC 4時間足KAMA(14)上向き」の日次一致率。
STEP1b 単体: breadth の五分位 → BTC h1 の先行 方向/量（H=24時間・72時間）。ブートストラップ付き。

実装注記:
  gold h1 は .loc["2018-01-01":] で切る（CLAUDE.md: 2018年以前は疎データ）。他銘柄はCLAUDE.md
  に疎データ警告が無いため全期間をそのまま使うが、下で日足換算のバー密度を出力して目視確認する。
  4時間足ゲートの日次への投影は、その日最後の4時間バーのゲート状態を採用（案4と同じ近似）。
  breadthのBTC h1への投影は、日次値をその日の全h1バーにffillする（値は前日終値までの情報で
  確定しているため先読みではない）。

Run:
  .venv/bin/python experiments/lab3_breadth.py --smoke
  .venv/bin/python experiments/lab3_breadth.py --full | tee experiments/out_lab3_breadth.txt
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import resample, kama_adaptive
from experiments.lab_common import (forward_direction, forward_move, quantile_labels,
                                    build_table, month_block_bootstrap, split_tables, fmt_table)

SYMBOLS = ["xauusd", "xagusd", "usousd", "nas100.r", "ger40.r", "us2000.r",
           "eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdjpy", "btcusd"]
DATA = "/home/angelbell/dev/auto-trade/data"
MIN_AVAIL = 8
H_LIST = [24, 72]
H_NAME = {24: "H=24本(24h)", 72: "H=72本(72h)"}


def daily_kama_rising(sym: str, smoke: bool) -> pd.Series:
    path = f"{DATA}/vantage_{sym}_h1.csv"
    d = load_mt5_csv(path)
    if sym == "xauusd":
        d = d.loc[GOLD_H1_START:]
    if smoke:
        d = d.tail(5000)
    dc = d["close"].resample("1D").last().dropna()
    kmg = kama_adaptive(dc, 14)
    rising = (kmg > kmg.shift(1)).shift(1)  # gate_kamaと同じ因果シフト
    bars_per_day = len(d) / max((d.index[-1] - d.index[0]).days, 1)
    print(f"    {sym:<10} h1本数={len(d):>7,}  日足換算本数/日={bars_per_day:.2f}  "
          f"span={d.index[0].date()}->{d.index[-1].date()}")
    return rising


def run(smoke: bool):
    print(f"\n{'='*70}\n案3 横断ブレッドス\n{'='*70}")
    print("  銘柄別データ密度:")
    ser = {}
    for sym in SYMBOLS:
        try:
            ser[sym] = daily_kama_rising(sym, smoke)
        except FileNotFoundError:
            print(f"    {sym}: ファイル無し（スキップ）")

    df = pd.DataFrame(ser)
    avail = df.notna().sum(axis=1)
    breadth = df.mean(axis=1, skipna=True)
    breadth = breadth.where(avail >= MIN_AVAIL)
    breadth = breadth.dropna()
    print(f"\n  breadth系列: n_days={len(breadth)}  (avail>={MIN_AVAIL}銘柄の日のみ)  "
          f"{breadth.index[0].date()} -> {breadth.index[-1].date()}")

    # --- BTC 4時間足KAMA(14)上向きゲート（案4と同じ組み方） ---
    dbtc = load_mt5_csv(f"{DATA}/vantage_btcusd_h1.csv")
    if smoke:
        dbtc = dbtc.tail(60000)
    d4 = resample(dbtc, "4h")
    kmg4 = kama_adaptive(d4["close"], 14)
    krise4 = (kmg4 > kmg4.shift(1)).shift(1)
    gate_daily = krise4.resample("1D").last()

    both = pd.DataFrame({"breadth": breadth, "ON": gate_daily}).dropna()
    both["ON"] = both["ON"].astype(bool)
    med = both["breadth"].median()
    agree = ((both["breadth"] > med) == both["ON"]).mean()
    rho, pval = spearmanr(both["breadth"], both["ON"].astype(int))
    print(f"\n  STEP1a 冗長性: breadth>中央値({med:.3f}) vs BTC4hKAMA上向き  n_days={len(both)}  "
          f"一致率={agree*100:.1f}%  スピアマンrho={rho:+.4f} (p={pval:.2e})")

    # --- STEP1b: breadth五分位 -> BTC h1 先行方向/量 ---
    h1 = load_mt5_csv(f"{DATA}/vantage_btcusd_h1.csv")
    if smoke:
        h1 = h1.tail(60000)
    breadth_h1 = breadth.reindex(h1.index, method="ffill")
    close = h1["close"]
    for H in H_LIST:
        dirH = forward_direction(close, H)
        moveH = forward_move(close, H)
        valid = breadth_h1.notna() & dirH.notna() & moveH.notna()
        labels = quantile_labels(breadth_h1[valid], 5).reindex(h1.index)
        table = build_table(labels, dirH, moveH)
        print(f"\n  -- STEP1b breadth五分位, {H_NAME[H]} --")
        print(fmt_table(table, "Q(breadth)"))
        boot = month_block_bootstrap(labels, dirH, moveH, top_label=5, bot_label=1)
        print(f"  Q5-Q1 方向差: 月次ブロック・ブートストラップ(n_boot={boot['n_boot']}, "
              f"月数={boot['n_months']}) 中央値={boot['diff_median']:+.5f} "
              f"95%CI=[{boot['diff_lo']:+.5f}, {boot['diff_hi']:+.5f}]")
        print(f"  Q5/Q1 動く量比:                  中央値={boot['ratio_median']:.3f} "
              f"95%CI=[{boot['ratio_lo']:.3f}, {boot['ratio_hi']:.3f}]")
        t_pre, t_post = split_tables(labels, dirH, moveH, "2022-01-01")
        print(f"  [pre]"); print(fmt_table(t_pre, "Q(breadth)"))
        print(f"  [post]"); print(fmt_table(t_post, "Q(breadth)"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    run(args.smoke or not args.full)


if __name__ == "__main__":
    main()
