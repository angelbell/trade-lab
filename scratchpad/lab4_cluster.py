"""案4: シグナル発生の群発度 — STEP1 単体測定・冗長性スクリーン。

シグナル = btc15m_L と同じ検出器（ZigZag 2×ATR・Pattern-B）の「ゲート無し」ブレイク確定
イベント（src.engine.detect.make_swings + pattern_b をそのままimport、execution/gateは通さず
時刻列だけを取る）。パラメータは src/engine/presets.BASE の swing/zz_k/atr/bo_window/wave に合わせた
（gate_kama等のゲート系はここでは適用しない＝仕様どおり「ゲート無し」）。

X = 各イベント時点での「過去14日間の同イベント数」（当該イベント自身を含まない、因果的）。
イベント集合をXの四分位で層別し、各層の先行 方向/量（H=96本=24時間）を表にする。

冗長性: Xを日次平均に落とした系列（イベントがある日だけの、その日の平均X）と、
「4時間足KAMA(14)上向きゲート」のON/OFF（BASE/book.pyのbtc_bo_kamaと同じ
gate_kama定義: kama>kama.shift(1) を1本シフトした因果版、日次には各日最後の4hバー値を採用）
の相関・一致率（X>全期間中央値 vs ON、日次）を1行。

対象: BTC 15分のみ（案の指定どおり）。

Run:
  .venv/bin/python scratchpad/lab4_cluster.py --smoke
  .venv/bin/python scratchpad/lab4_cluster.py --full | tee scratchpad/out_lab4_cluster.txt
"""
import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import spearmanr

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, kama_adaptive
from src.engine.detect import make_swings, pattern_b
from src.engine.presets import BASE
from scratchpad.lab_common import (forward_direction, forward_move, quantile_labels,
                                    build_table, month_block_bootstrap, fmt_table)

H = 96  # 24時間, 案4指定はH=96本のみ


def get_events(d: pd.DataFrame) -> pd.DatetimeIndex:
    args = SimpleNamespace(**{**BASE, "trend_ema": 0})  # trend_ema=0 => es=None => ゲート無し
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    sw = make_swings(h, l, c, a, args)
    setups = pattern_b(c, l, a, None, sw, args)
    idx = sorted(set(e_i for (e_i, pH1, pL0, pL2, iL0) in setups))
    return d.index[idx]


def past14d_count(events: pd.DatetimeIndex) -> pd.Series:
    ev = events.asi8
    win = pd.Timedelta(days=14).value
    out = np.zeros(len(ev), dtype=int)
    j0 = 0
    for i in range(len(ev)):
        while ev[j0] < ev[i] - win:
            j0 += 1
        out[i] = i - j0  # count of prior events in [t-14d, t), excluding self
    return pd.Series(out, index=events)


def run(smoke: bool):
    d = load_mt5_csv("/home/angelbell/dev/auto-trade/data/vantage_btcusd_m15.csv")
    if smoke:
        d = d.tail(20000)  # need enough history for 14d window + 24h forward + swing warm-up
    print(f"\n{'='*70}\n案4 シグナル群発度: BTC15m  bars={len(d):,}  {d.index[0]} -> {d.index[-1]}\n{'='*70}")

    events = get_events(d)
    print(f"  ゲート無しPattern-Bイベント数: {len(events)}  最初={events[0] if len(events) else None}  "
          f"最後={events[-1] if len(events) else None}")

    X = past14d_count(events)
    close = d["close"]
    dirH_full = forward_direction(close, H)
    moveH_full = forward_move(close, H)
    dirH = dirH_full.reindex(events)
    moveH = moveH_full.reindex(events)

    valid = X.notna() & dirH.notna() & moveH.notna()
    labels = quantile_labels(X[valid], 4)
    dirH_v = dirH[valid]; moveH_v = moveH[valid]

    table = build_table(labels, dirH_v, moveH_v)
    print(f"\n  -- X(過去14日イベント数) 四分位表, H=96本(24h) --")
    print(fmt_table(table, "Q(群発)"))

    q_top = int(labels.max()); q_bot = int(labels.min())
    boot = month_block_bootstrap(labels, dirH_v, moveH_v, top_label=q_top, bot_label=q_bot)
    print(f"  Q{q_top}-Q{q_bot} 方向差: 月次ブロック・ブートストラップ(n_boot={boot['n_boot']}, "
          f"月数={boot['n_months']}) 中央値={boot['diff_median']:+.5f} "
          f"95%CI=[{boot['diff_lo']:+.5f}, {boot['diff_hi']:+.5f}]")
    print(f"  Q{q_top}/Q{q_bot} 動く量比:      中央値={boot['ratio_median']:.3f} "
          f"95%CI=[{boot['ratio_lo']:.3f}, {boot['ratio_hi']:.3f}]")

    # 前後分割
    pre = labels.index < "2022-01-01"
    t_pre = build_table(labels[pre], dirH_v[pre], moveH_v[pre])
    t_post = build_table(labels[~pre], dirH_v[~pre], moveH_v[~pre])
    print(f"\n  -- 前後分割（2022-01-01, ラベルは全期間固定） --")
    print("  [pre]"); print(fmt_table(t_pre, "Q(群発)"))
    print("  [post]"); print(fmt_table(t_post, "Q(群発)"))

    # --- 冗長性: 日次平均X vs 4時間足KAMA(14)上向きゲート ---
    Xday = X.groupby(X.index.floor("D")).mean()  # イベントがある日だけ、その日の平均X
    d4 = resample(d, "4h")
    kmg = kama_adaptive(d4["close"], 14)
    krise = (kmg > kmg.shift(1)).shift(1)  # gate_kamaと同じ因果シフト（既存定義）
    gate_daily = krise.resample("1D").last()  # その日最後の4hバーのゲート状態

    both = pd.DataFrame({"X": Xday, "ON": gate_daily}).dropna()
    both["ON"] = both["ON"].astype(bool)
    rho, pval = spearmanr(both["X"], both["ON"].astype(int))
    med = both["X"].median()
    agree = (( both["X"] > med) == both["ON"]).mean()
    print(f"\n  冗長性: 日次平均X vs 4hKAMA(14)上向きゲート  n_days={len(both)}  "
          f"スピアマンrho={rho:+.4f} (p={pval:.2e})  "
          f"一致率(X>中央値 vs ON)={agree*100:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    run(args.smoke or not args.full)


if __name__ == "__main__":
    main()
