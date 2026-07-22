"""What does widening the stop from L2 to L0 actually buy, and what does it cost?

The user's read of the chart is CORRECT: widening the stop raises the win rate from 23% to 33%.
Some of those stop-outs really were noise. But total R falls from +445 to +278. So the fix costs
more than it saves, and the interesting question is WHERE the money goes.

The arithmetic, made visible:
    meanR = win% * avg_win_R - (1 - win%) * 1
  A wider stop moves BOTH terms. It raises win%, but because R is measured in units of the stop
  distance, doubling the stop halves the R of every win. The question is only which effect is bigger.

Two decompositions:
  T1  The identity above, for each stop mode: win%, avg win in R, avg win in DOLLARS, and the stop
      distance in dollars. If the dollar-win is unchanged and only the denominator grew, the "fix"
      is not a fix -- it is a re-labelling that also forces a smaller position.
  T2  Trade-by-trade, on the SAME entries: of the trades the tight (L2) stop kills, how many would
      the wide (L0) stop have SAVED, and what would those saved trades have earned? Against that,
      how much R do the wide stop's smaller positions give up on the trades that won anyway?
Run: .venv/bin/python experiments/stop_widen_autopsy.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3,
       "rr": 4.5, "fwd": 500, "fill_win": 200}


def leg(**over):
    with contextlib.redirect_stderr(io.StringIO()):
        d = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d, SimpleNamespace(**{**CFG, **over}))
    t = t.copy()
    t["Rnet"] = t["R"].values - 15.0 / t["risk"].values
    return t


def line(t, tag):
    R = t["Rnet"].values
    w = R > 0
    win = w.mean()
    avg_win_R = R[w].mean()
    # R is in units of the stop distance -> dollars won = R * risk
    dollars_win = (R[w] * t["risk"].values[w]).mean()
    print(f"  {tag:<34}{len(t):>5}{100*win:>7.1f}%{avg_win_R:>11.2f}R"
          f"{np.median(t['risk'].values):>12.0f}${dollars_win:>14,.0f}${R.mean():>10.3f}"
          f"{R.sum():>10.0f}")


def main():
    print("T1 -- 恒等式で見る:  meanR = 勝率 × (1勝あたりR) − (1−勝率) × 1")
    print("     R は『損切り幅を1とした単位』。損切りを2倍にすれば、同じ値幅の勝ちも R では半分になる。\n")
    print(f"  {'損切りの置き方':<34}{'n':>5}{'勝率':>8}{'1勝あたりR':>12}"
          f"{'損切り幅':>12}{'1勝の金額':>15}{'meanR':>10}{'totR':>10}")
    base = leg()
    line(base, "L2（現行・波2の安値）")
    for k in (0.5, 1.0, 2.0):
        line(leg(sl_b="band", sl_b_k=k), f"L2 − {k}×ATR")
    o1 = leg(sl_b="origin")
    line(o1, "L0（波の起点）目標も遠のく")
    o2 = leg(sl_b="origin", tgt_ref="l2")
    line(o2, "L0 ＋ 目標は現行の価格のまま")

    print("\n\nT2 -- 同じエントリーで直接対決（時刻でマッチング）")
    b = base.set_index("time"); o = o2.set_index("time")
    common = b.index.intersection(o.index)
    B, O = b.loc[common], o.loc[common]
    print(f"  両方が建てた同一トレード: {len(common)}本")
    killed = B["Rnet"] <= -0.9                      # 現行では損切りされた
    saved = killed & (O["Rnet"] > 0)                # 広い損切りなら勝ちに転じた
    still = killed & (O["Rnet"] <= 0)               # 広くしても結局負け（しかも損は同じ−1R）
    wonB = B["Rnet"] > 0                            # 現行で勝っていた
    print(f"\n  【広げて助かった分】現行で損切り → L0なら勝ち : {saved.sum():>4}本  "
          f"獲得 {O['Rnet'][saved].sum():>+8.1f}R   （現行では {B['Rnet'][saved].sum():+.1f}R）"
          f"  差引 {O['Rnet'][saved].sum() - B['Rnet'][saved].sum():+.1f}R")
    print(f"  【広げても助からない】現行で損切り → L0でも負け : {still.sum():>4}本  "
          f"{O['Rnet'][still].sum():>+8.1f}R  （現行では {B['Rnet'][still].sum():+.1f}R）"
          f"  差引 {O['Rnet'][still].sum() - B['Rnet'][still].sum():+.1f}R")
    print(f"  【勝ちトレードの目減り】現行で勝ち → L0だと玉が小さい: {wonB.sum():>4}本  "
          f"{O['Rnet'][wonB].sum():>+8.1f}R  （現行では {B['Rnet'][wonB].sum():+.1f}R）"
          f"  差引 {O['Rnet'][wonB].sum() - B['Rnet'][wonB].sum():+.1f}R")
    d = O["Rnet"].sum() - B["Rnet"].sum()
    print(f"\n  → 同一トレードでの総差引: {d:+.1f}R  "
          f"({'広げた方が良い' if d > 0 else '広げると損'})")
    print(f"     さらに、広げると建てられる本数が減る（占有時間が延びるため）: "
          f"{len(base)} → {len(o2)} 本")

    print("\n\nT3 -- 「刈られていた」の実態: 現行で損切りされた本のうち、L0 なら生き残ったのは何%か")
    print(f"  現行の損切り {killed.sum()} 本のうち、L0 で勝ちに転じたのは {saved.sum()} 本 "
          f"= **{100*saved.sum()/max(killed.sum(),1):.0f}%**")
    print(f"  残り {100*still.sum()/max(killed.sum(),1):.0f}% は、損切りを広げても結局負けた"
          f"（＝刈られたのではなく、方向が間違っていた）")


if __name__ == "__main__":
    main()
