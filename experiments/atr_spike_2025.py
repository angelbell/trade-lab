"""2025年の停滞の病巣を特定する。

分かっていること: 機構の摩耗ではない（クラウディング診断で6パネル全て P>0.12、USDJPY は26年平坦）。
コストでもない（2025のBTCスプレッドは 0.024% ＝ 全期間で最も狭い）。
∴ 残る仮説は「2025年の相場に固有の何か」か「ただの標本ノイズ」。

分解の順序:
  1. 年別の素性（本数・勝率・勝ちの平均R・負けの平均R・決着のしかた）
  2. **同じ年のランダム建て帰無との比較** ← ここが本丸。
     帰無も悪ければ「2025はロングに悪い年だった」＝エッジは無傷。
     帰無だけ良ければ「入口が壊れた」＝機構の問題。
  3. 相場側の素性（年間騰落・実現ボラ・上昇バーの割合）
帰無は同じ年から無作為に足を選び、**まったく同じ執行**（損切り=その足の安値・ATR×3トレール・fwd20）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402
from experiments.atr_spike_barspread import spread_series, wilder_atr, spikes   # noqa: E402

K, TRAIL, FWD = 2.0, 3.0, 20
NNULL = 400
RNG = np.random.default_rng(2025)


def prep(sym):
    d = load_mt5_csv(f"data/vantage_{sym}_h1.csv").loc["2022-01-01":]
    return d[~d.index.duplicated(keep="first")].sort_index()


def walk_idx(d, s, cost_series):
    """足の位置 s から、同じ執行でトレードを作る。"""
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if i + 1 < len(d) and o[i + 1] - l[i] > 0]
    if len(ent) < 10:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None:
        return None
    t = t.copy()
    t["rf"] = t["risk"] / t["e_px"]
    step = d.index[1] - d.index[0]
    cost = cost_series.reindex(t["time"] + step, method="ffill").to_numpy()
    t["pct"] = (t["R"] * t["risk"]) / t["e_px"] - cost
    t["R_net"] = t["pct"] / t["rf"]
    return t.dropna(subset=["pct"])


def sig_idx(d):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    m = (c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    return s[d.index.dayofweek.to_numpy()[s + 1] < 5]


if __name__ == "__main__":
    d = prep("btcusd")
    sp = spread_series("BTCUSD|h1")
    s = sig_idx(d)
    t = walk_idx(d, s, sp)
    t["y"] = t["time"].dt.year
    yrs = sorted(t["y"].unique())

    print("=== BTC 1時間ロング・年別の素性（実スプレッド課金）")
    print(f"  {'年':>5} {'N':>4} {'勝率':>7} {'1本R':>8} {'勝ちの平均R':>11} {'負けの平均R':>11} "
          f"{'PF':>6} {'最大勝ちR':>9} {'保有中央(h)':>11}")
    for y in yrs:
        g = t[t["y"] == y]
        r = g["R_net"].to_numpy()
        w, ls = r[r > 0], r[r < 0]
        p = g["pct"].to_numpy()
        pw, pl = p[p > 0].sum(), -p[p < 0].sum()
        print(f"  {y:>5} {len(r):>4} {(r>0).mean()*100:>6.1f}% {r.mean():>+8.3f} "
              f"{w.mean() if len(w) else np.nan:>+11.3f} {ls.mean() if len(ls) else np.nan:>+11.3f} "
              f"{pw/pl if pl>0 else np.nan:>6.2f} {r.max():>9.2f} "
              f"{np.median(g['hold'])*24:>11.0f}")

    print("\n=== 同じ年のランダム建て帰無と比べる（同じ本数・同じ執行）")
    print(f"  {'年':>5} {'実測 1本R':>10} {'帰無 中央値':>11} {'帰無 σ':>8} {'%ile':>7} "
          f"{'超過':>8} | {'相場: 年騰落':>12} {'上昇足%':>8}")
    o_, c_ = d["open"].to_numpy(), d["close"].to_numpy()
    yy = d.index.year.to_numpy()
    for y in yrs:
        g = t[t["y"] == y]
        obs = g["R_net"].mean()
        pool = np.flatnonzero((yy == y) & (np.arange(len(d)) < len(d) - 1))
        vals = []
        for _ in range(NNULL):
            pick = RNG.choice(pool, min(len(g), len(pool) // 2), replace=False)
            tn = walk_idx(d, np.sort(pick), sp)
            if tn is not None and len(tn) >= 10:
                vals.append(tn["R_net"].mean())
        vals = np.array(vals)
        px = d[d.index.year == y]["close"]
        ret = (px.iloc[-1] / px.iloc[0] - 1) * 100
        up = ((c_[yy == y] > o_[yy == y]).mean()) * 100
        print(f"  {y:>5} {obs:>+10.3f} {np.median(vals):>+11.3f} {vals.std():>8.3f} "
              f"{(vals < obs).mean()*100:>6.1f}% {obs-np.median(vals):>+8.3f} | "
              f"{ret:>+11.1f}% {up:>7.1f}%")

    print("\n=== 2025 は標本ノイズの範囲か（全期間から同じ本数を無作為に抜く）")
    allr = t["R_net"].to_numpy()
    n25 = int((t["y"] == 2025).sum())
    obs25 = t.loc[t["y"] == 2025, "R_net"].mean()
    draws = np.array([RNG.choice(allr, n25, replace=False).mean() for _ in range(5000)])
    print(f"  2025 実測 1本R={obs25:+.3f}（N={n25}）  全期間から N={n25} を抜いた分布: "
          f"中央値={np.median(draws):+.3f} σ={draws.std():.3f}")
    print(f"  → 2025 は下位 {(draws < obs25).mean()*100:.1f}%ile"
          f"（5%を割れば『ただのノイズ』では説明しにくい）")

    assert len(t) > 150, len(t)
    print(f"\nOK: N={len(t)}")
