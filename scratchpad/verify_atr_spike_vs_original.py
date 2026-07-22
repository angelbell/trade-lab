"""原典（BigBeluga の指標そのまま）と現状の最良を並べる。

原典の仕様:
  引き金 = 実体 > ATR(14)×2（指標の既定 atrMultInput=2.0）、方向は陽線/陰線
  出口   = ATR(14)×3 のシャンデリア・トレール（初期損切り = 安値 − 3×ATR）、利確目標なし、時間切れなし
  同方向の再シグナルは建玉中ロック（＝max_pos 1 で近似）
差分を1つずつ足して、どの変更が何を買ったのかを分解する。
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

COST = 0.0005


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


df = load_mt5_csv("data/vantage_btcusd_h1.csv")
o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
ap = wilder_atr(df).shift(1).to_numpy()
day = df.index.floor("D")
_pdh = df["high"].groupby(day).max().shift(1)
pdh_dist = (c - pd.Series(_pdh.reindex(df.index).to_numpy(),
                          index=df.index).ffill().to_numpy()) / ap
span = (df.index[-1] - df.index[0]).days / 365.25


def run(k, stop_mode, rr, trail, fwd, use_pdh):
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s_all = np.flatnonzero(hit)
    s_all = s_all[s_all + 1 < len(df)]
    if use_pdh:
        s_all = s_all[pdh_dist[s_all] > 0.0]
    ent = []
    for s in s_all:
        e = o[s + 1]
        st = l[s] if stop_mode == "bar" else l[s] - 3.0 * ap[s]
        if e - st > 0:
            ent.append((s, e, st, e + rr * (e - st), s))
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0,
                        trail_atr=trail, trail_n=14)
    t, _ = walk(df, ent, None, a)
    p = ((t["R"] * t["risk"] - COST * t["e_px"]) / t["e_px"]).to_numpy()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    run_l = best = 0
    for x in p:
        run_l = run_l + 1 if x <= 0 else 0
        best = max(best, run_l)
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    yr = pd.Series(p).groupby(t["time"].dt.year.values).sum()
    return dict(N=len(p), yr=len(p) / span, win=np.mean(p > 0) * 100, pf=w / ls,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd,
                ratio=p.sum() * 100 / dd, streak=best, pos=int((yr > 0).sum()), ny=len(yr))


ROWS = [
    ("1. 原典そのまま（3ATRトレール・目標なし・時間切れなし）", dict(k=2.0, stop_mode="chand", rr=1000.0, trail=3.0, fwd=500, use_pdh=False)),
    ("2. 原典・保有上限200本",                                  dict(k=2.0, stop_mode="chand", rr=1000.0, trail=3.0, fwd=200, use_pdh=False)),
    ("3. 損切りを拡大足の端へ（トレールのまま）",                dict(k=2.0, stop_mode="bar", rr=1000.0, trail=3.0, fwd=200, use_pdh=False)),
    ("4. 出口を固定RR4.5へ（変更点1）",                         dict(k=2.0, stop_mode="bar", rr=4.5, trail=0.0, fwd=200, use_pdh=False)),
    ("5. 保有上限を20本へ",                                     dict(k=2.0, stop_mode="bar", rr=4.5, trail=0.0, fwd=20, use_pdh=False)),
    ("6. 前日高値フィルタを追加（変更点2）＝現状最良",          dict(k=2.0, stop_mode="bar", rr=4.5, trail=0.0, fwd=20, use_pdh=True)),
]

print(f"{'':<44} {'N/年':>6} {'勝率':>6} {'PF':>6} {'平均%':>7} {'総%':>8} "
      f"{'maxDD%':>7} {'総/DD':>6} {'連敗':>5} {'黒字年':>7}")
for name, kw in ROWS:
    r = run(**kw)
    print(f"{name:<44} {r['yr']:6.1f} {r['win']:5.1f}% {r['pf']:6.2f} {r['mean']:+7.3f} "
          f"{r['tot']:+8.1f} {r['dd']:7.1f} {r['ratio']:6.2f} {r['streak']:5d} "
          f"{r['pos']:3d}/{r['ny']:<3d}")

# 検算: 現状最良が既知の値と一致すること
best = run(k=2.0, stop_mode="bar", rr=4.5, trail=0.0, fwd=20, use_pdh=True)
assert best["N"] == 386, best["N"]
assert 1.89 < best["pf"] < 1.91, best["pf"]
assert 16.0 < best["dd"] < 16.6, best["dd"]
print(f"\nOK: 現状最良 N={best['N']} PF={best['pf']:.2f} maxDD={best['dd']:.1f}% を再現")
