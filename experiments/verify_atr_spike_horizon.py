"""保有時間そのものが最大のレバーか。出口の種類と保有上限を直交させて測る。

出口3種:
  time  = 目標もトレールも無し。初期損切りだけ持ってN本で時間切れ（保有時間を単離）
  rr4.5 = 固定RR4.5
  tr3   = ATR×3 トレール
損切りはすべて拡大足の反対端。ロット0.01固定なので価格%・累積和。
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

EXITS = {"時間切れのみ": (1000.0, 0.0), "固定RR4.5": (4.5, 0.0), "ATR×3トレール": (1000.0, 3.0)}


def run(k, exit_name, fwd, use_pdh):
    rr, trail = EXITS[exit_name]
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s_all = np.flatnonzero(hit)
    s_all = s_all[s_all + 1 < len(df)]
    if use_pdh:
        s_all = s_all[pdh_dist[s_all] > 0.0]
    ent = []
    for s in s_all:
        e, st = o[s + 1], l[s]
        if e - st > 0:
            ent.append((s, e, st, e + rr * (e - st), s))
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=trail, trail_n=14)
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
    return dict(yr=len(p) / span, win=np.mean(p > 0) * 100, pf=w / ls, mean=p.mean() * 100,
                tot=p.sum() * 100, dd=dd, ratio=p.sum() * 100 / dd, streak=best,
                pos=int((yr > 0).sum()), ny=len(yr))


for use_pdh in (False, True):
    print(f"\n===== k=2.0  前日高値フィルタ={'あり' if use_pdh else 'なし'}  損切り=拡大足の端")
    for exit_name in EXITS:
        print(f"\n  --- 出口: {exit_name}")
        print(f"  {'保有上限':>8} {'N/年':>6} {'勝率':>6} {'PF':>6} {'平均%':>7} {'総%':>8} "
              f"{'maxDD%':>7} {'総/DD':>6} {'連敗':>5} {'黒字年':>7}")
        for fwd in (5, 10, 20, 30, 40, 60, 100, 200):
            r = run(2.0, exit_name, fwd, use_pdh)
            print(f"  {fwd:8d} {r['yr']:6.1f} {r['win']:5.1f}% {r['pf']:6.2f} {r['mean']:+7.3f} "
                  f"{r['tot']:+8.1f} {r['dd']:7.1f} {r['ratio']:6.2f} {r['streak']:5d} "
                  f"{r['pos']:3d}/{r['ny']:<3d}")

# 検算: 既知の現状最良（固定RR4.5・fwd20・フィルタあり）
b = run(2.0, "固定RR4.5", 20, True)
assert 1.89 < b["pf"] < 1.91, b["pf"]
assert 16.0 < b["dd"] < 16.6, b["dd"]
assert 41.5 < b["yr"] < 43.0, b["yr"]
print(f"\nOK: 現状最良 PF={b['pf']:.2f} maxDD={b['dd']:.1f}% 年{b['yr']:.1f}本 を再現")
