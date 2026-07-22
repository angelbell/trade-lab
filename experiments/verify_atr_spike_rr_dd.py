"""RR を伸ばすと DD はどうなるか（法則9の但し書きの実測）。

PF と平均% だけ見て RR6 を採ると、勝率低下でゴツゴツになる恐れがある。
maxDD と最大連敗を出して、総利益との比で見る。ロット0.01固定なので複利でなく累積%。
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


df = load_mt5_csv("data/vantage_xauusd_h1.csv") if False else load_mt5_csv("data/vantage_btcusd_h1.csv")
o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
ap = wilder_atr(df).shift(1).to_numpy()
day = df.index.floor("D")
pdh = df["high"].groupby(day).max().shift(1)
pdh = pd.Series(pdh.reindex(df.index).to_numpy(), index=df.index).ffill().to_numpy()
pdh_dist = (c - pdh) / ap
span = (df.index[-1] - df.index[0]).days / 365.25


def cell(k, rr, stop_mode, use_pdh):
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s_all = np.flatnonzero(hit)
    s_all = s_all[s_all + 1 < len(df)]
    if use_pdh:
        s_all = s_all[pdh_dist[s_all] > 0.0]
    ent = []
    for s in s_all:
        e = o[s + 1]
        st = l[s] if stop_mode == "A" else e - 2.0 * ap[s]
        if e - st > 0:
            ent.append((s, e, st, e + rr * (e - st), s))
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=0.0)
    t, _ = walk(df, ent, None, a)
    p = ((t["R"] * t["risk"] - COST * t["e_px"]) / t["e_px"]).to_numpy()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    # 最大連敗
    run_l = best = 0
    for x in p:
        run_l = run_l + 1 if x <= 0 else 0
        best = max(best, run_l)
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(N=len(p), per_yr=len(p) / span, win=np.mean(p > 0) * 100,
                pf=w / ls, mean=p.mean() * 100, tot=p.sum() * 100, dd=dd,
                ratio=p.sum() * 100 / dd if dd > 0 else np.nan, streak=best)


for k, tag in ((2.0, "k=2.0"), (1.5, "k=1.5")):
    for use_pdh in (False, True):
        print(f"\n===== {tag}  前日高値フィルタ={'あり(pdh>0)' if use_pdh else 'なし'}  保有上限20本")
        print(f"{'損切り':>6} {'RR':>5} {'N':>5} {'N/年':>6} {'勝率':>6} {'PF':>6} "
              f"{'平均%':>7} {'総%':>8} {'maxDD%':>7} {'総/DD':>6} {'最大連敗':>7}")
        for sm in ("A", "B"):
            for rr in (2.0, 3.0, 4.5, 6.0):
                r = cell(k, rr, sm, use_pdh)
                print(f"{sm:>6} {rr:5.1f} {r['N']:5d} {r['per_yr']:6.1f} {r['win']:5.1f}% "
                      f"{r['pf']:6.2f} {r['mean']:+7.3f} {r['tot']:+8.1f} {r['dd']:7.1f} "
                      f"{r['ratio']:6.2f} {r['streak']:7d}")

# 検算: 既知の基準セル（A系 RR3 k=2.0 フィルタ無し）
b = cell(2.0, 3.0, "A", False)
assert b["N"] == 556, b["N"]
assert 1.44 < b["pf"] < 1.46, b["pf"]
assert 22.0 < b["dd"] < 22.5, b["dd"]
print(f"\nOK: 基準セル N={b['N']} PF={b['pf']:.2f} maxDD={b['dd']:.1f}% を再現")
