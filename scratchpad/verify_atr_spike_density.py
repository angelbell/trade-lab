"""BTC h1 の密度の罠: 凍結仕様を「疎な時代」と「濃い時代」に分けて測り直す。

24時間市場なら h1 は年8760本。BTC は 2018:4835 / 2019:6235 / 2020:6231 / 2021:6689 と
24-45% の時間が欠けており、2022年以降で 7970-8674 本になる。ATR も前方20本も前日高値も、
欠けた時間の上では意味が変わる。アンカーがこの汚染を受けていないかを確認する。
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


def run(path, start, end, k=2.0, trail=3.0, fwd=20, use_pdh=True):
    d = load_mt5_csv(path)
    if start:
        d = d.loc[start:]
    if end:
        d = d.loc[:end]
    if len(d) < 2000:
        return None
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh_dist = (c - pd.Series(_pdh.reindex(d.index).to_numpy(),
                              index=d.index).ffill().to_numpy()) / ap
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s_all = np.flatnonzero(hit)
    s_all = s_all[s_all + 1 < len(d)]
    if use_pdh:
        s_all = s_all[pdh_dist[s_all] > 0.0]
    ent = []
    for s in s_all:
        e, st = o[s + 1], l[s]
        if e - st > 0:
            ent.append((s, e, st, e + 1000.0 * (e - st), s))
    if len(ent) < 20:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=trail, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 20:
        return None
    p = ((t["R"] * t["risk"] - COST * t["e_px"]) / t["e_px"]).to_numpy()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    span = (d.index[-1] - d.index[0]).days / 365.25
    yr = pd.Series(p).groupby(t["time"].dt.year.values).sum()
    return dict(N=len(p), per_yr=len(p) / span, win=np.mean(p > 0) * 100,
                pf=w / ls if ls > 0 else np.nan, mean=p.mean() * 100, tot=p.sum() * 100,
                dd=dd, ratio=p.sum() * 100 / dd if dd > 0 else np.nan,
                pos=int((yr > 0).sum()), ny=len(yr), bars=len(d))


BTC = "data/vantage_btcusd_h1.csv"
print("===== BTC h1 凍結仕様（ロング・ATR×3トレール・保有20本・前日高値フィルタ）を時代で割る")
print(f"{'期間':<26} {'足数':>7} {'N':>5} {'N/年':>6} {'勝率':>6} {'PF':>6} {'平均%':>7} "
      f"{'総%':>8} {'maxDD%':>7} {'総/DD':>6} {'黒字年':>7}")
for start, end, lab in ((None, None, "全期間 2017-2026"),
                        ("2018-01-01", "2021-12-31", "疎な時代 2018-2021"),
                        ("2022-01-01", None, "濃い時代 2022-2026"),
                        ("2019-01-01", None, "2019- （中間）")):
    r = run(BTC, start, end)
    if r is None:
        print(f"{lab:<26} データ不足")
        continue
    print(f"{lab:<26} {r['bars']:7d} {r['N']:5d} {r['per_yr']:6.1f} {r['win']:5.1f}% "
          f"{r['pf']:6.2f} {r['mean']:+7.3f} {r['tot']:+8.1f} {r['dd']:7.1f} "
          f"{r['ratio']:6.2f} {r['pos']:3d}/{r['ny']:<3d}")

print("\n===== 濃い時代（2022-01-01以降）で暗号資産を横並び。BTC以外は初見のデータ")
print(f"{'銘柄':<10} {'足数':>7} {'N':>5} {'N/年':>6} {'勝率':>6} {'PF':>6} {'平均%':>7} "
      f"{'総%':>8} {'maxDD%':>7} {'総/DD':>6} {'黒字年':>7}")
for s in ("btcusd", "ethusd", "xrpusd", "ltcusd", "bchusd", "trxusd",
          "solusd", "adausd", "dotusd", "bnbusd"):
    r = run(f"data/vantage_{s}_h1.csv", "2022-01-01", None)
    if r is None:
        print(f"{s:<10} データ不足")
        continue
    print(f"{s:<10} {r['bars']:7d} {r['N']:5d} {r['per_yr']:6.1f} {r['win']:5.1f}% "
          f"{r['pf']:6.2f} {r['mean']:+7.3f} {r['tot']:+8.1f} {r['dd']:7.1f} "
          f"{r['ratio']:6.2f} {r['pos']:3d}/{r['ny']:<3d}")

# 検算: 全期間の凍結仕様が既知のアンカーを再現すること
full = run(BTC, None, None)
assert 41.0 < full["per_yr"] < 43.0, full["per_yr"]
assert 2.05 < full["pf"] < 2.20, full["pf"]
assert 12.5 < full["dd"] < 14.0, full["dd"]
print(f"\nOK: 全期間アンカー N/年{full['per_yr']:.1f} PF{full['pf']:.2f} "
      f"maxDD{full['dd']:.1f}% を再現")
