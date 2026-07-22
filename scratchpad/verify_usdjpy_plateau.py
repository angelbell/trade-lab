"""参考セル（k2.5 / B系stop2.0ATR / ATR×3トレール / fwd20 / PDH>0 / 週足30MA上）が台地か尖りか。

掃引結果を見てから拾った3本目なので、隣接セルが揃って良いか（台地）、
そこだけ跳ねているか（尖り＝過剰当てはめ）を確かめる。
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

COST = 0.009
ACCOUNT, MAXRISK = 100000.0, 0.03


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


d = load_mt5_csv("data/vantage_usdjpy_h1.csv").loc["2000-01-01":]
o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
ap = wilder_atr(d).shift(1).to_numpy()
day = d.index.floor("D")
_pdh = d["high"].groupby(day).max().shift(1)
pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
wc = d["close"].resample("W").last().dropna()
reg = (wc > wc.rolling(30).mean()).shift(1).reindex(d.index, method="ffill").fillna(False).to_numpy()
span = (d.index[-1] - d.index[0]).days / 365.25


def cell(k=2.5, stopk=2.0, trail=3.0, fwd=20):
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s = s[reg[s]]
    ent = []
    for i in s:
        e = o[i + 1]
        st = l[i] if stopk == 0 else e - stopk * ap[i]
        if e - st > 0:
            ent.append((i, e, st, e + 1000.0 * (e - st), i))
    if len(ent) < 10:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=trail, trail_n=14)
    t, _ = walk(d, ent, None, a)
    px = (t["R"] * t["risk"] - COST).to_numpy()
    pct = px / t["e_px"].to_numpy()
    w, ls = pct[pct > 0].sum(), -pct[pct < 0].sum()
    med = np.median(t["risk"].to_numpy())
    lots = max(0.01, np.floor(ACCOUNT * MAXRISK / (med * 1000)) * 0.01)
    yen = px * 1000 * (lots / 0.01)
    eqy = np.cumsum(yen)
    yr = pd.Series(yen).groupby(t["time"].dt.year.values).sum()
    return dict(N=len(pct), pf=w / ls if ls > 0 else np.nan, mean=pct.mean() * 100,
                yen=yen.sum() / span, dd=float((np.maximum.accumulate(eqy) - eqy).max()),
                pos=int((yr > 0).sum()), ny=len(yr))


print("=== k × トレール倍率（stopk=2.0・fwd20 固定）  各セル: PF / N / 年間円")
trs = [2.0, 2.5, 3.0, 3.5, 4.0]
print(f"{'k':>6} " + " ".join(f"{'TR'+str(t):>18}" for t in trs))
for k in (2.0, 2.25, 2.5, 2.75, 3.0):
    row = []
    for tr in trs:
        r = cell(k=k, trail=tr)
        row.append(f"{r['pf']:5.2f}/{r['N']:4d}/{r['yen']:7,.0f}" if r else "        --        ")
    print(f"{k:6.2f} " + " ".join(f"{x:>18}" for x in row))

print("\n=== 損切り倍率（k2.5・TR3・fwd20）")
print(f"{'stopk':>8} {'N':>5} {'PF':>6} {'平均%':>8} {'年間円':>9} {'DD円':>9} {'黒字年':>7}")
for sk in (0, 1.5, 2.0, 2.5, 3.0):
    r = cell(stopk=sk)
    lab = "A系(端)" if sk == 0 else f"{sk:.1f}ATR"
    print(f"{lab:>8} {r['N']:5d} {r['pf']:6.2f} {r['mean']:+8.4f} {r['yen']:9,.0f} "
          f"{r['dd']:9,.0f} {r['pos']:3d}/{r['ny']:<3d}")

print("\n=== 保有上限（k2.5・stopk2.0・TR3）")
print(f"{'fwd':>6} {'N':>5} {'PF':>6} {'平均%':>8} {'年間円':>9} {'DD円':>9} {'黒字年':>7}")
for fw in (10, 20, 30, 40, 60, 100):
    r = cell(fwd=fw)
    print(f"{fw:6d} {r['N']:5d} {r['pf']:6.2f} {r['mean']:+8.4f} {r['yen']:9,.0f} "
          f"{r['dd']:9,.0f} {r['pos']:3d}/{r['ny']:<3d}")

b = cell()
assert 180 <= b["N"] <= 205, b["N"]
assert 2.30 < b["pf"] < 2.50, b["pf"]
print(f"\nOK: 参考セル N={b['N']} PF={b['pf']:.2f} を再現")
