"""ドル円レッグの直近の戦績を、ロング／ショート分離で年別に出す。

ロング = 採用仕様（k2.0 / 損切り2.0ATR / ATR×3トレール / fwd20 / 前日高値>0 / 週足30MA上）
ショート = その鏡像（mirror.invert 上で同じ規則）。ゲート有無の両方を出す。
🚨 反転フレームでは walk() 内部のコストが鏡像価格を使うので、cost=0 で回して外側で引く。
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
from src.engine.mirror import invert              # noqa: E402

COST = 0.009
LOTS = 0.09            # 10万円・リスク3%で採れる最大ロット（既測）


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


real = load_mt5_csv("data/vantage_usdjpy_h1.csv").loc["2000-01-01":]
inv = invert(real)
C = 2 * real["high"].max()

# 週足30MA は実フレームで作り、ショートでは向きを反転させる
wc = real["close"].resample("W").last().dropna()
up = (wc > wc.rolling(30).mean()).shift(1).reindex(real.index, method="ffill").fillna(False).to_numpy()


def leg(d, Cx, gate_mask, k=2.0, trail=3.0):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    if gate_mask is not None:
        s = s[gate_mask[s]]
    ent = []
    for i in s:
        e = o[i + 1]
        st = e - 2.0 * ap[i]
        if e - st > 0:
            ent.append((i, e, st, e + 1000.0 * (e - st), i))
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=trail, trail_n=14)
    t, _ = walk(d, ent, None, a)
    e_real = (Cx - t["e_px"]) if Cx is not None else t["e_px"]
    px = (t["R"] * t["risk"]).to_numpy() - COST
    return pd.DataFrame({"y": t["time"].dt.year.values,
                         "yen": px * 1000 * (LOTS / 0.01),
                         "pct": px / e_real.to_numpy() * 100})


CASES = [("ロング（採用仕様・週足30MA上）", real, None, up),
         ("ショート（鏡像・週足30MA下）", inv, C, ~up),
         ("ショート（鏡像・ゲート無し）", inv, C, None)]

res = {}
for lab, d, Cx, g in CASES:
    r = leg(d, Cx, g)
    res[lab] = r
    p = r["pct"].to_numpy()
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    print(f"{lab:<30} N={len(r):4d} 勝率={(p>0).mean()*100:5.1f}% "
          f"PF={w/ls:5.2f} 平均={p.mean():+.4f}% 通算={r['yen'].sum():+9,.0f}円")

print(f"\n--- 年別の円損益（{LOTS}ロット・10万円口座リスク3%相当）")
yrs = list(range(2018, 2027))
print(f"{'':<30} " + " ".join(f"{y:>8}" for y in yrs))
for lab in res:
    g = res[lab].groupby("y")["yen"].sum()
    print(f"{lab:<30} " + " ".join(
        f"{g.get(y, 0):>8,.0f}" for y in yrs))
print(f"{'（本数）ロング':<30} " + " ".join(
    f"{int((res[CASES[0][0]]['y']==y).sum()):>8}" for y in yrs))

print("\n--- 直近の区間別（ロング採用仕様）")
r = res[CASES[0][0]]
for lo, hi, lab in ((2018, 2026, "2018-2026"), (2022, 2026, "2022-2026"),
                    (2024, 2026, "2024-2026"), (2025, 2026, "2025-2026")):
    x = r[(r["y"] >= lo) & (r["y"] <= hi)]
    p = x["pct"].to_numpy()
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    print(f"  {lab:<12} N={len(x):3d} 勝率={(p>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else float('nan'):5.2f} 円={x['yen'].sum():+9,.0f}")

assert 300 <= len(res[CASES[0][0]]) <= 345, len(res[CASES[0][0]])
print("\nOK: ロングの母集団が既知の N≈322 と整合")
