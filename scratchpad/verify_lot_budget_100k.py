"""口座 10万円での実額比較。ロットは 0.01 刻みで「リスク3%以下」に丸める。

要点: サイズをリスクで揃えると「銘柄が安いこと」は利点にならない。
効くのは 1トレードの期待値÷損切り幅（＝R単位のエッジ）× 頻度 だけ。
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

ACCOUNT = 100000.0
FX = 150.0          # USDJPY レート（USD建て銘柄の円換算）
MAXRISK = 0.03


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def leg(path, cost_abs, start, k=2.0, skip_weekend=False):
    d = load_mt5_csv(path).loc[start:]
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    if skip_weekend:
        s = s[d.index.dayofweek.to_numpy()[np.minimum(s + 1, len(d) - 1)] < 5]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=3.0, trail_n=14)
    t, _ = walk(d, ent, None, a)
    span = (d.index[-1] - d.index[0]).days / 365.25
    return t["risk"].to_numpy(), (t["R"] * t["risk"] - cost_abs).to_numpy(), span, t


CASES = [
    # 名前, パス, 往復コスト(価格単位), 開始, 0.01ロットの通貨量, 円換算レート, 週末回避
    ("USDJPY", "data/vantage_usdjpy_h1.csv", 0.009, "2000-01-01", 1000.0, 1.0, False),
    ("BTC (2022-)", "data/vantage_btcusd_h1.csv", 15.0, "2022-01-01", 0.01, FX, True),
    ("ETH (2022-)", "data/vantage_ethusd_h1.csv", 2.0, "2022-01-01", 0.10, FX, True),
]

print(f"口座 ¥{ACCOUNT:,.0f} / リスク上限 {MAXRISK*100:.0f}% / ロットは0.01刻みで切り下げ\n")
print(f"{'銘柄':<13} {'N/年':>6} {'損切り中央値':>12} {'0.01の損失':>11} {'採用ロット':>9} "
      f"{'1本リスク':>10} {'1本期待値':>10} {'年間':>10} {'口座比':>7} {'最悪年':>10}")
for name, path, cost, start, units001, fx, skipwe in CASES:
    risk, pnl, span, t = leg(path, cost, start, skip_weekend=skipwe)
    loss001 = np.median(risk) * units001 * fx
    # 0.01ロット何個ぶん建てられるか → 0.01刻みへ
    lots = max(0.01, np.floor(ACCOUNT * MAXRISK / loss001) * 0.01)
    mult = lots / 0.01
    per_trade = pnl.mean() * units001 * fx * mult
    n_yr = len(pnl) / span
    yearly = per_trade * n_yr
    yr = pd.Series(pnl * units001 * fx * mult).groupby(t["time"].dt.year.values).sum()
    print(f"{name:<13} {n_yr:6.1f} {np.median(risk):12,.2f} ¥{loss001:10,.0f} {lots:9.2f} "
          f"¥{loss001*mult:9,.0f} ¥{per_trade:9,.0f} ¥{yearly:9,.0f} {yearly/ACCOUNT*100:6.1f}% "
          f"¥{yr.min():9,.0f}")

print("\n--- 年別の円損益（上の採用ロット）")
for name, path, cost, start, units001, fx, skipwe in CASES:
    risk, pnl, span, t = leg(path, cost, start, skip_weekend=skipwe)
    loss001 = np.median(risk) * units001 * fx
    lots = max(0.01, np.floor(ACCOUNT * MAXRISK / loss001) * 0.01)
    yr = pd.Series(pnl * units001 * fx * (lots / 0.01)).groupby(t["time"].dt.year.values).sum()
    print(f"{name:<13} " + " ".join(f"{y}:{v/1000:+.1f}k" for y, v in yr.items()))

r, p, sp, _ = leg("data/vantage_usdjpy_h1.csv", 0.009, "2000-01-01")
assert 600 <= len(p) <= 660, len(p)
print(f"\nOK: ドル円 N={len(p)} を再現")
