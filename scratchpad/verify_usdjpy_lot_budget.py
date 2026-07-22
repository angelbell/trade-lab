"""ドル円レッグを「円でいくら稼ぐか」に変換する。制約は証拠金ではなく1トレードの損失額。

口座 ¥15,277（本番・2026-07-22 時点のブリッジ health より）。
USDJPY 1ロット=100,000通貨 → 0.01ロット=1,000通貨。1pip(0.01円)=¥10。
BTC 1ロット=1BTC → 0.01ロット=0.01BTC。
損切り幅の分布から「リスク1%/2%/3%に収まる最大ロット」を出し、年間の円換算を出す。
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

ACCOUNT = 15277.0
USDJPY_NOW = 150.0


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def leg(path, k, cost_abs, riga=False, start=None):
    d = load_mt5_csv(path)
    if riga:
        idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
        d = d.set_index(idx)
    if start:
        d = d.loc[start:]
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=3.0, trail_n=14)
    t, _ = walk(d, ent, None, a)
    span = (d.index[-1] - d.index[0]).days / 365.25
    pnl_px = (t["R"] * t["risk"] - cost_abs).to_numpy()      # 価格単位の損益（コスト控除後）
    return t["risk"].to_numpy(), pnl_px, span, t


print("========== 1トレードの損切り幅と、口座 ¥15,277 に対するリスク")
rows = []

# --- USDJPY: 価格は円そのもの。1通貨あたり損益＝価格差(円)
risk_j, pnl_j, span_j, t_j = leg("data/vantage_usdjpy_h1.csv", 2.0, 0.009, start="2000-01-01")
med_stop_pips = np.median(risk_j) / 0.01
yen_risk_001 = np.median(risk_j) * 1000                      # 0.01ロット=1,000通貨
yen_per_trade_001 = pnl_j.mean() * 1000
print(f"USDJPY  損切り幅の中央値 {med_stop_pips:.1f}pip  0.01ロットの損失額 ¥{yen_risk_001:,.0f} "
      f"(口座の {yen_risk_001/ACCOUNT*100:.1f}%)  1トレード期待値 ¥{yen_per_trade_001:,.1f}")
rows.append(("USDJPY", risk_j, pnl_j, span_j, 1000.0, 1.0))

# --- BTC(Vantage): 価格はUSD。0.01ロット=0.01BTC。円換算は×150
risk_b, pnl_b, span_b, t_b = leg("data/vantage_btcusd_h1.csv", 2.0, 15.0, start="2022-01-01")
yen_risk_b = np.median(risk_b) * 0.01 * USDJPY_NOW
yen_per_trade_b = pnl_b.mean() * 0.01 * USDJPY_NOW
print(f"BTC     損切り幅の中央値 ${np.median(risk_b):,.0f}  0.01ロットの損失額 ¥{yen_risk_b:,.0f} "
      f"(口座の {yen_risk_b/ACCOUNT*100:.1f}%)  1トレード期待値 ¥{yen_per_trade_b:,.1f}")
rows.append(("BTC(2022-)", risk_b, pnl_b, span_b, 0.01, USDJPY_NOW))

print("\n========== リスク上限別の最大ロットと、年間の円換算")
print(f"{'銘柄':<12} {'リスク':>6} {'最大ロット':>10} {'1本の損失額':>12} "
      f"{'1本の期待値':>12} {'N/年':>6} {'年間期待値':>12} {'口座比':>8}")
for name, risk, pnl, span, units_per_001, fx in rows:
    n_yr = len(pnl) / span
    for pct in (0.01, 0.02, 0.03):
        budget = ACCOUNT * pct
        # 0.01ロットあたりの損失額（円）
        loss_001 = np.median(risk) * units_per_001 * fx
        lots = budget / loss_001 * 0.01
        per_trade = pnl.mean() * units_per_001 * fx * (lots / 0.01)
        yearly = per_trade * n_yr
        print(f"{name:<12} {pct*100:5.0f}% {lots:10.3f} ¥{budget:11,.0f} "
              f"¥{per_trade:11,.0f} {n_yr:6.1f} ¥{yearly:11,.0f} {yearly/ACCOUNT*100:7.1f}%")

# 年別の円換算（リスク2%固定・ドル円）
loss_001 = np.median(risk_j) * 1000
lots = ACCOUNT * 0.02 / loss_001 * 0.01
yr = pd.Series(pnl_j * 1000 * (lots / 0.01)).groupby(t_j["time"].dt.year.values).sum()
print(f"\nUSDJPY リスク2%（{lots:.3f}ロット）の年別円損益:")
print("  " + " ".join(f"{y}:{v:+,.0f}" for y, v in yr.items()))

assert 600 <= len(pnl_j) <= 660, len(pnl_j)
assert 20 < med_stop_pips < 80, med_stop_pips
print(f"\nOK: ドル円 N={len(pnl_j)} 損切り中央値{med_stop_pips:.1f}pip")
