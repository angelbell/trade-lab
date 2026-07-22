"""横展開で唯一 BTC 以外に生存したと報告された USDJPY h1 ロングの独立照合。

報告値: N=636(27年) N/年24.0 PF1.30 平均+0.048% maxDD8.9% 帰無%ile 99.8/100 黒字年17/27
コストは FX 往復 0.9pip 相当（USDJPY では 0.009円）を価格に対する割合として引く。
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


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def trigger(d, k, use_pdh):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    if use_pdh:
        s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    return s


def go(d, s_list, cost_abs):
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    ent = [(s, o[s + 1], l[s], o[s + 1] + 1000.0 * (o[s + 1] - l[s]), s)
           for s in s_list if o[s + 1] - l[s] > 0]
    if len(ent) < 10:
        return None, None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=3.0, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 10:
        return None, None
    p = ((t["R"] * t["risk"] - cost_abs) / t["e_px"]).to_numpy()
    return t, p


def rep(p):
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(N=len(p), win=np.mean(p > 0) * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd)


d = load_mt5_csv("data/vantage_usdjpy_h1.csv").loc["2000-01-01":]
COST = 0.009        # 往復 0.9pip
span = (d.index[-1] - d.index[0]).days / 365.25

t2, p2 = go(d, trigger(d, 2.0, True), COST)
r = rep(p2)
print(f"USDJPY h1 ロング k=2.0 前日高値フィルタあり（往復0.9pip）")
print(f"  N={r['N']} N/年={r['N']/span:.1f} 勝率={r['win']:.1f}% PF={r['pf']:.2f} "
      f"平均={r['mean']:+.4f}% 総={r['tot']:+.1f}% maxDD={r['dd']:.1f}%")

# 帰無: 素の全陽線（k=0・フィルタ無し）から同数・時間帯一致で抜く
pool = trigger(d, 0.0, False)
_, p0 = go(d, pool, COST)
hrs_pool = d.index.hour.to_numpy()[pool]
hrs_trig = d.index.hour.to_numpy()[trigger(d, 2.0, True)]
cnt = pd.Series(hrs_trig).value_counts()
rng = np.random.default_rng(23)
nm, npf = [], []
for _ in range(300):
    pick = []
    for hh, n in cnt.items():
        cand = pool[hrs_pool == hh]
        if len(cand):
            pick.extend(rng.choice(cand, size=min(int(n), len(cand)), replace=False))
    _, pn = go(d, np.sort(np.array(pick)), COST)
    if pn is None:
        continue
    q = rep(pn)
    nm.append(q["mean"]); npf.append(q["pf"])
nm, npf = np.array(nm), np.array(npf)
print(f"  素の母集団帰無: PF中央値={np.median(npf):.2f}±{npf.std(ddof=1):.2f} "
      f"平均={np.median(nm):+.4f}%±{nm.std(ddof=1):.4f}")
print(f"  → PF %ile={(npf < r['pf']).mean()*100:.1f}%  平均 %ile={(nm < r['mean']).mean()*100:.1f}%")

yr = pd.Series(p2).groupby(t2["time"].dt.year.values).sum()
print(f"  黒字年 {int((yr>0).sum())}/{len(yr)}")
print("  年別: " + " ".join(f"{y}:{v*100:+.1f}" for y, v in yr.items()))

# コスト感度（往復 0.9 / 1.8 / 2.7 pip）
print("\n  コスト梯子:")
for c in (0.009, 0.018, 0.027):
    _, pc = go(d, trigger(d, 2.0, True), c)
    q = rep(pc)
    print(f"    往復{c/0.01:.1f}pip: PF={q['pf']:.2f} 平均={q['mean']:+.4f}% maxDD={q['dd']:.1f}%")

assert 600 <= r["N"] <= 660, r["N"]
assert 1.20 < r["pf"] < 1.40, r["pf"]
print("\nOK: 報告値 N=636 PF1.30 の近傍を再現")
