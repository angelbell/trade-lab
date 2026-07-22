"""週末フィルタの検定。24時間データ（Binance）で「週末を持ち越さない」が帰無を超えるか。

診断では 週末をまたぐ玉 PF1.19（勝率38.1%）vs またがない玉 PF1.79（52.6%）だった。
これを取る/見送るの規則にして、同じ通過率のランダム間引き帰無を超えるかを見る。
フィルタは建てる時点の暦だけで決まる（先読みなし）。損益は入口価格に対する%。
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


def load(path):
    d = load_mt5_csv(path)
    idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    return d.set_index(idx)


def triggers(d, k=2.0):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    return s[(c[s] - pdh[s]) / ap[s] > 0.0]


def go(d, s_list):
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
    return t, ((t["R"] * t["risk"] - COST * t["e_px"]) / t["e_px"]).to_numpy()


def rep(p):
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(N=len(p), win=np.mean(p > 0) * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd)


def null_pct(p0, obs, reps=400, seed=31):
    rng = np.random.default_rng(seed)
    n = min(len(obs), len(p0))
    m, f = [], []
    for _ in range(reps):
        s = rng.choice(p0, size=n, replace=False)
        q = rep(s)
        m.append(q["mean"]); f.append(q["pf"])
    m, f = np.array(m), np.array(f)
    o = rep(obs)
    return (f < o["pf"]).mean() * 100, (m < o["mean"]).mean() * 100


for sym in ("btcusdt", "ethusdt"):
    d = load(f"data/binance_{sym}_h1.csv").loc["2018-01-01":]
    span = (d.index[-1] - d.index[0]).days / 365.25
    s_all = triggers(d)
    dow = d.index.dayofweek.to_numpy()
    # 建てる時点で確定する暦の条件（先読みなし）
    n = len(d)
    weekend_ahead = np.array([bool((dow[i + 1:min(i + 21, n)] >= 5).any()) for i in s_all])
    entry_dow = dow[np.minimum(s_all + 1, n - 1)]
    hour = d.index.hour.to_numpy()[np.minimum(s_all + 1, n - 1)]

    filts = {
        "フィルタ無し": np.ones(len(s_all), bool),
        "20時間先に土日を含まない": ~weekend_ahead,
        "金曜に建てない": entry_dow != 4,
        "金曜12時以降に建てない": ~((entry_dow == 4) & (hour >= 12)),
        "土日に建てない": entry_dow < 5,
    }
    t0, p0 = go(d, s_all)
    print(f"\n===== {sym.upper()} (Binance 24時間, 2018-2026, {span:.1f}年)")
    print(f"{'フィルタ':<26} {'通過':>6} {'N/年':>6} {'勝率':>6} {'PF':>6} {'平均%':>7} "
          f"{'総%':>8} {'maxDD%':>7} {'帰無%ile(PF,平均)':>18} {'黒字年':>7}")
    for name, mask in filts.items():
        t, p = go(d, s_all[mask])
        if p is None:
            continue
        r = rep(p)
        if name == "フィルタ無し":
            npf = nm = float("nan")
        else:
            npf, nm = null_pct(p0, p)
        yr = pd.Series(p).groupby(t["time"].dt.year.values).sum()
        print(f"{name:<26} {mask.mean()*100:5.1f}% {r['N']/span:6.1f} {r['win']:5.1f}% "
              f"{r['pf']:6.2f} {r['mean']:+7.3f} {r['tot']:+8.1f} {r['dd']:7.1f} "
              f"{npf:8.1f},{nm:8.1f} {int((yr>0).sum()):3d}/{len(yr):<3d}")

    # 時代別（週末のボラは時代で変わる）
    print("  --- 時代別（20時間先に土日を含まない版 vs 無し）")
    for a0, a1, lab in (("2018-01-01", "2021-12-31", "2018-2021"), ("2022-01-01", None, "2022-2026")):
        dd_ = d.loc[a0:a1] if a1 else d.loc[a0:]
        ss = triggers(dd_)
        dw = dd_.index.dayofweek.to_numpy(); nn = len(dd_)
        wa = np.array([bool((dw[i + 1:min(i + 21, nn)] >= 5).any()) for i in ss])
        for tag, m in (("無し", np.ones(len(ss), bool)), ("週末回避", ~wa)):
            _, pp = go(dd_, ss[m])
            if pp is None:
                continue
            r = rep(pp)
            print(f"    {lab} {tag:<8} N{r['N']:4d} 勝率{r['win']:5.1f}% PF{r['pf']:6.2f} "
                  f"平均{r['mean']:+7.3f}% maxDD{r['dd']:6.1f}%")

# 検算: BTC フィルタ無しが既知値を再現
d = load("data/binance_btcusdt_h1.csv").loc["2018-01-01":]
_, p = go(d, triggers(d))
r = rep(p)
assert 430 <= r["N"] <= 442, r["N"]
assert 1.50 < r["pf"] < 1.60, r["pf"]
print(f"\nOK: Binance BTC フィルタ無し N={r['N']} PF={r['pf']:.2f} を再現")
