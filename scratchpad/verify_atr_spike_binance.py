"""Vantage の 2018-2021（平日限定商品だった時代）の強さは実物か、構造の産物か。

Binance は 2017-08 以降ずっと本物の24時間なので、同じ期間を土日込みで測れば切り分けられる。
🚨 Binance は売買しているフィードではない（採用の証拠にはしない）。機構の検証にのみ使う。
時刻はブローカー時計（Europe/Riga）に合わせてから測る（前日高値の日境界と時間帯を Vantage と揃えるため）。
コストは比較可能性のため Vantage 実験と同じ割合 0.0005 を使う（Binance 現物の実費 0.2% とは別物）。
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


def load(path, to_broker):
    d = load_mt5_csv(path)
    if to_broker:
        # Binance は真のUTC。Vantage CSV は「ブローカー時刻をUTCと表記」した形なので、
        # 日境界と時間帯を揃えるために Riga のローカル時刻へ読み替える。
        idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
        d = d.set_index(idx)
    return d


def run(d, k=2.0, trail=3.0, fwd=20, use_pdh=True, start=None, end=None):
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
    ent = [(s, o[s + 1], l[s], o[s + 1] + 1000.0 * (o[s + 1] - l[s]), s)
           for s in s_all if o[s + 1] - l[s] > 0]
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
                pos=int((yr > 0).sum()), ny=len(yr))


SRC = {"Vantage BTC (CFD)": load("data/vantage_btcusd_h1.csv", False),
       "Binance BTCUSDT": load("data/binance_btcusdt_h1.csv", True),
       "Binance ETHUSDT": load("data/binance_ethusdt_h1.csv", True)}

print(f"{'フィード':<20} {'期間':<14} {'N':>5} {'N/年':>6} {'勝率':>6} {'PF':>6} "
      f"{'平均%':>7} {'総%':>8} {'maxDD%':>7} {'総/DD':>6} {'黒字年':>7}")
for name, d in SRC.items():
    for start, end, lab in (("2018-01-01", "2021-12-31", "2018-2021"),
                            ("2022-01-01", None, "2022-2026"),
                            ("2018-01-01", None, "2018-2026 通し")):
        r = run(d, start=start, end=end)
        if r is None:
            print(f"{name:<20} {lab:<14} データ不足")
            continue
        print(f"{name:<20} {lab:<14} {r['N']:5d} {r['per_yr']:6.1f} {r['win']:5.1f}% "
              f"{r['pf']:6.2f} {r['mean']:+7.3f} {r['tot']:+8.1f} {r['dd']:7.1f} "
              f"{r['ratio']:6.2f} {r['pos']:3d}/{r['ny']:<3d}")
    print()

# 検算: Vantage 側が既知の時代別の値を再現すること
v = SRC["Vantage BTC (CFD)"]
a1 = run(v, start="2018-01-01", end="2021-12-31")
a2 = run(v, start="2022-01-01")
assert 2.55 < a1["pf"] < 2.75, a1["pf"]
assert 1.52 < a2["pf"] < 1.66, a2["pf"]
print(f"OK: Vantage 疎な時代 PF={a1['pf']:.2f} / 濃い時代 PF={a2['pf']:.2f} を再現")
