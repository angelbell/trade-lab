"""BTC先導で絞ったアルト群を「1本の運用」として見る（口座寄与の段）。

手動執行なので採否を左右するのは % ではなく:
  - トレード解像度の maxDD（月次に潰してはならない＝法則8）
  - **同時に何本開くか**（横断面なので固まって出る恐れ）
  - 月あたりの本数と、無取引月の長さ
Vantage(2022-) と Binance(2018-2026) の両方で出す。等金額（1トレード=同じ名目）で合成する。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_leader_binance import (load as bload, spikes as bspikes,   # noqa: E402
                                                 leg as bleg, st)
from scratchpad.atr_spike_btc_leader import load as vload, spikes as vspikes, leg as vleg  # noqa: E402

COST = 0.0020          # 保守側（アルトの実スプレッド未実測。BTCは約0.023%）
V_USE = ["ethusd", "solusd", "adausd", "dotusd", "xrpusd", "ltcusd"]
B_USE = ["ethusdt", "solusdt", "adausdt", "dotusdt", "xrpusdt", "ltcusdt"]


def build(feed):
    if feed == "vantage":
        ld, sp, lg, use = vload, vspikes, vleg, V_USE
        btc = "btcusd"
    else:
        ld, sp, lg, use = bload, bspikes, bleg, B_USE
        btc = "btcusdt"
    b = sp(ld(btc))
    sB = (b | b.shift(1)).fillna(False)
    rows = []
    for s in use:
        t = lg(ld(s))
        if t is None:
            continue
        g = t["gross"].to_numpy() if "gross" in t else t["pct"].to_numpy()
        m = sB.reindex(t["time"]).fillna(False).to_numpy()
        rows.append(pd.DataFrame({"sym": s, "time": t["time"].values,
                                  "hold": t["hold"].values, "pct": g - COST})[m])
    R = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    return R


def concurrency(R):
    """建玉の重なり（保有は日数）。同時に開いている本数の最大と分布。"""
    ev = []
    for _, r in R.iterrows():
        ev.append((r["time"], 1))
        ev.append((r["time"] + pd.Timedelta(days=float(r["hold"])), -1))
    ev.sort()
    cur, mx, hist = 0, 0, {}
    for _, dlt in ev:
        cur += dlt
        mx = max(mx, cur)
        hist[cur] = hist.get(cur, 0) + 1
    return mx, hist


if __name__ == "__main__":
    for feed in ("vantage", "binance"):
        R = build(feed)
        p = R["pct"].to_numpy()
        eq = np.cumsum(p)
        dd = np.maximum.accumulate(eq) - eq
        span = (R["time"].iloc[-1] - R["time"].iloc[0]).days / 365.25
        mx, _ = concurrency(R)
        n, win, pf, mu = st(p)
        print(f"\n===== {feed.upper()}  6銘柄・BTC先導のみ・往復コスト{COST*100:.2f}%")
        print(f"  期間 {R['time'].iloc[0]:%Y-%m} 〜 {R['time'].iloc[-1]:%Y-%m} ({span:.1f}年)")
        print(f"  N={n}（年{n/span:.0f}本）勝率={win:.1f}% PF={pf:.2f} 平均={mu:+.3f}% "
              f"通算={p.sum()*100:+.0f}%")
        print(f"  トレード解像度 maxDD={dd.max()*100:.1f}%（名目に対する率）  "
              f"通算/DD={p.sum()/dd.max():.2f}")
        print(f"  最大同時建玉={mx}本  中央保有={np.median(R['hold'])*24:.0f}時間")
        mo = R.set_index("time").resample("ME")["pct"].agg(["count", "sum"])
        zero = int((mo["count"] == 0).sum())
        print(f"  月別: 中央{mo['count'].median():.0f}本/月 最大{int(mo['count'].max())}本 "
              f"無取引月={zero}/{len(mo)}  黒字月={int((mo['sum']>0).sum())}/{int((mo['count']>0).sum())}")
        yr = R.set_index("time").resample("YE")["pct"].agg(["count", "sum"])
        print("  年別: " + " ".join(f"{i.year}:{r['sum']*100:+.0f}%/{int(r['count'])}"
                                    for i, r in yr.iterrows()))
        print(f"  銘柄別の寄与: " + " ".join(
            f"{s[:3]}:{R[R['sym']==s]['pct'].sum()*100:+.0f}%/{len(R[R['sym']==s])}"
            for s in R["sym"].unique()))
        if feed == "binance":
            keep = (n, pf, mx)

    assert keep[1] > 1.3, keep
    assert keep[2] <= 6, keep
    print(f"\nOK: Binance 6銘柄 N={keep[0]} PF={keep[1]:.2f} 最大同時建玉={keep[2]}本")
