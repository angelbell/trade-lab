"""同時建玉に上限を課したときに何が失われるか（手動執行では6本同時は現実的でない）。

由来: BTC先導で絞った6銘柄は最大同時建玉6本＝BTCが動いた瞬間に全部が火を噴く。
      名目に対する maxDD 51-55% はその帰結。%表示は口座の痛みではないので R 単位にも直す。

上限のかけ方は先着順（時刻の早い順に建て、埋まっていたら見送る）＝実際にできること。
比較の相手は「無作為に同じ本数だけ間引く」（法則7: 選択ルールは運の選別器）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_leader_binance import (load as bload, spikes as bspikes,   # noqa: E402
                                                 leg as bleg, st)
from experiments.atr_spike_btc_leader import load as vload, spikes as vspikes, leg as vleg  # noqa: E402

COST = 0.0020
RNG = np.random.default_rng(555)
NBOOT = 1000
V_USE = ["ethusd", "solusd", "adausd", "dotusd", "xrpusd", "ltcusd"]
B_USE = ["ethusdt", "solusdt", "adausdt", "dotusdt", "xrpusdt", "ltcusdt"]


def build(feed):
    if feed == "vantage":
        ld, sp, lg, use, btc = vload, vspikes, vleg, V_USE, "btcusd"
    else:
        ld, sp, lg, use, btc = bload, bspikes, bleg, B_USE, "btcusdt"
    b = sp(ld(btc))
    sB = (b | b.shift(1)).fillna(False)
    rows = []
    for s in use:
        t = lg(ld(s))
        if t is None:
            continue
        g = t["gross"].to_numpy() if "gross" in t else t["pct"].to_numpy()
        risk_frac = (t["risk"] / t["e_px"]).to_numpy()      # 損切り幅（価格に対する率）
        m = sB.reindex(t["time"]).fillna(False).to_numpy()
        rows.append(pd.DataFrame({"sym": s, "time": t["time"].values,
                                  "hold": t["hold"].values, "pct": g - COST,
                                  "rf": risk_frac})[m])
    return pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)


def apply_cap(R, cap):
    """先着順で上限まで建てる。返り値は採用行の boolean。"""
    open_until = []
    keep = np.zeros(len(R), dtype=bool)
    for i, r in enumerate(R.itertuples()):
        t = r.time
        open_until = [x for x in open_until if x > t]
        if len(open_until) < cap:
            keep[i] = True
            open_until.append(t + pd.Timedelta(days=float(r.hold)))
    return keep


def metrics(R, keep):
    p = R.loc[keep, "pct"].to_numpy()
    rr = (R.loc[keep, "pct"] / R.loc[keep, "rf"]).to_numpy()      # R 単位
    eq, eqr = np.cumsum(p), np.cumsum(rr)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    ddr = float((np.maximum.accumulate(eqr) - eqr).max())
    n, win, pf, mu = st(p)
    return dict(n=n, win=win, pf=pf, mu=mu, tot=p.sum() * 100, dd=dd * 100,
                totR=rr.sum(), ddR=ddr, score=p.sum() / dd if dd > 0 else np.nan,
                scoreR=rr.sum() / ddr if ddr > 0 else np.nan)


if __name__ == "__main__":
    for feed in ("binance", "vantage"):
        R = build(feed)
        span = (R["time"].iloc[-1] - R["time"].iloc[0]).days / 365.25
        print(f"\n===== {feed.upper()}  6銘柄・BTC先導・往復{COST*100:.2f}%  "
              f"（{span:.1f}年・損切り幅の中央値 {R['rf'].median()*100:.2f}%）")
        print(f"  {'上限':>6} {'N':>5} {'年本数':>7} {'勝率':>7} {'PF':>6} {'平均%':>8} "
              f"{'通算%':>8} {'DD%':>7} {'通算R':>8} {'DD(R)':>7} {'R通算/DD':>9} {'帰無%ile':>9}")
        full = metrics(R, np.ones(len(R), dtype=bool))
        for cap in (1, 2, 3, 4, 6):
            keep = apply_cap(R, cap)
            m = metrics(R, keep)
            # 間引き帰無: 同じ本数を無作為に残したときの R通算/DD
            k = int(keep.sum())
            null = []
            for _ in range(NBOOT):
                sel = np.zeros(len(R), dtype=bool)
                sel[RNG.choice(len(R), k, replace=False)] = True
                mm = metrics(R, sel)
                null.append(mm["scoreR"])
            null = np.array(null)
            pct = (null[np.isfinite(null)] < m["scoreR"]).mean() * 100
            print(f"  {cap:>6} {m['n']:>5} {m['n']/span:>7.0f} {m['win']:>6.1f}% {m['pf']:>6.2f} "
                  f"{m['mu']:>+8.3f} {m['tot']:>+8.0f} {m['dd']:>7.1f} {m['totR']:>+8.1f} "
                  f"{m['ddR']:>7.1f} {m['scoreR']:>9.2f} {pct:>8.1f}")
        print(f"  {'上限なし':>6} {full['n']:>5} {full['n']/span:>7.0f} {full['win']:>6.1f}% "
              f"{full['pf']:>6.2f} {full['mu']:>+8.3f} {full['tot']:>+8.0f} {full['dd']:>7.1f} "
              f"{full['totR']:>+8.1f} {full['ddR']:>7.1f} {full['scoreR']:>9.2f}")
        print(f"  口座換算（上限2本・1トレードのリスクを口座の r% とすると maxDD ≒ "
              f"{metrics(R, apply_cap(R, 2))['ddR']:.1f} × r%）")
        if feed == "binance":
            keep2 = metrics(R, apply_cap(R, 2))

    assert keep2["pf"] > 1.3, keep2
    assert keep2["ddR"] > 0
    print(f"\nOK: Binance 上限2本 N={keep2['n']} PF={keep2['pf']:.2f} DD={keep2['ddR']:.1f}R")
