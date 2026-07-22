"""確認された3セルのゲート後の年別と、ブロック・ブートストラップ。

Binance 2018-2026 で確認されたもの（しきい値はいずれも拡張窓の50%点）:
  BTC ロング × ER高（帰無%ile 98.3）· BTC ショート × ER低（98.2）· ETH ロング × ER高（97.8）
落ちたもの: ETH ショート × ER低（49.1／3分位が単調でない）
ここではゲート後の年別と、経路当てはめでないかの確認（巡回ブロック・ブートストラップ）を出す。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_er_binance import (load, spikes, build, attach,   # noqa: E402
                                             COST, er_series, WIN)

NBOOT = 1000
RNG = np.random.default_rng(9090)


def sc(r):
    if len(r) < 5:
        return np.nan
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return r.sum() / dd if dd > 0 else np.nan


if __name__ == "__main__":
    B, E = load("btcusdt"), load("ethusdt")
    sB = spikes(B)
    lead = (sB | sB.shift(1)).fillna(False)
    erB, erE = er_series(B["close"], WIN), er_series(E["close"], WIN)

    CASES = [("BTC ロング × ER高", B, "long", erB, None, "btcusdt", True),
             ("BTC ショート × ER低", B, "short", erB, None, "btcusdt", False),
             ("ETH ロング × ER高", E, "long", erE, lead, "ethusdt", True)]
    for nm, d, side, e, ld, sym, high in CASES:
        T = attach(build(d, side, COST[sym], lead=ld), e)
        T["on"] = (T["er"] >= T["x50"]) if high else (T["er"] < T["x50"])
        span = (T["time"].max() - T["time"].min()).days / 365.25
        print(f"\n===== {nm}  （助走後 {span:.1f}年）")
        for lab, g in (("ゲート無し", T), ("ゲート有り", T[T["on"]])):
            r = g["R_net"].to_numpy()
            p = g["pct"].to_numpy()
            w, ls = p[p > 0].sum(), -p[p < 0].sum()
            eq = np.cumsum(r)
            dd = float((np.maximum.accumulate(eq) - eq).max())
            print(f"  {lab:<10} N={len(r):4d} 年{len(r)/span:3.0f}本 "
                  f"勝率={(p>0).mean()*100:5.1f}% PF={w/ls if ls>0 else np.nan:5.2f} "
                  f"1本R={r.mean():+.3f} totR={r.sum():+7.1f} DD={dd:5.1f}R 比={r.sum()/dd:6.2f}")
        yy = T.groupby(T["time"].dt.year)
        print("  年別 1本R（ゲート無し → 有り）:")
        line = []
        for y, g in yy:
            on = g[g["on"]]
            line.append(f"{y}:{g['R_net'].mean():+.2f}→"
                        f"{on['R_net'].mean() if len(on) >= 3 else np.nan:+.2f}/{len(on)}")
        print("    " + " ".join(line))

        # ブロック・ブートストラップ
        T2 = T.copy()
        T2["mo"] = T2["time"].dt.to_period("M")
        months = sorted(T2["mo"].unique())
        bymo = {m: g for m, g in T2.groupby("mo")}
        nm_ = len(months)
        out = []
        for b in (1, 3, 6, 12):
            win_ = ok = 0
            for _ in range(NBOOT):
                need = int(np.ceil(nm_ / b))
                starts = RNG.integers(0, nm_, size=need)
                pa, po = [], []
                for st in starts:
                    blk = [months[(st + i) % nm_] for i in range(b)]
                    gs = [bymo[m] for m in blk if m in bymo]
                    if not gs:
                        continue
                    g = pd.concat(gs, ignore_index=True)
                    pa.append(g["R_net"].to_numpy())
                    po.append(g.loc[g["on"], "R_net"].to_numpy())
                if not pa:
                    continue
                sa, so = sc(np.concatenate(pa)), sc(np.concatenate(po))
                if np.isfinite(sa) and np.isfinite(so):
                    ok += 1
                    win_ += int(so > sa)
            out.append(f"{b}か月:{win_/max(ok,1)*100:.1f}%")
        print("  ブロック・ブートストラップ（totR/DD が勝つ割合）: " + " → ".join(out))

    print("\n（ブロックを長くするほど上がれば本物）")
