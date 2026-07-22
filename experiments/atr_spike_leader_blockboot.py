"""巡回ブロック・ブートストラップ（棄却基準7の後半）。

間引き帰無は「同じ価格経路の上で、同じ本数を無作為に残す」ことしか訊いていない。
真の改善はブロックを長くするほど勝率が上がり、経路当てはめは上がらない。1/3/6/12か月で確かめる。

2つの主張を別々にかける:
  A. 先導フィルタ（BTC同時のみ vs 全トレード）
  B. 同時建玉の上限（先着順で1本/2本 vs 上限なし）
比較の指標は **totR / maxDD(R)**（固定ロットなので R で数え、DD も R で測る）。
本数が変わる操作なので、PF や平均% で比べると「悪い玉を捨てた」だけで勝ってしまう。

手続き: 暦の「月」を単位に、b か月の連続ブロックを巡回で復元抽出し、全期間ぶんつなぐ。
上限は**ブロックの中**で適用する（ブロック内は実時間の並びなので規則が意味を持つ）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_leader_cap import apply_cap, V_USE, B_USE          # noqa: E402
from experiments.atr_spike_leader_binance import (load as bload, spikes as bspikes,   # noqa: E402
                                                 leg as bleg)
from experiments.atr_spike_btc_leader import (load as vload, spikes as vspikes,       # noqa: E402
                                             leg as vleg)

NBOOT = 1000
COST = 0.0020
RNG = np.random.default_rng(20260722)


def prep(feed):
    """先導フィルタ前の母集団（BTC同時の印つき）。"""
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
        rf = (t["risk"] / t["e_px"]).to_numpy()
        rows.append(pd.DataFrame({"sym": s, "time": t["time"].values,
                                  "hold": t["hold"].values, "pct": g - COST, "rf": rf,
                                  "lead": sB.reindex(t["time"]).fillna(False).to_numpy()}))
    R = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    R["R"] = R["pct"] / R["rf"]
    R["mo"] = R["time"].dt.to_period("M")
    return R


def score(r):
    """totR / maxDD(R)。"""
    if len(r) < 5:
        return np.nan
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return r.sum() / dd if dd > 0 else np.nan


def capped(g, k):
    g = g.sort_values("time").reset_index(drop=True)
    return g.loc[apply_cap(g, k), "R"].to_numpy()


def run_feed(feed):
    R = prep(feed)
    months = sorted(R["mo"].unique())
    bymo = {mo: g for mo, g in R.groupby("mo")}
    nm = len(months)
    L = R[R["lead"]].sort_values("time").reset_index(drop=True)
    print(f"\n===== {feed.upper()}  母集団 N={len(R)}  {months[0]}〜{months[-1]}（{nm}か月）")
    print(f"  実測の totR/maxDD(R): 全トレード {score(R['R'].to_numpy()):5.2f}  →  "
          f"先導のみ {score(L['R'].to_numpy()):5.2f}  →  "
          f"先導＋上限2本 {score(capped(L, 2)):5.2f}  ·  上限1本 {score(capped(L, 1)):5.2f}")
    print(f"\n  {'ブロック':>9} | {'A 先導フィルタ':>16} | {'B1 上限1本':>14} | {'B2 上限2本':>14}")
    for b in (1, 3, 6, 12):
        winA = winB1 = winB2 = okA = okB1 = okB2 = 0
        for _ in range(NBOOT):
            need = int(np.ceil(nm / b))
            starts = RNG.integers(0, nm, size=need)
            pa, pl, p1, p2 = [], [], [], []
            for stt in starts:
                blk = [months[(stt + i) % nm] for i in range(b)]
                gs = [bymo[mo] for mo in blk if mo in bymo]
                if not gs:
                    continue
                g = pd.concat(gs, ignore_index=True).sort_values("time").reset_index(drop=True)
                pa.append(g["R"].to_numpy())
                gl = g[g["lead"]].reset_index(drop=True)
                if len(gl) == 0:
                    continue
                pl.append(gl["R"].to_numpy())
                p1.append(capped(gl, 1))
                p2.append(capped(gl, 2))
            if not pl:
                continue
            s_all, s_lead = score(np.concatenate(pa)), score(np.concatenate(pl))
            s1, s2 = score(np.concatenate(p1)), score(np.concatenate(p2))
            if np.isfinite(s_all) and np.isfinite(s_lead):
                okA += 1
                winA += int(s_lead > s_all)
            if np.isfinite(s_lead) and np.isfinite(s1):
                okB1 += 1
                winB1 += int(s1 > s_lead)
            if np.isfinite(s_lead) and np.isfinite(s2):
                okB2 += 1
                winB2 += int(s2 > s_lead)
        print(f"  {b:>7}か月 | {winA/max(okA,1)*100:>15.1f}% | "
              f"{winB1/max(okB1,1)*100:>13.1f}% | {winB2/max(okB2,1)*100:>13.1f}%")
    return R


if __name__ == "__main__":
    Rb = run_feed("binance")
    Rv = run_feed("vantage")
    print("\n（読み方: ブロックを長くするほど勝率が上がれば本物。"
          "\n  1か月でだけ高くて12か月で落ちるなら、短い経路のつなぎ方に当てはめている）")
    assert len(Rb) > 500 and len(Rv) > 300, (len(Rb), len(Rv))
    print(f"\nOK: Binance N={len(Rb)} / Vantage N={len(Rv)}")
