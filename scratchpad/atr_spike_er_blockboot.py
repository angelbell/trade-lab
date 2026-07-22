"""ERゲートの本番の検定: 巡回ブロック・ブートストラップ。

間引き帰無は通った（BTC 窓120 上位67%で %ile 95.7、ETH で 95.0）が、それは
「同じ価格経路の上で同じ本数を無作為に残す」ことしか訊いていない。同時建玉上限の件で、
間引き帰無を通ってブロック・ブートストラップで落ちる例を見たばかり。

真の改善はブロックを長くするほど勝率が上がる。1/3/6/12か月で確かめる。
指標は totR / maxDD(R)（本数が変わる操作なので PF や平均で比べない）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_er_gate import build, er_series          # noqa: E402
from scratchpad.atr_spike_barspread import spikes                  # noqa: E402

NBOOT = 1000
RNG = np.random.default_rng(3141)


def score(r):
    if len(r) < 5:
        return np.nan
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return r.sum() / dd if dd > 0 else np.nan


if __name__ == "__main__":
    dB, tB = build("btcusd")
    sB = spikes(dB)
    lead = (sB | sB.shift(1)).fillna(False)
    dE, tE = build("ethusd", lead=lead)

    for nm, d, t in (("BTCUSD", dB, tB), ("ETHUSD(BTC確認)", dE, tE)):
        e = er_series(d["close"], 120)
        t = t.copy()
        t["er"] = e.reindex(t["time"]).to_numpy()
        t = t.dropna(subset=["er"])
        t["mo"] = t["time"].dt.to_period("M")
        months = sorted(t["mo"].unique())
        bymo = {m: g for m, g in t.groupby("mo")}
        nm_ = len(months)
        thr67 = t["er"].quantile(0.33)
        thr50 = t["er"].quantile(0.50)
        print(f"\n===== {nm}  N={len(t)}  {months[0]}〜{months[-1]}（{nm_}か月）")
        print(f"  実測 totR/DD: 全体 {score(t['R_net'].to_numpy()):.2f} → "
              f"ER上位67% {score(t.loc[t['er']>=thr67,'R_net'].to_numpy()):.2f} → "
              f"ER上位50% {score(t.loc[t['er']>=thr50,'R_net'].to_numpy()):.2f}")
        print(f"  {'ブロック':>9} | {'ER上位67%が勝つ割合':>22} | {'ER上位50%が勝つ割合':>22}")
        for b in (1, 3, 6, 12):
            w67 = w50 = ok67 = ok50 = 0
            for _ in range(NBOOT):
                need = int(np.ceil(nm_ / b))
                starts = RNG.integers(0, nm_, size=need)
                pa, p67, p50 = [], [], []
                for st in starts:
                    blk = [months[(st + i) % nm_] for i in range(b)]
                    gs = [bymo[m] for m in blk if m in bymo]
                    if not gs:
                        continue
                    g = pd.concat(gs, ignore_index=True)
                    pa.append(g["R_net"].to_numpy())
                    p67.append(g.loc[g["er"] >= thr67, "R_net"].to_numpy())
                    p50.append(g.loc[g["er"] >= thr50, "R_net"].to_numpy())
                if not pa:
                    continue
                s_all = score(np.concatenate(pa))
                s67 = score(np.concatenate(p67))
                s50 = score(np.concatenate(p50))
                if np.isfinite(s_all) and np.isfinite(s67):
                    ok67 += 1
                    w67 += int(s67 > s_all)
                if np.isfinite(s_all) and np.isfinite(s50):
                    ok50 += 1
                    w50 += int(s50 > s_all)
            print(f"  {b:>7}か月 | {w67/max(ok67,1)*100:>21.1f}% | "
                  f"{w50/max(ok50,1)*100:>21.1f}%")

    print("\n（読み方: ブロックを長くするほど勝率が上がれば本物。"
          "\n  横ばい・低下なら短い経路のつなぎ方に当てはめている＝棄却）")
