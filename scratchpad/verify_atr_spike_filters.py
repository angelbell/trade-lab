"""(b) の見出しを独立照合: 前日高値フィルタとアジア時間の間引き帰無。

計測係は「ロング k2.0 で pdh_dist>0 が %ile 100/100」「k1.5 で アジア∩pdh>0 が 97/100」
「ショートは両軸とも帰無割れ」と報告した。ここを自前で組み直して確かめる。
帰無は同じ通過率をフィルタ無し母集団からランダムに残す操作（価格%・400回）。
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
from src.engine.mirror import invert              # noqa: E402

COST = 0.0005


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


df = load_mt5_csv("data/vantage_btcusd_h1.csv")
inv = invert(df)
C = 2 * df["high"].max()


def prep(d):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    return o, h, l, c, ap, (c - pdh) / ap


def run(d, s_list, rr, pf, Cx):
    do, dl = d["open"].to_numpy(), d["low"].to_numpy()
    ent = []
    for s in s_list:
        e, st = do[s + 1], dl[s]
        if e - st > 0:
            ent.append((s, e, st, e + rr * (e - st), s))
    if not ent:
        return None
    a = SimpleNamespace(pullback_frac=pf, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=0.0)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) == 0:
        return None
    er = (Cx - t["e_px"]) if Cx is not None else t["e_px"]
    return t.assign(pnl=((t["R"] * t["risk"] - COST * er) / er))


def pf_of(p):
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / ls) if ls > 0 else float("nan")


def nullpct(p0, obs, reps=400, seed=17):
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(obs) / len(p0) * len(p0))))
    n = min(len(obs), len(p0))
    m, f = [], []
    for _ in range(reps):
        s = rng.choice(p0, size=n, replace=False)
        m.append(s.mean()); f.append(pf_of(s))
    m, f = np.array(m), np.array(f)
    return (float((f < pf_of(obs)).mean() * 100), float((m < obs.mean()).mean() * 100))


CELLS = [("ロング k2.0 RR3", df, None, +1, 2.0, 3.0, 0.0),
         ("ロング k1.5 RR4.5", df, None, +1, 1.5, 4.5, 0.0),
         ("ショート k2.0 RR3 指0.5", inv, C, -1, 2.0, 3.0, 0.5),
         ("ショート k1.5 RR4.5 指0.382", inv, C, -1, 1.5, 4.5, 0.382)]

for name, d, Cx, direction, k, rr, pf in CELLS:
    o, h, l, c, ap, pdh_dist = prep(d)
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s_all = np.flatnonzero(hit)
    s_all = s_all[s_all + 1 < len(d)]
    hrs = d.index.hour.to_numpy()
    base = run(d, s_all, rr, pf, Cx)
    p0 = base["pnl"].to_numpy()
    print(f"\n===== {name}  基準 N={len(p0)} 勝率={np.mean(p0>0)*100:.1f}% "
          f"PF={pf_of(p0):.2f} 平均={p0.mean()*100:+.3f}%")
    filts = {"アジア0-7時": hrs[s_all] < 8,
             "pdh>-1.0": pdh_dist[s_all] > -1.0,
             "pdh>0.0": pdh_dist[s_all] > 0.0,
             "アジア∩pdh>0": (hrs[s_all] < 8) & (pdh_dist[s_all] > 0.0)}
    for fname, mask in filts.items():
        t = run(d, s_all[mask], rr, pf, Cx)
        p = t["pnl"].to_numpy()
        fpc, mpc = nullpct(p0, p)
        yr = t.assign(y=t["time"].dt.year).groupby("y")["pnl"].sum()
        pos = int((yr > 0).sum())
        print(f"  {fname:<14} 通過{mask.mean()*100:5.1f}% N{len(p):4d} 勝率{np.mean(p>0)*100:5.1f}% "
              f"PF{pf_of(p):5.2f} 平均{p.mean()*100:+7.3f}% "
              f"帰無%ile(PF{fpc:5.1f} 平均{mpc:5.1f}) プラス年{pos}/{len(yr)}")

# 検算: 基準セルの既知値
b = run(df, np.flatnonzero((df["close"].to_numpy() - df["open"].to_numpy()
                            > wilder_atr(df).shift(1).to_numpy() * 2.0)
                           & (df["close"].to_numpy() > df["open"].to_numpy())), 3.0, 0.0, None)
assert len(b) == 556, len(b)
assert 1.44 < pf_of(b["pnl"].to_numpy()) < 1.46, pf_of(b["pnl"].to_numpy())
print("\nOK: trail_atr 追加後も基準セル N=556 / PF1.45 を再現")
