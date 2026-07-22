"""見出しセルのベータ点検と年別の広がり。

問い: 「拡大足＋押し目指値」の期待値は、同じ損切り/目標規則で**素の全陽線（全陰線）**から
同数を建てた場合を超えるか。超えなければ拡大足条件は何も足しておらず、BTCの上昇ドリフト
（ロング）を拾っているだけ。時間帯分布は一致させる。
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


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build(d, k, rr):
    atr_prev = wilder_atr(d).shift(1).to_numpy()
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    hit = (c - o > atr_prev * k) & (c > o) & np.isfinite(atr_prev)
    out = []
    for s in np.flatnonzero(hit):
        if s + 1 >= len(d):
            continue
        e, stop = o[s + 1], l[s]
        if e - stop > 0:
            out.append((s, e, stop, e + rr * (e - stop), s))
    return out


def go(d, ent, pf, rr, fwd=20, fill_win=200, cost=0.0005, C=None):
    args = SimpleNamespace(pullback_frac=pf, fill_win=fill_win, fwd=fwd, cost=0.0,
                           max_pos=1, swap_pct=0.0, tp1_frac=0.0, exec_split=0)
    t, _ = walk(d, ent, None, args)
    if t is None or len(t) == 0:
        return None
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]
    pnl = t["R"] * t["risk"] - cost * e_real
    return t.assign(pnl_pct=pnl / e_real)


def pf_of(p):
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("nan")


df = load_mt5_csv("data/vantage_btcusd_h1.csv")
inv = invert(df)
C = 2 * df["high"].max()

CELLS = [("long  成行", df, None, 2.0, 3.0, 0.00),
         ("long  成行", df, None, 1.5, 4.5, 0.00),
         ("short 指値0.5", inv, C, 2.0, 3.0, 0.50),
         ("short 指値0.382", inv, C, 1.5, 4.5, 0.382)]

rng = np.random.default_rng(3)
for name, d, Cx, k, rr, pf in CELLS:
    t = go(d, build(d, k, rr), pf, rr, C=Cx)
    obs_mean = t["pnl_pct"].mean()
    obs_pf = pf_of(t["pnl_pct"].to_numpy())

    # 帰無: 素の全陽線（反転フレームでは全陰線）を同じ規則で建て、時間帯分布を一致させて同数抽出
    pool = build(d, 0.0, rr)
    pool_hours = np.array([d.index[e[0]].hour for e in pool])
    trig_hours = np.array([d.index[e[0]].hour for e in build(d, k, rr)])
    cnt = pd.Series(trig_hours).value_counts()
    null_m, null_pf = [], []
    for _ in range(200):
        pick = []
        for hh, n in cnt.items():
            cand = np.flatnonzero(pool_hours == hh)
            if len(cand) == 0:
                continue
            pick.extend(rng.choice(cand, size=min(int(n), len(cand)), replace=False))
        sub = [pool[j] for j in sorted(pick)]
        tn = go(d, sub, pf, rr, C=Cx)
        if tn is None:
            continue
        null_m.append(tn["pnl_pct"].mean())
        null_pf.append(pf_of(tn["pnl_pct"].to_numpy()))
    null_m, null_pf = np.array(null_m), np.array(null_pf)

    print(f"\n===== {name}  k={k} RR={rr} pf={pf}")
    print(f"  実測: N={len(t)}  PF={obs_pf:.2f}  平均={obs_mean*100:+.3f}%")
    print(f"  素の母集団帰無(同数・時間帯一致): PF中央値={np.median(null_pf):.2f}±{null_pf.std(ddof=1):.2f} "
          f"平均={np.median(null_m)*100:+.3f}%±{null_m.std(ddof=1)*100:.3f}")
    print(f"  → PF %ile={(null_pf < obs_pf).mean()*100:.1f}%  平均 %ile={(null_m < obs_mean).mean()*100:.1f}%")

    t = t.assign(y=t["time"].dt.year)
    yr = t.groupby("y")["pnl_pct"].agg(N="size", 平均=lambda x: x.mean() * 100, PF=lambda x: pf_of(x.to_numpy()))
    print("  年別:", " ".join(f"{int(y)}:N{int(r.N)}/PF{r.PF:.2f}" for y, r in yr.iterrows()))

# 検算: 素の母集団プールが引き金より十分大きいこと／ロング成行の既知値の再現
assert len(build(df, 0.0, 3.0)) > 20000, len(build(df, 0.0, 3.0))
tl = go(df, build(df, 2.0, 3.0), 0.0, 3.0)
assert len(tl) == 556, len(tl)
assert 0.38 < tl["pnl_pct"].mean() * 100 < 0.41, tl["pnl_pct"].mean() * 100
print("\nOK: ロング成行 k=2.0 RR3 は N=556 / 平均+0.393% を再現")
