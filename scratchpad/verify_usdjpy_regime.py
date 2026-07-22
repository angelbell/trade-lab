"""ドル円レッグのレジーム条件付けが本物か。3つの検定を同時にかける。

1. ランダム間引き帰無（同じ通過率をランダムに残す）を超えるか＝ゲートが効いたのか、ただの間引きか
2. 上位3年(2001/2013/2022)を除いてもゲートは効くか＝3年を選び直しているだけではないか
3. 前後半（2000-2012 / 2013-2026）の両方で効くか
BTC では全トレンドゲートが帰無割れした（拡大足自身がトレンド検出器で冗長）。ドル円は管理相場で
トレンドが稀なので、ゲートが「欠けている文脈」を補う可能性がある＝法則3の予言を検定する形。
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

COST = 0.009          # 往復 0.9pip


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


d = load_mt5_csv("data/vantage_usdjpy_h1.csv").loc["2000-01-01":]
o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
ap = wilder_atr(d).shift(1).to_numpy()
day = d.index.floor("D")
_pdh = d["high"].groupby(day).max().shift(1)
pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()

# --- レジーム（すべて確定値のみ。日足/週足は resample→shift→ffill）
dc = d["close"].resample("1D").last().dropna()
sma200 = dc.rolling(200).mean()
reg_sma = (dc > sma200).shift(1).reindex(d.index, method="ffill").fillna(False).to_numpy()
wc = d["close"].resample("W").last().dropna()
w30 = wc.rolling(30).mean()
reg_w30 = (wc > w30).shift(1).reindex(d.index, method="ffill").fillna(False).to_numpy()
datr = wilder_atr(d.resample("1D").agg({"open": "first", "high": "max",
                                        "low": "min", "close": "last"}).dropna())
reg_vol = (datr > datr.rolling(756).median()).shift(1).reindex(d.index,
                                                               method="ffill").fillna(False).to_numpy()


def trig(k=2.0, use_pdh=True):
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    return s[(c[s] - pdh[s]) / ap[s] > 0.0] if use_pdh else s


def go(s_list):
    ent = [(s, o[s + 1], l[s], o[s + 1] + 1000.0 * (o[s + 1] - l[s]), s)
           for s in s_list if o[s + 1] - l[s] > 0]
    if len(ent) < 10:
        return None, None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=3.0, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 10:
        return None, None
    return t, ((t["R"] * t["risk"] - COST) / t["e_px"]).to_numpy()


def pf_of(p):
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / ls) if ls > 0 else float("nan")


def drop_null(p0, obs, reps=400, seed=41):
    rng = np.random.default_rng(seed)
    n = min(len(obs), len(p0))
    f, m = [], []
    for _ in range(reps):
        s = rng.choice(p0, size=n, replace=False)
        f.append(pf_of(s)); m.append(s.mean())
    f, m = np.array(f), np.array(m)
    return (f < pf_of(obs)).mean() * 100, (m < obs.mean()).mean() * 100


s_all = trig()
GATES = {"日足SMA200上": reg_sma, "週足30MA上": reg_w30, "円ボラ高": reg_vol}
BIG3 = {2001, 2013, 2022}

for lab, sel in (("全期間 2000-2026", None),
                 ("上位3年を除く", "no3"),
                 ("前半 2000-2012", ("2000", "2012")),
                 ("後半 2013-2026", ("2013", "2026"))):
    if sel is None:
        ss = s_all
    elif sel == "no3":
        ss = s_all[~np.isin(d.index.year.to_numpy()[s_all], list(BIG3))]
    else:
        yy = d.index.year.to_numpy()[s_all]
        ss = s_all[(yy >= int(sel[0])) & (yy <= int(sel[1]))]
    t0, p0 = go(ss)
    if p0 is None:
        continue
    print(f"\n===== {lab}   ゲート無し: N={len(p0)} 勝率{np.mean(p0>0)*100:.1f}% "
          f"PF={pf_of(p0):.2f} 平均={p0.mean()*100:+.4f}%")
    for gname, g in GATES.items():
        sub = ss[g[ss]]
        t, p = go(sub)
        if p is None:
            continue
        npf, nm = drop_null(p0, p)
        yr = pd.Series(p).groupby(t["time"].dt.year.values).sum()
        print(f"  {gname:<12} 通過{len(sub)/len(ss)*100:5.1f}% N{len(p):4d} "
              f"勝率{np.mean(p>0)*100:5.1f}% PF{pf_of(p):5.2f} 平均{p.mean()*100:+.4f}% "
              f"帰無%ile(PF{npf:5.1f} 平均{nm:5.1f}) 黒字年{int((yr>0).sum())}/{len(yr)}")

# 検算
_, pa = go(s_all)
assert len(pa) == 636, len(pa)
assert 1.29 < pf_of(pa) < 1.32, pf_of(pa)
print(f"\nOK: アンカー N={len(pa)} PF={pf_of(pa):.2f} を再現")
