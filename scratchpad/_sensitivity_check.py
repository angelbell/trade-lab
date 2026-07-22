"""ベータ検定(上位3年除外)の帰無%ileがseed/reps依存で不安定でないかの感度チェック。"""
SCREEN = "atr_spike_btc_h1"
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from types import SimpleNamespace
from src.data_loader import load_mt5_csv
from src.engine.walk import walk

COST = 0.009


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def trigger(d, k, use_pdh):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit); s = s[s + 1 < len(d)]
    if use_pdh: s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    return s


def go(d, s_list, cost_abs):
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    ent = [(s, o[s + 1], l[s], o[s + 1] + 1000.0 * (o[s + 1] - l[s]), s) for s in s_list if o[s + 1] - l[s] > 0]
    if len(ent) < 10: return None, None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=20, cost=0.0, max_pos=1, swap_pct=0.0,
                         tp1_frac=0.0, exec_split=0, trail_atr=3.0, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 10: return None, None
    p = ((t["R"] * t["risk"] - cost_abs) / t["e_px"]).to_numpy()
    return t, p


def rep(p):
    eq = np.cumsum(p); dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(N=len(p), pf=w / ls if ls > 0 else np.nan, mean=p.mean() * 100)


d = load_mt5_csv("data/vantage_usdjpy_h1.csv").loc["2000-01-01":]
s_all = trigger(d, 2.0, True)
t_all, p_all = go(d, s_all, COST)
yr_pnl = pd.Series(p_all).groupby(t_all["time"].dt.year.values).sum().sort_values(ascending=False)
top3 = list(yr_pnl.index[:3])
yrs_sig = d.index[s_all].year.to_numpy()
s_ex = s_all[~np.isin(yrs_sig, top3)]
_, p_ex = go(d, s_ex, COST)
o = rep(p_ex)
pool = trigger(d, 0.0, False)
hrs_pool = d.index.hour.to_numpy()[pool]
hrs_trig = d.index.hour.to_numpy()[s_ex]
cnt = pd.Series(hrs_trig).value_counts()
for seed, reps in [(23, 400), (99, 400), (7, 600)]:
    rng = np.random.default_rng(seed)
    npf, nm = [], []
    for _ in range(reps):
        pick = []
        for hh, n in cnt.items():
            cand = pool[hrs_pool == hh]
            if len(cand): pick.extend(rng.choice(cand, size=min(int(n), len(cand)), replace=False))
        _, pn = go(d, np.sort(np.array(pick)), COST)
        if pn is None: continue
        q = rep(pn); npf.append(q["pf"]); nm.append(q["mean"])
    npf, nm = np.array(npf), np.array(nm)
    print(f"seed={seed} reps={reps}: PF%ile={(npf < o['pf']).mean()*100:.1f} "
          f"mean%ile={(nm < o['mean']).mean()*100:.1f}  null PF med={np.median(npf):.3f} "
          f"mean med={np.median(nm):+.4f}")

assert o["N"] == 552, o["N"]
print("OK")
