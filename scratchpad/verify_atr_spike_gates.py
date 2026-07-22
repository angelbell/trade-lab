"""ゲート実験の独立照合＋ゲートの因果性テスト。

照合するのは 1. ロング側で4時間足KAMAがゲート無しを超えるか（間引き帰無の%ile）
2. ショート側でゲートが本当に帰無割れするか（構造法則11の予言と反対なので念入りに）
3. ゲートが先読みしていないか（末尾を切り落としても過去のゲート値が変わらないこと）。
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
from src.engine.gates import gate_kama, gate_sma  # noqa: E402


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build(d, k, rr, gate=None):
    atr_prev = wilder_atr(d).shift(1).to_numpy()
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    hit = (c - o > atr_prev * k) & (c > o) & np.isfinite(atr_prev)
    if gate is not None:
        hit = hit & gate
    out = []
    for s in np.flatnonzero(hit):
        if s + 1 >= len(d):
            continue
        e, stop = o[s + 1], l[s]
        if e - stop > 0:
            out.append((s, e, stop, e + rr * (e - stop), s))
    return out


def go(d, ent, pf, cost=0.0005, C=None):
    args = SimpleNamespace(pullback_frac=pf, fill_win=200, fwd=20, cost=0.0,
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

GATES = {
    "none": None,
    "kama_4h": lambda d: gate_kama(d, SimpleNamespace(gate_kama=14, gate_kama_tf="4h")),
    "kama_1d": lambda d: gate_kama(d, SimpleNamespace(gate_kama=14, gate_kama_tf="1D")),
    "sma_1d150": lambda d: gate_sma(d, SimpleNamespace(daily_sma=150, gate_tf="1D",
                                                      daily_slope_k=0, ext_cap=0))[0],
}

rng = np.random.default_rng(5)
for side, d, Cx, pf, k, rr in (("long", df, None, 0.0, 2.0, 3.0),
                               ("short", inv, C, 0.5, 2.0, 3.0)):
    base = go(d, build(d, k, rr), pf, C=Cx)
    p0 = base["pnl_pct"].to_numpy()
    print(f"\n===== {side} k={k} RR={rr} pf={pf}  ゲート無し: N={len(p0)} "
          f"PF={pf_of(p0):.2f} 平均={p0.mean()*100:+.3f}%")
    print(f"{'ゲート':>10} {'ON%':>6} {'N':>5} {'勝率':>6} {'PF':>6} {'平均%':>8} "
          f"{'帰無PF':>7} {'PF%ile':>7} {'平均%ile':>8}")
    for name, fn in GATES.items():
        if fn is None:
            continue
        g = fn(d)
        t = go(d, build(d, k, rr, gate=g), pf, C=Cx)
        p = t["pnl_pct"].to_numpy()
        q = len(p) / len(p0)
        n = max(1, int(round(q * len(p0))))
        nm, npf = [], []
        for _ in range(400):
            s = rng.choice(p0, size=n, replace=False)
            nm.append(s.mean()); npf.append(pf_of(s))
        nm, npf = np.array(nm), np.array(npf)
        print(f"{name:>10} {g.mean()*100:5.1f}% {len(p):5d} {(p>0).mean()*100:5.1f}% "
              f"{pf_of(p):6.2f} {p.mean()*100:+8.3f} {np.nanmedian(npf):7.2f} "
              f"{(npf < pf_of(p)).mean()*100:6.1f}% {(nm < p.mean()).mean()*100:7.1f}%")

# 因果性テスト: 末尾を切り落としてもそれ以前のゲート値が変わらないこと（＝先読みしていない）
cut = "2023-06-30"
for name, fn in GATES.items():
    if fn is None:
        continue
    full = pd.Series(fn(df), index=df.index).loc[:"2023-06-01"]
    trunc = pd.Series(fn(df.loc[:cut]), index=df.loc[:cut].index).loc[:"2023-06-01"]
    same = int((full.values == trunc.values).sum())
    print(f"[因果性] {name}: 一致 {same}/{len(full)}")
    assert same == len(full), (name, same, len(full))

# 検算: ゲート無しの既知値
tl = go(df, build(df, 2.0, 3.0), 0.0)
assert len(tl) == 556, len(tl)
assert 1.44 < pf_of(tl["pnl_pct"].to_numpy()) < 1.46, pf_of(tl["pnl_pct"].to_numpy())
ts = go(inv, build(inv, 2.0, 3.0), 0.5, C=C)
assert len(ts) == 457, len(ts)
print("\nOK: ゲート無しの基準セル（ロングN=556/PF1.45・ショートN=457）を再現、全ゲートが因果的")
