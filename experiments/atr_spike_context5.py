"""第4段 STEP1: 拡大足に「トレンド以外の文脈」5軸の情報があるか（閾値なしの分位測定）。

トレンド系ゲートは全滅済み（拡大足自身がトレンド検出器＝冗長）。ここで測るのは
「どこで起きたか」の軸。閾値は切らない。単調な勾配が無い軸はこの時点で落とす。

🚨 変数はすべて**実フレーム**で計算し、ショート版は明示的に符号を反転させる。
反転フレームで比率（乖離%など）を作ると鏡像化が壊れる（x_conventions#mirror-cost-overcharge の兄弟）。
損益は入口価格に対する%（ロット0.01固定）。R では測らない。
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


df = load_mt5_csv("data/vantage_btcusd_h1.csv")
inv = invert(df)
C = 2 * df["high"].max()
o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
atr_prev = wilder_atr(df).shift(1).to_numpy()

# --- 文脈変数（すべて確定値のみ。日足は resample→shift(1)→ffill で先読み無し）
day = df.index.floor("D")
pdh = df["high"].groupby(day).max().shift(1).reindex(df.index, method=None)
pdl = df["low"].groupby(day).min().shift(1).reindex(df.index, method=None)
pdh = pd.Series(pdh.to_numpy(), index=df.index).ffill()
pdl = pd.Series(pdl.to_numpy(), index=df.index).ffill()
dclose = df["close"].groupby(day).last()
sma150 = dclose.rolling(150).mean().shift(1).reindex(df.index).ffill()
roll_hi = df["high"].rolling(20).max().shift(1).to_numpy()
roll_lo = df["low"].rolling(20).min().shift(1).to_numpy()
rng = (h - l)
atr_pct = pd.Series(atr_prev, index=df.index).rolling(100).rank(pct=True).to_numpy()

VARS_L = {
    "前日高値との距離": (c - pdh.to_numpy()) / atr_prev,
    "日足SMA150乖離%": (c - sma150.to_numpy()) / sma150.to_numpy() * 100,
    "直近20本高値超え": (c - roll_hi) / atr_prev,
    "終値位置(締まり)": np.divide(c - l, rng, out=np.full_like(c, np.nan), where=rng > 0),
    "ATR分位(収縮明け)": atr_pct,
}
VARS_S = {
    "前日安値との距離": (pdl.to_numpy() - c) / atr_prev,
    "日足SMA150乖離%": -(c - sma150.to_numpy()) / sma150.to_numpy() * 100,
    "直近20本安値割れ": (roll_lo - c) / atr_prev,
    "終値位置(締まり)": np.divide(h - c, rng, out=np.full_like(c, np.nan), where=rng > 0),
    "ATR分位(収縮明け)": atr_pct,
}


def trigger_idx(k, direction):
    hit = ((c - o > atr_prev * k) if direction > 0 else (o - c > atr_prev * k))
    hit &= ((c > o) if direction > 0 else (c < o)) & np.isfinite(atr_prev)
    s = np.flatnonzero(hit)
    return s[s + 1 < len(df)]


def run(d, s_list, rr, pf, Cx):
    """引き金位置の配列から entries を組んで walk。損益は価格%。"""
    do, dl = d["open"].to_numpy(), d["low"].to_numpy()
    ent = []
    for s in s_list:
        e, stop = do[s + 1], dl[s]
        if e - stop > 0:
            ent.append((s, e, stop, e + rr * (e - stop), s))
    if not ent:
        return None
    args = SimpleNamespace(pullback_frac=pf, fill_win=200, fwd=20, cost=0.0,
                           max_pos=1, swap_pct=0.0, tp1_frac=0.0, exec_split=0)
    t, _ = walk(d, ent, None, args)
    if t is None or len(t) == 0:
        return None
    e_real = (Cx - t["e_px"]) if Cx is not None else t["e_px"]
    pnl = t["R"] * t["risk"] - 0.0005 * e_real
    return (pnl / e_real).to_numpy()


def pf_of(p):
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / ls) if ls > 0 else float("nan")


CELLS = [("ロング k2.0 RR3", df, None, +1, 2.0, 3.0, 0.0),
         ("ロング k1.5 RR4.5", df, None, +1, 1.5, 4.5, 0.0),
         ("ショート k2.0 RR3 指値0.5", inv, C, -1, 2.0, 3.0, 0.5),
         ("ショート k1.5 RR4.5 指値0.382", inv, C, -1, 1.5, 4.5, 0.382)]

for name, d, Cx, direction, k, rr, pf in CELLS:
    s_all = trigger_idx(k, direction)
    base = run(d, s_all, rr, pf, Cx)
    print(f"\n===== {name}   フィルタ無し: N={len(base)} 勝率={np.mean(base>0)*100:.1f}% "
          f"PF={pf_of(base):.2f} 平均={base.mean()*100:+.3f}%")
    V = VARS_L if direction > 0 else VARS_S
    for vname, arr in V.items():
        v = arr[s_all]
        ok = np.isfinite(v)
        if ok.sum() < 100:
            continue
        qs = np.quantile(v[ok], [0.2, 0.4, 0.6, 0.8])
        row = []
        for b in range(5):
            lo = -np.inf if b == 0 else qs[b - 1]
            hi = np.inf if b == 4 else qs[b]
            sel = s_all[ok & (v > lo) & (v <= hi)] if b > 0 else s_all[ok & (v <= qs[0])]
            p = run(d, sel, rr, pf, Cx)
            row.append((len(p) if p is not None else 0,
                        pf_of(p) if p is not None else float("nan"),
                        p.mean() * 100 if p is not None else float("nan")))
        print(f"  {vname:<18} 分位1→5  " +
              " | ".join(f"N{n:3d} PF{f:5.2f} {m:+.3f}%" for n, f, m in row))

# 検算: 既知の基準セル
b1 = run(df, trigger_idx(2.0, +1), 3.0, 0.0, None)
assert len(b1) == 556, len(b1)
assert 1.44 < pf_of(b1) < 1.46, pf_of(b1)
b2 = run(inv, trigger_idx(2.0, -1), 3.0, 0.5, C)
assert len(b2) == 457, len(b2)
assert 1.26 < pf_of(b2) < 1.28, pf_of(b2)
# 変数の値域
cp = VARS_L["終値位置(締まり)"]
cp = cp[np.isfinite(cp)]
assert cp.min() >= 0.0 and cp.max() <= 1.0, (cp.min(), cp.max())
ap = atr_pct[np.isfinite(atr_pct)]
assert ap.min() >= 0.0 and ap.max() <= 1.0, (ap.min(), ap.max())
print("\nOK: 基準セル(556/1.45・457/1.27)を再現、close_pos と atr_pctile は[0,1]")
