"""抜けていた2つ: 時間帯の情報と、ATR×3トレール出口の対照。

トレールは engine の walk() に無い出口なので、この場で前方走査を書く。**同じループの中に
固定RR版も実装し、それが正典の N=556 / PF1.45 を再現することを assert する**（新しい実行系を
黙って分岐させないための紐付け）。用途は対照の計測だけで、運用経路には使わない。
成行(pf=0)のロングに限定＝約定は open[s+1] 決め打ちなので指値の複雑さが入らない。
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


def trig(d, k, sign):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    return s[s + 1 < len(d)], ap


def ents(d, s_list, rr):
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    out = []
    for s in s_list:
        e, st = o[s + 1], l[s]
        if e - st > 0:
            out.append((s, e, st, e + rr * (e - st), s))
    return out


def eng(d, e_list, pf, Cx):
    a = SimpleNamespace(pullback_frac=pf, fill_win=200, fwd=20, cost=0.0,
                        max_pos=1, swap_pct=0.0, tp1_frac=0.0, exec_split=0)
    t, _ = walk(d, e_list, None, a)
    er = (Cx - t["e_px"]) if Cx is not None else t["e_px"]
    return ((t["R"] * t["risk"] - COST * er) / er).to_numpy()


def pf_of(p):
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / ls) if ls > 0 else float("nan")


def line(tag, p):
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    return (f"{tag:<22} N{len(p):4d} 勝率{np.mean(p>0)*100:5.1f}% PF{pf_of(p):6.2f} "
            f"平均{p.mean()*100:+7.3f}% 総{p.sum()*100:+8.1f}% maxDD{dd:6.1f}%")


# ---------------- 1. 時間帯（ブローカー時刻） ----------------
print("========== 時間帯（引き金足の時刻・ブローカー時刻）==========")
for name, d, Cx, k, rr, pf in (("ロング k2.0 RR3", df, None, 2.0, 3.0, 0.0),
                               ("ロング k1.5 RR4.5", df, None, 1.5, 4.5, 0.0),
                               ("ショート k2.0 RR3 指0.5", inv, C, 2.0, 3.0, 0.5),
                               ("ショート k1.5 RR4.5 指0.382", inv, C, 1.5, 4.5, 0.382)):
    s_all, _ = trig(d, k, +1)
    base = eng(d, ents(d, s_all, rr), pf, Cx)
    print(f"\n--- {name}  全体: {line('', base)}")
    hrs = d.index.hour.to_numpy()[s_all]
    for lo, hi, lab in ((0, 8, "アジア 0-7時"), (8, 16, "欧州 8-15時"), (16, 24, "NY 16-23時")):
        sel = s_all[(hrs >= lo) & (hrs < hi)]
        p = eng(d, ents(d, sel, rr), pf, Cx)
        print("   " + line(lab, p))
    # 1時間刻み（当たった時間帯の年別安定性を見るための素データ）
    row = []
    for hh in range(24):
        sel = s_all[hrs == hh]
        if len(sel) < 15:
            row.append(f"{hh:02d}:--")
            continue
        p = eng(d, ents(d, sel, rr), pf, Cx)
        row.append(f"{hh:02d}:{pf_of(p):.2f}")
    print("   時間別PF " + " ".join(row))

# ---------------- 2. トレール出口の対照（ロング成行のみ） ----------------
print("\n\n========== 出口: 固定RR vs ATRトレール（ロング成行）==========")
o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
atr = wilder_atr(df).to_numpy()


def walk_exit(s_list, rr, mode, mult, fwd):
    """同じループで固定RRとトレールを実装。mode='rr' が正典の紐付け対象。"""
    out = []
    busy = -1
    for s in s_list:
        if s <= busy:
            continue
        e_bar, e = s + 1, o[s + 1]
        stop = l[s]
        risk = e - stop
        if risk <= 0:
            continue
        tgt = e + rr * risk
        px, xj = None, min(e_bar + fwd, len(c) - 1)
        for j in range(e_bar + 1, min(e_bar + 1 + fwd, len(c))):
            if mode == "trail":
                stop = max(stop, l[j - 1] - atr[j - 1] * mult)   # 確定足で更新
            if l[j] <= stop:                                     # 同足は損切り優先
                px, xj = stop, j
                break
            if mode == "rr" and h[j] >= tgt:
                px, xj = tgt, j
                break
            if mode == "trail" and c[j] < stop:
                px, xj = c[j], j
                break
        if px is None:
            px = c[xj]
        out.append((px - e - COST * e) / e)
        busy = xj
    return np.array(out)


for k, rr in ((2.0, 3.0), (1.5, 4.5)):
    s_all, _ = trig(df, k, +1)
    print(f"\n--- k={k} RR={rr}")
    print("   " + line(f"固定RR{rr} fwd20(正典)", eng(df, ents(df, s_all, rr), 0.0, None)))
    for fwd in (20, 200):
        print("   " + line(f"固定RR{rr} fwd{fwd}", walk_exit(s_all, rr, "rr", 0, fwd)))
        for m in (2.0, 3.0, 4.0):
            print("   " + line(f"ATR×{m}トレール fwd{fwd}", walk_exit(s_all, rr, "trail", m, fwd)))

# 紐付け: 同じループの固定RR版が正典 walk() と一致すること
s2, _ = trig(df, 2.0, +1)
canon = eng(df, ents(df, s2, 3.0), 0.0, None)
mine = walk_exit(s2, 3.0, "rr", 0, 20)
assert len(canon) == 556, len(canon)
assert abs(len(mine) - 556) <= 2, len(mine)
assert abs(pf_of(mine) - pf_of(canon)) < 0.05, (pf_of(mine), pf_of(canon))
print(f"\nOK: 自前ループの固定RR版 N={len(mine)} PF={pf_of(mine):.2f} が "
      f"正典 walk() N={len(canon)} PF={pf_of(canon):.2f} と一致")
