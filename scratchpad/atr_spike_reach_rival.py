"""対立仮説: 引き金の正体は「実体の大きさ」ではなく「終値が自分の安値から遠いこと（射程）」ではないか。

クラウディング診断の距離一致帰無で、k=2.0 の続き率リフトが後半 ほぼゼロになった。
帰無＝同じ (close-low)/ATR を持つ【非】拡大足。ならばトレード水準で直接ぶつける。

比較する引き金（他は全て同一: 損切り=引き金足の安値・ATR×3トレール・fwd20・前日高値>0・土日建て禁止）:
  A 拡大足      : 実体 > ATR*k かつ陽線                    ← 現行採用
  B 射程のみ    : (close-low)/ATR > r（実体・方向を問わない）
  C 射程＋陽線  : B かつ close > open
  D 射程＋小実体: B かつ 実体 < ATR*0.5（＝下ヒゲ足だけ）   ← A の補集合に近い側
r は各銘柄で「A と本数がそろう値」を探索して固定する（本数を合わせないと比較にならない）。

判定: B/C が A と同等以上なら、引き金はより単純で本数の多い形に置換できる。
      A > B なら「大陽線であること」自体に情報がある。
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

K = 2.0
FWD, TRAIL = 20, 3.0


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def prep(path):
    d = load_mt5_csv(path)
    idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    d = d.set_index(idx).loc["2018-01-01":]
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    return d, o, h, l, c, ap, pdh


def sel(d, o, h, l, c, ap, pdh, kind, thr, lo=2018, hi=2026):
    body = c - o
    reach = np.where(np.isfinite(ap) & (ap > 0), (c - l) / ap, np.nan)
    if kind == "A":
        m = (body > ap * thr) & (c > o)
    elif kind == "B":
        m = reach > thr
    elif kind == "C":
        m = (reach > thr) & (c > o)
    elif kind == "D":
        m = (reach > thr) & (body < ap * 0.5)
    m = m & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    yy = d.index.year.to_numpy()[s]
    return s[(yy >= lo) & (yy <= hi)]


def run(d, o, l, c, ap, s, cost):
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 15:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 15:
        return None
    p = ((t["R"] * t["risk"] - cost) / t["e_px"]).to_numpy()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    yr = pd.Series(p).groupby(t["time"].dt.year.values).sum()
    return dict(n=len(p), win=(p > 0).mean() * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd * 100,
                score=p.sum() / dd if dd > 0 else np.nan,
                pos=int((yr > 0).sum()), ny=len(yr))


def match_thr(d, o, h, l, c, ap, pdh, kind, target):
    """本数が target に最も近くなる閾値を返す。"""
    best, bt = None, None
    for r in np.arange(0.5, 6.01, 0.05):
        n = len(sel(d, o, h, l, c, ap, pdh, kind, r))
        if best is None or abs(n - target) < abs(best - target):
            best, bt = n, r
    return bt, best


def line(lab, r):
    if r is None:
        return f"  {lab:<26}   --"
    return (f"  {lab:<26} N={r['n']:4d} 勝率={r['win']:5.1f}% PF={r['pf']:5.2f} "
            f"平均={r['mean']:+.3f}% 総={r['tot']:+7.1f}% DD={r['dd']:6.1f}% "
            f"総/DD={r['score']:5.2f} 黒字年{r['pos']}/{r['ny']}")


if __name__ == "__main__":
    keep = {}
    for sym, cost in (("btcusdt", 15.0), ("ethusdt", 2.0)):
        d, o, h, l, c, ap, pdh = prep(f"data/binance_{sym}_h1.csv")
        sA = sel(d, o, h, l, c, ap, pdh, "A", K)
        rA = run(d, o, l, c, ap, sA, cost)
        print(f"\n===== {sym.upper()} 1H  Binance 2018-2026  (本数を A={rA['n']} に合わせる)")
        print(line(f"A 拡大足 body>{K}ATR", rA))
        keep[sym] = rA
        for kind, nm in (("B", "B 射程のみ"), ("C", "C 射程＋陽線"), ("D", "D 射程＋小実体")):
            thr, got = match_thr(d, o, h, l, c, ap, pdh, kind, len(sA))
            s = sel(d, o, h, l, c, ap, pdh, kind, thr)
            print(line(f"{nm} reach>{thr:.2f}ATR", run(d, o, l, c, ap, s, cost)))
        # 重なりの度合い（同じトレードを選んでいるだけではないか）
        thrC, _ = match_thr(d, o, h, l, c, ap, pdh, "C", len(sA))
        sC = sel(d, o, h, l, c, ap, pdh, "C", thrC)
        ov = len(np.intersect1d(sA, sC))
        print(f"  → A と C の重なり: {ov} 本 ({ov/len(sA)*100:.0f}% of A, "
              f"{ov/max(len(sC),1)*100:.0f}% of C)")

        print(f"  --- 前半(2018-2021) / 後半(2022-2026) 別")
        for kind, thr, nm in (("A", K, "A 拡大足"), ("C", thrC, "C 射程＋陽線")):
            for lo, hi, tag in ((2018, 2021, "前半"), (2022, 2026, "後半")):
                s = sel(d, o, h, l, c, ap, pdh, kind, thr, lo, hi)
                print(line(f"{nm} {tag}", run(d, o, l, c, ap, s, cost)))

    assert 300 <= keep["btcusdt"]["n"] <= 700, keep["btcusdt"]["n"]
    assert 1.2 < keep["btcusdt"]["pf"] < 2.2, keep["btcusdt"]["pf"]
    print(f"\nOK: BTC の A系が既知帯 (N={keep['btcusdt']['n']}, PF={keep['btcusdt']['pf']:.2f}) と整合")
