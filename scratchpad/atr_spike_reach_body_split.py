"""射程を固定して、実体の有無だけを変える（前回 D は本数合わせで閾値を下げ、損切り幅まで変えてしまった）。

射程 reach = (close-low)/ATR は【損切り幅そのもの】。これを揃えないと比較にならない。
同じ reach 帯の中で:
  L 実体あり : body > ATR*2.0   （＝現行の拡大足）
  S 実体小   : body < ATR*1.0   （＝下ヒゲ足・上ヒゲ足）
  M 中間     : その間
本数は自然に不揃いになるので、そのまま出す（揃えるほうが嘘になる）。
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

FWD, TRAIL = 20, 3.0
RNG = np.random.default_rng(7)


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


def run(d, o, l, c, s, cost):
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 12:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 12:
        return None
    p = ((t["R"] * t["risk"] - cost) / t["e_px"]).to_numpy()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    yr = pd.Series(p).groupby(t["time"].dt.year.values).sum()
    return dict(n=len(p), win=(p > 0).mean() * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd * 100,
                score=p.sum() / dd if dd > 0 else np.nan, pos=int((yr > 0).sum()), ny=len(yr),
                risk_med=float(np.median((np.array([e[1] - e[2] for e in ent]) /
                                          np.array([e[1] for e in ent])))) * 100)


def line(lab, r):
    if r is None:
        return f"  {lab:<24}   本数不足"
    return (f"  {lab:<24} N={r['n']:4d} 勝率={r['win']:5.1f}% PF={r['pf']:5.2f} "
            f"平均={r['mean']:+.3f}% 総={r['tot']:+7.1f}% DD={r['dd']:6.1f}% "
            f"総/DD={r['score']:6.2f} 損切幅中央={r['risk_med']:.2f}% 黒字年{r['pos']}/{r['ny']}")


def base(d, o, h, l, c, ap, pdh, reach_lo, reach_hi, body_lo, body_hi):
    reach = np.where(np.isfinite(ap) & (ap > 0), (c - l) / ap, np.nan)
    body = np.where(np.isfinite(ap) & (ap > 0), (c - o) / ap, np.nan)
    m = ((reach >= reach_lo) & (reach < reach_hi) &
         (body >= body_lo) & (body < body_hi) & np.isfinite(ap))
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    return s


if __name__ == "__main__":
    hold = {}
    for sym, cost in (("btcusdt", 15.0), ("ethusdt", 2.0)):
        d, o, h, l, c, ap, pdh = prep(f"data/binance_{sym}_h1.csv")
        print(f"\n===== {sym.upper()} 1H  射程帯を固定して実体だけを変える")
        for rlo, rhi in ((2.0, 3.0), (3.0, 4.5), (4.5, 99.0)):
            print(f"  -- 射程 {rlo}〜{rhi if rhi < 90 else '∞'} ATR（損切り幅がほぼ同じ層）")
            for lab, blo, bhi in (("実体 >2.0ATR (拡大足)", 2.0, 99.0),
                                  ("実体 1.0-2.0ATR", 1.0, 2.0),
                                  ("実体 <1.0ATR (ヒゲ足)", -99.0, 1.0)):
                s = base(d, o, h, l, c, ap, pdh, rlo, rhi, blo, bhi)
                print(line(lab, run(d, o, l, c, s, cost)))
        # 全射程まとめ（>=2.0ATR）
        print("  -- 射程 >=2.0ATR 全体")
        for lab, blo, bhi in (("実体 >2.0ATR (拡大足)", 2.0, 99.0),
                              ("実体 <1.0ATR (ヒゲ足)", -99.0, 1.0)):
            s = base(d, o, h, l, c, ap, pdh, 2.0, 99.0, blo, bhi)
            r = run(d, o, l, c, s, cost)
            print(line(lab, r))
            hold[(sym, "big" if blo == 2.0 else "wick")] = r
    a, w = hold[("btcusdt", "big")], hold[("btcusdt", "wick")]
    e = hold[("ethusdt", "wick")]
    assert a["n"] > 200 and a["pf"] > 1.3, (a["n"], a["pf"])
    assert w["pf"] < 1.0 and e["pf"] < 1.0, (w["pf"], e["pf"])
    print(f"\nOK: 射程>=2.0ATR を揃えても 拡大足 PF={a['pf']:.2f} > ヒゲ足 "
          f"BTC {w['pf']:.2f}/ETH {e['pf']:.2f} ＝実体そのものに情報がある")
