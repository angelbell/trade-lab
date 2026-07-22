"""時間のホールドアウト: 前半だけでパラメータを選び、そのまま後半に当てる。

私たちは BTC 一本の上で 5段ぶんの分岐選択（ゲート・文脈軸・閾値・出口・保有時間）を歩いた。
銘柄の切り出し（ETH/SOL/ADA/DOT は未使用データ）は通ったが、時間の切り出しは未実施。
Binance は 2018-2026 の 8.9年ある（真の24時間）ので、ここで初めて測れる。

手続き:
  IS  = 2018-01-01..2021-12-31 だけを見て、格子の中から最良を選ぶ
  OOS = 2022-01-01.. に、その設定を**再調整せずに**当てる
  比較 = (a) OOS を見て選んだ後知恵の最良 (b) 全設定の OOS 中央値 (c) 現行採用設定
  判定 = IS で選んだものが OOS 分布の上位に来るか（来なければ選択自体が過剰当てはめ）
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402


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


def run(d, o, l, c, ap, pdh, k, trail, fwd, use_pdh, skip_we, cost, lo, hi):
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    if use_pdh:
        s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    if skip_we:
        s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    yy = d.index.year.to_numpy()[s]
    s = s[(yy >= lo) & (yy <= hi)]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 15:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=trail, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 15:
        return None
    p = ((t["R"] * t["risk"] - cost) / t["e_px"]).to_numpy()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(n=len(p), pf=w / ls if ls > 0 else np.nan, mean=p.mean() * 100,
                tot=p.sum() * 100, dd=dd * 100, score=p.sum() / dd if dd > 0 else np.nan)


GRID = list(itertools.product((1.5, 2.0, 2.5), (2.0, 3.0, 4.0), (10, 20, 40),
                              (True, False), (True, False)))
CUR = (2.0, 3.0, 20, True, True)          # 現行採用設定

for sym, cost in (("btcusdt", 15.0), ("ethusdt", 2.0)):
    d, o, h, l, c, ap, pdh = prep(f"data/binance_{sym}_h1.csv")
    rows = []
    for k, tr, fw, up, sw in GRID:
        a = run(d, o, l, c, ap, pdh, k, tr, fw, up, sw, cost, 2018, 2021)
        b = run(d, o, l, c, ap, pdh, k, tr, fw, up, sw, cost, 2022, 2026)
        if a and b:
            rows.append(dict(cfg=(k, tr, fw, up, sw), is_score=a["score"], is_pf=a["pf"],
                             oos_score=b["score"], oos_pf=b["pf"], oos_mean=b["mean"],
                             oos_n=b["n"], oos_tot=b["tot"]))
    R = pd.DataFrame(rows)
    best_is = R.loc[R["is_score"].idxmax()]
    best_oos = R.loc[R["oos_score"].idxmax()]
    cur = R[R["cfg"] == CUR].iloc[0]
    rank = (R["oos_score"] < best_is["oos_score"]).mean() * 100

    print(f"\n===== {sym.upper()}  格子 {len(R)} 構成")
    print(f"  IS(2018-21)で選んだ最良      : {best_is['cfg']} "
          f"IS総/DD={best_is['is_score']:.2f} → OOS PF={best_is['oos_pf']:.2f} "
          f"平均{best_is['oos_mean']:+.3f}% 総/DD={best_is['oos_score']:.2f} (n={int(best_is['oos_n'])})")
    print(f"  後知恵でOOSを見て選んだ最良  : {best_oos['cfg']} OOS総/DD={best_oos['oos_score']:.2f}")
    print(f"  現行の採用設定               : {CUR} OOS PF={cur['oos_pf']:.2f} "
          f"総/DD={cur['oos_score']:.2f}")
    print(f"  全構成の OOS総/DD  中央値={R['oos_score'].median():.2f} "
          f"上位25%={R['oos_score'].quantile(0.75):.2f} 下位25%={R['oos_score'].quantile(0.25):.2f}")
    print(f"  → IS選択が OOS 分布の何%ileか: {rank:.0f}%ile"
          f"   （50%付近なら選択に予測力なし＝過剰当てはめ）")
    print(f"  OOS で PF>1 の構成: {(R['oos_pf']>1).mean()*100:.0f}%  "
          f"IS と OOS の総/DD の順位相関: {R['is_score'].corr(R['oos_score'], method='spearman'):+.2f}")

print("\n（判定: 順位相関が正で IS選択が OOS 上位なら、掃引は形を捉えている。"
      "\n  相関ゼロ付近なら、私たちが選んだのは前半の運）")
