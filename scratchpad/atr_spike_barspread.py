"""拡大足が出た【その足そのもの】の実スプレッドで、レッグを計算し直す。

これまでのコストの扱い:
  第1版 一律 0.05〜0.20% を仮定（未実測。実は 1.5〜50倍 甘かった）
  第2版 デモ口座の tick を1点だけ実測（閑散な時間帯・現在値のみ）
  第3版（本節）**MT5 の履歴バーが持つ per-bar spread を全期間ぶん取得**
        → 拡大足が出た瞬間のスプレッドで、1トレードずつ課金する

これで「拡大足の瞬間はスプレッドが開くのでは」という最後の未測定変数が消える。

執行の型（CLAUDE.md）: バーは bid ベース、買いは ask で入る ∴ 往復の価格距離コスト＝1×スプレッド。
建てるのは引き金足の次バー始値なので、課金するのは**その次バーのスプレッド**。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402

K, TRAIL = 2.0, 3.0
SPREADS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "bar_spreads.json")))


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def spread_series(key):
    """スプレッドを『価格に対する率』の Series（ブローカー時刻）で返す。"""
    o = SPREADS[key]
    d = pd.DataFrame(o["rates"])
    t = pd.to_datetime(d["time"], unit="s", utc=True)
    pct = d["spread"].to_numpy() / 10 ** o["digits"] / d["close"].to_numpy()
    s = pd.Series(pct, index=t)
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_px(sym, tf, start):
    return load_mt5_csv(f"data/vantage_{sym}_{tf}.csv").loc[start:]


def spikes(d):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    return pd.Series((c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0), index=d.index)


def leg(d, fwd, cost_series=None, flat=None, lead=None):
    """cost_series: 建玉バー（i+1）のスプレッド率。flat: 一律コスト率（比較用）。"""
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    m = (c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    if lead is not None:
        lm = lead.reindex(d.index).fillna(False).to_numpy()
        s = s[lm[s]]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 15:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 15:
        return None
    t = t.copy()
    t["rf"] = t["risk"] / t["e_px"]
    gross = (t["R"] * t["risk"]) / t["e_px"]
    if flat is not None:
        cost = np.full(len(t), flat)
    else:
        # 引き金足の次バー（＝約定バー）のスプレッド。無ければ前方の直近値。
        nxt = t["time"] + (d.index[1] - d.index[0])
        cost = cost_series.reindex(nxt, method="ffill").to_numpy()
    t["cost"] = cost
    t["pct"] = gross - cost
    t["R_net"] = t["pct"] / t["rf"]
    return t.dropna(subset=["pct"])


def show(t, span, lab):
    p = t["pct"].to_numpy()
    r = t["R_net"].to_numpy()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    print(f"    {lab:<22} N={len(p):4d} 年{len(p)/span:3.0f}本 勝率={(p>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本R={r.mean():+.3f} "
          f"totR={r.sum():+7.1f} DD={dd:6.1f}R 比={r.sum()/dd if dd>0 else np.nan:6.2f}")
    return dict(pf=w / ls if ls > 0 else np.nan, meanR=r.mean(),
                ratio=r.sum() / dd if dd > 0 else np.nan, n=len(p))


ALTS = ["ethusd", "solusd", "adausd", "dotusd", "xrpusd", "ltcusd",
        "bchusd", "bnbusd", "trxusd"]

if __name__ == "__main__":
    btc = load_px("btcusd", "h1", "2022-01-01")
    sB = spikes(btc)
    lead = (sB | sB.shift(1)).fillna(False)
    span = (btc.index[-1] - btc.index[0]).days / 365.25
    spb = spread_series("BTCUSD|h1")

    print(f"=== 1時間足・実スプレッド課金（Vantage 2022-・{span:.1f}年）")
    print("  -- BTCUSD（確認条件なし）")
    tb = leg(btc, 20, cost_series=spb)
    show(leg(btc, 20, flat=0.0020), span, "一律0.20%")
    show(leg(btc, 20, flat=0.00038), span, "一律0.038%(現値)")
    rb = show(tb, span, "★実スプレッド")
    print(f"       課金されたスプレッドの中央値={tb['cost'].median()*100:.4f}% "
          f"（1Rの{tb['cost'].median()/tb['rf'].median()*100:.0f}%）"
          f" 90%点={tb['cost'].quantile(0.9)*100:.4f}%")

    alive = []
    for sym in ALTS:
        key = f"{sym.upper()}|h1"
        if key not in SPREADS:
            continue
        d = load_px(sym, "h1", "2022-01-01")
        sp = spread_series(key)
        t = leg(d, 20, cost_series=sp, lead=lead)
        if t is None:
            print(f"  -- {sym.upper()}: 本数不足")
            continue
        print(f"  -- {sym.upper()}（BTC同時確認つき）")
        show(leg(d, 20, flat=0.0020, lead=lead), span, "一律0.20%")
        r = show(t, span, "★実スプレッド")
        med = t["cost"].median()
        print(f"       課金スプレッド 中央値={med*100:.3f}% "
              f"（1Rの{med/t['rf'].median()*100:.0f}%） 90%点={t['cost'].quantile(0.9)*100:.3f}%")
        if r["pf"] > 1.2 and r["ratio"] > 2.0:
            alive.append(sym)
    print(f"\n  実スプレッドでの生存: {alive if alive else 'なし'}")

    assert rb["n"] > 150, rb
    print(f"\nOK: BTC 実スプレッド PF={rb['pf']:.2f} 比={rb['ratio']:.2f}")
