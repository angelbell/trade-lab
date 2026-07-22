"""時間足ラダー: この機構は1時間足だけのものか。5m/15m/1h/4h/1d を並べる。

これまで拡大足の研究は全部1時間足の上でやっていた（常設ルール＝全時間足を並べて見る、に対する抜け）。
台帳の関連する事実:
  ❌ btc15m_L のレシピを5分足へ ＝ コストでなく【シグナル】が壊れた（素PF 2.03→1.36）
  「5m/15m はスプレッドで生死」
  一方で btc15m_L 自体はブックの生命線＝15分足そのものが死んでいるわけではない

∴ 見るべきは「PF が落ちるか」ではなく **落ちる理由がコストかシグナルか**。両方を分けて出す:
  - 素（コスト0）の PF/1本R  ＝ シグナルの質
  - 損切り幅の中央値（価格に対する%）と、往復コストが 1R の何%を食うか ＝ コストの効き方
コストを 1R の何%かで見れば、時間足を落とすと同じスプレッドが効いてくる様子が直接見える。

仕様は無調整で移す（k2.0・損切り=引き金足の安値・ATR(14)×3トレール・fwd20本・前日高値>0・土日除外）。
4h/1d は h1 から再標本化（ブローカー時刻のラベルなので日境界も Vantage 慣行に合う）。
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

K, TRAIL = 2.0, 3.0
COSTS = (0.0005, 0.0020)          # 往復。BTC実測は約0.023%、アルトは未実測なので保守側も併記


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def brokerize(d):
    idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    d = d.set_index(idx)
    return d[~d.index.duplicated(keep="first")].sort_index()


def get(sym, tf):
    """tf の足を返す。無ければ h1 から再標本化。無理なら None。"""
    fn = f"data/binance_{sym}_{tf}.csv"
    if os.path.exists(fn):
        return brokerize(load_mt5_csv(fn)).loc["2018-01-01":]
    if tf in ("h4", "d1"):
        h = brokerize(load_mt5_csv(f"data/binance_{sym}_h1.csv")).loc["2018-01-01":]
        rule = "4h" if tf == "h4" else "1D"
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        for v in ("tick_volume", "volume"):
            if v in h.columns:
                agg[v] = "sum"
        return h.resample(rule).agg(agg).dropna()
    return None


def spikes(d):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    return pd.Series((c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0), index=d.index)


def leg(d, fwd, lead=None):
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
    if len(ent) < 20:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 20:
        return None
    t = t.copy()
    t["rf"] = (t["risk"] / t["e_px"])          # 損切り幅（価格に対する率）
    t["gross"] = (t["R"] * t["risk"]) / t["e_px"]
    return t


def stats(t, cost, span):
    p = t["gross"].to_numpy() - cost
    r = p / t["rf"].to_numpy()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(n=len(p), ny=len(p) / span, win=(p > 0).mean() * 100,
                pf=w / ls if ls > 0 else np.nan, meanR=r.mean(),
                totR=r.sum(), ddR=dd, ratio=r.sum() / dd if dd > 0 else np.nan)


TFS = [("m5", "5分", 20), ("m15", "15分", 20), ("h1", "1時間", 20),
       ("h4", "4時間", 20), ("d1", "日足", 20)]

if __name__ == "__main__":
    btc_sp = {}
    for sym in ("btcusdt", "ethusdt"):
        print(f"\n{'='*104}\n===== {sym.upper()}  仕様は無調整で移す（k2.0・端stop・ATR×3トレール・fwd20本）")
        print(f"  {'足':>6} {'バー/年':>8} {'N':>6} {'本/年':>7} {'勝率':>7} "
              f"{'素PF':>6} {'素1本R':>8} | {'損切幅%':>8} {'1Rに占めるコスト 0.05/0.20%':>26} "
              f"| {'PF@0.05':>8} {'R@0.05':>8} {'比@0.05':>8} "
              f"| {'PF@0.20':>8} {'R@0.20':>8} {'比@0.20':>8}")
        for tf, lab, fwd in TFS:
            d = get(sym, tf)
            if d is None:
                print(f"  {lab:>6} データ無し（取得が必要）")
                continue
            if sym == "btcusdt":
                btc_sp[tf] = spikes(d)
            span = (d.index[-1] - d.index[0]).days / 365.25
            bars_y = len(d) / span
            t = leg(d, fwd)
            if t is None:
                print(f"  {lab:>6} {bars_y:>8.0f} 本数不足")
                continue
            g0 = stats(t, 0.0, span)
            g1 = stats(t, COSTS[0], span)
            g2 = stats(t, COSTS[1], span)
            rf = float(np.median(t["rf"].to_numpy()))
            print(f"  {lab:>6} {bars_y:>8.0f} {g0['n']:>6} {g0['ny']:>7.0f} {g0['win']:>6.1f}% "
                  f"{g0['pf']:>6.2f} {g0['meanR']:>+8.3f} | {rf*100:>7.2f}% "
                  f"{COSTS[0]/rf*100:>12.0f}% {COSTS[1]/rf*100:>12.0f}% "
                  f"| {g1['pf']:>8.2f} {g1['meanR']:>+8.3f} {g1['ratio']:>8.2f} "
                  f"| {g2['pf']:>8.2f} {g2['meanR']:>+8.3f} {g2['ratio']:>8.2f}")

    print(f"\n{'='*104}\n===== ETHUSDT に BTC 同時確認をかける（同じ時間足の BTC 拡大足・同足or1本前）")
    print(f"  {'足':>6} {'確認N':>7} {'本/年':>7} {'勝率':>7} {'PF@0.20':>9} {'1本R@0.20':>11} "
          f"{'totR/DD':>9} | {'無確認との差(1本R)':>20}")
    for tf, lab, fwd in TFS:
        d = get("ethusdt", tf)
        if d is None or tf not in btc_sp:
            print(f"  {lab:>6} データ無し")
            continue
        span = (d.index[-1] - d.index[0]).days / 365.25
        lead = (btc_sp[tf] | btc_sp[tf].shift(1)).fillna(False)
        t_all = leg(d, fwd)
        t_led = leg(d, fwd, lead=lead)
        if t_all is None or t_led is None:
            print(f"  {lab:>6} 本数不足")
            continue
        a, b = stats(t_led, COSTS[1], span), stats(t_all, COSTS[1], span)
        print(f"  {lab:>6} {a['n']:>7} {a['ny']:>7.0f} {a['win']:>6.1f}% {a['pf']:>9.2f} "
              f"{a['meanR']:>+11.3f} {a['ratio']:>9.2f} | {a['meanR']-b['meanR']:>+19.3f}")

    d = get("btcusdt", "h1")
    t = leg(d, 20)
    assert t is not None and len(t) > 250, None if t is None else len(t)
    print(f"\nOK: BTC 1時間足の基準 N={len(t)} を再現")
