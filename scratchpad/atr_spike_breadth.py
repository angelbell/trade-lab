"""横断面: 同じ1時間に何銘柄が同時に拡大足を出したかは、取る/見送るの材料になるか。

動機: 時系列の文脈（トレンドゲート・時間帯・ボラ・文脈5軸）は暗号資産では全滅した。理由は
      「拡大足自身がトレンド検出器だから冗長」。ならば**冗長でない文脈は横断面にあるはず**。
      これは一度も測っていない軸。固定ロットなので「取る/見送る」のしきい値として出す。

A. Binance BTC⇄ETH の相互（2018-2026・8.9年・真の24時間）
   同時＝同じ足 or 直前の足で相手も拡大足。「同時」「単独」に割って、間引き帰無と比べる。
B. Vantage 暗号資産10銘柄の広がり（2022-・商品仕様の変更後）
   その足で拡大足を出した銘柄数を数え、BTC/ETH レッグを広がりで層別する。

🚨 間引き帰無が必須（法則7）。フィルタが玉を減らすだけで見栄えが良くなる分を差し引く。
   帰無は「同じ本数を無作為に残す」を2000回。%ile が 95 を超えて初めて情報がある。
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

FWD, TRAIL, K = 20, 3.0, 2.0
NBOOT = 2000
RNG = np.random.default_rng(4242)


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def prep(path, utc, start):
    d = load_mt5_csv(path)
    if utc:
        # Binance は UTC 保存。ブローカー時刻のラベルに直す（前日高値・週末の定義を Vantage に合わせる）。
        # 夏時間の折り返しで年1本ほど重複が出るので落とす（両銘柄に同じ処理＝整合は保たれる）。
        idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
        d = d.set_index(idx)
        d = d[~d.index.duplicated(keep="first")].sort_index()
    return d.loc[start:]


def spike_series(d):
    """その足が拡大足か（陽線・実体>ATR*K）。時刻つき boolean。"""
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    m = (c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    return pd.Series(m, index=d.index)


def leg_trades(d, cost, we=True):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    m = (c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    if we:
        s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    t = t.copy()
    t["pct"] = (t["R"] * t["risk"] - cost) / t["e_px"]
    # 成行では walk が e_bar=i（引き金足そのもの）を記録するので time が引き金足の時刻
    t["sig_time"] = t["time"]
    assert t["sig_time"].isin(d.index[s]).all(), "引き金足の時刻が母集団と一致しない"
    return t


def stats(p):
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(p), win=(p > 0).mean() * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd * 100,
                score=p.sum() / dd if dd > 0 else np.nan)


def dropnull(all_p, keep_p):
    """同じ本数を無作為に残す帰無に対する %ile（平均% と PF の両方）。"""
    k = len(keep_p)
    if k < 12 or k >= len(all_p):
        return None
    idx = np.arange(len(all_p))
    mu, pf = [], []
    for _ in range(NBOOT):
        q = all_p[RNG.choice(idx, k, replace=False)]
        mu.append(q.mean())
        w, ls = q[q > 0].sum(), -q[q < 0].sum()
        pf.append(w / ls if ls > 0 else np.nan)
    mu, pf = np.array(mu), np.array(pf)
    return ((mu < keep_p.mean()).mean() * 100,
            (pf[np.isfinite(pf)] < stats(keep_p)["pf"]).mean() * 100)


def report(lab, all_p, sub_p):
    if len(sub_p) < 12:
        print(f"    {lab:<26} 本数不足 (N={len(sub_p)})")
        return
    s = stats(sub_p)
    dn = dropnull(all_p, sub_p)
    tag = "" if dn is None else f" 帰無%ile 平均={dn[0]:5.1f} PF={dn[1]:5.1f}"
    print(f"    {lab:<26} N={s['n']:4d} 勝率={s['win']:5.1f}% PF={s['pf']:5.2f} "
          f"平均={s['mean']:+.3f}% DD={s['dd']:5.1f}% 総/DD={s['score']:5.2f}{tag}")


if __name__ == "__main__":
    print("=== A. Binance BTC⇄ETH の相互（2018-2026）")
    B = prep("data/binance_btcusdt_h1.csv", True, "2018-01-01")
    E = prep("data/binance_ethusdt_h1.csv", True, "2018-01-01")
    sB, sE = spike_series(B), spike_series(E)
    keepA = {}
    for nm, d, cost, other in (("BTC", B, 15.0, sE), ("ETH", E, 2.0, sB)):
        t = leg_trades(d, cost)
        p = t["pct"].to_numpy()
        oth = other.reindex(t["sig_time"]).fillna(False).to_numpy()
        oth1 = other.shift(1).reindex(t["sig_time"]).fillna(False).to_numpy()
        co = oth | oth1
        print(f"  -- {nm} レッグ（相手＝{'ETH' if nm=='BTC' else 'BTC'}）")
        report("全トレード", p, p)
        report("同時（同足）", p, p[oth])
        report("同時（同足 or 直前）", p, p[co])
        report("単独（相手は静か）", p, p[~co])
        keepA[nm] = (p, co)

    print("\n=== B. Vantage 暗号資産の広がり（2022-・10銘柄）")
    SYM = ["btcusd", "ethusd", "xrpusd", "ltcusd", "bchusd", "trxusd",
           "solusd", "adausd", "dotusd", "bnbusd"]
    sers = {}
    for s in SYM:
        try:
            d = prep(f"data/vantage_{s}_h1.csv", False, "2022-01-01")
            sers[s] = spike_series(d)
        except Exception as ex:                                    # noqa: BLE001
            print(f"    ({s} 読み込み不可: {ex})")
    W = pd.DataFrame(sers).fillna(False)
    breadth = W.sum(axis=1)
    print(f"    銘柄数={W.shape[1]}  足数={len(W)}  広がりの分布: "
          f"中央値={breadth.median():.0f} 平均={breadth.mean():.2f} σ={breadth.std():.2f} "
          f"90%点={breadth.quantile(0.9):.0f} 最大={breadth.max():.0f}")
    keepB = {}
    for nm, sym, cost in (("BTC", "btcusd", 15.0), ("ETH", "ethusd", 2.0)):
        d = prep(f"data/vantage_{sym}_h1.csv", False, "2022-01-01")
        t = leg_trades(d, cost)
        p = t["pct"].to_numpy()
        br = breadth.reindex(t["sig_time"]).fillna(1).to_numpy()
        print(f"  -- {nm} レッグ（自分を含む同時銘柄数）")
        report("全トレード", p, p)
        for lo, hi, lab in ((1, 1, "自分だけ (=1)"), (2, 2, "2銘柄"), (3, 4, "3-4銘柄"),
                            (5, 99, "5銘柄以上")):
            report(lab, p, p[(br >= lo) & (br <= hi)])
        report("2銘柄以上（累積）", p, p[br >= 2])
        report("3銘柄以上（累積）", p, p[br >= 3])
        keepB[nm] = (p, br)

    pb, co = keepA["BTC"]
    assert 300 <= len(pb) <= 400, len(pb)
    assert stats(pb)["pf"] > 1.4, stats(pb)["pf"]
    print(f"\nOK: BTC の母集団 N={len(pb)} PF={stats(pb)['pf']:.2f} を再現 "
          f"（同時率 {co.mean()*100:.0f}%）")
