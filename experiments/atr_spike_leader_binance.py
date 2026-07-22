"""先導者仮説を独立フィード（Binance・真の24時間・2018-2026）で確認する。

Vantage 2022- で得た結果: アルト9銘柄プール N=1281、BTC同時 +0.913%/PF1.87 vs BTC静か +0.032%/PF1.02、
帰無%ile 100、符号一致 8/9、5年とも同符号、実体3層×ボラ3層 6/6、往復0.20%でも PF1.85。
穴は「独立フィードが無い」「4.5年しかない」の2つ。Binance のバルクで両方を埋める。

🔑 2018-2021 は Vantage では測れない期間（暗号資産が平日限定の商品だった）＝**本物の時間ホールドアウト**。
時刻はブローカー時刻のラベルに直す（前日高値・週末の定義を Vantage と揃えるため）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402

FWD, TRAIL, K = 20, 3.0, 2.0
NBOOT = 2000
RNG = np.random.default_rng(1234)
ALTS = ["ethusdt", "xrpusdt", "ltcusdt", "bchusdt", "trxusdt",
        "solusdt", "adausdt", "dotusdt", "bnbusdt"]


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def load(sym):
    d = load_mt5_csv(f"data/binance_{sym}_h1.csv")
    idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    d = d.set_index(idx)
    d = d[~d.index.duplicated(keep="first")].sort_index()
    return d.loc["2018-01-01":]


def spikes(d):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    return pd.Series((c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0), index=d.index)


def leg(d):
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
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 15:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None:
        return None
    t = t.copy()
    t["gross"] = (t["R"] * t["risk"]) / t["e_px"]
    return t


def st(p):
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return (len(p), (p > 0).mean() * 100, w / ls if ls > 0 else np.nan, p.mean() * 100)


def dropnull(allp, keep):
    k = len(keep)
    if k < 12 or k >= len(allp):
        return None
    mu, pf = [], []
    for _ in range(NBOOT):
        q = allp[RNG.choice(len(allp), k, replace=False)]
        mu.append(q.mean())
        w, ls = q[q > 0].sum(), -q[q < 0].sum()
        pf.append(w / ls if ls > 0 else np.nan)
    mu, pf = np.array(mu), np.array(pf)
    return ((mu < keep.mean()).mean() * 100,
            (pf[np.isfinite(pf)] < st(keep)[2]).mean() * 100)


def block(R, cost, lab):
    p = R["gross"].to_numpy() - cost
    b = R["btc"].to_numpy()
    a, o = p[b], p[~b]
    dn = dropnull(p, a)
    print(f"  {lab:<22} 全体 N={len(p):4d} PF={st(p)[2]:5.2f} {st(p)[3]:+.3f}%  |  "
          f"BTC同時 N={len(a):4d} 勝率{st(a)[1]:5.1f}% PF={st(a)[2]:5.2f} {st(a)[3]:+.3f}%  |  "
          f"静か N={len(o):4d} PF={st(o)[2]:5.2f} {st(o)[3]:+.3f}%  |  帰無%ile={dn[0]:5.1f}/{dn[1]:5.1f}")


if __name__ == "__main__":
    btc = load("btcusdt")
    sB = (spikes(btc) | spikes(btc).shift(1)).fillna(False)

    recs = []
    for sym in ALTS:
        d = load(sym)
        t = leg(d)
        if t is None:
            print(f"  {sym}: 本数不足")
            continue
        recs.append(pd.DataFrame({"sym": sym, "gross": t["gross"].to_numpy(),
                                  "y": t["time"].dt.year.values,
                                  "btc": sB.reindex(t["time"]).fillna(False).to_numpy()}))
    R = pd.concat(recs, ignore_index=True)
    print(f"=== Binance アルト9銘柄プール 2018-2026  N={len(R)}  BTC同時 {R['btc'].mean()*100:.0f}%")

    for cost in (0.0005, 0.0020):
        print(f"\n--- 往復コスト {cost*100:.2f}%")
        block(R, cost, "全期間 2018-2026")
        block(R[R["y"] <= 2021], cost, "IS  2018-2021")
        block(R[R["y"] >= 2022], cost, "OOS 2022-2026")

    print("\n=== 銘柄別（コスト0.05%・全期間）")
    sgn = 0
    tot = 0
    for sym in ALTS:
        s = R[R["sym"] == sym]
        a = s.loc[s["btc"], "gross"].to_numpy() - 0.0005
        o = s.loc[~s["btc"], "gross"].to_numpy() - 0.0005
        if len(a) < 20 or len(o) < 20:
            print(f"  {sym:<10} 本数不足")
            continue
        tot += 1
        sgn += int(a.mean() > o.mean())
        print(f"  {sym:<10} 同時 N={len(a):4d} PF={st(a)[2]:5.2f} {st(a)[3]:+.3f}%  |  "
              f"静か N={len(o):4d} PF={st(o)[2]:5.2f} {st(o)[3]:+.3f}%  |  差={st(a)[3]-st(o)[3]:+.3f}%")
    print(f"  符号の一致: {sgn}/{tot} → 二項検定 P={sps.binomtest(sgn, tot, 0.5).pvalue:.4f}")

    print("\n=== 年別（コスト0.05%・平均%／本数）")
    ys = sorted(R["y"].unique())
    print("    " + " ".join(f"{y:>15}" for y in ys))
    for lab, m in (("BTC同時", R["btc"]), ("BTC静か", ~R["btc"])):
        cells = []
        for y in ys:
            q = R[m & (R["y"] == y)]["gross"].to_numpy() - 0.0005
            cells.append(f"{q.mean()*100:+.3f}%/{len(q):3d}" if len(q) >= 8 else "      --     ")
        print(f"  {lab:<8}" + " ".join(f"{x:>15}" for x in cells))

    a = R.loc[R["btc"], "gross"].to_numpy() - 0.0005
    o = R.loc[~R["btc"], "gross"].to_numpy() - 0.0005
    assert len(R) > 2000, len(R)
    assert st(a)[2] > st(o)[2], (st(a)[2], st(o)[2])
    print(f"\nOK: Binance でも BTC同時 PF={st(a)[2]:.2f} > 静か PF={st(o)[2]:.2f} (N={len(R)})")
