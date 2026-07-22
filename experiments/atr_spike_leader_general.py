"""先導者仮説は暗号資産の内輪の話か、それとも横断面という新しい層か。

暗号資産で見つけたもの: アルトの拡大足は「BTCも同時に拡大足か」で PF1.87 vs 1.02 に割れる
（2フィード・帰無%ile 99-100・符号一致8/9・実体3層×ボラ3層 6/6・偽薬ゼロ）。

一般化の検定: 同じ手続きを他の一家に当てる。各一家の各銘柄について
「**同じ足か直前の足で、同じ一家の他の銘柄も拡大足を出したか**」で割り、間引き帰無と比べる。
一家 = 貴金属(金/銀/銅[+白金]) · 指数(NAS100/GER40[+US2000]) · FX対ドル(EUR/GBP/AUD/NZD) ·
      FXドル建て(USDJPY/USDCAD)。

🔑 ここが本番: 金・指数・原油は17銘柄横展開で全滅した（素の拡大足がランダム建て帰無を超えない）。
   **横断面の条件付けで生き返るなら、これは暗号資産の話ではなく層の話**。生き返らないなら
   「暗号資産という一家の内部事情」に格下げする。

🚨 金 h1 は 2018-01-01 以降のみ（疎データの罠）。一家の中で期間をそろえる。
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
RNG = np.random.default_rng(31337)

# (一家名, 開始日, [(銘柄, 往復コスト率)])
FAMILIES = [
    ("貴金属 (金/銀/銅)", "2018-01-01",
     [("xauusd", 0.0003), ("xagusd", 0.0010), ("copper-cr", 0.0010)]),
    ("貴金属+白金", "2022-03-01",
     [("xauusd", 0.0003), ("xagusd", 0.0010), ("copper-cr", 0.0010), ("xptusd.r", 0.0015)]),
    ("指数 (NAS100/GER40)", "2016-01-01",
     [("nas100.r", 0.0002), ("ger40.r", 0.0002)]),
    ("指数+US2000", "2020-05-01",
     [("nas100.r", 0.0002), ("ger40.r", 0.0002), ("us2000.r", 0.0003)]),
    ("FX 対ドル (EUR/GBP/AUD/NZD)", "2000-01-01",
     [("eurusd", 0.0001), ("gbpusd", 0.0001), ("audusd", 0.0001), ("nzdusd", 0.0001)]),
    ("FX ドル建て (JPY/CAD)", "2000-01-01",
     [("usdjpy", 0.0001), ("usdcad", 0.0001)]),
]


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def load(sym, start):
    d = load_mt5_csv(f"data/vantage_{sym}_h1.csv").loc[start:]
    return d[~d.index.duplicated(keep="first")].sort_index()


def spikes(d):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    return pd.Series((c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0), index=d.index)


def leg(d, cost):
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
    t["pct"] = (t["R"] * t["risk"]) / t["e_px"] - cost
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


def show(lab, allp, sub):
    if len(sub) < 15:
        print(f"    {lab:<26} 本数不足 N={len(sub)}")
        return None
    n, w, pf, mu = st(sub)
    dn = dropnull(allp, sub)
    tg = "" if dn is None else f"  帰無%ile 平均={dn[0]:5.1f} PF={dn[1]:5.1f}"
    print(f"    {lab:<26} N={n:5d} 勝率={w:5.1f}% PF={pf:5.2f} 平均={mu:+.3f}%{tg}")
    return mu


if __name__ == "__main__":
    summary = []
    for fam, start, members in FAMILIES:
        sp, tr = {}, {}
        for sym, cost in members:
            try:
                d = load(sym, start)
            except Exception as ex:                                # noqa: BLE001
                print(f"  {sym}: 読込不可 {ex}")
                continue
            sp[sym] = spikes(d)
            t = leg(d, cost)
            if t is not None:
                tr[sym] = t
        if len(tr) < 2:
            print(f"\n===== {fam}: 銘柄が足りない")
            continue
        print(f"\n===== {fam}  ({start}- ・{len(tr)}銘柄)")
        recs = []
        for sym, t in tr.items():
            others = [sp[s] for s in sp if s != sym]
            lead = others[0].copy() * False
            for o in others:
                lead = lead | o.reindex(lead.index).fillna(False)
            lead = (lead | lead.shift(1)).fillna(False)
            m = lead.reindex(t["time"]).fillna(False).to_numpy()
            recs.append(pd.DataFrame({"sym": sym, "pct": t["pct"].to_numpy(),
                                      "y": t["time"].dt.year.values, "lead": m}))
        R = pd.concat(recs, ignore_index=True)
        allp = R["pct"].to_numpy()
        show("全トレード（素の母集団）", allp, allp)
        a = R.loc[R["lead"], "pct"].to_numpy()
        b = R.loc[~R["lead"], "pct"].to_numpy()
        ma = show("一家の他銘柄も同時", allp, a)
        mb = show("自分だけ（他は静か）", allp, b)
        print(f"    {'--- 銘柄別 ---':<26} 同時率={R['lead'].mean()*100:.0f}%")
        sgn, tot = 0, 0
        for sym in tr:
            s = R[R["sym"] == sym]
            x = s.loc[s["lead"], "pct"].to_numpy()
            y = s.loc[~s["lead"], "pct"].to_numpy()
            if len(x) < 15 or len(y) < 15:
                print(f"      {sym:<14} 本数不足 ({len(x)}/{len(y)})")
                continue
            tot += 1
            sgn += int(x.mean() > y.mean())
            print(f"      {sym:<14} 同時 N={len(x):4d} PF={st(x)[2]:5.2f} {st(x)[3]:+.3f}%"
                  f"  |  静か N={len(y):4d} PF={st(y)[2]:5.2f} {st(y)[3]:+.3f}%"
                  f"  |  差={st(x)[3]-st(y)[3]:+.3f}%")
        if tot >= 2:
            print(f"      符号の一致 {sgn}/{tot}"
                  f"  二項P={sps.binomtest(sgn, tot, 0.5).pvalue:.3f}")
        if ma is not None and mb is not None:
            summary.append((fam, len(allp), st(allp)[2], st(a)[2], st(b)[2], ma - mb, sgn, tot))

    print("\n" + "=" * 100)
    print("=== まとめ（暗号資産の既測値を最下段に置く）")
    print(f"  {'一家':<30} {'N':>6} {'素PF':>6} {'同時PF':>7} {'静かPF':>7} {'差(平均%)':>10} {'符号一致':>8}")
    for fam, n, pf0, pfa, pfb, dif, sgn, tot in summary:
        print(f"  {fam:<30} {n:>6} {pf0:>6.2f} {pfa:>7.2f} {pfb:>7.2f} {dif:>+10.3f} {sgn:>4}/{tot}")
    print(f"  {'暗号資産アルト9 (Vantage 2022-)':<30} {1281:>6} {1.33:>6.2f} {1.87:>7.2f} "
          f"{1.02:>7.2f} {0.881:>+10.3f} {8:>4}/9")
    print(f"  {'暗号資産アルト9 (Binance 2018-)':<30} {2137:>6} {1.49:>6.2f} {1.97:>7.2f} "
          f"{1.30:>7.2f} {0.554:>+10.3f} {8:>4}/9")

    assert len(summary) >= 4, len(summary)
    print(f"\nOK: {len(summary)} 一家を測定")
