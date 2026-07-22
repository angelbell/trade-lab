"""先導者仮説: アルトの拡大足は「BTC も同時に拡大足か」で選別できるか。

由来（`atr_spike_breadth.py`）: 広がりで層別すると BTC と ETH で符号が逆になった。
両方を説明する筋書きは「効いているのは銘柄数ではなく、その動きに BTC が入っているか」だけ。
  BTC 単独 = BTC が先導 = 良い / ETH 単独（BTC 静か）= 支えの無いアルト単独動 = 悪い
  ETH＋多数 = BTC も入っている = 良い
アルト9銘柄すべてに当てて、標本を9倍にする（銘柄ホールドアウトが検定に内蔵される）。

🚨 間引き帰無（同じ本数を無作為に残す×2000）を必ず併記する。
🚨 コストはアルトの実測が無いので価格の 0.05%（往復）で統一する。BTC の $15 は約0.023% なので
   アルトには保守側。コストを変えても符号は変わらないことを別途確認する。
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
COST_PCT = 0.0005
NBOOT = 2000
RNG = np.random.default_rng(99)
ALTS = ["ethusd", "xrpusd", "ltcusd", "bchusd", "trxusd",
        "solusd", "adausd", "dotusd", "bnbusd"]


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def load(sym):
    return load_mt5_csv(f"data/vantage_{sym}_h1.csv").loc["2022-01-01":]


def spikes(d):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    return pd.Series((c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0), index=d.index)


def leg(d, cost_pct=COST_PCT):
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
    t["pct"] = (t["R"] * t["risk"]) / t["e_px"] - cost_pct
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


def row(lab, allp, sub):
    if len(sub) < 12:
        print(f"    {lab:<24} 本数不足 N={len(sub)}")
        return
    n, w, pf, mu = st(sub)
    dn = dropnull(allp, sub)
    tg = "" if dn is None else f" 帰無%ile 平均={dn[0]:5.1f} PF={dn[1]:5.1f}"
    print(f"    {lab:<24} N={n:4d} 勝率={w:5.1f}% PF={pf:5.2f} 平均={mu:+.3f}%{tg}")


if __name__ == "__main__":
    btc = load("btcusd")
    sB = spikes(btc)
    sB_any = (sB | sB.shift(1).fillna(False))

    pooled_w, pooled_wo, per_sym = [], [], []
    print("=== 銘柄ごと（BTC が同時に拡大足か で割る・Vantage 2022-）")
    print(f"    {'銘柄':<10} {'全体':>22} | {'BTC同時':>26} | {'BTC静か':>26}")
    for sym in ALTS:
        d = load(sym)
        t = leg(d)
        if t is None:
            print(f"    {sym:<10} 本数不足")
            continue
        p = t["pct"].to_numpy()
        with_btc = sB_any.reindex(t["time"]).fillna(False).to_numpy()
        a, b = p[with_btc], p[~with_btc]
        pooled_w.append(a)
        pooled_wo.append(b)
        f = lambda q: f"N={len(q):3d} PF={st(q)[2]:4.2f} 平均{st(q)[3]:+.3f}%" if len(q) >= 8 else "  --  "
        per_sym.append((sym, st(a)[3] if len(a) >= 8 else np.nan,
                        st(b)[3] if len(b) >= 8 else np.nan))
        print(f"    {sym:<10} {f(p):>22} | {f(a):>26} | {f(b):>26}")

    W = np.concatenate(pooled_w)
    O = np.concatenate(pooled_wo)
    ALL = np.concatenate([W, O])
    print(f"\n=== アルト9銘柄プール（同一仕様・コスト0.05%）  母集団 N={len(ALL)}")
    row("全トレード", ALL, ALL)
    row("BTC も同時に拡大足", ALL, W)
    row("BTC は静か", ALL, O)

    sgn = [(s, a, b) for s, a, b in per_sym if np.isfinite(a) and np.isfinite(b)]
    win = sum(1 for _, a, b in sgn if a > b)
    from scipy import stats as sps
    print(f"\n  符号の一致: {win}/{len(sgn)} 銘柄で「BTC同時」の平均%が上回る "
          f"→ 二項検定(p=0.5) P={sps.binomtest(win, len(sgn), 0.5).pvalue:.4f}")

    print("\n=== BTC レッグ自身（他銘柄の広がりで割る・再掲の確認）")
    tb = leg(btc)
    pb = tb["pct"].to_numpy()
    alt_sp = pd.DataFrame({s: spikes(load(s)) for s in ALTS}).fillna(False)
    alt_any = (alt_sp.any(axis=1) | alt_sp.any(axis=1).shift(1).fillna(False))
    wa = alt_any.reindex(tb["time"]).fillna(False).to_numpy()
    row("全トレード", pb, pb)
    row("アルトも同時", pb, pb[wa])
    row("BTC 単独", pb, pb[~wa])

    print("\n=== 年別（アルトプール・平均%／本数）")
    recs = []
    for sym in ALTS:
        t = leg(load(sym))
        if t is None:
            continue
        wb = sB_any.reindex(t["time"]).fillna(False).to_numpy()
        recs.append(pd.DataFrame({"y": t["time"].dt.year.values,
                                  "pct": t["pct"].to_numpy(), "w": wb}))
    R = pd.concat(recs, ignore_index=True)
    print("年   " + "  ".join(f"{y:>16}" for y in sorted(R['y'].unique())))
    for lab, mask in (("BTC同時", R["w"]), ("BTC静か", ~R["w"])):
        cells = []
        for y in sorted(R["y"].unique()):
            q = R[mask & (R["y"] == y)]["pct"].to_numpy()
            cells.append(f"{q.mean()*100:+.3f}%/{len(q):3d}" if len(q) >= 8 else "     --     ")
        print(f"    {lab:<8}" + "  ".join(f"{x:>16}" for x in cells))

    assert len(ALL) > 800, len(ALL)
    assert st(ALL)[2] > 1.0, st(ALL)[2]
    print(f"\nOK: アルトプール N={len(ALL)} PF={st(ALL)[2]:.2f}")
