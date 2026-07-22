"""効率比(ER)ゲートの検定。2025年を見てから拾った変数なので、関門を全部通す。

由来: 2025年の停滞の病巣は「往復」だった（年ER 0.002＝他年の1/10以下、ボラは平年並み）。
      直前120本のERで3分位に切ると 1本R が +0.175 / +0.178 / +0.630 と単調。

🚨 後から拾った変数なので必須の関門:
  1. 窓としきい値のスイープ（台地か単独の尖りか）
  2. 間引き帰無（同じ本数を無作為に残す×2000）— 玉を減らしただけではないか
  3. 銘柄ホールドアウト（ETH で同じ符号か）
  4. 年別 ON%（1つの時代だけではないか）
  5. 巡回ブロック・ブートストラップ（経路当てはめではないか）
ER は方向を問わない量（下降トレンドでも高い）ので、既に全滅している方向ゲート
（KAMA・SMA150）とは別物である点に注意。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_2025 import prep, sig_idx, walk_idx        # noqa: E402
from experiments.atr_spike_barspread import spread_series, spikes     # noqa: E402

NBOOT = 2000
RNG = np.random.default_rng(77)


def er_series(c_ser, win):
    """効率比: |純変化| / Σ|1本の変化|。確定足のみ（shift(1)）。"""
    net = (c_ser - c_ser.shift(win)).abs()
    path = c_ser.diff().abs().rolling(win).sum()
    return (net / path).shift(1)


def stats(p, r):
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(p), win=(p > 0).mean() * 100, pf=w / ls if ls > 0 else np.nan,
                meanR=r.mean(), totR=r.sum(), ddR=dd,
                ratio=r.sum() / dd if dd > 0 else np.nan)


def dropnull(all_p, all_r, keep_mask):
    k = int(keep_mask.sum())
    if k < 12 or k >= len(all_p):
        return None
    mu, pf = [], []
    for _ in range(NBOOT):
        sel = RNG.choice(len(all_p), k, replace=False)
        q = all_p[sel]
        mu.append(all_r[sel].mean())
        w, ls = q[q > 0].sum(), -q[q < 0].sum()
        pf.append(w / ls if ls > 0 else np.nan)
    mu, pf = np.array(mu), np.array(pf)
    obs = stats(all_p[keep_mask], all_r[keep_mask])
    return ((mu < obs["meanR"]).mean() * 100,
            (pf[np.isfinite(pf)] < obs["pf"]).mean() * 100)


def build(sym, lead=None):
    d = prep(sym)
    sp = spread_series(f"{sym.upper()}|h1")
    s = sig_idx(d)
    if lead is not None:
        lm = lead.reindex(d.index).fillna(False).to_numpy()
        s = s[lm[s]]
    t = walk_idx(d, s, sp)
    t["y"] = t["time"].dt.year
    return d, t


if __name__ == "__main__":
    dB, tB = build("btcusd")
    sB = spikes(dB)
    lead = (sB | sB.shift(1)).fillna(False)
    dE, tE = build("ethusd", lead=lead)

    for nm, d, t in (("BTCUSD", dB, tB), ("ETHUSD(BTC確認)", dE, tE)):
        c = d["close"]
        print(f"\n===== {nm}  母集団 N={len(t)}")
        print(f"  {'窓':>5} {'しきい':>8} {'残す率':>7} {'N':>5} {'勝率':>7} {'PF':>6} "
              f"{'1本R':>8} {'比':>7} {'帰無%ile 平均/PF':>18}")
        allp = t["pct"].to_numpy()
        allr = t["R_net"].to_numpy()
        base = stats(allp, allr)
        print(f"  {'—':>5} {'全体':>8} {'100%':>7} {base['n']:>5} {base['win']:>6.1f}% "
              f"{base['pf']:>6.2f} {base['meanR']:>+8.3f} {base['ratio']:>7.2f}")
        for win in (60, 120, 240, 480):
            e = er_series(c, win)
            t[f"er{win}"] = e.reindex(t["time"]).to_numpy()
            v = t[f"er{win}"]
            for qq in (0.33, 0.50, 0.67):
                thr = v.quantile(qq)
                m = (v >= thr).to_numpy()
                if m.sum() < 20:
                    continue
                st = stats(allp[m], allr[m])
                dn = dropnull(allp, allr, m)
                tg = "" if dn is None else f"{dn[0]:>8.1f} /{dn[1]:>6.1f}"
                print(f"  {win:>5} {'上位'+str(int((1-qq)*100))+'%':>8} "
                      f"{m.mean()*100:>6.0f}% {st['n']:>5} {st['win']:>6.1f}% {st['pf']:>6.2f} "
                      f"{st['meanR']:>+8.3f} {st['ratio']:>7.2f} {tg:>18}")

    # 年別 ON% と成績（BTC・窓120・上位2/3）
    print("\n=== 年別（BTC・窓120・上位67%を残す）")
    c = dB["close"]
    e = er_series(c, 120)
    tB["er"] = e.reindex(tB["time"]).to_numpy()
    thr = tB["er"].quantile(0.33)
    tB["on"] = tB["er"] >= thr
    print(f"  しきい値={thr:.4f}")
    print(f"  {'年':>5} {'全体N':>6} {'ON N':>6} {'ON%':>6} {'全体1本R':>10} {'ON 1本R':>10} "
          f"{'OFF 1本R':>10}")
    for y in sorted(tB["y"].unique()):
        g = tB[tB["y"] == y]
        on, off = g[g["on"]], g[~g["on"]]
        print(f"  {y:>5} {len(g):>6} {len(on):>6} {g['on'].mean()*100:>5.0f}% "
              f"{g['R_net'].mean():>+10.3f} "
              f"{on['R_net'].mean() if len(on) else np.nan:>+10.3f} "
              f"{off['R_net'].mean() if len(off) else np.nan:>+10.3f}")

    assert len(tB) > 150, len(tB)
    print(f"\nOK: BTC N={len(tB)} / ETH N={len(tE)}")
