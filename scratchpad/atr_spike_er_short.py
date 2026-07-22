"""ERゲートをショートに当てる。ERは方向を問わない量なので、ショートにも効く可能性がある。

前提（実スプレッド課金・Vantage 2022-）:
  ショートは最良でも 成行 PF1.07・比2.38（ロングは 1.39・8.32）＝素では不採用
  ERゲートはロングで全関門を通過（DD減少 99.3%・利益増 69.9%）
∴ 「ショートが死んでいたのは、往復相場で建てていたからではないか」を検定する。

2つの流儀:
  A 方向なし ER: |純変化|/Σ|変化| が高い（＝どちらかに走った後）
  B 方向つき ER: 上に加えて【純変化が負】（＝下に走った後）
しきい値は拡張窓の分位（先読みを入れない）。
比較は必ず「同じ本数を無作為に残す」帰無と。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from scratchpad.atr_spike_short_barspread import run    # noqa: E402
from scratchpad.atr_spike_barspread import spread_series  # noqa: E402
from scratchpad.atr_spike_er_gate import er_series       # noqa: E402

USDJPY, LOT = 150.0, 0.01
NBOOT = 2000
RNG = np.random.default_rng(808)
WARM = pd.Timedelta(days=365)


def dropnull(all_p, all_r, mask):
    k = int(mask.sum())
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
    obs_r = all_r[mask].mean()
    qq = all_p[mask]
    w, ls = qq[qq > 0].sum(), -qq[qq < 0].sum()
    obs_pf = w / ls if ls > 0 else np.nan
    return ((mu < obs_r).mean() * 100,
            (pf[np.isfinite(pf)] < obs_pf).mean() * 100)


def rep(lab, g, span, allp=None, allr=None, mask=None):
    p, r = g["pct"].to_numpy(), g["R_net"].to_numpy()
    y = p * g["e_px_real"].to_numpy() * LOT * USDJPY
    eq = np.cumsum(y)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    tag = ""
    if mask is not None:
        dn = dropnull(allp, allr, mask)
        if dn:
            tag = f"  帰無%ile 平均={dn[0]:5.1f} PF={dn[1]:5.1f}"
    print(f"  {lab:<28} N={len(p):4d} 年{len(p)/span:3.0f}本 勝率={(p>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本R={r.mean():+.3f} "
          f"年間={y.sum()/span:+8,.0f}円 DD={dd:7,.0f}円{tag}")


if __name__ == "__main__":
    btc = load_mt5_csv("data/vantage_btcusd_h1.csv").loc["2022-01-01":]
    btc = btc[~btc.index.duplicated(keep="first")].sort_index()
    sp = spread_series("BTCUSD|h1")
    c = btc["close"]
    C = 2 * btc["high"].max()

    er = er_series(c, 120)                       # |純変化|/Σ|変化|（方向なし）
    net = (c - c.shift(120)).shift(1)            # 純変化（符号つき・確定足）

    for side, pf in (("short", 0.0), ("long", 0.0)):
        t = run(btc, pf, cost_series=sp, side=side)
        t = t.copy()
        t["e_px_real"] = (C - t["e_px"]) if side == "short" else t["e_px"]
        t["er"] = er.reindex(t["time"]).to_numpy()
        t["net"] = net.reindex(t["time"]).to_numpy()
        t = t.dropna(subset=["er", "net"]).sort_values("time").reset_index(drop=True)
        t["exp50"] = t["er"].expanding(min_periods=20).quantile(0.50).shift(1)
        t = t[t["time"] >= t["time"].iloc[0] + WARM]
        span = (t["time"].max() - t["time"].min()).days / 365.25
        allp, allr = t["pct"].to_numpy(), t["R_net"].to_numpy()
        print(f"\n===== BTC 1時間 {'ショート（成行）' if side=='short' else 'ロング（成行）'}"
              f"  助走後 {span:.1f}年")
        rep("全部取る", t, span)
        # 3分位で単調性を先に見る
        q = t["er"].quantile([1/3, 2/3]).to_numpy()
        for lo, hi, nm in ((-np.inf, q[0], "ER 低"), (q[0], q[1], "ER 中"), (q[1], np.inf, "ER 高")):
            g = t[(t["er"] >= lo) & (t["er"] < hi)]
            if len(g) >= 12:
                rep(f"  {nm}", g, span)
        mA = (t["er"] >= t["exp50"]).to_numpy()
        rep("A 方向なしER（拡張窓中央）", t[mA], span, allp, allr, mA)
        mB = mA & (t["net"].to_numpy() < 0)
        if mB.sum() >= 12:
            rep("B 方向つきER（下に走った後）", t[mB], span, allp, allr, mB)
        mC = mA & (t["net"].to_numpy() > 0)
        if mC.sum() >= 12:
            rep("C 方向つきER（上に走った後）", t[mC], span, allp, allr, mC)

    print("\n（ロングは比較用。ショートで帰無%ile 95 を超え、かつ年間の円がプラスで初めて候補）")
