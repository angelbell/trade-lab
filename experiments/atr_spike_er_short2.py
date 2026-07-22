"""ショートの「低ER（往復相場）」が本物か。帰無とブロック・ブートストラップで詰める。

前段:
  ショートは ER 低 PF1.41 / 中 1.21 / 高 0.81 ＝ ロングと符号が逆
  ロングのゲートをそのまま当てると帰無%ile 7.2（有意に悪い）
∴ ショート専用のゲート「ER が拡張窓の中央値を【下回る】ときだけ建てる」を検定する。
しきい値は拡張窓（先読みなし）。比較は間引き帰無＋巡回ブロック・ブートストラップ。
0.01ロット固定なので判定は年間の円と maxDD の円。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from experiments.atr_spike_short_barspread import run    # noqa: E402
from experiments.atr_spike_barspread import spread_series  # noqa: E402
from experiments.atr_spike_er_gate import er_series       # noqa: E402
from experiments.atr_spike_er_short import dropnull       # noqa: E402

USDJPY, LOT = 150.0, 0.01
WARM = pd.Timedelta(days=365)
NBOOT = 1000
RNG = np.random.default_rng(1212)


def metrics(y):
    if len(y) < 5:
        return np.nan, np.nan
    eq = np.cumsum(y)
    return y.sum(), float((np.maximum.accumulate(eq) - eq).max())


if __name__ == "__main__":
    btc = load_mt5_csv("data/vantage_btcusd_h1.csv").loc["2022-01-01":]
    btc = btc[~btc.index.duplicated(keep="first")].sort_index()
    sp = spread_series("BTCUSD|h1")
    C = 2 * btc["high"].max()
    t = run(btc, 0.0, cost_series=sp, side="short").copy()
    t["e_px_real"] = C - t["e_px"]

    print("=== ショート専用ERゲート（窓を振る・拡張窓の分位で切る・先読みなし）")
    print(f"  {'窓':>5} {'分位':>7} {'残す率':>7} {'N':>5} {'年本数':>7} {'勝率':>7} {'PF':>6} "
          f"{'年間':>10} {'DD':>9} {'帰無%ile 平均/PF':>18}")
    best = None
    for win in (60, 120, 240, 480):
        e = er_series(btc["close"], win)
        u = t.copy()
        u["er"] = e.reindex(u["time"]).to_numpy()
        u = u.dropna(subset=["er"]).sort_values("time").reset_index(drop=True)
        for q in (0.33, 0.50, 0.67):
            u[f"x{q}"] = u["er"].expanding(min_periods=20).quantile(q).shift(1)
        w = u[u["time"] >= u["time"].iloc[0] + WARM]
        span = (w["time"].max() - w["time"].min()).days / 365.25
        allp, allr = w["pct"].to_numpy(), w["R_net"].to_numpy()
        ally = allp * w["e_px_real"].to_numpy() * LOT * USDJPY
        if win == 60:
            ta, da = metrics(ally)
            print(f"  {'—':>5} {'全部':>7} {'100%':>7} {len(w):>5} {len(w)/span:>7.0f} "
                  f"{(allp>0).mean()*100:>6.1f}% "
                  f"{(allp[allp>0].sum()/-allp[allp<0].sum()):>6.2f} {ta/span:>+9,.0f}円 "
                  f"{da:>8,.0f}円")
        for q in (0.33, 0.50, 0.67):
            m = (w["er"] < w[f"x{q}"]).to_numpy()        # 【下回る】ときだけ建てる
            if m.sum() < 20:
                continue
            g = w[m]
            p = g["pct"].to_numpy()
            y = p * g["e_px_real"].to_numpy() * LOT * USDJPY
            tt, dd = metrics(y)
            dn = dropnull(allp, allr, m)
            pf = p[p > 0].sum() / -p[p < 0].sum() if (p < 0).any() else np.nan
            print(f"  {win:>5} {'下位'+str(int(q*100))+'%':>7} {m.mean()*100:>6.0f}% "
                  f"{len(g):>5} {len(g)/span:>7.0f} {(p>0).mean()*100:>6.1f}% {pf:>6.2f} "
                  f"{tt/span:>+9,.0f}円 {dd:>8,.0f}円 "
                  f"{dn[0]:>8.1f} /{dn[1]:>6.1f}" if dn else "")
            if dn and (best is None or dn[0] > best[0]):
                best = (dn[0], win, q, w, m, span)

    if best is None:
        print("\n  候補なし")
        sys.exit(0)
    _, win, q, w, m, span = best
    print(f"\n=== 最良セル（窓{win}・下位{int(q*100)}%）でブロック・ブートストラップ")
    P = w.copy()
    P["on"] = m
    P["yen"] = P["pct"].to_numpy() * P["e_px_real"].to_numpy() * LOT * USDJPY
    P["mo"] = P["time"].dt.to_period("M")
    months = sorted(P["mo"].unique())
    bymo = {mm: g for mm, g in P.groupby("mo")}
    nm = len(months)
    print(f"  {'ブロック':>9} | {'通算の円が増える割合':>22} | {'DDの円が減る割合':>20}")
    for b in (1, 3, 6, 12):
        w1 = w2 = ok = 0
        for _ in range(NBOOT):
            need = int(np.ceil(nm / b))
            starts = RNG.integers(0, nm, size=need)
            pa, po = [], []
            for st in starts:
                blk = [months[(st + i) % nm] for i in range(b)]
                gs = [bymo[mm] for mm in blk if mm in bymo]
                if not gs:
                    continue
                g = pd.concat(gs, ignore_index=True)
                pa.append(g["yen"].to_numpy())
                po.append(g.loc[g["on"], "yen"].to_numpy())
            if not pa:
                continue
            ta, da = metrics(np.concatenate(pa))
            to, do = metrics(np.concatenate(po))
            if not (np.isfinite(ta) and np.isfinite(to)):
                continue
            ok += 1
            w1 += int(to > ta)
            w2 += int(do < da)
        print(f"  {b:>7}か月 | {w1/max(ok,1)*100:>21.1f}% | {w2/max(ok,1)*100:>19.1f}%")

    print("\n  年別（円）")
    for lab, g in (("全部取る", P), ("ERゲート", P[P["on"]])):
        yy = g.assign(y=g["time"].dt.year).groupby("y")["yen"].agg(["sum", "count"])
        print(f"    {lab:<10}" + " ".join(f"{y}:{r['sum']:+,.0f}/{int(r['count'])}"
                                          for y, r in yy.iterrows()))
