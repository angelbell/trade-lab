"""ロング（高ER）＋ショート（低ER）を1つの系として合成する。

分かったこと（窓120・拡張窓の分位・先読みなし・実スプレッド課金）:
  ロング: ER が中央値を【上回る】ときだけ建てる（帰無%ile 97.2、ブロック12か月でDD 99.3%）
  ショート: ER が中央値を【下回る】ときだけ建てる（帰無%ile 91.0/96.2）
  → 2つは定義上ほぼ排他なので、レジームを分担する可能性がある。
検定: 窓120のショートでブロック・ブートストラップを取り直し、合成の年間円・DD・年別を出す。
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

USDJPY, LOT = 150.0, 0.01
WARM = pd.Timedelta(days=365)
NBOOT = 1000
RNG = np.random.default_rng(6161)
WIN = 120


def metrics(y):
    if len(y) < 5:
        return np.nan, np.nan
    eq = np.cumsum(y)
    return y.sum(), float((np.maximum.accumulate(eq) - eq).max())


def side_frame(btc, sp, side):
    C = 2 * btc["high"].max()
    t = run(btc, 0.0, cost_series=sp, side=side).copy()
    t["e_px_real"] = (C - t["e_px"]) if side == "short" else t["e_px"]
    e = er_series(btc["close"], WIN)
    t["er"] = e.reindex(t["time"]).to_numpy()
    t = t.dropna(subset=["er"]).sort_values("time").reset_index(drop=True)
    t["x50"] = t["er"].expanding(min_periods=20).quantile(0.50).shift(1)
    t["x67"] = t["er"].expanding(min_periods=20).quantile(0.67).shift(1)
    t = t[t["time"] >= t["time"].iloc[0] + WARM].copy()
    t["yen"] = t["pct"].to_numpy() * t["e_px_real"].to_numpy() * LOT * USDJPY
    t["side"] = side
    t["on"] = (t["er"] >= t["x50"]) if side == "long" else (t["er"] < t["x67"])
    return t


def rep(lab, y, span):
    tt, dd = metrics(y)
    if not np.isfinite(tt):
        print(f"  {lab:<30} 本数不足")
        return
    w, ls = y[y > 0].sum(), -y[y < 0].sum()
    print(f"  {lab:<30} N={len(y):4d} 年{len(y)/span:3.0f}本 勝率={(y>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本={y.mean():+7,.0f}円 "
          f"年間={tt/span:+9,.0f}円 DD={dd:8,.0f}円")


def boot(P, lab):
    P = P.copy()
    P["mo"] = P["time"].dt.to_period("M")
    months = sorted(P["mo"].unique())
    bymo = {m: g for m, g in P.groupby("mo")}
    nm = len(months)
    print(f"  {lab}")
    print(f"  {'ブロック':>9} | {'通算の円が増える割合':>22} | {'DDの円が減る割合':>20}")
    for b in (1, 3, 6, 12):
        w1 = w2 = ok = 0
        for _ in range(NBOOT):
            need = int(np.ceil(nm / b))
            starts = RNG.integers(0, nm, size=need)
            pa, po = [], []
            for st in starts:
                blk = [months[(st + i) % nm] for i in range(b)]
                gs = [bymo[m] for m in blk if m in bymo]
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


if __name__ == "__main__":
    btc = load_mt5_csv("data/vantage_btcusd_h1.csv").loc["2022-01-01":]
    btc = btc[~btc.index.duplicated(keep="first")].sort_index()
    sp = spread_series("BTCUSD|h1")
    L = side_frame(btc, sp, "long")
    S = side_frame(btc, sp, "short")
    span = (max(L["time"].max(), S["time"].max())
            - min(L["time"].min(), S["time"].min())).days / 365.25

    print(f"=== BTC 1時間・窓{WIN}・0.01ロット固定・実スプレッド（助走後 {span:.1f}年）")
    rep("ロング 全部", L["yen"].to_numpy(), span)
    rep("ロング ER高のみ", L.loc[L["on"], "yen"].to_numpy(), span)
    rep("ショート 全部", S["yen"].to_numpy(), span)
    rep("ショート ER低のみ", S.loc[S["on"], "yen"].to_numpy(), span)

    print()
    boot(S, f"ショート（窓{WIN}・ER下位67%）のブロック・ブートストラップ")

    print("\n=== 合成")
    for lab, gl, gs in (("両方 全部取る", L, S),
                        ("ロングのみ ER高", L[L["on"]], None),
                        ("両方 ERゲートあり", L[L["on"]], S[S["on"]])):
        parts = [pd.DataFrame({"time": gl["time"].values, "yen": gl["yen"].to_numpy()})]
        if gs is not None:
            parts.append(pd.DataFrame({"time": gs["time"].values, "yen": gs["yen"].to_numpy()}))
        P = pd.concat(parts, ignore_index=True).sort_values("time")
        rep(lab, P["yen"].to_numpy(), span)

    print("\n  -- 年別（円）")
    for lab, gl, gs in (("両方 全部取る", L, S),
                        ("ロングのみ ER高", L[L["on"]], None),
                        ("両方 ERゲート", L[L["on"]], S[S["on"]])):
        parts = [pd.DataFrame({"y": gl["time"].dt.year.values, "yen": gl["yen"].to_numpy()})]
        if gs is not None:
            parts.append(pd.DataFrame({"y": gs["time"].dt.year.values, "yen": gs["yen"].to_numpy()}))
        P = pd.concat(parts, ignore_index=True)
        yy = P.groupby("y")["yen"].agg(["sum", "count"])
        print(f"    {lab:<16}" + " ".join(f"{y}:{r['sum']:+,.0f}/{int(r['count'])}"
                                          for y, r in yy.iterrows()))

    ov = (L.loc[L["on"], "time"].isin(S.loc[S["on"], "time"])).sum()
    print(f"\n  ロングON と ショートON の時刻の重なり: {ov} 本（排他性の確認）")
    assert len(L) > 150 and len(S) > 150, (len(L), len(S))
    print(f"\nOK: L={len(L)} S={len(S)}")
