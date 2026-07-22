"""ERゲートを Binance の長い履歴（2018-2026・8.5年）で、ロング・ショート両方について検定する。

Vantage は 2022- の 3.4年（助走後）しかなく、ETH ロングは「向きは一致するが %ile 69-78」で未確定、
ETH ショートは符号が逆だった。標本を 2.5倍にして決着させる。

コストは Vantage の実測スプレッドを一律で当てる（Binance に per-bar spread は無い）:
  BTC 0.05%（実測の課金中央値 0.029% より保守側）· ETH 0.15%（同 0.127% より保守側）
時刻はブローカー時刻のラベルに直す（前日高値・週末の定義を Vantage に合わせる）。
ロングは ETH のみ BTC同時確認つき（確立した仕様）。ショートは素。
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
from src.engine.mirror import invert              # noqa: E402
from scratchpad.atr_spike_barspread import wilder_atr             # noqa: E402
from scratchpad.atr_spike_er_gate import er_series                # noqa: E402
from scratchpad.atr_spike_er_short import dropnull                # noqa: E402

K, TRAIL, FWD, WIN = 2.0, 3.0, 20, 120
WARM = pd.Timedelta(days=365)
COST = {"btcusdt": 0.0005, "ethusdt": 0.0015}
NBOOT = 1000
RNG = np.random.default_rng(4242)


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


def build(real, side, cost, lead=None):
    d = invert(real) if side == "short" else real
    C = 2 * real["high"].max() if side == "short" else None
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    m = (c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    if side == "long":                      # 前日高値フィルタはロング専用（鏡像は帰無割れ）
        s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    if lead is not None:
        lm = lead.reindex(d.index).fillna(False).to_numpy()
        s = s[lm[s]]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    t = t.copy()
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]
    t["rf"] = t["risk"] / e_real
    t["pct"] = (t["R"] * t["risk"]) / e_real - cost
    t["R_net"] = t["pct"] / t["rf"]
    return t


def attach(t, e):
    t = t.copy()
    t["er"] = e.reindex(t["time"]).to_numpy()
    t = t.dropna(subset=["er"]).sort_values("time").reset_index(drop=True)
    for q in (33, 50, 67):
        t[f"x{q}"] = t["er"].expanding(min_periods=20).quantile(q / 100).shift(1)
    return t[t["time"] >= t["time"].iloc[0] + WARM].copy()


def rep(lab, g, span, allp=None, allr=None, mask=None):
    p, r = g["pct"].to_numpy(), g["R_net"].to_numpy()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    tag = ""
    if mask is not None:
        dn = dropnull(allp, allr, mask)
        if dn:
            tag = f"  帰無%ile 平均={dn[0]:5.1f} PF={dn[1]:5.1f}"
    print(f"  {lab:<28} N={len(p):4d} 年{len(p)/span:3.0f}本 勝率={(p>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本R={r.mean():+.3f} "
          f"totR={r.sum():+7.1f} DD={dd:6.1f}R{tag}")


if __name__ == "__main__":
    B, E = load("btcusdt"), load("ethusdt")
    sB = spikes(B)
    lead = (sB | sB.shift(1)).fillna(False)
    erB, erE = er_series(B["close"], WIN), er_series(E["close"], WIN)

    CASES = [("BTC ロング", B, "long", erB, None, "btcusdt", True),
             ("BTC ショート", B, "short", erB, None, "btcusdt", False),
             ("ETH ロング（BTC確認）", E, "long", erE, lead, "ethusdt", True),
             ("ETH ショート", E, "short", erE, None, "ethusdt", False)]
    keep = {}
    for nm, d, side, e, ld, sym, want_high in CASES:
        T = attach(build(d, side, COST[sym], lead=ld), e)
        span = (T["time"].max() - T["time"].min()).days / 365.25
        allp, allr = T["pct"].to_numpy(), T["R_net"].to_numpy()
        print(f"\n===== {nm}  Binance 2018-2026（助走後 {span:.1f}年）")
        rep("全部取る", T, span)
        q = T["er"].quantile([1/3, 2/3]).to_numpy()
        for lo, hi, lb in ((-np.inf, q[0], "ER 低"), (q[0], q[1], "ER 中"), (q[1], np.inf, "ER 高")):
            g = T[(T["er"] >= lo) & (T["er"] < hi)]
            if len(g) >= 15:
                rep(f"  {lb}", g, span)
        for qq in (33, 50, 67):
            thr = T[f"x{qq}"].to_numpy()
            m = (T["er"].to_numpy() >= thr) if want_high else (T["er"].to_numpy() < thr)
            if m.sum() < 20:
                continue
            d_ = "上回る" if want_high else "下回る"
            rep(f"  ゲート {qq}%点を{d_}", T[m], span, allp, allr, m)
            keep[(nm, qq)] = (T, m, span)
        print("  年別 1本R: " + " ".join(
            f"{y}:{g:+.2f}" for y, g in T.groupby(T["time"].dt.year)["R_net"].mean().items()))

    print("\n（判定: Vantage 2022- で見えた符号——ロング=高ER・ショート=低ER——が"
          "\n  8.5年の独立フィードでも帰無 95 を超えるか）")
    assert len(keep) > 0
    print(f"\nOK: {len(keep)} セル")
