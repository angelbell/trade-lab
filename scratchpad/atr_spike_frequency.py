"""頻度のレバーを、0.01ロット固定の実額で並べ直す。

🚨 判定軸の訂正: 0.01ロット固定では比（totR/DD）は使えない。滑らかさを賭け率に変換できないので、
   見るべきは【年間の円】【年本数】【maxDDの円】。比を最適化すると本数と総額を捨てることになる。

振る軸（すべて実スプレッド課金・Vantage の板）:
  時間足 1時間 / 15分（BTCのみ・fwd は壁時計20時間にそろえる）
  引き金 k = 1.5 / 1.75 / 2.0 / 2.5    ← セッション初期に 2.0 で凍結したまま見直していない
  ERゲート あり / なし
  銘柄 BTC / ETH（ETHはBTC同時確認つき・1時間のみ）
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
from scratchpad.atr_spike_barspread import spread_series, wilder_atr   # noqa: E402
from scratchpad.atr_spike_er_gate import er_series                     # noqa: E402

USDJPY, LOT, TRAIL = 150.0, 0.01, 3.0
WARM = pd.Timedelta(days=365)


def load(sym, tf):
    d = load_mt5_csv(f"data/vantage_{sym}_{tf}.csv").loc["2022-01-01":]
    return d[~d.index.duplicated(keep="first")].sort_index()


def spikes(d, k):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    return pd.Series((c - o > ap * k) & (c > o) & np.isfinite(ap) & (ap > 0), index=d.index)


def leg(d, k, fwd, cost_series, lead=None):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    m = (c - o > ap * k) & (c > o) & np.isfinite(ap) & (ap > 0)
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
    if t is None:
        return None
    t = t.copy()
    step = d.index[1] - d.index[0]
    cost = cost_series.reindex(t["time"] + step, method="ffill").to_numpy()
    t["pct"] = (t["R"] * t["risk"]) / t["e_px"] - cost
    t["yen"] = t["pct"] * t["e_px"] * LOT * USDJPY
    return t.dropna(subset=["yen"])


def gate(t, d, win):
    e = er_series(d["close"], win)
    t = t.copy()
    t["er"] = e.reindex(t["time"]).to_numpy()
    t = t.dropna(subset=["er"]).sort_values("time").reset_index(drop=True)
    t["x50"] = t["er"].expanding(min_periods=20).quantile(0.50).shift(1)
    return t[t["time"] >= t["time"].iloc[0] + WARM].copy()


def show(lab, y, span):
    if len(y) < 10:
        print(f"  {lab:<34} 本数不足")
        return
    eq = np.cumsum(y)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = y[y > 0].sum(), -y[y < 0].sum()
    print(f"  {lab:<34} {len(y)/span:>6.0f} {(y>0).mean()*100:>6.1f}% "
          f"{w/ls if ls>0 else np.nan:>6.2f} {y.mean():>+7,.0f}円 "
          f"{y.sum()/span:>+10,.0f}円 {dd:>9,.0f}円")


if __name__ == "__main__":
    HDR = (f"  {'構成':<34} {'年本数':>6} {'勝率':>7} {'PF':>6} {'1本':>8} "
           f"{'年間':>11} {'maxDD':>10}")
    b1 = load("btcusd", "h1")
    sp1 = spread_series("BTCUSD|h1")
    b15 = load("btcusd", "m15")
    sp15 = spread_series("BTCUSD|m15")
    e1 = load("ethusd", "h1")
    spe = spread_series("ETHUSD|h1")

    print("=== BTC 1時間（fwd20＝壁時計20時間）")
    print(HDR)
    store = {}
    for k in (1.5, 1.75, 2.0, 2.5):
        t = leg(b1, k, 20, sp1)
        if t is None:
            continue
        T = gate(t, b1, 120)
        span = (T["time"].max() - T["time"].min()).days / 365.25
        show(f"k={k} ゲート無し", T["yen"].to_numpy(), span)
        show(f"k={k} ERゲート有り", T.loc[T["er"] >= T["x50"], "yen"].to_numpy(), span)
        store[("btc1h", k)] = (T, span)

    print("\n=== BTC 15分（fwd80＝壁時計20時間）")
    print(HDR)
    for k in (1.5, 1.75, 2.0, 2.5):
        t = leg(b15, k, 80, sp15)
        if t is None:
            continue
        T = gate(t, b15, 480)          # 15分の120本は5時間なので、壁時計を合わせて480本
        span = (T["time"].max() - T["time"].min()).days / 365.25
        show(f"k={k} ゲート無し", T["yen"].to_numpy(), span)
        show(f"k={k} ERゲート有り", T.loc[T["er"] >= T["x50"], "yen"].to_numpy(), span)
        store[("btc15m", k)] = (T, span)

    print("\n=== ETH 1時間（BTC同時確認つき）")
    print(HDR)
    sB = spikes(b1, 2.0)
    lead = (sB | sB.shift(1)).fillna(False)
    for k in (1.5, 1.75, 2.0, 2.5):
        t = leg(e1, k, 20, spe, lead=lead)
        if t is None:
            continue
        T = gate(t, e1, 120)
        span = (T["time"].max() - T["time"].min()).days / 365.25
        show(f"k={k} ゲート無し", T["yen"].to_numpy(), span)
        store[("eth1h", k)] = (T, span)

    print("\n=== 組み合わせ（ゲート無し・k を各構成の最良近辺で）")
    print(HDR)
    for combo, keys in (("BTC 1時間 k1.5 のみ", [("btc1h", 1.5)]),
                        ("BTC 1時間 k1.5 ＋ ETH 1時間 k1.5", [("btc1h", 1.5), ("eth1h", 1.5)]),
                        ("BTC 15分 k1.5 のみ", [("btc15m", 1.5)]),
                        ("BTC 1時間 k1.5 ＋ BTC 15分 k2.0",
                         [("btc1h", 1.5), ("btc15m", 2.0)]),
                        ("BTC 1時間+15分+ETH（すべて k1.5）",
                         [("btc1h", 1.5), ("btc15m", 1.5), ("eth1h", 1.5)])):
        parts, sp_ = [], 0
        ok = True
        for key in keys:
            if key not in store:
                ok = False
                break
            T, s = store[key]
            parts.append(pd.DataFrame({"time": T["time"].values, "yen": T["yen"].to_numpy()}))
            sp_ = max(sp_, s)
        if not ok:
            continue
        P = pd.concat(parts, ignore_index=True).sort_values("time")
        show(combo, P["yen"].to_numpy(), sp_)

    assert len(store) > 5, len(store)
    print(f"\nOK: {len(store)} 構成")
