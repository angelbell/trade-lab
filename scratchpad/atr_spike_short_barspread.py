"""ショートの唯一の生き残り候補（BTC h1・0.5戻り売り指値）を、実スプレッド課金で検定し直す。

台帳の現状:
  BTC h1 ショート 成行 PF1.05 → 0.5戻り売り指値 PF1.27・maxDD 56%→19%（法則11の実例）
  ゲート（4h/日足KAMA）は帰無割れ・前日安値フィルタも帰無の下側＝どちらも入れてはいけない
  ドル円/FX/アルトのショートはすべて死
∴ 残っているのは BTC h1 の戻り売り指値だけ。PF1.27 は薄いので実コストで消える恐れがある。

鏡像は engine の invert() に一本化。
🚨 反転フレームでは walk() 内部のコストが鏡像価格を使うので cost=0 で回し、
   実価格（C − e_px）に対して外側で課金する（x_conventions#mirror-cost-overcharge）。
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
from scratchpad.atr_spike_barspread import spread_series, wilder_atr   # noqa: E402

K, TRAIL, FWD = 2.0, 3.0, 20


def run(real, pf, cost_series=None, flat=None, side="short", use_pdl=False):
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
    if use_pdl:
        s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 15:
        return None
    a = SimpleNamespace(pullback_frac=pf, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 15:
        return None
    t = t.copy()
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]     # 実価格に戻す
    t["rf"] = t["risk"] / e_real
    gross = (t["R"] * t["risk"]) / e_real
    if flat is not None:
        cost = np.full(len(t), flat)
    else:
        # 成行(pf=0)は引き金足の次バーで約定、指値(pf>0)は t["time"] がその約定バー
        step = d.index[1] - d.index[0]
        look = t["time"] + step if pf == 0 else t["time"]
        cost = cost_series.reindex(look, method="ffill").to_numpy()
    t["cost"] = cost
    t["pct"] = gross - cost
    t["R_net"] = t["pct"] / t["rf"]
    return t.dropna(subset=["pct"])


def show(t, span, lab):
    if t is None:
        print(f"    {lab:<26} 本数不足")
        return None
    p = t["pct"].to_numpy()
    r = t["R_net"].to_numpy()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    print(f"    {lab:<26} N={len(p):4d} 年{len(p)/span:3.0f}本 勝率={(p>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本R={r.mean():+.3f} totR={r.sum():+7.1f} "
          f"DD={dd:6.1f}R 比={r.sum()/dd if dd>0 else np.nan:6.2f}")
    return dict(pf=w / ls if ls > 0 else np.nan, ratio=r.sum() / dd if dd > 0 else np.nan,
                n=len(p), meanR=r.mean())


if __name__ == "__main__":
    btc = load_mt5_csv("data/vantage_btcusd_h1.csv").loc["2022-01-01":]
    btc = btc[~btc.index.duplicated(keep="first")].sort_index()
    span = (btc.index[-1] - btc.index[0]).days / 365.25
    sp = spread_series("BTCUSD|h1")

    print(f"=== BTC 1時間足ショート・実スプレッド課金（Vantage 2022-・{span:.1f}年）")
    print("  -- 戻り売り指値の深さを振る（ゲート無し・前日安値フィルタ無し＝台帳どおり）")
    keep = {}
    for pfrac in (0.0, 0.3, 0.5, 0.7):
        lab = "成行" if pfrac == 0 else f"戻り売り指値 {pfrac}"
        show(run(btc, pfrac, flat=0.0020), span, f"{lab}（一律0.20%）")
        keep[pfrac] = show(run(btc, pfrac, cost_series=sp), span, f"{lab}（★実スプレッド）")
        print()

    print("  -- 参考: 前日安値フィルタを入れた場合（台帳では帰無の下側＝禁止）")
    show(run(btc, 0.5, cost_series=sp, use_pdl=True), span, "0.5指値＋前日安値割れ")

    print("\n  -- 参考: 同じ物差しでのロング（成行・実スプレッド）")
    show(run(btc, 0.0, cost_series=sp, side="long"), span, "ロング成行")

    print("\n  -- 年別の1本R（ショート 0.5指値・実スプレッド）")
    t = run(btc, 0.5, cost_series=sp)
    yy = t.groupby(t["time"].dt.year)["R_net"].agg(["mean", "count"])
    print("     " + " ".join(f"{y}:{r['mean']:+.3f}/{int(r['count'])}" for y, r in yy.iterrows()))
    print(f"     課金スプレッド 中央値={t['cost'].median()*100:.4f}% "
          f"（1Rの{t['cost'].median()/t['rf'].median()*100:.0f}%）"
          f"  損切幅中央={t['rf'].median()*100:.2f}%")

    assert keep[0.5] is not None
    print(f"\nOK: ショート0.5指値 実スプレッド PF={keep[0.5]['pf']:.2f} 比={keep[0.5]['ratio']:.2f}")
