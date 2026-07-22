"""k=1.75 は台地か尖りか。Binance 8.5年で確かめる。

Vantage 2022-（3.4年）で k を振ると 1.5→+16,307円 / 1.75→+18,693円 / 2.0→+12,734円 /
2.5→+4,973円 と、1.75 が最良に見えた。だが同じデータの上で掃引して拾った値なので、
台地（隣も良い）か単独の尖り（当てはめ）かを、独立フィード・2.5倍の期間で確かめる。

刻みを細かく（1.25〜3.0を0.25刻み）振り、totR と 1本R と本数を並べる。
コストは実測スプレッドを一律（BTC 0.05%）。壁時計20時間（1時間足 fwd20）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_er_binance import load, build, attach, COST   # noqa: E402
from scratchpad.atr_spike_er_gate import er_series                      # noqa: E402
import scratchpad.atr_spike_er_binance as EB                            # noqa: E402

WIN = 120


def run_k(d, k, cost, e):
    EB.K = k                       # build() が参照する引き金の強さを差し替える
    t = build(d, "long", cost)
    T = attach(t, e)
    return T


if __name__ == "__main__":
    B = load("btcusdt")
    e = er_series(B["close"], WIN)
    orig = EB.K
    print("=== BTC 1時間ロング・Binance 2018-2026（助走後）・コスト0.05%・実額は1コインあたり")
    print(f"  {'k':>6} {'N':>5} {'年本数':>7} {'勝率':>7} {'PF':>6} {'1本R':>8} "
          f"{'totR':>8} {'DD(R)':>7} {'比':>7} | {'ERゲート有り: 年本数/1本R/totR':>32}")
    rows = []
    for k in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0):
        T = run_k(B, k, COST["btcusdt"], e)
        if T is None or len(T) < 30:
            print(f"  {k:>6.2f} 本数不足")
            continue
        span = (T["time"].max() - T["time"].min()).days / 365.25
        p, r = T["pct"].to_numpy(), T["R_net"].to_numpy()
        eq = np.cumsum(r)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        w, ls = p[p > 0].sum(), -p[p < 0].sum()
        g = T[T["er"] >= T["x50"]]
        gr = g["R_net"].to_numpy()
        rows.append((k, len(p), len(p) / span, r.sum()))
        print(f"  {k:>6.2f} {len(p):>5} {len(p)/span:>7.0f} {(p>0).mean()*100:>6.1f}% "
              f"{w/ls if ls>0 else np.nan:>6.2f} {r.mean():>+8.3f} {r.sum():>+8.1f} "
              f"{dd:>7.1f} {r.sum()/dd if dd>0 else np.nan:>7.2f} | "
              f"{len(gr)/span:>10.0f}本 {gr.mean():>+8.3f} {gr.sum():>+8.1f}")
    EB.K = orig

    print("\n=== 年別の1本R（k を絞って）")
    for k in (1.5, 1.75, 2.0, 2.5):
        T = run_k(B, k, COST["btcusdt"], e)
        yy = T.groupby(T["time"].dt.year)["R_net"].agg(["mean", "count"])
        print(f"  k={k}: " + " ".join(f"{y}:{r['mean']:+.2f}/{int(r['count'])}"
                                      for y, r in yy.iterrows()))
    EB.K = orig

    assert len(rows) >= 6, len(rows)
    tot = {k: t for k, _, _, t in rows}
    print(f"\n  totR の並び: " + " ".join(f"k{k}:{t:+.0f}" for k, t in sorted(tot.items())))
    print("  （台地なら隣接する k も同程度。単独の尖りなら両隣が谷）")
    print(f"\nOK: {len(rows)} セル")
