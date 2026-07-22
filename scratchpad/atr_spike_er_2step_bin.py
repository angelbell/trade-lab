"""2段サイズ（ER高0.02 / ER低0.01）を Binance 8.5年で確かめる。

Vantage 3.4年では、同じ maxDD にそろえた年間の円が
  全部0.01 = 全部0.02（対照どおり完全一致）→ ER高0.02/低0.01 で +27〜39%、0.03/0.01 で +40〜58%
だった。期間固有でないかを、独立フィード・2.5倍の期間で確かめる。

判定は「同じ maxDD にそろえたときの通算R」（Binance では円換算しない。1コインあたりで見る）。
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


def stat(lots, r):
    x = r * (lots / 0.01)
    x = x[lots > 0]
    if len(x) < 10:
        return None
    eq = np.cumsum(x)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(x), tot=x.sum(), dd=dd,
                ratio=x.sum() / dd if dd > 0 else np.nan)


if __name__ == "__main__":
    B = load("btcusdt")
    e = er_series(B["close"], 120)
    orig = EB.K
    for k in (1.5, 2.0):
        EB.K = k
        T = attach(build(B, "long", COST["btcusdt"]), e)
        span = (T["time"].max() - T["time"].min()).days / 365.25
        r = T["R_net"].to_numpy()
        on = (T["er"] >= T["x50"]).to_numpy()
        print(f"\n===== BTC 1時間 k={k}  Binance（助走後 {span:.1f}年・N={len(T)}・"
              f"ER高の割合 {on.mean()*100:.0f}%）")
        CASES = [("全部 0.01", np.full(len(T), 0.01)),
                 ("全部 0.02", np.full(len(T), 0.02)),
                 ("ER高0.02 / ER低0.01", np.where(on, 0.02, 0.01)),
                 ("ER高0.03 / ER低0.01", np.where(on, 0.03, 0.01)),
                 ("ER高0.02 / ER低0", np.where(on, 0.02, 0.0)),
                 ("ER高0.01 / ER低0", np.where(on, 0.01, 0.0))]
        base = stat(np.full(len(T), 0.01), r)
        print(f"  {'構成':<24} {'年本数':>6} {'通算R':>9} {'DD(R)':>8} {'通算/DD':>8} "
              f"{'同DDに揃えた通算R':>18} {'対 全部0.01':>12}")
        for lab, lots in CASES:
            s = stat(lots, r)
            if s is None:
                continue
            scaled = s["tot"] * (base["dd"] / s["dd"]) if s["dd"] > 0 else np.nan
            print(f"  {lab:<24} {s['n']/span:>6.0f} {s['tot']:>+9.1f} {s['dd']:>8.1f} "
                  f"{s['ratio']:>8.2f} {scaled:>+18.1f} "
                  f"{(scaled/base['tot']-1)*100:>+11.0f}%")
    EB.K = orig
    print("\n（全部0.02 が 全部0.01 と同じ『同DDに揃えた通算R』になるのが対照。"
          "\n  それを超えた分だけが ER の情報）")
