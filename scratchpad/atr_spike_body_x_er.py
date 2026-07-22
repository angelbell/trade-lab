"""「実体の大きさ」×「ER」の2次元で層別する。両方採用＝この2軸のサイズ表と同じ。

k=2.0 のトレードは k=1.5 の部分集合なので、2構成を同時に回すと重なった玉に 0.01+0.02=0.03 が乗る。
∴「両方採用」＝ サイズ表:
   実体>2.0ATR かつ ER高 → 0.03 ／ それ以外(実体>1.5ATR) → 0.01

問い: 実体とERの【積】に、それぞれ単独にはない情報があるか。
  ER単独（k=1.5の母集団で）＝ +1%（ただのレバレッジ）
  実体単独（kを上げる）＝ 通算Rが減る
  両方そろったとき＝？
まず素の2次元層別（本数と1本R）を見て、それからサイズ表を対照つきで検定する。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_er_binance import load, build, attach, COST   # noqa: E402
from scratchpad.atr_spike_er_gate import er_series                      # noqa: E402
from scratchpad.atr_spike_barspread import wilder_atr                   # noqa: E402
import scratchpad.atr_spike_er_binance as EB                            # noqa: E402


def stat(lots, r):
    x = r * (lots / 0.01)
    x = x[lots > 0]
    if len(x) < 10:
        return None
    eq = np.cumsum(x)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(x), tot=x.sum(), dd=dd)


if __name__ == "__main__":
    B = load("btcusdt")
    e = er_series(B["close"], 120)
    orig = EB.K
    EB.K = 1.5                       # 母集団は k=1.5（広いほう）
    T = attach(build(B, "long", COST["btcusdt"]), e)
    EB.K = orig

    # 引き金足の実体（ATR単位）を復元
    ap = wilder_atr(B).shift(1)
    body = ((B["close"] - B["open"]) / ap)
    T["body"] = body.reindex(T["time"]).to_numpy()
    T = T.dropna(subset=["body"]).reset_index(drop=True)
    span = (T["time"].max() - T["time"].min()).days / 365.25
    on = (T["er"] >= T["x50"]).to_numpy()
    r = T["R_net"].to_numpy()

    print(f"=== BTC 1時間・Binance（{span:.1f}年・N={len(T)}）  実体 × ER の2次元層別")
    print(f"  {'実体(ATR単位)':<16} {'ER低: N / 1本R / 通算R':>28} {'ER高: N / 1本R / 通算R':>28}")
    bands = [(1.5, 2.0), (2.0, 2.5), (2.5, 3.5), (3.5, 99)]
    for lo, hi in bands:
        m = (T["body"] >= lo) & (T["body"] < hi)
        line = []
        for sel, lab in ((~on, "低"), (on, "高")):
            q = r[m & sel]
            line.append(f"N={len(q):4d} {q.mean():+.3f} {q.sum():+7.1f}" if len(q) >= 8
                        else "        --         ")
        print(f"  {lo:.1f}〜{hi if hi < 90 else '∞':<11} {line[0]:>28} {line[1]:>28}")
    print(f"  {'合計':<16} " +
          f"N={int((~on).sum()):4d} {r[~on].mean():+.3f} {r[~on].sum():+7.1f}".rjust(28) +
          f"N={int(on.sum()):4d} {r[on].mean():+.3f} {r[on].sum():+7.1f}".rjust(28))

    print("\n=== サイズ表を対照つきで検定（同じ maxDD に揃えた通算R）")
    strong = (T["body"] >= 2.0).to_numpy()
    CASES = [
        ("全部 0.01（基準）", np.full(len(T), 0.01)),
        ("全部 0.02（対照）", np.full(len(T), 0.02)),
        ("【両方採用】実体>2.0かつER高→0.03 / 他0.01", np.where(strong & on, 0.03, 0.01)),
        ("実体>2.0かつER高→0.02 / 他0.01", np.where(strong & on, 0.02, 0.01)),
        ("実体>2.0のみ→0.02 / 他0.01", np.where(strong, 0.02, 0.01)),
        ("ER高のみ→0.02 / 他0.01", np.where(on, 0.02, 0.01)),
        ("実体>2.0かつER高→0.02 / 他0（絞り込み）", np.where(strong & on, 0.02, 0.0)),
    ]
    base = stat(np.full(len(T), 0.01), r)
    print(f"  {'構成':<40} {'年本数':>6} {'通算R':>9} {'DD(R)':>8} "
          f"{'同DD揃え':>10} {'対 基準':>9}")
    for lab, lots in CASES:
        s = stat(lots, r)
        if s is None:
            continue
        sc = s["tot"] * (base["dd"] / s["dd"]) if s["dd"] > 0 else np.nan
        print(f"  {lab:<40} {s['n']/span:>6.0f} {s['tot']:>+9.1f} {s['dd']:>8.1f} "
              f"{sc:>+10.1f} {(sc/base['tot']-1)*100:>+8.0f}%")

    assert len(T) > 300, len(T)
    print(f"\nOK: N={len(T)}")
