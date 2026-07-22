"""0.02 版（通常0.01・強いシグナルで0.02）がブロック検定でも本物か。

段の高さは 0.03 だと Binance で同DD +12%、0.02 だと +7%。0.02 のほうが DD が27%小さい。
まだ 0.02 版のブロック・ブートストラップを取っていないので、ここで確かめる。
あわせて「強い」の定義を3通り比べる:
   ER だけ / 実体だけ / 実体かつER（積）
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_er_binance import load, build, attach, COST   # noqa: E402
from experiments.atr_spike_er_gate import er_series                      # noqa: E402
from experiments.atr_spike_barspread import wilder_atr                   # noqa: E402
import experiments.atr_spike_er_binance as EB                            # noqa: E402

NBOOT = 1000
RNG = np.random.default_rng(20260722)


def ratio(r):
    if len(r) < 5:
        return np.nan
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return r.sum() / dd if dd > 0 else np.nan


if __name__ == "__main__":
    B = load("btcusdt")
    e = er_series(B["close"], 120)
    orig = EB.K
    EB.K = 1.5
    T = attach(build(B, "long", COST["btcusdt"]), e)
    EB.K = orig
    ap = wilder_atr(B).shift(1)
    T["body"] = ((B["close"] - B["open"]) / ap).reindex(T["time"]).to_numpy()
    T = T.dropna(subset=["body"]).reset_index(drop=True)
    T["mo"] = T["time"].dt.to_period("M")
    span = (T["time"].max() - T["time"].min()).days / 365.25
    r = T["R_net"].to_numpy()

    er_hi = (T["er"] >= T["x50"]).to_numpy()
    strong = (T["body"] >= 2.0).to_numpy()
    DEFS = [("ER だけ", er_hi),
            ("実体>2.0 だけ", strong),
            ("実体>2.0 かつ ER高（積）", strong & er_hi)]

    print(f"=== BTC 1時間・Binance（{span:.1f}年・N={len(T)}）  段は 0.02/0.01")
    base = ratio(r)
    print(f"  {'「強い」の定義':<26} {'厚い玉の割合':>12} {'通算R':>9} {'DD(R)':>8} "
          f"{'通算/DD':>9} {'対 全部0.01':>11}")
    print(f"  {'全部 0.01（基準）':<26} {'—':>12} {r.sum():>+9.1f} "
          f"{float((np.maximum.accumulate(np.cumsum(r))-np.cumsum(r)).max()):>8.1f} "
          f"{base:>9.2f} {'—':>11}")
    for lab, m in DEFS:
        w = np.where(m, 2.0, 1.0)
        x = r * w
        print(f"  {lab:<26} {m.mean()*100:>11.0f}% {x.sum():>+9.1f} "
              f"{float((np.maximum.accumulate(np.cumsum(x))-np.cumsum(x)).max()):>8.1f} "
              f"{ratio(x):>9.2f} {(ratio(x)/base-1)*100:>+10.0f}%")

    print("\n=== 巡回ブロック・ブートストラップ（通算/DD が 全部0.01 に勝つ割合）")
    months = sorted(T["mo"].unique())
    bymo = {m: g for m, g in T.groupby("mo")}
    nm = len(months)
    print(f"  {'構成':<34} " + " ".join(f"{b}か月".rjust(9) for b in (1, 3, 6, 12)))
    for lab, m in DEFS + [("実体かつER高 → 0.03（参考）", strong & er_hi)]:
        mult = 3.0 if "0.03" in lab else 2.0
        w = np.where(m, mult, 1.0)
        outs = []
        for b in (1, 3, 6, 12):
            win = ok = 0
            for _ in range(NBOOT):
                need = int(np.ceil(nm / b))
                starts = RNG.integers(0, nm, size=need)
                idx = []
                for st in starts:
                    for mm in [months[(st + i) % nm] for i in range(b)]:
                        if mm in bymo:
                            idx.append(bymo[mm].index.to_numpy())
                if not idx:
                    continue
                I = np.concatenate(idx)
                ra, rb = ratio(r[I] * w[I]), ratio(r[I])
                if np.isfinite(ra) and np.isfinite(rb):
                    ok += 1
                    win += int(ra > rb)
            outs.append(f"{win/max(ok,1)*100:8.1f}%")
        tag = lab if "0.03" in lab else f"{lab} → 0.02"
        print(f"  {tag:<34} " + " ".join(outs))

    print("\n（50%付近＝ただの倍がけ。ブロックを長くするほど上がれば本物）")
    assert len(T) > 300, len(T)
    print(f"\nOK: N={len(T)}")
