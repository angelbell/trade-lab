"""最後の未検定軸: 「BTC 同時」はアルトの【ショート】も救うか。

暗号資産のショートはこれまで全滅してきた（拡大足レッグの鏡像・btc15m_S を除く）。だが
「BTC 同時」は今日見つけたばかりの新しい変数で、構造法則11は「ドリフトに対する向きで
ゲートの速さもフィルタの厳しさも反転する」と言っている。∴ 向きを変えて測り直す価値がある。

鏡像の作り方は engine の invert() に一本化（自前で符号を反転させない）。
引き金 = 陰線の拡大足（実体の絶対値 > ATR*k）、位置条件 = 前日安値より下、損切り = 引き金足の高値。
先導条件 = BTC も同じ足か±1本で【陰線の】拡大足。

🚨 反転フレームでは walk() 内部のコストが鏡像価格を使う（e_px ≈ 実価格の3倍）ので、
   cost=0 で回して外側で実価格に対して引く（x_conventions#mirror-cost-overcharge）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.walk import walk                  # noqa: E402
from src.engine.mirror import invert              # noqa: E402
from scratchpad.atr_spike_leader_binance import (load, spikes, st, dropnull,   # noqa: E402
                                                 wilder_atr, ALTS, K, FWD, TRAIL)

COST = 0.0005


def leg_short(real):
    """鏡像フレームでロングとして解き、実価格に戻して % 損益を返す。"""
    d = invert(real)
    C = 2 * real["high"].max()
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)     # 鏡像の前日高値 = 実の前日安値
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    m = (c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 20:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None:
        return None
    t = t.copy()
    e_real = C - t["e_px"]                            # 実価格に戻す
    t["pct"] = (t["R"] * t["risk"]) / e_real - COST
    return t


def spikes_short(real):
    """実フレームでの陰線の拡大足（＝鏡像での陽線の拡大足）。"""
    return spikes(invert(real))


if __name__ == "__main__":
    btcR = load("btcusdt")
    sBs = spikes_short(btcR)
    sBs1 = (sBs | sBs.shift(1) | sBs.shift(-1)).fillna(False)   # ±1本
    sBs_use = (sBs | sBs.shift(1)).fillna(False)                # 運用可（同足 or 1本前）
    sBl = spikes(btcR)
    sBl_use = (sBl | sBl.shift(1)).fillna(False)

    recs = []
    for sym in ALTS:
        real = load(sym)
        t = leg_short(real)
        if t is None:
            print(f"  {sym}: 本数不足")
            continue
        recs.append(pd.DataFrame({
            "sym": sym, "pct": t["pct"].to_numpy(), "y": t["time"].dt.year.values,
            "btc_s": sBs_use.reindex(t["time"]).fillna(False).to_numpy(),
            "btc_l": sBl_use.reindex(t["time"]).fillna(False).to_numpy()}))
    R = pd.concat(recs, ignore_index=True)
    allp = R["pct"].to_numpy()
    n, w, pf, mu = st(allp)
    print(f"=== Binance アルト9銘柄・ショート 2018-2026  N={n} 勝率={w:.1f}% "
          f"PF={pf:.2f} 平均={mu:+.3f}%（コスト{COST*100:.2f}%）")
    print(f"  （参考: 同じ銘柄のロング母集団は N=2137 PF=1.49 平均+0.679%）")

    print(f"\n  {'条件':<30} {'N':>6} {'勝率':>7} {'PF':>6} {'平均%':>9} {'帰無%ile 平均/PF':>18}")
    for lab, m in (("全ショート", np.ones(len(R), dtype=bool)),
                   ("BTC も陰線の拡大足（同足or1本前）", R["btc_s"].to_numpy()),
                   ("BTC は陰線を出していない", ~R["btc_s"].to_numpy()),
                   ("BTC が【陽線の】拡大足（逆行）", R["btc_l"].to_numpy())):
        q = allp[m]
        if len(q) < 20:
            print(f"  {lab:<30} 本数不足 {len(q)}")
            continue
        dn = dropnull(allp, q) if m.sum() < len(R) else None
        tg = "" if dn is None else f" {dn[0]:>8.1f} /{dn[1]:>6.1f}"
        print(f"  {lab:<30} {len(q):>6} {st(q)[1]:>6.1f}% {st(q)[2]:>6.2f} {st(q)[3]:>+9.3f}{tg}")

    print("\n=== 銘柄別（BTC も陰線 vs 出していない）")
    sgn, tot = 0, 0
    for sym in R["sym"].unique():
        s = R[R["sym"] == sym]
        a = s.loc[s["btc_s"], "pct"].to_numpy()
        b = s.loc[~s["btc_s"], "pct"].to_numpy()
        if len(a) < 15 or len(b) < 15:
            print(f"  {sym:<10} 本数不足 ({len(a)}/{len(b)})")
            continue
        tot += 1
        sgn += int(a.mean() > b.mean())
        print(f"  {sym:<10} 同時 N={len(a):4d} PF={st(a)[2]:5.2f} {st(a)[3]:+.3f}%  |  "
              f"静か N={len(b):4d} PF={st(b)[2]:5.2f} {st(b)[3]:+.3f}%  |  "
              f"差={st(a)[3]-st(b)[3]:+.3f}%")
    if tot >= 2:
        print(f"  符号の一致 {sgn}/{tot}  二項P={sps.binomtest(sgn, tot, 0.5).pvalue:.3f}"
              f"   （ロングは 8/9・P=0.039）")

    print("\n=== 年別（BTC も陰線・平均%／本数）")
    ys = sorted(R["y"].unique())
    for lab, m in (("BTC同時ショート", R["btc_s"]), ("BTC静かショート", ~R["btc_s"])):
        cells = []
        for y in ys:
            q = R[m & (R["y"] == y)]["pct"].to_numpy()
            cells.append(f"{q.mean()*100:+.2f}%/{len(q):3d}" if len(q) >= 8 else "     --    ")
        print(f"  {lab:<16}" + " ".join(f"{x:>13}" for x in cells))

    assert len(R) > 800, len(R)
    print(f"\nOK: ショート母集団 N={len(R)} PF={pf:.2f}（ロングの 1.49 と比較）")
