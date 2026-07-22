"""2025の病巣の続き: 敵は「方向」ではなく「動かないこと」か。

前段で分かったこと:
  同年ランダム建て帰無に対する超過は 2022 +1.165 / 2023 +0.746 / 2024 +0.638 /
  **2025 +0.606** / 2026 +0.701 ＝ **エッジは劣化していない**。落ちたのは土台。
  かつ「下げ相場だから」ではない（2022 は BTC −65% でこのレッグの最高の年 +0.520R）。

∴ 仮説: 効くのは【動いたかどうか】で、方向ではない。レンジ年（2025 −6.1%）が敵。
測る量（すべて年別、拡大足とは独立に相場側の素性として）:
  実現ボラ（時間足リターンのσ・年率）
  効率比 ER = |終値の純変化| / Σ|1本ごとの変化|（1に近いほどトレンド的、0に近いほど往復）
  拡大足の発生率（そもそも引き金が出たか）
  拡大足の後の巡行幅（安値を割る前に伸びた幅の中央値・ATR単位）
最後に、レッグの1本Rを ER・ボラで層別して、関係が単調かを見る。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_2025 import prep, sig_idx, walk_idx      # noqa: E402
from scratchpad.atr_spike_barspread import spread_series, wilder_atr   # noqa: E402


def er(x):
    """効率比: 純変化 / 絶対変化の合計。"""
    dx = np.diff(x)
    s = np.abs(dx).sum()
    return abs(x[-1] - x[0]) / s if s > 0 else np.nan


if __name__ == "__main__":
    d = prep("btcusd")
    sp = spread_series("BTCUSD|h1")
    s = sig_idx(d)
    t = walk_idx(d, s, sp)
    t["y"] = t["time"].dt.year

    c = d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    o, h, l = (d[x].to_numpy() for x in ("open", "high", "low"))
    yy = d.index.year.to_numpy()
    ret = np.diff(np.log(c), prepend=np.nan)

    print("=== 相場側の素性（拡大足とは独立に）")
    print(f"  {'年':>5} {'年騰落':>9} {'実現ボラ(年率)':>13} {'効率比ER':>9} "
          f"{'拡大足の発生率':>13} {'拡大足後の巡行幅中央(ATR)':>22}")
    rows = {}
    for y in sorted(np.unique(yy)):
        m = yy == y
        px = c[m]
        vol = np.nanstd(ret[m]) * np.sqrt(24 * 365) * 100
        e = er(px)
        raw = (c - o > ap * 2.0) & (c > o) & np.isfinite(ap)
        rate = raw[m].mean() * 100
        # 拡大足後の巡行幅（安値を割るまでの最大幅・ATR単位）
        idx = np.flatnonzero(raw & m)
        mfe = []
        for i in idx[:2000]:
            end = min(i + 200, len(c) - 1)
            top = -np.inf
            j = i + 1
            while j <= end:
                top = max(top, h[j])
                if l[j] <= l[i]:
                    break
                j += 1
            if np.isfinite(top) and ap[i] > 0:
                mfe.append((top - c[i]) / ap[i])
        rows[y] = dict(ret=(px[-1] / px[0] - 1) * 100, vol=vol, er=e, rate=rate,
                       mfe=float(np.median(mfe)) if mfe else np.nan)
        print(f"  {y:>5} {rows[y]['ret']:>+8.1f}% {vol:>12.1f}% {e:>9.3f} "
              f"{rate:>12.2f}% {rows[y]['mfe']:>22.2f}")

    print("\n  レッグの1本R との並び（年別）")
    print(f"  {'年':>5} {'1本R':>8} {'超過(帰無比)':>12} {'ER':>7} {'ボラ':>8} {'巡行幅':>8}")
    exc = {2022: 1.165, 2023: 0.746, 2024: 0.638, 2025: 0.606, 2026: 0.701}
    for y in sorted(t["y"].unique()):
        g = t[t["y"] == y]["R_net"].mean()
        r = rows[y]
        print(f"  {y:>5} {g:>+8.3f} {exc.get(y, np.nan):>+12.3f} {r['er']:>7.3f} "
              f"{r['vol']:>7.1f}% {r['mfe']:>8.2f}")

    print("\n=== トレードを ER・ボラで層別（年をまたいでプールし、直前120本で測った値）")
    win = 120
    cs = pd.Series(c, index=d.index)
    roll_er = cs.rolling(win).apply(lambda x: er(x.to_numpy()), raw=False).shift(1)
    roll_vol = pd.Series(ret, index=d.index).rolling(win).std().shift(1) * np.sqrt(24 * 365) * 100
    t["er"] = roll_er.reindex(t["time"]).to_numpy()
    t["vol"] = roll_vol.reindex(t["time"]).to_numpy()
    tt = t.dropna(subset=["er", "vol"])
    for col, lab in (("er", "効率比ER（直前120本）"), ("vol", "実現ボラ（直前120本）")):
        q = tt[col].quantile([1/3, 2/3]).to_numpy()
        print(f"  -- {lab}")
        for lo, hi, nm in ((-np.inf, q[0], "低"), (q[0], q[1], "中"), (q[1], np.inf, "高")):
            g = tt[(tt[col] >= lo) & (tt[col] < hi)]
            r = g["R_net"].to_numpy()
            p = g["pct"].to_numpy()
            w, ls = p[p > 0].sum(), -p[p < 0].sum()
            print(f"     {nm}（{lo if np.isfinite(lo) else 0:.3f}〜"
                  f"{hi if np.isfinite(hi) else 99:.3f}） N={len(r):3d} "
                  f"勝率={(r>0).mean()*100:5.1f}% PF={w/ls if ls>0 else np.nan:5.2f} "
                  f"1本R={r.mean():+.3f}")

    assert len(tt) > 150, len(tt)
    print(f"\nOK: N={len(tt)}")
