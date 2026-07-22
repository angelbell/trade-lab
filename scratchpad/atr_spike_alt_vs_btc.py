"""レッグの存在理由そのものを疑う: BTC が参加している瞬間なら、BTC を建てればいいのではないか。

いま分かっていること: アルトの拡大足は「BTC も同時に拡大足」のときだけ効く（全関門を通過）。
だが、その条件が満たされる瞬間には **BTC 自身も拡大足を出している＝BTC のレッグも建てられる**。
∴ アルトを建てる意味があるのは、次のどちらかが成り立つときだけ:
   (1) 同じ瞬間に、アルトのほうが BTC より良い
   (2) アルトの結果が BTC と十分に独立していて、束ねると滑らかになる
どちらも成り立たないなら、この一式は BTC の劣化コピーで、枠は BTC に使うべき。

手動執行では同時に持てるのは1〜2本なので、問いはこう言い換えられる:
**確認済みシグナルが出た瞬間、その枠にどの銘柄を入れるべきか。**

比較（すべて同一仕様・往復コスト0.20%・totR と maxDD(R) で見る）:
  a. BTC 単独レッグ
  b. アルト（BTC確認つき）・上限2本
  c. BTC ＋ アルト（BTC確認つき）・合計上限2本
  d. 同時刻のペア（アルト vs その瞬間の BTC）の結果相関
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_leader_binance import load, spikes, leg, st          # noqa: E402,F401
from scratchpad.atr_spike_leader_cap import apply_cap, B_USE, V_USE            # noqa: E402
from scratchpad.atr_spike_btc_leader import (load as vload, spikes as vspikes,  # noqa: E402
                                             leg as vleg)

COST = 0.0020


def score(r):
    if len(r) < 5:
        return np.nan
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(r), totR=r.sum(), ddR=dd, ratio=r.sum() / dd if dd > 0 else np.nan,
                meanR=r.mean(), win=(r > 0).mean() * 100)


def build(feed):
    if feed == "binance":
        ld, sp, lg, use, btc = load, spikes, leg, B_USE, "btcusdt"
    else:
        ld, sp, lg, use, btc = vload, vspikes, vleg, V_USE, "btcusd"
    b = sp(ld(btc))
    sB = (b | b.shift(1)).fillna(False)

    def frame(sym, mark):
        t = lg(ld(sym))
        if t is None:
            return None
        g = t["gross"].to_numpy() if "gross" in t else t["pct"].to_numpy()
        rf = (t["risk"] / t["e_px"]).to_numpy()
        f = pd.DataFrame({"sym": sym, "time": t["time"].values, "hold": t["hold"].values,
                          "R": (g - COST) / rf})
        f["lead"] = sB.reindex(t["time"]).fillna(False).to_numpy() if mark else True
        return f

    bt = frame(btc, False)
    alts = [frame(s, True) for s in use]
    alts = pd.concat([a[a["lead"]] for a in alts if a is not None], ignore_index=True)
    return bt.sort_values("time").reset_index(drop=True), \
        alts.sort_values("time").reset_index(drop=True)


def capped(g, k):
    g = g.sort_values("time").reset_index(drop=True)
    return g.loc[apply_cap(g, k)]


if __name__ == "__main__":
    for feed in ("binance", "vantage"):
        BT, AL = build(feed)
        span = (max(BT["time"].max(), AL["time"].max())
                - min(BT["time"].min(), AL["time"].min())).days / 365.25
        print(f"\n===== {feed.upper()}  ({span:.1f}年・往復{COST*100:.2f}%)")
        print(f"  {'構成':<34} {'N':>5} {'年本数':>7} {'勝率':>7} {'1本のR':>8} "
              f"{'totR':>8} {'DD(R)':>7} {'totR/DD':>8}")
        rows = [("a. BTC 単独レッグ", BT),
                ("b. アルト(BTC確認)・上限2本", capped(AL, 2)),
                ("b'. アルト(BTC確認)・上限1本", capped(AL, 1))]
        both = pd.concat([BT, AL], ignore_index=True).sort_values("time").reset_index(drop=True)
        rows.append(("c. BTC＋アルト・合計上限2本", capped(both, 2)))
        rows.append(("c'. BTC＋アルト・合計上限1本", capped(both, 1)))
        # BTC を優先して枠を埋める（同時刻なら BTC を先に）
        both2 = both.copy()
        both2["prio"] = (both2["sym"] != BT["sym"].iloc[0]).astype(int)
        both2 = both2.sort_values(["time", "prio"]).reset_index(drop=True)
        rows.append(("d. BTC優先＋アルト・上限2本", capped(both2, 2)))
        for lab, g in rows:
            s = score(g["R"].to_numpy())
            print(f"  {lab:<34} {s['n']:>5} {s['n']/span:>7.0f} {s['win']:>6.1f}% "
                  f"{s['meanR']:>+8.3f} {s['totR']:>+8.1f} {s['ddR']:>7.1f} {s['ratio']:>8.2f}")

        # 同じ瞬間のペア: アルトのトレードと、同時刻±1本で建った BTC トレード
        bt_idx = pd.Series(BT["R"].to_numpy(), index=pd.DatetimeIndex(BT["time"]))
        pair_a, pair_b = [], []
        for _, r in AL.iterrows():
            w = bt_idx[(bt_idx.index >= r["time"] - pd.Timedelta(hours=1)) &
                       (bt_idx.index <= r["time"] + pd.Timedelta(hours=1))]
            if len(w):
                pair_a.append(r["R"])
                pair_b.append(w.iloc[0])
        pair_a, pair_b = np.array(pair_a), np.array(pair_b)
        if len(pair_a) > 30:
            print(f"  --- 同時刻ペア N={len(pair_a)}（アルトの何%が BTC と同時に建つか: "
                  f"{len(pair_a)/len(AL)*100:.0f}%）")
            print(f"      アルト側 1本のR={pair_a.mean():+.3f}  BTC側 1本のR={pair_b.mean():+.3f}  "
                  f"差={pair_a.mean()-pair_b.mean():+.3f}")
            print(f"      結果の相関 Pearson={np.corrcoef(pair_a, pair_b)[0,1]:+.2f}  "
                  f"符号一致率={np.mean(np.sign(pair_a)==np.sign(pair_b))*100:.0f}%")
            print(f"      アルトが BTC を上回った割合={np.mean(pair_a>pair_b)*100:.0f}%")
        # BTC が建たない確認済みアルト（＝BTC のレッグが前日高値等で落ちた瞬間）
        lone = AL[~AL["time"].isin(pd.DatetimeIndex(BT["time"]))]
        if len(lone) > 30:
            s = score(lone["R"].to_numpy())
            print(f"  --- BTC が建たなかった瞬間のアルト N={s['n']} 勝率={s['win']:.1f}% "
                  f"1本のR={s['meanR']:+.3f} totR={s['totR']:+.1f}")

    print("\n（判定: c/d が a を明確に上回らないなら、アルト一式は BTC の劣化コピー。"
          "\n  枠は BTC に使い、アルトは畳むべき）")

    BT, AL = build("binance")
    assert len(BT) > 200 and len(AL) > 400, (len(BT), len(AL))
    print(f"\nOK: Binance の BTC 単独 N={len(BT)} / 確認済みアルト N={len(AL)}")
