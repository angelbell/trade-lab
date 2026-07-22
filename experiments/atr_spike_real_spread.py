"""実測スプレッドを入れ直して、アルト群のレッグが何本生き残るかを出す。

これまでの検証はアルトの往復コストを 0.05〜0.20% と仮定していた（未実測なので保守側のつもりだった）。
Vantage デモ口座で実測したところ、仮定より 1.5〜50倍 広い銘柄が並んだ。
∴ 銘柄ごとの実測値を入れて、レッグ単位・ポートフォリオ単位で判定し直す。

🚨 実測は 2026-07-22 06:23 サーバ時刻（最も閑散な時間帯）の1点。活発な時間帯では狭くなるはずなので、
   ここでの結果は【下限】。時間帯の分布は sample_crypto_spread.py で別途採取する。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_btc_leader import load, spikes, leg, st, ALTS   # noqa: E402

# 2026-07-22 06:23 サーバ時刻の実測（往復＝ask-bid、暗号資産は手数料ゼロ）
SPREAD = {"btcusd": 0.00038, "ethusd": 0.00132, "trxusd": 0.00307, "bnbusd": 0.00481,
          "solusd": 0.00691, "xrpusd": 0.00719, "bchusd": 0.01500, "ltcusd": 0.02205,
          "adausd": 0.03676, "dotusd": 0.10047}
ASSUMED = 0.0020


def run(sym, cost, lead=None):
    t = leg(load(sym), cost_pct=0.0)
    if t is None:
        return None
    p = t["pct"].to_numpy() - cost           # leg() は cost_pct=0 で素を返す
    rf = None
    if lead is not None:
        m = lead.reindex(t["time"]).fillna(False).to_numpy()
        p = p[m]
        t = t[m]
    if len(p) < 15:
        return None
    # 損切り幅（価格に対する率）を復元して R 単位も出す
    gross = t["pct"].to_numpy()
    rf = np.where(gross != 0, np.abs((t["R"] * t["risk"] / t["e_px"]).to_numpy() /
                                     np.where(t["R"] != 0, t["R"], np.nan)), np.nan)
    rf = np.nanmedian(rf)
    r = p / rf
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return dict(n=len(p), win=(p > 0).mean() * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, rf=rf * 100, meanR=r.mean(), totR=r.sum(), ddR=dd,
                ratio=r.sum() / dd if dd > 0 else np.nan,
                costR=cost / rf * 100)


if __name__ == "__main__":
    b = spikes(load("btcusd"))
    lead = (b | b.shift(1)).fillna(False)
    span = 4.5
    print("=== BTC 同時確認つきアルト・レッグを、仮定コストと実測スプレッドで比べる（Vantage 2022-）")
    print(f"  {'銘柄':<9} {'N':>5} {'損切幅%':>8} {'実測ｽﾌﾟﾚｯﾄﾞ':>11} {'1Rに占める':>10} | "
          f"{'仮定0.20%: PF/1本R/比':>26} | {'実測: PF/1本R/比':>26} {'判定':>6}")
    alive, dead = [], []
    for sym in ALTS:
        a = run(sym, ASSUMED, lead)
        r = run(sym, SPREAD[sym], lead)
        if a is None or r is None:
            print(f"  {sym:<9} 本数不足")
            continue
        ok = "生存" if r["pf"] > 1.2 and r["ratio"] > 2.0 else "死"
        (alive if ok == "生存" else dead).append(sym)
        print(f"  {sym:<9} {a['n']:>5} {a['rf']:>7.2f}% {SPREAD[sym]*100:>10.3f}% "
              f"{r['costR']:>9.0f}% | {a['pf']:>7.2f} {a['meanR']:>+7.3f}R {a['ratio']:>7.2f} | "
              f"{r['pf']:>7.2f} {r['meanR']:>+7.3f}R {r['ratio']:>7.2f} {ok:>6}")
    print(f"\n  生存: {alive if alive else 'なし'}")
    print(f"  死  : {dead if dead else 'なし'}")

    print("\n=== BTC 自身（参考・確認条件なし）")
    for sym in ("btcusd",):
        a = run(sym, ASSUMED)
        r = run(sym, SPREAD[sym])
        print(f"  {sym:<9} N={a['n']:4d} 損切幅={a['rf']:.2f}%  "
              f"仮定0.20%: PF={a['pf']:.2f} 比={a['ratio']:.2f}  →  "
              f"実測{SPREAD[sym]*100:.3f}%: PF={r['pf']:.2f} 比={r['ratio']:.2f}")

    print("\n=== 生存銘柄だけでポートフォリオを組み直す")
    rows = []
    for sym in ALTS + ["btcusd"]:
        c = SPREAD[sym]
        rf_probe = run(sym, c, lead if sym != "btcusd" else None)
        if rf_probe is None or rf_probe["pf"] <= 1.2 or rf_probe["ratio"] <= 2.0:
            continue
        t = leg(load(sym), cost_pct=0.0)
        m = (lead.reindex(t["time"]).fillna(False).to_numpy()
             if sym != "btcusd" else np.ones(len(t), dtype=bool))
        p = t["pct"].to_numpy()[m] - c
        rows.append(pd.DataFrame({"sym": sym, "time": t["time"].values[m],
                                  "R": p / (rf_probe["rf"] / 100)}))
    if rows:
        P = pd.concat(rows, ignore_index=True).sort_values("time")
        r = P["R"].to_numpy()
        eq = np.cumsum(r)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        print(f"  採用 {sorted(P['sym'].unique())}")
        print(f"  N={len(r)}（年{len(r)/span:.0f}本）勝率={(r>0).mean()*100:.1f}% "
              f"1本R={r.mean():+.3f} totR={r.sum():+.1f} DD={dd:.1f}R 比={r.sum()/dd:.2f}")

    assert len(dead) >= 3, dead
    print(f"\nOK: 実測スプレッドで {len(dead)} 銘柄が脱落")
