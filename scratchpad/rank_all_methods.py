"""手元の全手法を【共通の物差し】で並べる。

🚨 そのまま比べてはいけない理由:
   ブックの6レッグは inv-vol 配分後の CAGR/DD で報告されている（重みが違う）
   Spike Rider は 0.01ロット固定の円で報告している
   PF も R基準と価格%基準では別物
∴ 全部を【1トレード＝同じ賭け（1R）・R単位】にそろえて、PF・maxDD(R)・年本数を出す。

含めるもの:
  採用ブックの6レッグ（research/book.py の純R系列。運用仕様＝Pine が実際に発注する形）
  Spike Rider（BTC 1H・k1.5・実スプレッド課金）— 段なし／段あり(0.02/0.01) の両方
  参考: ETH 1H（BTC同時確認つき）
⚠️ btc15m_L の系列には PDHソフト0.5 のサイズ写像が既に掛かっている（記録どおり）。
   Spike Rider の「段あり」も同様にサイズ写像入り。素の比較は「段なし」の行で見ること。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.book import get_book_legs                              # noqa: E402
from scratchpad.atr_spike_frequency import load, leg, gate           # noqa: E402
from scratchpad.atr_spike_barspread import spread_series, wilder_atr, spikes  # noqa: E402


def stats(r, idx, lab):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return None
    span = (idx.max() - idx.min()).days / 365.25
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = r[r > 0].sum(), -r[r < 0].sum()
    # 最長連敗
    run = mx = 0
    for x in r:
        run = run + 1 if x < 0 else 0
        mx = max(mx, run)
    return dict(lab=lab, n=len(r), span=span, ny=len(r) / span,
                win=(r > 0).mean() * 100, pf=w / ls if ls > 0 else np.nan,
                meanR=r.mean(), totR=r.sum(), dd=dd,
                score=r.sum() / dd if dd > 0 else np.nan, ddy=dd / (r.sum() / span) if r.sum() > 0 else np.nan,
                streak=mx)


if __name__ == "__main__":
    rows = []
    print("採用ブックの6レッグを構築中…", file=sys.stderr)
    for name, s in get_book_legs().items():
        st = stats(s.values, pd.DatetimeIndex(s.index), name)
        if st:
            rows.append(st)

    print("Spike Rider を構築中…", file=sys.stderr)
    b1 = load("btcusd", "h1")
    T = gate(leg(b1, 1.5, 20, spread_series("BTCUSD|h1")), b1, 120)
    ap = wilder_atr(b1).shift(1)
    T["body"] = ((b1["close"] - b1["open"]) / ap).reindex(T["time"]).to_numpy()
    T = T.dropna(subset=["body"]).reset_index(drop=True)
    # 純R = 価格%損益 ÷ 損切り幅（価格に対する率）
    T["R_net"] = T["pct"].to_numpy() / (T["risk"].to_numpy() / T["e_px"].to_numpy())
    idx = pd.DatetimeIndex(T["time"])
    rows.append(stats(T["R_net"].values, idx, "spike_rider（段なし）"))
    hit = ((T["body"] >= 2.0) & (T["er"] >= T["x50"])).to_numpy()
    rows.append(stats(T["R_net"].values * np.where(hit, 2.0, 1.0), idx,
                      "spike_rider（段0.02/0.01）"))

    e1 = load("ethusd", "h1")
    sB = spikes(b1)
    lead = (sB | sB.shift(1)).fillna(False)
    tE = leg(e1, 1.5, 20, spread_series("ETHUSD|h1"), lead=lead)
    tE["R_net"] = tE["pct"] / (tE["risk"] / tE["e_px"])
    rows.append(stats(tE["R_net"].values, pd.DatetimeIndex(tE["time"]), "eth_spike（参考）"))

    rows = [r for r in rows if r]
    print("\n" + "=" * 108)
    print("=== 全手法を【1トレード＝同じ賭け・R単位】でそろえた比較")
    print("=" * 108)
    for key, title in (("pf", "PF の上位"), ("score", "通算R ÷ maxDD(R) の上位"),
                       ("dd", "maxDD(R) の小さい順")):
        rev = key != "dd"
        srt = sorted([r for r in rows if np.isfinite(r[key])],
                     key=lambda r: r[key], reverse=rev)
        print(f"\n--- {title}")
        print(f"  {'#':>2} {'手法':<26} {'期間':>6} {'年本数':>7} {'勝率':>7} {'PF':>6} "
              f"{'1本R':>8} {'通算R':>8} {'maxDD(R)':>9} {'通算/DD':>8} {'最長連敗':>8}")
        for i, r in enumerate(srt[:5], 1):
            print(f"  {i:>2} {r['lab']:<26} {r['span']:>5.1f}年 {r['ny']:>7.0f} "
                  f"{r['win']:>6.1f}% {r['pf']:>6.2f} {r['meanR']:>+8.3f} {r['totR']:>+8.1f} "
                  f"{r['dd']:>9.1f} {r['score']:>8.2f} {r['streak']:>8}")

    print("\n--- 全手法（年本数の多い順・参考）")
    print(f"  {'手法':<26} {'期間':>6} {'年本数':>7} {'勝率':>7} {'PF':>6} "
          f"{'1本R':>8} {'通算R':>8} {'maxDD(R)':>9} {'通算/DD':>8}")
    for r in sorted(rows, key=lambda r: -r["ny"]):
        print(f"  {r['lab']:<26} {r['span']:>5.1f}年 {r['ny']:>7.0f} {r['win']:>6.1f}% "
              f"{r['pf']:>6.2f} {r['meanR']:>+8.3f} {r['totR']:>+8.1f} {r['dd']:>9.1f} "
              f"{r['score']:>8.2f}")

    assert len(rows) >= 8, len(rows)
    print(f"\nOK: {len(rows)} 手法")
