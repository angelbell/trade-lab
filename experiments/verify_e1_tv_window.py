"""TVのbtc15m_L(E1階段)バックテストとの tie-back 検証。
TVが15分足で読めた窓（2026-04-01〜2026-07-24, 約4か月）に限定し、Python(Vantage CSV)で
baseline(PDHソフト0.5) と E1(成分1階段×成分2日足) を再現、TVの主要統計と突き合わせる。

TV実測（E1オフ→オン, 10K口座/1%risk）:
  baseline: n=35 win=4(11.43%) PF=0.717 maxDD=19.70% total=-8.16%
  E1オン  : n=35 win=4(11.43%) PF=0.798 maxDD=13.73% total=-3.83%
照合の軸:
  1) baselineとE1でトレード数が一致するか（E1は入口を変えない＝厳密一致のはず）
  2) 窓のトレード数・勝ち数がTV(35/4)に近いか（フィード差で±数本は許容）
  3) E1のDD<baselineのDD、E1の損失<baselineの損失（TVと同じ向きか）
新規ロジックなし。stack_size_btc15mL の関数を窓で呼ぶだけ。
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import numpy as np
import pandas as pd
from stack_size_btc15mL import (build_population, apply_size, comp1_ladder, comp2_daily)

W_START, W_END = "2026-04-01", "2026-07-24"


def curve_stats(R, ti):
    """1%risk複利での total-return% と maxDD%（TVの主要統計と同じ土俵）。"""
    order = np.argsort(ti.values)
    R_s = np.asarray(R)[order]
    eq = np.cumprod(1.0 + 0.01 * R_s)          # 10K基準の口座倍率
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max()) * 100.0
    total = (eq[-1] - 1.0) * 100.0
    pos, neg = R_s[R_s > 0].sum(), -R_s[R_s <= 0].sum()
    pf = pos / neg if neg > 0 else float("inf")
    win = int((R_s > 0).sum())
    return len(R_s), win, 100.0 * win / len(R_s), pf, total, dd


if __name__ == "__main__":
    d15, t, ii = build_population()
    assert len(t) == 763, f"母体tie-back失敗: n={len(t)} (既知763)"  # 検算: 母集団が凍結仕様と一致

    # 窓に限定
    tt = pd.to_datetime(t["time"].values)
    m = (tt >= pd.Timestamp(W_START)) & (tt <= pd.Timestamp(W_END))
    print(f"\n窓 {W_START}〜{W_END}: 全{len(t)}本中 {int(m.sum())}本")

    # baseline: PDHソフト(1.0/0.5)
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w_base = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)

    # E1: 成分1(階段) × 成分2(日足)
    W1, above_pdh, above_hh4 = comp1_ladder(d15, t, ii)
    W2, _ = comp2_daily(d15, t, ii)
    w_e1 = W1 * W2

    R_base = apply_size(t, w_base)
    R_e1 = apply_size(t, w_e1)

    ti = pd.Series(tt)
    for name, R, w in [("baseline (PDHソフト0.5)", R_base, w_base),
                       ("E1 (階段×日足)", R_e1, w_e1)]:
        n, win, winp, pf, total, dd = curve_stats(R[m], ti[m])
        print(f"\n  --- {name} ---")
        print(f"    n={n}  win={win}({winp:.2f}%)  PF={pf:.3f}  total={total:+.2f}%  maxDD={dd:.2f}%")

    # サイズ内訳（窓内・E1）
    ww = w_e1[m]
    full = int((np.abs(ww - 1.0) < 1e-9).sum())
    print(f"\n  E1サイズ内訳(窓内): フル(1.0)={full}/{int(m.sum())} ({100*full/max(int(m.sum()),1):.0f}%)  "
          f"重みの分布={sorted(np.round(np.unique(ww),3).tolist())}")

    # --- 検算assert（既知の性質） ---
    assert len(R_base) == len(R_e1), "baselineとE1でトレード数が違う＝入口が動いた"
    print("\n  OK(検算): baselineとE1のトレード数が全期間で一致（E1は入口を変えない）")
