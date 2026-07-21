"""E1 採否の実務資料: btc15m_L サイズスタックの「形 × 賭け率」換算表。

形: baseline(現行PDHソフト0.5) / フル(ICT弱0.5) / フル(ICT弱0.25) / フル(ICT弱0=見送り)。
賭け率 r = 重み1.0の玉に張る基本リスク%（スタックの重みはこれに乗算される）。

各セル: ブートストラップ中央値maxDD（賭け率は必ずここから決める・実測1経路は運を含む）、
95%点DD、実測(単一経路)のCAGR/maxDD、CAGR/中央値DD。想定実DD=中央値×1.5〜2（CLAUDE.md）。

母集団・会計は stack_size_btc15mL.py と同一（コスト$15+スワップ30%/年込み、tie-back付き）。
Run: .venv/bin/python scratchpad/E1_bet_table.py 2>/dev/null | tee scratchpad/out_E1_bet_table.txt
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.arbiter import cd, Boot
from src.engine.size import pdh_soft
from stack_size_btc15mL import (build_population, apply_size, comp1_ladder, comp2_daily,
                                comp3_ict, comp3_ict_variant)

RISKS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]   # 基本リスク%/玉（重み1.0のとき）
NB = 1000
MINLOT_BTC = 0.01                               # Vantage BTCUSD の最小玉（要ターミナル確認）
USDJPY_ASSUMED = 150.0                          # 円口座→USD換算の仮置き
ACCOUNTS_JPY = [500_000, 1_000_000, 3_000_000, 5_000_000]


def dd_dist(boot, s):
    """全ブートストラップ経路の maxDD 分布（中央値・95%点）。"""
    mk = s.index.to_period("M")
    by = {m: s.values[mk == m] for m in boot.months}
    n = len(s)
    out = np.empty(len(boot.layout))
    for i, seq in enumerate(boot.layout):
        v = np.concatenate([by[boot.months[j]] for j in seq])[:n]
        eq = np.cumprod(1.0 + v)
        pk = np.maximum.accumulate(eq)
        out[i] = ((pk - eq) / pk).max() * 100.0
    return float(np.median(out)), float(np.percentile(out, 95))


def main():
    d15, t, ii = build_population()
    ti = pd.DatetimeIndex(t["time"])
    days = max((ti[-1] - ti[0]).days, 1)

    W_base, _ = pdh_soft(d15, t)
    W1, ap, ah = comp1_ladder(d15, t, ii)
    W2, dn = comp2_daily(d15, t, ii)
    W3, AB = comp3_ict(d15, t, ii, x=48)
    forms = {
        "baseline(PDHソフト)":  (W_base, apply_size(t, W_base)),
        "フル(ICT弱0.5)":       (W1 * W2 * W3, apply_size(t, W1 * W2 * W3)),
        "フル(ICT弱0.25)":      (W1 * W2 * comp3_ict_variant(AB, 0.25),
                                 apply_size(t, W1 * W2 * comp3_ict_variant(AB, 0.25))),
        "フル(ICT弱0=見送り)":   (W1 * W2 * comp3_ict_variant(AB, 0.0),
                                 apply_size(t, W1 * W2 * comp3_ict_variant(AB, 0.0))),
    }

    months = sorted(set(ti.to_period("M")))
    boot = Boot(months, nb=NB, k=3, seed=20260717)

    print(f"\n形 × 賭け率の換算表（n=763・7.7年・コスト$15+スワップ込み・ブートストラップ{NB}経路）")
    print("実運用の賭け率は『中央値DD ×1.5〜2 ≦ 自分が耐えられるDD』から逆算する（実測1経路のDDは運を含む）\n")
    hdr = f"  {'形':<20}{'r%':>6}{'CAGR実測':>10}{'DD実測':>8}{'DD中央値':>10}{'DD95%':>8}{'想定実DD':>12}{'CAGR/DD中央':>12}"
    print(hdr)
    for name, (W, R) in forms.items():
        for r in RISKS:
            s = pd.Series(R * (r / 100.0), index=ti)
            cagr, dd1 = cd(s.values, days)
            ddm, dd95 = dd_dist(boot, s)
            print(f"  {name:<20}{r:>6.2f}{cagr:>+9.1f}%{dd1:>7.1f}%{ddm:>9.1f}%{dd95:>7.1f}%"
                  f"{ddm*1.5:>6.1f}-{ddm*2:.1f}%{cagr/max(ddm,1e-9):>12.2f}")
        print()

    # ---- 最小玉の実行可能性（小口座の床）----
    print("最小玉チェック（仮定: 最小玉0.01BTC・USDJPY=150。玉サイズ= 口座×r%×重み ÷ 損切り幅$）")
    print(f"  損切り幅$の分布: 中央値 {np.median(t['risk']):.0f}  10%点 {np.percentile(t['risk'],10):.0f}  90%点 {np.percentile(t['risk'],90):.0f}")
    W_full = forms["フル(ICT弱0.5)"][0]
    for r in (0.5, 1.0):
        line = []
        for acct in ACCOUNTS_JPY:
            usd = acct / USDJPY_ASSUMED
            lots = usd * (r / 100.0) * W_full / t["risk"].values
            below = 100.0 * np.mean(lots < MINLOT_BTC)
            line.append(f"{acct//10000:>4}万円: {below:4.1f}%")
        print(f"  r={r:.1f}% で最小玉を下回る玉の割合   " + "  ".join(line))
    print("  （下回る玉は最小玉で張る＝そのぶん実効リスクが設計より大きくなる。割合が2桁なら口座かrを上げる）")


if __name__ == "__main__":
    main()
