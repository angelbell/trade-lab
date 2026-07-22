"""ICT 再設計ライン — 項目4 Step1 FVG-MSS の審判（フェーズ4）。

ict_fvg_mss_ablation.py で FVG段が濃縮して見えたセルについて、
「本当に仕事をしているか」を CLAUDE.md チェックリスト通りに刺す:
  1. プラセボ窓 +4/8/12h のプレミアム（窓非依存の偽約定でないか）
  2. random-drop null（同じ本数をランダムに残すのに対し、FVG選別が totR/DD で勝つか＝濃縮の必要条件）
  3. 巡回ブロック・ブートストラップ 1/3/6/12か月（別の月の並びでも totR>0 か）
  4. 時代別（4era）・IS/OOS（一時代のベータでないか）

対象セル（ablation で PF>1・単調 or 大幅改善だったもの）:
  eurusd long / usdjpy long / audusd long / btcusd long（各 stage b/c/d）
  gbpusd/nzdusd/usdcad/gold と全 short は ablation で悪化 or 損益分岐未満のため対象外。

Run: .venv/bin/python experiments/ict_fvg_mss_audit.py [--smoke]
"""
import sys, io, contextlib, argparse
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, F_CANON, RR_CANON, walk, stats
from ict_population import canonical_setups, load_prepped
from ict_audit import placebo_premium, random_drop_null, block_boot

ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]
STAGES = [("b_FVGexist", True, 0.00), ("c_FVG0.15", True, 0.15), ("d_FVG0.25", True, 0.25)]
CANDIDATES = ["eurusd", "usdjpy", "audusd", "btcusd"]   # long のみ ablation で生存


def eras_of(tr):
    return " ".join(
        f"{sum(x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b):+7.1f}"
        if any(a <= pd.Timestamp(x[0]).year <= b for x in tr) else "    n/a"
        for a, b in ERAS)


def audit_cell(df, tarr, dates, name, side, span, lab, uf, fm):
    sp, cost = MODEL[name]
    S_base = canonical_setups(df, tarr, dates, 0, use_fvg=False, fvg_min_atr=0.0)
    S_fvg = canonical_setups(df, tarr, dates, 0, use_fvg=uf, fvg_min_atr=fm)
    tr_base = walk(df, S_base, F_CANON, RR_CANON, BUF, sp, cost, side)
    tr_fvg = walk(df, S_fvg, F_CANON, RR_CANON, BUF, sp, cost, side)
    st = stats(tr_fvg, span)
    if st is None:
        print(f"  [{name} {side} {lab}] n<10, skip")
        return

    print(f"\n  --- {name} {side} {lab} ---")
    print(f"  素の数値: n={st['n']} n/yr={st['npy']:.1f} win%={st['win']:.1f} meanR={st['net']:+.3f} "
          f"PF={st['pf']:.2f} totR/DD={st['rdd']:.2f} IS={st['IS']:+.0f} OOS={st['OOS']:+.0f}")

    # 1. プラセボ窓
    pp = placebo_premium(df, tarr, dates, name, side, span, use_fvg=uf, fvg_min_atr=fm)
    r0, r8 = pp[0], pp[8]
    prem8 = (r0["net"] - r8["net"]) if (r0 and r8) else float("nan")
    print(f"  プラセボ窓: " + "  ".join(
        f"+{sh}h(n={pp[sh]['n'] if pp[sh] else 0},net={pp[sh]['net'] if pp[sh] else float('nan'):+.3f})"
        for sh in (0, 4, 8, 12)))
    print(f"  窓プレミアム(0h - 8h) = {prem8:+.3f}")

    # 2. random-drop null（母集団=stage-a 生MSS、同数kをランダム残し）
    null = random_drop_null(tr_base, st["n"], nrep=2000)
    pct = 100.0 * (null < st["rdd"]).mean()
    print(f"  random-drop null(母集団 raw-MSS n={len(tr_base)}, k={st['n']}): "
          f"null中央値totR/DD={np.median(null):+.2f} 実測%ile={pct:.1f}")

    # 3. 巡回ブロック・ブートストラップ
    bb = {m: block_boot(tr_fvg, m) for m in (1, 3, 6, 12)}
    print(f"  ブロックブートストラップ P(totR>0): " + "  ".join(
        f"{m}mo={bb[m]:.0f}%" for m in (1, 3, 6, 12)))

    # 4. 時代別
    print(f"  時代別({'/'.join(f'{a}-{b}' for a,b in ERAS)}): {eras_of(tr_fvg)}")


def main(smoke=False):
    for name in CANDIDATES:
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.25):]
        print(f"\n{'='*100}\n{name} (span={span}年)\n{'='*100}")
        for lab, uf, fm in STAGES:
            audit_cell(df, tarr, dates, name, "long", span, lab, uf, fm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
