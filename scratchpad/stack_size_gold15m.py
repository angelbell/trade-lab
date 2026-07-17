"""E5: 直交サイズ変数の統合スタックを gold15m へ移植（btc15m_Lで確立したE1と同じ裁定器）。
報告形式・裁定器はstack_size_btc15mL.pyと完全同一(Boot.equal_dd_cagr同DD CAGR・40seedランダムnull・
巡回ブロック1/3/6/12mo・逆ダミー・年別totR)。

母集団(凍結): gold15m 正典 = BASE+daily_sma150/slope10+ext_cap8.0+pullback_frac0.25+fill_win200
  (rr=BASE既定4.0)、cost=$0.6（本カード指定）。tie-back: 生run n=325・meanR≈+0.585。
  ベースライン=現行形(goldにはPDHソフト無し=素、W=1.0全trade)。

サイズ写像:
  成分1 PDH x HH4H階段: 約定価格が前日高値の上 かつ 直近確定4Hスイング高値の上=1.0/片方=0.5/無し=0.25
    (stack_size_btc15mL.comp1_ladder/hh4h_series をそのまま流用、goldのd15を渡すだけ)。
  成分2 日足レジーム: 日足終値[確定]<SMA150 で x0.75 (comp2_daily を流用)。
    ※ホストの入口ゲート自体が「日足SMA150上向き」を要求するため、成分2のあり率はゲートと縮退し
    ≈0%になる可能性がある --- 実測を必ず印字する。
  成分3 ICTラベルAのみ(X=48, 狩りのみ): 無しでx0.5 (compute_labels の labelA のみ使用、Bはgoldで
    死んでいると確定済みのため不使用)。

流用（車輪の再発明禁止）: breakout_wave.{run,resample}／radar_gate_race.BASE／
stack_size_btc15mL.{hh4h_series, comp1_ladder, comp1_ladder_reverse, comp2_daily,
comp2_daily_reverse, cell_report}／ict_size_transplant.{compute_labels, RISK_PCT,
block_boot_beat}／arb_common.{Boot, cd}。

Run: .venv/bin/python scratchpad/stack_size_gold15m.py [--smoke] 2>&1 | tee scratchpad/out_stack_size_gold15m.txt
"""
import sys, io, contextlib, argparse, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import numpy as np
import pandas as pd

from arb_common import Boot, cd
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from ict_size_transplant import compute_labels, RISK_PCT, block_boot_beat
from stack_size_btc15mL import (hh4h_series, comp1_ladder, comp1_ladder_reverse,
                                comp2_daily, comp2_daily_reverse, cell_report)

ROOT = "/home/angelbell/dev/auto-trade"
COST_GOLD = 0.6
NB_MAIN = 200


def build_population():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    CFG = {**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0, "pullback_frac": 0.25,
           "fill_win": 200, "fwd": 500}
    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d15, SimpleNamespace(**CFG))
    print(f"  tie-back(生run): n={len(t)}  meanR={t['R'].mean():+.3f}  (既知: n=325, meanR≈+0.585)")
    ii = d15.index.get_indexer(t["time"])
    return d15, t, ii


def apply_size(t, W):
    """cost=$0.6固定、swap無し(gold正典どおり)。Wはbtc版と同じ一般化(risk=t.risk/W)。"""
    risk = t["risk"].values / W
    R = t["R"].values * W - COST_GOLD / risk
    return R


def comp3_ict_A_only(d15, t, ii, x=48):
    labelA, labelB = compute_labels(d15, t, ii, x)
    W = np.where(labelA, 1.0, 0.5)
    return W, labelA


def comp3_variant(labelA, weak_w):
    return np.where(labelA, 1.0, weak_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("母集団 tie-back (gold15m)")
    print("#" * 110)
    d15, t, ii = build_population()
    if args.smoke:
        m = pd.DatetimeIndex(t["time"]) >= pd.Timestamp("2024-01-01", tz=pd.DatetimeIndex(t["time"]).tz)
        t = t[m].reset_index(drop=True); ii = ii[m]
    ti = pd.DatetimeIndex(t["time"])

    print("\n" + "#" * 110)
    print("ベースライン（現行形=goldにはPDHソフト無し、W=1.0全trade）")
    print("#" * 110)
    W_base = np.ones(len(t))
    R_base = apply_size(t, W_base)
    s_base = cell_report("baseline (no sizing)", R_base, ti)

    months = sorted(s_base.index.to_period("M").unique())
    boot = Boot(months, nb=NB_MAIN, k=3, seed=20260717)
    D0 = boot.dd_median(s_base)
    cagr_base, _ = boot.equal_dd_cagr(s_base, D0)
    print(f"    同DD({D0:.2f}%)でのbaseline CAGR = {cagr_base:+.1f}%  (以降はこのD0に揃えて比較)")

    W1, above_pdh, above_hh4 = comp1_ladder(d15, t, ii)
    W2, down_at_b = comp2_daily(d15, t, ii)
    W3, labelA = comp3_ict_A_only(d15, t, ii, x=48)
    print(f"\n  成分あり率: HTF階段(both)={100*np.mean(above_pdh&above_hh4):.1f}%  "
          f"(above_pdh単独={100*np.mean(above_pdh):.1f}%  above_hh4単独={100*np.mean(above_hh4):.1f}%)  "
          f"日足↓={100*np.mean(down_at_b):.1f}%  ICT(狩りA)あり={100*np.mean(labelA):.1f}%")

    def eval_and_report(label, W):
        R = apply_size(t, W)
        s = cell_report(label, R, ti)
        cagr, scale = boot.equal_dd_cagr(s, D0)
        print(f"    同DD({D0:.2f}%)でのCAGR = {cagr:+.1f}%  (対baseline差={cagr-cagr_base:+.1f}pt)  "
              f"スケール={scale:.3f}")
        return s, cagr

    print("\n" + "#" * 110)
    print("成分単独x3")
    print("#" * 110)
    s1, cagr1 = eval_and_report("成分1単独: HTF階段(PDH x HH4H)", W1)
    s2, cagr2 = eval_and_report("成分2単独: 日足レジーム(SMA150)", W2)
    s3, cagr3 = eval_and_report("成分3単独: ICTラベル(狩りAのみ,X=48)", W3)

    print("\n" + "#" * 110)
    print("フルスタック（1x2x3）")
    print("#" * 110)
    Wfull = W1 * W2 * W3
    s_full, cagr_full = eval_and_report("フルスタック", Wfull)

    print("\n" + "#" * 110)
    print("フルから1成分ずつ抜いた3形")
    print("#" * 110)
    s_no1, cagr_no1 = eval_and_report("フル - 成分1(HTF階段抜き)", W2 * W3)
    s_no2, cagr_no2 = eval_and_report("フル - 成分2(日足抜き)", W1 * W3)
    s_no3, cagr_no3 = eval_and_report("フル - 成分3(ICT抜き)", W1 * W2)
    print(f"\n    限界利得(フル - (フル-成分)): 成分1={cagr_full-cagr_no1:+.1f}pt  "
          f"成分2={cagr_full-cagr_no2:+.1f}pt  成分3={cagr_full-cagr_no3:+.1f}pt")

    print("\n" + "#" * 110)
    print("外挿: フルスタックでICT成分だけ x0.25 / x0 に振った2形")
    print("#" * 110)
    W_ict025 = comp3_variant(labelA, 0.25)
    Wz = W1 * W2 * np.where(labelA, 1.0, 1e-9)
    s_e025, cagr_e025 = eval_and_report("フル、ICT弱=0.25", W1 * W2 * W_ict025)
    s_e0, cagr_e0 = eval_and_report("フル、ICT弱=0(実質フィルタ)", Wz)

    print("\n" + "#" * 110)
    print("同数ランダムサイズnull(40回): フルスタックの重みの多重集合をシャッフルして張り直す")
    print("#" * 110)
    null_diffs = []
    for seed in range(40):
        perm = np.random.default_rng(20260717 + seed).permutation(len(Wfull))
        W_shuf = Wfull[perm]
        R_shuf = apply_size(t, W_shuf)
        s_shuf = pd.Series(R_shuf * RISK_PCT, index=ti).sort_index()
        cagr_shuf, _ = boot.equal_dd_cagr(s_shuf, D0)
        null_diffs.append(cagr_shuf - cagr_base)
    null_diffs = np.array(null_diffs)
    real_diff = cagr_full - cagr_base
    pct = 100 * np.mean(null_diffs < real_diff)
    print(f"    実測差(フル-baseline)={real_diff:+.1f}pt  null帯=[{np.percentile(null_diffs,2.5):+.1f},"
          f"{np.percentile(null_diffs,97.5):+.1f}]pt (中央値={np.median(null_diffs):+.1f})  -> {pct:.0f}%ile")

    print("\n" + "#" * 110)
    print("巡回ブロック・ブートストラップ 1/3/6/12mo: P(フルスタックがbaselineを同DD-CAGRで上回る)")
    print("#" * 110)
    for k in (1, 3, 6, 12):
        p = block_boot_beat(s_base, s_full, months, k, nb=300)
        print(f"    k={k:>2}mo: P={p:.0f}%")

    print("\n" + "#" * 110)
    print("逆ダミー: 写像を反転（階段0.25<->1.0、日足↓でx1.25、ラベル無しでx1.5）")
    print("#" * 110)
    W1_rev = comp1_ladder_reverse(above_pdh, above_hh4)
    W2_rev = comp2_daily_reverse(down_at_b)
    W3_rev = comp3_variant(labelA, 1.5)
    s_rev, cagr_rev = eval_and_report("逆ダミー フルスタック", W1_rev * W2_rev * W3_rev)
    print(f"\n    逆ダミー vs baseline: {'機構整合(悪化,OK)' if cagr_rev < cagr_base else '⚠️機構不整合(悪化せず)'}")
    print(f"    逆ダミー vs フルスタック: {'機構整合(フルが優位,OK)' if cagr_full > cagr_rev else '⚠️機構不整合'}")

    print("\n" + "#" * 110)
    print("事前登録判定材料（生数字のみ）")
    print("#" * 110)
    print(f"    フル対baseline差={real_diff:+.1f}pt  null%ile={pct:.0f}  "
          f"12moブロックP={block_boot_beat(s_base, s_full, months, 12, nb=300):.0f}%")


if __name__ == "__main__":
    main()
