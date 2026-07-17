"""E2: ER72h(拡大窓パーセンタイル)を btc15m_L スタック(E1)の第4成分として追加する。
裁定器・報告形式は stack_size_btc15mL.py と完全同一。ホストは E1 フルスタック(階段x日足x
ICT弱0.5)をアンカーとして再現する。

第4成分(先読み禁止): 約定バー直前の確定足[fill-1]で ER72h = |close.diff(288)| /
  close.diff().abs().rolling(288).sum() （15分足288本=72時間、research.regime_gate_lab.er と同じ
  Kaufman効率比の定義をそのまま流用）。この値を「過去の全約定トレードのER値」に対する拡大窓
  パーセンタイルへ変換する（未来のトレードを含めない・最低1年の履歴が貯まるまでは m=1.0固定）。
  写像: pct<25 -> x0.7 / 25<=pct<50 -> x0.85 / pct>=50 -> x1.0 （凍結・掃引しない）。

流用（車輪の再発明禁止）: stack_size_btc15mL.{build_population, apply_size, comp1_ladder,
comp2_daily, comp3_ict, cell_report}／ict_size_transplant.{RISK_PCT, block_boot_beat}／
research.regime_gate_lab.er／arb_common.{Boot, cd}。

Run: .venv/bin/python scratchpad/stack_er_btc15mL.py [--smoke] 2>&1 | tee scratchpad/out_stack_er_btc15mL.txt
"""
import sys, io, contextlib, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import numpy as np
import pandas as pd

from arb_common import Boot, cd
from research.regime_gate_lab import er
from ict_size_transplant import RISK_PCT, block_boot_beat
from stack_size_btc15mL import (build_population, apply_size, comp1_ladder, comp2_daily,
                                comp3_ict, cell_report)

NB_MAIN = 200
N_ER = 288   # 15min x 288 = 72h


def er_percentile_component(d15, t, ii, ti):
    """拡大窓パーセンタイル(過去の約定トレードのER値のみ、未来を見ない)。1年未満はm=1.0固定。"""
    er_full = er(d15["close"], N_ER).values
    er_at_fill = np.array([er_full[b - 1] if b >= 1 and np.isfinite(er_full[b - 1]) else np.nan
                           for b in ii])
    order = np.argsort(ti.values)
    ti_sorted = ti[order]
    er_sorted = er_at_fill[order]
    n = len(er_sorted)
    pct_sorted = np.full(n, np.nan)
    year1_cut = ti_sorted[0] + pd.Timedelta(days=365.25)
    for i in range(n):
        if ti_sorted[i] < year1_cut:
            continue
        pool = er_sorted[:i]
        pool = pool[~np.isnan(pool)]
        if len(pool) < 30 or np.isnan(er_sorted[i]):
            continue
        pct_sorted[i] = 100 * np.mean(pool <= er_sorted[i])
    W4_sorted = np.where(np.isnan(pct_sorted), 1.0,
                         np.where(pct_sorted < 25, 0.7, np.where(pct_sorted < 50, 0.85, 1.0)))
    # 元のtiの順序へ戻す
    inv = np.argsort(order)
    W4 = W4_sorted[inv]
    n_scored = int((~np.isnan(pct_sorted)).sum())
    return W4, n_scored, n


def er_percentile_reverse(d15, t, ii, ti):
    er_full = er(d15["close"], N_ER).values
    er_at_fill = np.array([er_full[b - 1] if b >= 1 and np.isfinite(er_full[b - 1]) else np.nan
                           for b in ii])
    order = np.argsort(ti.values)
    ti_sorted = ti[order]
    er_sorted = er_at_fill[order]
    n = len(er_sorted)
    pct_sorted = np.full(n, np.nan)
    year1_cut = ti_sorted[0] + pd.Timedelta(days=365.25)
    for i in range(n):
        if ti_sorted[i] < year1_cut:
            continue
        pool = er_sorted[:i]
        pool = pool[~np.isnan(pool)]
        if len(pool) < 30 or np.isnan(er_sorted[i]):
            continue
        pct_sorted[i] = 100 * np.mean(pool <= er_sorted[i])
    W4r_sorted = np.where(np.isnan(pct_sorted), 1.0,
                          np.where(pct_sorted < 25, 1.3, np.where(pct_sorted < 50, 1.15, 1.0)))
    inv = np.argsort(order)
    return W4r_sorted[inv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("母集団 tie-back + E1フルスタック(アンカー)の再現")
    print("#" * 110)
    d15, t, ii = build_population()
    if args.smoke:
        m = pd.DatetimeIndex(t["time"]) >= pd.Timestamp("2024-01-01", tz=pd.DatetimeIndex(t["time"]).tz)
        t = t[m].reset_index(drop=True); ii = ii[m]
    ti = pd.DatetimeIndex(t["time"])

    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w_pdh = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
    R_base_pdh = apply_size(t, w_pdh)
    s_base_pdh = cell_report("baseline (PDH soft 0.5, 参考)", R_base_pdh, ti)

    W1, above_pdh, above_hh4 = comp1_ladder(d15, t, ii)
    W2, down_at_b = comp2_daily(d15, t, ii)
    W3, AB = comp3_ict(d15, t, ii, x=48)
    Wfull_anchor = W1 * W2 * W3
    s_anchor = cell_report("E1フルスタック(アンカー)", apply_size(t, Wfull_anchor), ti)

    months = sorted(s_anchor.index.to_period("M").unique())
    boot = Boot(months, nb=NB_MAIN, k=3, seed=20260717)
    D0 = boot.dd_median(s_base_pdh)   # stack_size_btc15mL.py と同じ基準点(PDHソフトのみの中央値DD)
                                      # に揃える --- +20.3ptの再現チェックのため、E1フル自身のDDでなく
                                      # こちらをD0にする(以降の比較も全てこのD0に統一)
    cagr_anchor, _ = boot.equal_dd_cagr(s_anchor, D0)
    cagr_base_pdh, _ = boot.equal_dd_cagr(s_base_pdh, D0)
    print(f"\n    同DD({D0:.2f}%、PDHソフトのみ=baselineの中央値maxDDに揃える。"
          f"stack_size_btc15mL.py と同じ基準点)基準:")
    print(f"      E1フルスタック CAGR = {cagr_anchor:+.1f}%")
    print(f"      (参考)PDHソフトのみ CAGR = {cagr_base_pdh:+.1f}%  "
          f"(E1フル-PDHソフト差={cagr_anchor-cagr_base_pdh:+.1f}pt "
          f"--- ict_size_transplant/stack_size_btc15mL の+20.3ptの再現チェック)")

    def eval_and_report(label, W):
        R = apply_size(t, W)
        s = cell_report(label, R, ti)
        cagr, scale = boot.equal_dd_cagr(s, D0)
        print(f"    同DD({D0:.2f}%)でのCAGR = {cagr:+.1f}%  (対E1フル差={cagr-cagr_anchor:+.1f}pt)  "
              f"スケール={scale:.3f}")
        return s, cagr

    print("\n" + "#" * 110)
    print("第4成分: ER72h 拡大窓パーセンタイル")
    print("#" * 110)
    W4, n_scored, n_total = er_percentile_component(d15, t, ii, ti)
    print(f"  スコア付与率(1年経過後・pctile算出可): {n_scored}/{n_total} ({100*n_scored/n_total:.1f}%)")
    print(f"  W4の内訳: x0.7={100*np.mean(W4==0.7):.1f}%  x0.85={100*np.mean(W4==0.85):.1f}%  "
          f"x1.0(pct>=50 or 未スコア)={100*np.mean(W4==1.0):.1f}%")

    print("\n  フル+ER:")
    s_plus, cagr_plus = eval_and_report("フル+ER", Wfull_anchor * W4)

    print("\n  ER単体をbaseline(PDHソフト)へ掛けた形(参考):")
    R_er_on_base = apply_size(t, w_pdh * W4)
    s_er_base = cell_report("ERのみ x PDHソフト(参考)", R_er_on_base, ti)
    cagr_er_base, _ = boot.equal_dd_cagr(s_er_base, D0)
    print(f"    同DD({D0:.2f}%)でのCAGR = {cagr_er_base:+.1f}%  (対PDHソフトのみ差={cagr_er_base-cagr_base_pdh:+.1f}pt)")

    print("\n" + "#" * 110)
    print("逆ダミー: ER写像反転(pct<25->x1.3 / 25-50->x1.15 / >=50->x1.0)")
    print("#" * 110)
    W4_rev = er_percentile_reverse(d15, t, ii, ti)
    s_rev, cagr_rev = eval_and_report("フル+ER逆ダミー", Wfull_anchor * W4_rev)
    print(f"\n    逆ダミー vs E1フル: {'機構整合(悪化,OK)' if cagr_rev < cagr_anchor else '⚠️機構不整合(悪化せず)'}")
    print(f"    逆ダミー vs フル+ER: {'機構整合(フル+ERが優位,OK)' if cagr_plus > cagr_rev else '⚠️機構不整合'}")

    print("\n" + "#" * 110)
    print("同数ランダムサイズnull(40回): ER成分(W4)の倍率多重集合をシャッフル、他成分は固定")
    print("#" * 110)
    null_diffs = []
    for seed in range(40):
        perm = np.random.default_rng(20260717 + seed).permutation(len(W4))
        W4_shuf = W4[perm]
        R_shuf = apply_size(t, Wfull_anchor * W4_shuf)
        s_shuf = pd.Series(R_shuf * RISK_PCT, index=ti).sort_index()
        cagr_shuf, _ = boot.equal_dd_cagr(s_shuf, D0)
        null_diffs.append(cagr_shuf - cagr_anchor)
    null_diffs = np.array(null_diffs)
    real_diff = cagr_plus - cagr_anchor
    pct = 100 * np.mean(null_diffs < real_diff)
    print(f"    実測差(フル+ER - E1フル)={real_diff:+.1f}pt  "
          f"null帯=[{np.percentile(null_diffs,2.5):+.1f},{np.percentile(null_diffs,97.5):+.1f}]pt "
          f"(中央値={np.median(null_diffs):+.1f})  -> {pct:.0f}%ile")

    print("\n" + "#" * 110)
    print("巡回ブロック・ブートストラップ 1/3/6/12mo: P(フル+ER > E1フル、同DD-CAGR基準)")
    print("#" * 110)
    for k in (1, 3, 6, 12):
        p = block_boot_beat(s_anchor, s_plus, months, k, nb=300)
        print(f"    k={k:>2}mo: P={p:.0f}%")

    print("\n" + "#" * 110)
    print("事前登録判定材料（生数字のみ）")
    print("#" * 110)
    print(f"    限界利得(フル+ER - E1フル)={real_diff:+.1f}pt  null%ile={pct:.0f}  "
          f"12moブロックP={block_boot_beat(s_anchor, s_plus, months, 12, nb=300):.0f}%")


if __name__ == "__main__":
    main()
