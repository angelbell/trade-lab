"""gold Donchian(20)ロング（無条件・ゲート無し）の再検定 — 測定の単位と目的変数を変える。

仕様カード（凍結・2026-07-29、前回 experiments/donchian20_anatomy_gold.py の続き）に基づく。
パラメータは一切動かさない。エントリー/出口の定義・コスト・データ区間は前回と完全同一で、
donchian20_sar_gold.simulate_long_gate（正確には donchian20_long_gate_gold.simulate_long_gate）
donchian20_anatomy_gold.build_b_axes / build_c_axes / quantile_table / degenerate_split /
label_shuffle_null / top3_vs_all_bootstrap 等を import してそのまま使う（自前の再実装はしない）。

ロジックの平文説明:
  (a) 節: 損益を「初期リスク幅に対する倍率R」で測り直す。
      R = pnl_pct / risk_pct、risk_pct = (約定値 - 建て時点で有効な20本安値レベル)/約定値*100
      （donchian20_anatomy_gold.build_c_axes の C4_initial_risk_pct と同一定義、そこから取る）。
      %建てで見えていた相関が R建てでも残るか＝「入口の質」、消えるか＝「サイズの変数（初期リスク幅
      が単に大きい/小さいだけ）」を軸ごとに判定する。あわせて「固定ロット(%建て)」と「初期リスク一定
      サイジング(R建て、1トレード=1R)」の年率/DDを比較する反実仮想を出す
      （これは新しいエッジではなく賭け率の写像であることを明記する）。
  (b) 節: 目的変数を「保有本数>=20（20本安値による決済より前に、理論上生存できる最短本数に
      達したか）」の二値に変える。既に確定済みの通り、保有20本未満のトレードは全て負けなので、
      「勝敗」の情報のほとんどはこの二値に集約されている。9軸それぞれについて、5分位ごとの生存率、
      分位番号との Spearman ρ、軸を単独スコアとした生存予測の ROC-AUC、ラベル並べ替え帰無1000回
      での実測AUCの分位を出す。多重比較(17検定)の参考線(Bonferroni相当)も併記する。

未解決の仕様上の曖昧さ（自己判断で埋めた箇所。すべてここに明記する）:
  1. AUCの算出は sklearn.metrics.roc_auc_score を使用（scipy/pandas-ta同様、既存環境に入っている
     標準ライブラリの利用であり「自前の再実装」ではないと判断。ロジック上の疑義があれば指摘してほしい）。
  2. AUC帰無検定は「生存ラベルを並べ替え、軸の値は固定」で1000回行う（軸値を並べ替えても数学的に
     同じ帰無分布になるが、生存ラベル側を動かす方が「観測された軸値の並びは固定、ラベル割当だけが
     偶然」という帰無仮説の直接表現として分かりやすいためこちらを採用した）。
  3. §(a).4 の「同じmaxDDにそろえて年率で比較」は、R建て累積系列を定数kでスケールし
     （k = maxDD%建て / maxDDR建て）、そのkを掛けたR建て年率を「%相当」として単純併記する
     （複利ではなく単純比例のスケーリング。トレード解像度の資産曲線を線形にスケールするだけなので、
     単純比例で正確。複利化するとこの操作自体が別の仮定を持ち込むため、線形のままにした）。
  4. §(b).3 の分位番号とAUCは同じ5分位境界(qcut)を共有するが、9軸のうち退化軸(C1等)は
     analyze_axis_v2 が2群フォールバックへ落ちるため、Spearman(分位番号,生存率)は前回同様
     「分位テーブル無し」として結果に含めない。ただしAUCは連続値のスコアなので退化軸でも
     計算できる限り計算する（0が最頻値でもAUCの定義上は問題なく動く）。
  5. §(b).4 のブートストラップは「事前登録の合格線(片側90%ile/両側95%ile)を通った軸」にのみ適用する
     （Bonferroni参考線は多重比較の警告としてのみ使い、判定そのものには使わない。仕様カード原文の
     「合格線を通った軸があれば」の合格線＝事前登録の主基準と解釈した）。
"""
SCREEN = "donchian20_sar_gold_h1"

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from donchian20_sar_gold import (
    LENGTH, START, load_tf, resample_4h, apply_cost, years_span,
    max_drawdown, summarize,
)
from donchian20_long_gate_gold import (
    simulate_long_gate, pf_pct, pf_from_array, atr14_causal,
    daily_pnl_series, cagr_dd_ratio_from_seq, block_bootstrap_diff,
)
from donchian20_anatomy_gold import (
    build_b_axes, build_c_axes, assign_and_align,
    top3_vs_all_bootstrap,
    HYPOTHESIS, DISPLAY_NAME, BASELINE, BASELINE_YEARS, N_PER_YEAR_ABS_TOL,
    CENTRAL, N_NULL_REPS, RNG_SEED, BLOCK_MONTHS,
)

SURVIVAL_BARS = 20  # bars_held >= 20 を「生存」と定義

# 多重比較: h1 9軸 + 4h 8軸 = 17検定。Bonferroni相当の参考線。
N_TOTAL_TESTS = 17
BONF_ALPHA = 0.05 / N_TOTAL_TESTS
BONF_PCTILE_TWO_SIDED = (1 - BONF_ALPHA) * 100  # 両側 |AUC-0.5| 用
BONF_PCTILE_ONE_SIDED = (1 - BONF_ALPHA) * 100  # 片側 AUC 用（同じalphaの家族内消費と解釈）


# ==================================================================
# 汎用: 5分位ビン割当（軸値だけから決まる。値配列に依らず共通のビン境界を使う）
# ==================================================================

def make_bins(axis_vals, n_bins=5, min_per_bin=5):
    axis_vals = np.asarray(axis_vals, float)
    valid = np.isfinite(axis_vals)
    n_excl = int((~valid).sum())
    v = axis_vals[valid]
    if len(v) < n_bins * min_per_bin:
        return None
    try:
        bin_idx_valid = pd.qcut(v, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    if np.all(pd.isna(bin_idx_valid)):
        return None
    k = int(np.nanmax(bin_idx_valid)) + 1
    if k < 3:
        return None
    return dict(valid=valid, bin_idx_valid=bin_idx_valid, k=k, n_excl=n_excl)


# ==================================================================
# (a) R建て 5分位テーブル（n, win%, PF(R), meanR, medianR, totR, 平均本数, 生存率%）
# ==================================================================

def quantile_table_v2(axis_vals, val_arr, bars_held, n_bins=5):
    """quantile_table(donchian20_anatomy_gold)の一般化版。val_arrはpnl_pctでもRでもよい
    （PF/mean/totはすべて符号付き合計比という同じ演算なので単位に依らず正しい）。
    死亡率(<=5本負け)の代わりに20本生存率%を出す。中央値も追加。"""
    b = make_bins(axis_vals, n_bins)
    if b is None:
        return None, int((~np.isfinite(np.asarray(axis_vals, float))).sum())
    axis_vals = np.asarray(axis_vals, float)
    val_arr = np.asarray(val_arr, float)
    bars_held = np.asarray(bars_held, float)
    valid, bin_idx_valid, k, n_excl = b["valid"], b["bin_idx_valid"], b["k"], b["n_excl"]
    p, bh = val_arr[valid], bars_held[valid]
    rows = []
    for i in range(k):
        sel = bin_idx_valid == i
        pp = p[sel]
        n = len(pp)
        if n == 0:
            rows.append(dict(bin=i + 1, n=0, win_pct=np.nan, pf=np.nan, mean=np.nan,
                              median=np.nan, tot=np.nan, avg_bars=np.nan, survive_pct=np.nan))
            continue
        rows.append(dict(
            bin=i + 1, n=n, win_pct=float((pp > 0).mean() * 100), pf=pf_from_array(pp),
            mean=float(pp.mean()), median=float(np.median(pp)), tot=float(pp.sum()),
            avg_bars=float(bh[sel].mean()), survive_pct=float((bh[sel] >= SURVIVAL_BARS).mean() * 100),
        ))
    return rows, n_excl


def degenerate_split_v2(axis_vals, val_arr, bars_held):
    axis_vals = np.asarray(axis_vals, float)
    val_arr = np.asarray(val_arr, float)
    bars_held = np.asarray(bars_held, float)
    valid = np.isfinite(axis_vals) & np.isfinite(val_arr)
    v, p, bh = axis_vals[valid], val_arr[valid], bars_held[valid]
    if len(v) == 0:
        return None
    vals, counts = np.unique(v, return_counts=True)
    mode_val = float(vals[np.argmax(counts)])
    mode_mask = v == mode_val
    other_mask = ~mode_mask

    def stats(mask):
        pp = p[mask]
        bb = bh[mask]
        n = len(pp)
        if n == 0:
            return dict(n=0, share=0.0, win_pct=np.nan, pf=np.nan, mean=np.nan,
                        median=np.nan, tot=np.nan, survive_pct=np.nan)
        return dict(n=n, share=float(n / len(v) * 100), win_pct=float((pp > 0).mean() * 100),
                    pf=pf_from_array(pp), mean=float(pp.mean()), median=float(np.median(pp)),
                    tot=float(pp.sum()), survive_pct=float((bb >= SURVIVAL_BARS).mean() * 100))

    return dict(mode_val=mode_val, n_valid=len(v),
                group_mode=stats(mode_mask), group_other=stats(other_mask))


def spearman_bins_v2(rows, key="mean"):
    valid_rows = [r for r in rows if r["n"] > 0]
    if len(valid_rows) < 3:
        return np.nan, np.nan
    xs = [r["bin"] for r in valid_rows]
    ys = [r[key] for r in valid_rows]
    rho, pval = spearmanr(xs, ys)
    return float(rho), float(pval)


def print_qtable_v2(rows, n_excl, axis_name, val_label):
    disp = DISPLAY_NAME.get(axis_name, axis_name)
    print(f"\n    -- {disp} 5分位テーブル（{val_label}建て、NaN除外n={n_excl}） --")
    print(f"    {'分位':>4}{'n':>6}{'win%':>7}{f'PF({val_label})':>10}{f'mean{val_label}':>10}"
          f"{f'median{val_label}':>12}{f'tot{val_label}':>10}{'平均本数':>9}{'生存率%':>8}")
    for r in rows:
        print(f"    {r['bin']:>4}{r['n']:>6}{r['win_pct']:>7.1f}{r['pf']:>10.3f}"
              f"{r['mean']:>10.4f}{r['median']:>12.4f}{r['tot']:>10.2f}"
              f"{r['avg_bars']:>9.1f}{r['survive_pct']:>8.1f}")


def print_degenerate_v2(ds, axis_name, val_label):
    disp = DISPLAY_NAME.get(axis_name, axis_name)
    print(f"\n    -- {disp}: 値の重複により5分位に分割不能（{val_label}建て、"
          f"値={ds['mode_val']:.4g}がn={ds['group_mode']['n']}本, "
          f"全体のn={ds['n_valid']}本中{ds['group_mode']['share']:.1f}%） --")
    print(f"    {'群':>12}{'n':>6}{'シェア%':>8}{'win%':>7}{f'PF({val_label})':>10}"
          f"{f'mean{val_label}':>10}{f'median{val_label}':>12}{'生存率%':>8}")
    gm, go = ds["group_mode"], ds["group_other"]
    print(f"    {f'={ds['mode_val']:.4g}':>12}{gm['n']:>6}{gm['share']:>8.1f}{gm['win_pct']:>7.1f}"
          f"{gm['pf']:>10.3f}{gm['mean']:>10.4f}{gm['median']:>12.4f}{gm['survive_pct']:>8.1f}")
    print(f"    {f'!={ds['mode_val']:.4g}':>12}{go['n']:>6}{go['share']:>8.1f}{go['win_pct']:>7.1f}"
          f"{go['pf']:>10.3f}{go['mean']:>10.4f}{go['median']:>12.4f}{go['survive_pct']:>8.1f}")


def analyze_axis_v2(axis_vals, val_arr, bars_held, name, val_label):
    rows, n_excl = quantile_table_v2(axis_vals, val_arr, bars_held)
    if rows is not None:
        print_qtable_v2(rows, n_excl, name, val_label)
        rho, pval = spearman_bins_v2(rows, "mean")
        print(f"      Spearman ρ(分位番号, mean{val_label}) = {rho:.3f} (p={pval:.3f})")
        return rows, n_excl
    ds = degenerate_split_v2(axis_vals, val_arr, bars_held)
    if ds is not None:
        print_degenerate_v2(ds, name, val_label)
    else:
        print(f"\n    -- {DISPLAY_NAME.get(name, name)}: サンプル不足（{val_label}建て） --")
    return None, n_excl


# ==================================================================
# (b) 生存二値検定: AUC + ラベル並べ替え帰無
# ==================================================================

def auc_permutation_test(axis_vals, survive, n_reps, rng):
    axis_vals = np.asarray(axis_vals, float)
    survive = np.asarray(survive, bool)
    valid = np.isfinite(axis_vals)
    v, y = axis_vals[valid], survive[valid].astype(int)
    if len(np.unique(y)) < 2 or len(v) < 25:
        return None
    auc_actual = float(roc_auc_score(y, v))
    null_aucs = np.empty(n_reps)
    for r in range(n_reps):
        y_perm = rng.permutation(y)
        null_aucs[r] = roc_auc_score(y_perm, v)
    one_sided_pctile = float((null_aucs < auc_actual).mean() * 100)
    two_sided_pctile = float((np.abs(null_aucs - 0.5) < abs(auc_actual - 0.5)).mean() * 100)
    return dict(auc=auc_actual, n=len(v), null_median=float(np.median(null_aucs)),
                null_std=float(null_aucs.std(ddof=1)),
                one_sided_pctile=one_sided_pctile, two_sided_pctile=two_sided_pctile)


def survival_quantile_table(axis_vals, pp, R, bars_held, survive, n_bins=5):
    b = make_bins(axis_vals, n_bins)
    if b is None:
        return None
    axis_vals = np.asarray(axis_vals, float)
    pp = np.asarray(pp, float)
    R = np.asarray(R, float)
    bars_held = np.asarray(bars_held, float)
    survive = np.asarray(survive, bool)
    valid, bin_idx_valid, k = b["valid"], b["bin_idx_valid"], b["k"]
    p_v, R_v, s_v = pp[valid], R[valid], survive[valid]
    rows = []
    for i in range(k):
        sel = bin_idx_valid == i
        n = int(sel.sum())
        if n == 0:
            rows.append(dict(bin=i + 1, n=0, survive_pct=np.nan, surv_win_pct=np.nan,
                              surv_meanR=np.nan, nonsurv_mean_pct=np.nan))
            continue
        surv_mask = s_v[sel]
        surv_pp = p_v[sel][surv_mask]
        surv_R = R_v[sel][surv_mask]
        nonsurv_pp = p_v[sel][~surv_mask]
        rows.append(dict(
            bin=i + 1, n=n, survive_pct=float(surv_mask.mean() * 100),
            surv_win_pct=float((surv_pp > 0).mean() * 100) if len(surv_pp) else np.nan,
            surv_meanR=float(np.nanmean(surv_R)) if len(surv_R) else np.nan,
            nonsurv_mean_pct=float(nonsurv_pp.mean()) if len(nonsurv_pp) else np.nan,
        ))
    return rows


def print_survival_table(rows, name):
    disp = DISPLAY_NAME.get(name, name)
    print(f"\n    -- {disp}: 5分位 × 生存率 --")
    print(f"    {'分位':>4}{'n':>6}{'生存率%':>8}{'生存者win%':>11}{'生存者meanR':>12}{'非生存者mean%':>14}")
    for r in rows:
        print(f"    {r['bin']:>4}{r['n']:>6}{r['survive_pct']:>8.1f}{r['surv_win_pct']:>11.1f}"
              f"{r['surv_meanR']:>12.4f}{r['nonsurv_mean_pct']:>14.4f}")


def spearman_survival(rows):
    valid_rows = [r for r in rows if r["n"] > 0 and np.isfinite(r["survive_pct"])]
    if len(valid_rows) < 3:
        return np.nan, np.nan
    xs = [r["bin"] for r in valid_rows]
    ys = [r["survive_pct"] for r in valid_rows]
    rho, pval = spearmanr(xs, ys)
    return float(rho), float(pval)


# ==================================================================
# メイン
# ==================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="直近2年だけで通し稼働確認")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    h1_full = load_tf("h1")
    b_axes = build_b_axes(h1_full)

    results = {}
    for tf in ("h1", "4h"):
        df = load_tf(tf)
        if args.smoke:
            df = df.loc[df.index[-1] - pd.Timedelta(days=730):]
        yrs = years_span(df)
        atr_arr = atr14_causal(df).values

        trades_raw, open_trade, diag = simulate_long_gate(df, gate_arr=None)
        central = apply_cost(trades_raw, *CENTRAL)
        cs = summarize(central, yrs)
        if cs is not None:
            cs["pf_pct"] = pf_pct(central)

        b_aligned = {}
        for name in ("B1_ADX14", "B2_ER20", "B3_daily_donch_atr", "B5_VR5"):
            b_aligned[name] = assign_and_align(b_axes[name], df.index)
        b_aligned["B4_4h_donch_atr"] = assign_and_align(b_axes["B4_4h_donch_atr"], df.index)

        c_axes_df = build_c_axes(df, central, atr_arr)

        pp = np.array([t["pnl_pct"] for t in central])
        bars = np.array([t["bars_held"] for t in central], dtype=float)
        risk_pct = c_axes_df["C4_initial_risk_pct"].values
        risk_valid = np.isfinite(risk_pct) & (risk_pct > 0)
        n_invalid_risk = int((~risk_valid).sum())
        R = np.full(len(pp), np.nan)
        R[risk_valid] = pp[risk_valid] / risk_pct[risk_valid]
        survive = bars >= SURVIVAL_BARS

        results[tf] = dict(df=df, yrs=yrs, atr_arr=atr_arr, trades_raw=trades_raw,
                            open_trade=open_trade, diag=diag, central=central, cs=cs,
                            b_aligned=b_aligned, c_axes_df=c_axes_df, pp=pp, bars=bars,
                            risk_pct=risk_pct, R=R, n_invalid_risk=n_invalid_risk, survive=survive)

    # =================== §0 基準線一致確認 ===================
    print("=" * 100)
    print("§0 基準線の一致確認（数値assert、cost=$0.3/slip=$0.1）")
    print("=" * 100)
    if not args.smoke:
        for tf in ("h1", "4h"):
            cs = results[tf]["cs"]
            base = BASELINE[tf]
            checks = [
                ("n_per_year", cs["n_per_year"], base["n_per_year"]),
                ("pf_pct", cs["pf_pct"], base["pf_pct"]),
                ("tot_pct_per_year", cs["tot_pct_per_year"], base["tot_pct_per_year"]),
                ("maxdd_pct", cs["maxdd_pct"], base["maxdd_pct"]),
            ]
            for name, got, exp in checks:
                if name == "n_per_year":
                    abs_err = abs(got - exp)
                    assert abs_err < N_PER_YEAR_ABS_TOL, \
                        f"{tf}/{name}: got={got} expected={exp} abs_err={abs_err}"
                else:
                    rel_err = abs(got - exp) / abs(exp)
                    assert rel_err < 1e-3, f"{tf}/{name}: got={got} expected={exp} rel_err={rel_err}"
            yrs = results[tf]["yrs"]
            rel_err_yrs = abs(yrs - BASELINE_YEARS) / BASELINE_YEARS
            assert rel_err_yrs < 1e-3, f"{tf}/yrs: got={yrs} expected={BASELINE_YEARS}"
            print(f"  [OK] {tf}: n/年={cs['n_per_year']:.2f} PF%={cs['pf_pct']:.3f} "
                  f"年率%={cs['tot_pct_per_year']:.2f} maxDD%={cs['maxdd_pct']:.2f} 期間={yrs:.2f}年 "
                  f"（基準線と1e-3以内で一致）")
    else:
        print("  [SKIP] --smoke モードのためデータが短縮されており基準線とは一致しない（想定通り）")

    b_names_all = ["B1_ADX14", "B2_ER20", "B3_daily_donch_atr", "B4_4h_donch_atr", "B5_VR5"]
    c_names = ["C1_breakout_jump_atr", "C2_channel_width_atr", "C3_prior_slope_atr", "C4_initial_risk_pct"]

    def axes_for_tf(tf):
        names = b_names_all if tf == "h1" else [n for n in b_names_all if n != "B4_4h_donch_atr"]
        return names, c_names

    def axis_values(tf, name):
        r = results[tf]
        if name.startswith("B"):
            return np.array([r["b_aligned"][name][t["entry_i"]] for t in r["central"]])
        return r["c_axes_df"][name].values

    # =================== §(a) R建て再測定 ===================
    print("\n" + "=" * 100)
    print("§(a) R建て再測定（R = pnl_pct / risk_pct、risk_pct=C4_initial_risk_pctと同一定義）")
    print("=" * 100)

    for tf in ("h1", "4h"):
        r = results[tf]
        print(f"\n### TF={tf}  n_invalid_risk_pct（除外）={r['n_invalid_risk']}")
        pp, R = r["pp"], r["R"]
        Rv = R[np.isfinite(R)]
        win_pct = float((pp > 0).mean() * 100)
        print("  -- §(a).1 全体: %建て vs R建て --")
        print(f"    {'':>10}{'n':>6}{'win%':>7}{'meanR/mean%':>12}{'medianR/median%':>16}"
              f"{'stdR/std%':>10}{'PF':>8}{'合計':>10}")
        print(f"    {'%建て':>10}{len(pp):>6}{win_pct:>7.1f}{pp.mean():>12.4f}{np.median(pp):>16.4f}"
              f"{pp.std(ddof=1):>10.4f}{pf_from_array(pp):>8.3f}{pp.sum():>10.2f}")
        print(f"    {'R建て':>10}{len(Rv):>6}{float((Rv>0).mean()*100):>7.1f}{Rv.mean():>12.4f}"
              f"{np.median(Rv):>16.4f}{Rv.std(ddof=1):>10.4f}{pf_from_array(Rv):>8.3f}{Rv.sum():>10.2f}")

    print("\n  -- §(a).2 9軸すべての5分位表（R建て） --")
    axis_r_rows = {}
    for tf in ("h1", "4h"):
        r = results[tf]
        names_b, names_c = axes_for_tf(tf)
        print(f"\n### TF={tf}")
        axis_r_rows[tf] = {}
        if tf == "4h":
            print("\n    -- B4_4h_donch_atr: 4hでは建て足のチャネル幅/ATR(C2)と定義上同一のため計算しない --")
        for name in names_b + names_c:
            av = axis_values(tf, name)
            rows, n_excl = analyze_axis_v2(av, r["R"], r["bars"], name, "R")
            axis_r_rows[tf][name] = (av, rows, n_excl)

    print("\n  -- §(a).3 %建て vs R建て 相関の比較 --")
    axis_pct_rows = {}
    comparison_summary = {}
    for tf in ("h1", "4h"):
        r = results[tf]
        names_b, names_c = axes_for_tf(tf)
        print(f"\n### TF={tf}")
        print(f"    {'軸':<24}{'ρ(%建て)':>10}{'ρ(R建て)':>10}{'符号変化/消失':>16}")
        axis_pct_rows[tf] = {}
        comparison_summary[tf] = {}
        for name in names_b + names_c:
            av = axis_values(tf, name)
            rows_pct, _ = quantile_table_v2(av, r["pp"], r["bars"])
            rows_R = axis_r_rows[tf][name][1]
            axis_pct_rows[tf][name] = rows_pct
            rho_pct = spearman_bins_v2(rows_pct, "mean")[0] if rows_pct is not None else np.nan
            rho_R = spearman_bins_v2(rows_R, "mean")[0] if rows_R is not None else np.nan
            if not np.isfinite(rho_pct) or not np.isfinite(rho_R):
                verdict = "分位不能につき比較不可"
            elif abs(rho_pct) >= 0.8 and abs(rho_R) < 0.8:
                verdict = "R建てで消失(単調性喪失)"
            elif abs(rho_pct) < 0.8 and abs(rho_R) >= 0.8:
                verdict = "R建てで新規出現"
            elif np.sign(rho_pct) != np.sign(rho_R) and abs(rho_pct) >= 0.3 and abs(rho_R) >= 0.3:
                verdict = "符号反転"
            else:
                verdict = "変化なし"
            comparison_summary[tf][name] = dict(rho_pct=rho_pct, rho_R=rho_R, verdict=verdict)
            disp = DISPLAY_NAME.get(name, name)
            print(f"    {disp:<24}{rho_pct:>10.3f}{rho_R:>10.3f}{verdict:>16}")

        c4 = comparison_summary[tf].get("C4_initial_risk_pct")
        if c4:
            print(f"    [C4_initial_risk_pct結論] ρ(%建て)={c4['rho_pct']:.3f} → ρ(R建て)={c4['rho_R']:.3f}: "
                  f"{c4['verdict']}。"
                  f"{'消える＝サイズの変数（リスク一定に揃えれば済む）' if 'R建てで消失' in c4['verdict'] else '残る＝入口の質の変数（サイズを揃えても効く）'}")

    print("\n  -- §(a).4 サイジング反実仮想: 固定ロット(%建て累積) vs 初期リスク一定(R建て累積、1トレード=1R) --")
    print("     ⚠️ これは新しいエッジではなく賭け率の写像である（R建て累積=暗黙に「毎回リスクを一定に揃える」")
    print("     というサイジング・ルールを仮定している）。トレード解像度のmaxDDで比較する。")
    for tf in ("h1", "4h"):
        r = results[tf]
        pp, R, yrs = r["pp"], r["R"], r["yrs"]
        cum_pct = np.cumsum(pp)
        cum_R = np.cumsum(R[np.isfinite(R)])
        maxdd_pct = max_drawdown(cum_pct)
        maxdd_R = max_drawdown(cum_R)
        ann_pct = pp.sum() / yrs
        ann_R = np.nansum(R) / yrs
        ratio_pct = ann_pct / maxdd_pct if maxdd_pct > 0 else np.nan
        ratio_R = ann_R / maxdd_R if maxdd_R > 0 else np.nan
        k = maxdd_pct / maxdd_R if maxdd_R > 0 else np.nan
        ann_R_scaled = ann_R * k if np.isfinite(k) else np.nan
        print(f"\n    ### TF={tf}")
        print(f"    {'系列':>12}{'年率(native)':>14}{'maxDD(native)':>15}{'年率/DD':>10}")
        print(f"    {'固定ロット%建て':>12}{ann_pct:>14.3f}{maxdd_pct:>15.3f}{ratio_pct:>10.3f}")
        print(f"    {'初期リスク一定R建て':>12}{ann_R:>14.3f}{maxdd_R:>15.3f}{ratio_R:>10.3f}"
              f"  (単位=R/年, R)")
        print(f"    [同DD揃え(スケールk={k:.4f})] R建てをmaxDD%建てに揃えた時の年率%相当="
              f"{ann_R_scaled:.3f}%  vs 固定ロット%建ての年率={ann_pct:.3f}%")

    # =================== §(b) 20本生存の二値検定 ===================
    print("\n" + "=" * 100)
    print(f"§(b) 目的変数を「保有本数>={SURVIVAL_BARS}本(生存)」の二値に変えた検定")
    print("=" * 100)

    for tf in ("h1", "4h"):
        r = results[tf]
        base_rate = float(r["survive"].mean() * 100)
        print(f"\n### TF={tf}  生存基本率={base_rate:.2f}%  (期待値: h1≈67.2%, 4h≈69.9%)")

        print("  -- §(b).1 分解: mean = P(生存)*E[生存時] + (1-P)*E[非生存時] --")
        pp, R, survive = r["pp"], r["R"], r["survive"]
        P = survive.mean()
        e_surv_pct = pp[survive].mean()
        e_nonsurv_pct = pp[~survive].mean()
        recon_pct = P * e_surv_pct + (1 - P) * e_nonsurv_pct
        print(f"    %建て: P={P:.4f} E[%|生存]={e_surv_pct:.4f} E[%|非生存]={e_nonsurv_pct:.4f} "
              f"→ 再構成mean%={recon_pct:.4f} (実測mean%={pp.mean():.4f})")
        Rv_surv = R[survive & np.isfinite(R)]
        Rv_nonsurv = R[(~survive) & np.isfinite(R)]
        e_surv_R = np.nanmean(Rv_surv) if len(Rv_surv) else np.nan
        e_nonsurv_R = np.nanmean(Rv_nonsurv) if len(Rv_nonsurv) else np.nan
        recon_R = P * e_surv_R + (1 - P) * e_nonsurv_R
        Rall = R[np.isfinite(R)]
        print(f"    R建て: P={P:.4f} E[R|生存]={e_surv_R:.4f} E[R|非生存]={e_nonsurv_R:.4f} "
              f"→ 再構成meanR={recon_R:.4f} (実測meanR={Rall.mean():.4f})")

    print("\n  -- §(b).2 9軸すべての5分位×生存率表 --")
    survival_rows_all = {}
    for tf in ("h1", "4h"):
        r = results[tf]
        names_b, names_c = axes_for_tf(tf)
        print(f"\n### TF={tf}")
        survival_rows_all[tf] = {}
        for name in names_b + names_c:
            av = axis_values(tf, name)
            rows = survival_quantile_table(av, r["pp"], r["R"], r["bars"], r["survive"])
            survival_rows_all[tf][name] = rows
            if rows is None:
                print(f"\n    -- {DISPLAY_NAME.get(name, name)}: 分位不能（退化）--")
            else:
                print_survival_table(rows, name)

    print("\n  -- §(b).3 各軸の検定: Spearman(分位,生存率) / AUC / 並べ替え帰無1000回 --")
    print(f"     多重比較注記: h1 9軸 + 4h {len(axes_for_tf('4h')[0]) + len(axes_for_tf('4h')[1])}軸 "
          f"= {N_TOTAL_TESTS}検定。95%ile(両側)/90%ile(片側)を1本超えるのは偶然の期待どおり"
          f"（1検定あたり有意水準5%×{N_TOTAL_TESTS}検定で家族内error率が跳ね上がる）。"
          f"Bonferroni相当の参考線 = {BONF_PCTILE_TWO_SIDED:.2f}%ile を併記する。")

    passed_axes = {}
    for tf in ("h1", "4h"):
        r = results[tf]
        names_b, names_c = axes_for_tf(tf)
        print(f"\n### TF={tf}")
        passed_axes[tf] = []
        for name in names_b + names_c:
            av = axis_values(tf, name)
            hyp = HYPOTHESIS[name]
            rows = survival_rows_all[tf][name]
            rho, pval = spearman_survival(rows) if rows is not None else (np.nan, np.nan)
            auc_res = auc_permutation_test(av, r["survive"], N_NULL_REPS, rng)
            disp = DISPLAY_NAME.get(name, name)
            if auc_res is None:
                print(f"  {disp}: AUC計算不能（サンプル不足/生存が一様）")
                continue
            kind_label = "片側(90%ile)" if hyp["kind"] == "one_sided" else "両側(95%ile,|AUC-0.5|)"
            if hyp["kind"] == "one_sided":
                pctile = auc_res["one_sided_pctile"]
                passed_primary = pctile >= 90
                passed_bonf = pctile >= BONF_PCTILE_ONE_SIDED
            else:
                pctile = auc_res["two_sided_pctile"]
                passed_primary = pctile >= 95
                passed_bonf = pctile >= BONF_PCTILE_TWO_SIDED
            print(f"  {disp} [{kind_label}]: Spearman(分位,生存率)ρ={rho:.3f} "
                  f"AUC={auc_res['auc']:.4f}(n={auc_res['n']}) 帰無中央値={auc_res['null_median']:.4f} "
                  f"std={auc_res['null_std']:.4f} 実測分位={pctile:.1f}%ile "
                  f"→ 主基準{'合格' if passed_primary else '不合格'} / "
                  f"Bonferroni参考線{'合格' if passed_bonf else '不合格'}")
            if passed_primary:
                passed_axes[tf].append(name)

    print("\n  -- §(b).4 上位3分位のみ vs 全建て の巡回ブロック・ブートストラップ --")
    any_passed = any(len(v) > 0 for v in passed_axes.values())
    if not any_passed:
        print("    通過ゼロ（§(b).3の主基準を通過した軸が1本も無い）。ブートストラップは省略する。")
    else:
        for tf in ("h1", "4h"):
            if not passed_axes[tf]:
                continue
            r = results[tf]
            print(f"\n    ### TF={tf}  合格軸: {[DISPLAY_NAME.get(n, n) for n in passed_axes[tf]]}")
            for name in passed_axes[tf]:
                av = axis_values(tf, name)
                disp = DISPLAY_NAME.get(name, name)
                print(f"\n    -- {disp} --")
                bs = top3_vs_all_bootstrap(av, r["central"], r["df"], 5, BLOCK_MONTHS, N_NULL_REPS, rng)
                if bs is None:
                    print("      分位不足またはトレード不足でスキップ")
                    continue
                prev_frac = None
                monotone_up = True
                for bm in BLOCK_MONTHS:
                    res = bs[bm]
                    if res is None:
                        print(f"      ブロック{bm:>2d}か月: 有効サンプル無し")
                        continue
                    print(f"      ブロック{bm:>2d}か月: 差>0の割合={res['pos_frac']:5.1f}% "
                          f"中央値={res['median']:7.3f} std={res['std']:7.3f} (有効{res['n_eff']}/{N_NULL_REPS})")
                    if prev_frac is not None and res["pos_frac"] < prev_frac:
                        monotone_up = False
                    prev_frac = res["pos_frac"]
                print(f"      [結論] ブロックを長くするほど割合が単調に上がる={monotone_up}"
                      f"（上がらない場合は経路当てはめの疑い）")

    print("\n" + "=" * 100)
    print("完了")
    print("=" * 100)


if __name__ == "__main__":
    main()
