"""gold Donchian(20)ロングのトレード母集団の解剖（死因診断）。パラメータ最適化はしない。

仕様カード（凍結・2026-07-29）に基づく。既存 experiments/donchian20_long_gate_gold.py の
simulate_long_gate / apply_cost / summarize / atr14_causal / exit_anatomy 等をそのまま import
して使う（自前の再実装はしない）。ゲートは使わない（無条件のロングのみ・20本安値でフラット）。

ロジックの平文説明:
  エントリー・出口の定義は donchian20_long_gate_gold.simulate_long_gate と完全に同一
  （20本高値の確定足ブレイクを次足以降有効な逆指値としてロング、20本安値で手仕舞いフラット）。
  本スクリプトはそのトレード列を「なぜ勝ち/なぜ負けたか」の切り口で解剖する:
    A. 無条件の解剖（保有本数分布・即死率・保有本数バケット・MAE/MFE）。
    B. 上位足のレンジ判定軸5種（ADX/ER/ドンチャン幅比/VR）で5分位に分けた強度勾配。
    C. 建て時点のシグナル強度4種（飛び幅・チャネル幅・直前傾き・初期リスク幅）で同様。
    D. Bの最強軸×即死率のクロス表、最下位2分位の損失説明力。
  判定は「フィルタとしての合否」ではなく「強度勾配としての単調性」で行う
  （分位番号とmean%のSpearman順位相関|ρ|≥0.8のみ、ラベルシャッフル帰無とブロックブートストラップに通す）。

未解決の仕様上の曖昧さ（自己判断で埋めた箇所。すべてここに明記する）:
  1. B.5 日足VR(5) のローリング窓幅W。コーディネーター回答(2026-07-29): W=60日でよい
     （research/regime_statedet.py L176 の既存先例 rolling_vr(r, W=60, q) に倣う）。感度分析は不要。
     表のヘッダには窓幅を明示し "VR(5, W=60d)" と表記する。
  2. B.4「4Hドンチャン20幅/4H ATR14」。コーディネーター回答: TF=4hでは建て足の直近レンジ
     （C2: チャネル幅(hh-ll)/ATR）と定義上完全に同一になるため、TF=4hの実行ではB.4を計算せず
     「C2と同一のためスキップ」と1行出力する。B.4はTF=h1でのみ算出する（多重比較を無駄に増やさないため）。
  3. B/Cの各軸の仮説の向き。コーディネーター回答: 一部は片側、残りは両側（下記HYPOTHESIS参照）。
     片側（正の相関が主仮説そのもの、90%ileで合格）: 日足ADX(14)・日足ER(20)・日足VR(5)・
     直前の傾き/ATR(C3)。両側（事前の向きが決まらない、|ρ|≥0.8かつ両側帰無95%ileで合格）:
     日足ドンチャン幅/ATR(B3)・4Hドンチャン幅/ATR(B4,h1のみ)・チャネル幅(hh-ll)/ATR(C2)・
     ブレイクの飛び/ATR(C1)・初期リスク幅%(C4)。
  4. 保有本数(bars_held)は simulate_long_gate の定義通り (exit_i - entry_i) をそのまま使う
     （エントリー確定バーからエグジット確定バーまでのバー差）。
  5. §0のn/年のみ絶対誤差0.05に緩和（コーディネーター指示、下記BASELINE_TOL参照）:
     カード側の丸め起因（22.2は22.155768の丸め。他の3指標=PF%・年率%・maxDD%は相対誤差1e-3のまま）。
  6. C1（ブレイクの飛び）は多くのトレードで厳密に0（水準ちょうど約定＝ギャップ無し）になり
     5分位に分割できない退化ケースが発生しうる。この場合は分位テーブルの代わりに
     「値=最頻値の群」と「それ以外の群」の2群比較（n・シェア%・win%・PF%・mean%）を出す
     （degenerate_split。C1に限らず、他の軸が退化した場合にも同じフォールバックを適用する）。
"""
SCREEN = "donchian20_sar_gold_h1"

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import spearmanr

from donchian20_sar_gold import (
    LENGTH, START, load_tf, resample_4h, apply_cost, years_span,
    max_drawdown, summarize,
)
from donchian20_long_gate_gold import (
    simulate_long_gate, pf_pct, pf_from_array, atr14_causal, exit_anatomy,
    daily_pnl_series, cagr_dd_ratio_from_seq, block_bootstrap_diff, summ_stats,
)

N_NULL_REPS = 1000
RNG_SEED = 20260729
BLOCK_MONTHS = [3, 6, 12]
CENTRAL = (0.3, 0.1)
VR_WINDOW = 60  # 仕様上の曖昧さ(1) 参照

# 既知の基準線（仕様カード §0、cost=$0.3/slip=$0.1）
BASELINE = {
    "h1": dict(n_per_year=97.6, pf_pct=1.453, tot_pct_per_year=14.86, maxdd_pct=13.12),
    "4h": dict(n_per_year=22.2, pf_pct=2.009, tot_pct_per_year=13.43, maxdd_pct=17.38),
}
BASELINE_YEARS = 7.81
# n/年のみ絶対誤差0.05に緩和（カード側の丸め起因＝22.2は22.155768の丸め、コーディネーター指示2026-07-29）。
# 他の3指標(PF%/年率%/maxDD%)は仕様通り相対誤差1e-3のまま。
N_PER_YEAR_ABS_TOL = 0.05

# B/C 軸の仮説区分（コーディネーター回答, 仕様上の曖昧さ(3) 参照）。
# "one_sided": 正の相関そのものが主仮説（トレンドが強いほど良い＝レンジで死ぬ）。合格線=90%ile。
# "two_sided": 事前の向きが決まらない。合格線=|ρ|≥0.8 かつ 両側帰無95%ile。
HYPOTHESIS = {
    "B1_ADX14":            dict(kind="one_sided", sign=+1),
    "B2_ER20":             dict(kind="one_sided", sign=+1),
    "B3_daily_donch_atr":  dict(kind="two_sided", sign=0),
    "B4_4h_donch_atr":     dict(kind="two_sided", sign=0),
    "B5_VR5":              dict(kind="one_sided", sign=+1),
    "C1_breakout_jump_atr": dict(kind="two_sided", sign=0),
    "C2_channel_width_atr": dict(kind="two_sided", sign=0),
    "C3_prior_slope_atr":  dict(kind="one_sided", sign=+1),
    "C4_initial_risk_pct": dict(kind="two_sided", sign=0),
}

# 表示名（B5はVRの窓幅を明示）
DISPLAY_NAME = {"B5_VR5": "B5_VR(5, W=60d)"}


# ==================================================================
# 因果的な日足/4H系列の構築（確定値のみ・shift(1)して次バーから使用可）
# ==================================================================

def atr_raw(df):
    """atr14_causal と同じTR定義だが、shiftしない生の(その足を含む)ATR。
    B軸はここで一括してshift(1)するため、二重shiftを避けるために生の版を別途用意する。"""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def make_daily_ohlc(h1_df):
    o = h1_df["open"].resample("1D").first()
    h = h1_df["high"].resample("1D").max()
    l = h1_df["low"].resample("1D").min()
    c = h1_df["close"].resample("1D").last()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()


def assign_and_align(raw_series, target_index):
    """raw_series[t] は「t足の確定値まで使った値」。shift(1)して「t+1足から使用可」にした上で
    target_indexへffillで整列する（build_daily_gates と同じ型）。"""
    assigned = raw_series.shift(1)
    aligned = assigned.reindex(target_index, method="ffill")
    return aligned.values.astype(float)


def build_b_axes(h1_df):
    daily = make_daily_ohlc(h1_df)
    dc = daily["close"]

    adx_raw = ta.adx(daily["high"], daily["low"], daily["close"], length=14)["ADX_14"]

    er_raw = (dc - dc.shift(20)).abs() / (dc - dc.shift(1)).abs().rolling(20).sum()

    d_atr = atr_raw(daily)
    d_donch_w = daily["high"].rolling(LENGTH).max() - daily["low"].rolling(LENGTH).min()
    donch_atr_raw = d_donch_w / d_atr

    r = np.log(dc / dc.shift(1))
    vr_vals = pd.Series(r.values, index=r.index).rolling(VR_WINDOW).apply(
        lambda s: _variance_ratio(s, 5), raw=True)
    vr_raw = vr_vals

    h4 = resample_4h(h1_df)
    h4_atr = atr_raw(h4)
    h4_donch_w = h4["high"].rolling(LENGTH).max() - h4["low"].rolling(LENGTH).min()
    h4_donch_atr_raw = h4_donch_w / h4_atr

    return dict(
        daily_index=daily.index,
        B1_ADX14=adx_raw, B2_ER20=er_raw, B3_daily_donch_atr=donch_atr_raw,
        B5_VR5=vr_raw, h4_index=h4.index, B4_4h_donch_atr=h4_donch_atr_raw,
    )


def _variance_ratio(r, q):
    r = np.asarray(r, float)
    if len(r) < q + 2:
        return np.nan
    v1 = np.var(r, ddof=1)
    agg = np.convolve(r, np.ones(q), "valid")
    vq = np.var(agg, ddof=1) / q
    return vq / v1 if v1 > 0 else np.nan


# ==================================================================
# C軸（建て時点のシグナル強度）
# ==================================================================

def build_c_axes(df, trades_c, atr_arr):
    hh_lvl = df["high"].rolling(LENGTH).max().shift(1).values
    ll_lvl = df["low"].rolling(LENGTH).min().shift(1).values
    close = df["close"].values

    rows = []
    for t in trades_c:
        ei = t["entry_i"]
        a = atr_arr[ei]
        jump = (t["entry_raw"] - hh_lvl[ei]) / a if a and np.isfinite(a) and a > 0 else np.nan
        width = (hh_lvl[ei] - ll_lvl[ei]) / a if a and np.isfinite(a) and a > 0 else np.nan
        if ei - 1 >= 20 and np.isfinite(atr_arr[ei - 1]) and atr_arr[ei - 1] > 0:
            slope = (close[ei - 1] - close[ei - 1 - 20]) / atr_arr[ei - 1]
        else:
            slope = np.nan
        risk_pct = (t["entry_raw"] - ll_lvl[ei]) / t["entry_raw"] * 100
        rows.append(dict(C1_breakout_jump_atr=jump, C2_channel_width_atr=width,
                          C3_prior_slope_atr=slope, C4_initial_risk_pct=risk_pct))
    return pd.DataFrame(rows)


# ==================================================================
# 汎用: 5分位テーブル・Spearman・帰無・ブロックブートストラップ
# ==================================================================

def quantile_table(axis_vals, pnl_pct, bars_held, n_bins=5):
    axis_vals = np.asarray(axis_vals, float)
    pnl_pct = np.asarray(pnl_pct, float)
    bars_held = np.asarray(bars_held, float)
    valid = np.isfinite(axis_vals)
    n_excl = int((~valid).sum())
    v, p, b = axis_vals[valid], pnl_pct[valid], bars_held[valid]
    if len(v) < n_bins * 5:
        return None, n_excl
    try:
        bin_idx = pd.qcut(v, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None, n_excl
    if np.all(pd.isna(bin_idx)):
        # 値がほぼ一定（重複だらけ）で分位境界を作れない退化ケース
        return None, n_excl
    k = int(np.nanmax(bin_idx)) + 1
    rows = []
    for i in range(k):
        sel = bin_idx == i
        pp = p[sel]
        n = len(pp)
        if n == 0:
            rows.append(dict(bin=i + 1, n=0, win_pct=np.nan, pf_pct=np.nan, mean_pct=np.nan,
                              tot_pct=np.nan, avg_bars=np.nan, death_rate=np.nan))
            continue
        win_pct = float((pp > 0).mean() * 100)
        pfv = pf_from_array(pp)
        mean_pct = float(pp.mean())
        tot_pct = float(pp.sum())
        avg_bars = float(b[sel].mean())
        death = float(((b[sel] <= 5) & (pp < 0)).mean() * 100)
        rows.append(dict(bin=i + 1, n=n, win_pct=win_pct, pf_pct=pfv, mean_pct=mean_pct,
                          tot_pct=tot_pct, avg_bars=avg_bars, death_rate=death))
    return rows, n_excl


def degenerate_split(axis_vals, pnl_pct, bars_held):
    """quantile_tableが退化(分位不可)を返した軸のフォールバック。
    最頻値（境界に張り付く値。例: ブレイクの飛び=0＝水準ちょうど約定）を基準に2群化し、
    n・シェア%・win%・PF%・mean%を両群で比較する（情報を落とさず測る）。"""
    axis_vals = np.asarray(axis_vals, float)
    pnl_pct = np.asarray(pnl_pct, float)
    valid = np.isfinite(axis_vals)
    v, p = axis_vals[valid], pnl_pct[valid]
    if len(v) == 0:
        return None
    vals, counts = np.unique(v, return_counts=True)
    mode_val = float(vals[np.argmax(counts)])
    mode_mask = v == mode_val
    other_mask = ~mode_mask

    def stats(mask):
        pp = p[mask]
        n = len(pp)
        if n == 0:
            return dict(n=0, share=0.0, win_pct=np.nan, pf_pct=np.nan, mean_pct=np.nan, tot_pct=np.nan)
        return dict(n=n, share=float(n / len(v) * 100), win_pct=float((pp > 0).mean() * 100),
                    pf_pct=pf_from_array(pp), mean_pct=float(pp.mean()), tot_pct=float(pp.sum()))

    return dict(mode_val=mode_val, n_valid=len(v),
                group_mode=stats(mode_mask), group_other=stats(other_mask))


def print_degenerate_split(ds, axis_name):
    print(f"\n    -- {axis_name}: 値の重複により5分位に分割不能"
          f"（値={ds['mode_val']:.4g}がn={ds['group_mode']['n']}本, "
          f"全体のn={ds['n_valid']}本中{ds['group_mode']['share']:.1f}%） --")
    print(f"    {'群':>12}{'n':>6}{'シェア%':>8}{'win%':>7}{'PF%':>8}{'mean%':>8}{'総%寄与':>9}")
    gm, go = ds["group_mode"], ds["group_other"]
    print(f"    {f'={ds['mode_val']:.4g}':>12}{gm['n']:>6}{gm['share']:>8.1f}{gm['win_pct']:>7.1f}"
          f"{gm['pf_pct']:>8.3f}{gm['mean_pct']:>8.4f}{gm['tot_pct']:>9.2f}")
    print(f"    {f'≠{ds['mode_val']:.4g}':>12}{go['n']:>6}{go['share']:>8.1f}{go['win_pct']:>7.1f}"
          f"{go['pf_pct']:>8.3f}{go['mean_pct']:>8.4f}{go['tot_pct']:>9.2f}")


def spearman_bins(rows):
    valid_rows = [r for r in rows if r["n"] > 0]
    if len(valid_rows) < 3:
        return np.nan, np.nan
    xs = [r["bin"] for r in valid_rows]
    ys = [r["mean_pct"] for r in valid_rows]
    rho, pval = spearmanr(xs, ys)
    return float(rho), float(pval)


def label_shuffle_null(axis_vals, pnl_pct, n_bins, n_reps, rng):
    """分位割当を固定し、pnl_pctだけラベルシャッフル。実測=上位2分位mean%-下位2分位mean%。
    片側/両側どちらの判定にも使えるよう、帰無分布そのもの(null_diffs)も返す
    （片側=percentile(null_diffs<actual)、両側=percentile(|null_diffs|<|actual|)は呼び出し側で計算）。"""
    axis_vals = np.asarray(axis_vals, float)
    pnl_pct = np.asarray(pnl_pct, float)
    valid = np.isfinite(axis_vals)
    v, p = axis_vals[valid], pnl_pct[valid]
    bin_idx = pd.qcut(v, n_bins, labels=False, duplicates="drop")
    if np.all(pd.isna(bin_idx)):
        return None
    k = int(np.nanmax(bin_idx)) + 1
    if k < n_bins:
        return None
    top_sel = bin_idx == (k - 1)
    bot_sel = bin_idx == 0
    actual = p[top_sel].mean() - p[bot_sel].mean()
    null_diffs = np.empty(n_reps)
    for r in range(n_reps):
        shuf = rng.permutation(p)
        null_diffs[r] = shuf[top_sel].mean() - shuf[bot_sel].mean()
    one_sided_pctile = float((null_diffs < actual).mean() * 100)
    two_sided_pctile = float((np.abs(null_diffs) < abs(actual)).mean() * 100)
    return dict(actual=float(actual), null_median=float(np.median(null_diffs)),
                null_std=float(null_diffs.std(ddof=1)),
                one_sided_pctile=one_sided_pctile, two_sided_pctile=two_sided_pctile)


def top3_vs_all_bootstrap(axis_vals, trades_c, df, n_bins, block_months_list, n_reps, rng):
    axis_vals = np.asarray(axis_vals, float)
    valid = np.isfinite(axis_vals)
    idxs_valid = np.where(valid)[0]
    v = axis_vals[valid]
    bin_idx_valid = pd.qcut(v, n_bins, labels=False, duplicates="drop")
    if np.all(pd.isna(bin_idx_valid)):
        return None
    k = int(np.nanmax(bin_idx_valid)) + 1
    if k < n_bins:
        return None
    top_local = bin_idx_valid >= (k - 3)  # 上位3分位
    top_global = np.zeros(len(trades_c), dtype=bool)
    top_global[idxs_valid[top_local]] = True

    date_index = pd.date_range(df.index[0].normalize(), df.index[-1].normalize(), freq="D")
    daily_all = daily_pnl_series(trades_c, date_index)
    top3_trades = [t for t, ok in zip(trades_c, top_global) if ok]
    if len(top3_trades) < 5:
        return None
    daily_top3 = daily_pnl_series(top3_trades, date_index)

    out = {}
    for bm in block_months_list:
        diffs = block_bootstrap_diff(daily_all, daily_top3, bm, n_reps, rng)
        if len(diffs) == 0:
            out[bm] = None
            continue
        out[bm] = dict(pos_frac=float((diffs > 0).mean() * 100),
                        median=float(np.median(diffs)), std=float(diffs.std(ddof=1)),
                        n_eff=len(diffs))
    return out


# ==================================================================
# 表出力ヘルパー
# ==================================================================

def analyze_axis(axis_vals, pp, bars, name):
    """quantile_tableを試み、実質的な生存分位数が3未満（退化: 値の重複が支配的）なら
    5分位テーブルの代わりにdegenerate_split(2群比較)を出す。両方とも印字し、
    退化時はrows=Noneを返す（判定手続き側は「分位テーブル無し」として一律スキップに回す）。"""
    rows, n_excl = quantile_table(axis_vals, pp, bars)
    n_live = sum(1 for r in rows if r["n"] > 0) if rows else 0
    disp = DISPLAY_NAME.get(name, name)
    if rows is None or n_live < 3:
        ds = degenerate_split(axis_vals, pp, bars)
        if ds is not None:
            print_degenerate_split(ds, disp)
        else:
            print(f"\n    -- {disp}: サンプル不足（NaN除外n={n_excl}） --")
        return None, n_excl
    print_qtable(rows, n_excl, name)
    rho, pval = spearman_bins(rows)
    print(f"      Spearman ρ(分位番号, mean%) = {rho:.3f} (p={pval:.3f})")
    return rows, n_excl


def print_qtable(rows, n_excl, axis_name):
    disp = DISPLAY_NAME.get(axis_name, axis_name)
    print(f"\n    -- {disp} 5分位テーブル（NaN除外n={n_excl}） --")
    print(f"    {'分位':>4}{'n':>6}{'win%':>7}{'PF%':>8}{'mean%':>8}{'総%寄与':>9}{'平均本数':>9}{'即死率%':>8}")
    for r in rows:
        print(f"    {r['bin']:>4}{r['n']:>6}{r['win_pct']:>7.1f}{r['pf_pct']:>8.3f}"
              f"{r['mean_pct']:>8.4f}{r['tot_pct']:>9.2f}{r['avg_bars']:>9.1f}{r['death_rate']:>8.1f}")


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

        # B軸の整列（このTFのdf.indexへ）
        b_aligned = {}
        for name in ("B1_ADX14", "B2_ER20", "B3_daily_donch_atr", "B5_VR5"):
            b_aligned[name] = assign_and_align(b_axes[name], df.index)
        b_aligned["B4_4h_donch_atr"] = assign_and_align(b_axes["B4_4h_donch_atr"], df.index)

        c_axes_df = build_c_axes(df, central, atr_arr)

        results[tf] = dict(df=df, yrs=yrs, atr_arr=atr_arr, trades_raw=trades_raw,
                            open_trade=open_trade, diag=diag, central=central, cs=cs,
                            b_aligned=b_aligned, c_axes_df=c_axes_df)

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
                    # n/年のみ絶対誤差0.05（カード側の丸め起因: 22.2は22.155768の丸めで、
                    # 相対誤差1e-3ではこの丸め幅を吸収できないため。コーディネーター指示2026-07-29）
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

    # =================== §A トレード解剖（無条件） ===================
    print("\n" + "=" * 100)
    print("§A トレード解剖（無条件）")
    print("=" * 100)

    for tf in ("h1", "4h"):
        r = results[tf]
        central = r["central"]
        n_tot = len(central)
        bars = np.array([t["bars_held"] for t in central], dtype=float)
        pp = np.array([t["pnl_pct"] for t in central])
        win_mask = pp > 0

        print(f"\n### TF={tf}  n={n_tot}  win%={float(win_mask.mean()*100):.2f}")

        print("  -- A.1 保有本数の分布（全体/勝ち/負け） --")
        for name, mask in (("全体", np.ones(n_tot, dtype=bool)), ("勝ち", win_mask), ("負け", ~win_mask)):
            sub = bars[mask]
            if len(sub) == 0:
                print(f"    {name}: n=0")
                continue
            qs = np.percentile(sub, [10, 25, 50, 75, 90])
            print(f"    {name}: n={len(sub):>4d} mean={sub.mean():6.2f} median={np.median(sub):6.1f} "
                  f"q10={qs[0]:5.1f} q25={qs[1]:5.1f} q50={qs[2]:5.1f} q75={qs[3]:5.1f} q90={qs[4]:5.1f}")

        print("\n  -- A.2 即死率テーブル（保有本数≤T かつ負け） --")
        print(f"    {'T':>4}{'即死率%':>9}{'層n':>6}{'層総%寄与':>10}{'層平均%':>9}")
        for T in (2, 3, 5, 10):
            sel = (bars <= T) & (pp < 0)
            rate = float(sel.mean() * 100)
            layer_n = int(sel.sum())
            layer_tot = float(pp[sel].sum()) if layer_n else 0.0
            layer_mean = float(pp[sel].mean()) if layer_n else float("nan")
            print(f"    {T:>4}{rate:>9.2f}{layer_n:>6d}{layer_tot:>10.2f}{layer_mean:>9.4f}")

        print("\n  -- A.3 保有本数バケット別 --")
        buckets = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
        tot_all = float(pp.sum())
        print(f"    {'bucket':>9}{'n':>6}{'win%':>7}{'PF%':>8}{'mean%':>8}{'総%寄与':>9}{'割合%':>8}")
        bucket_rows = []
        for lo, hi in buckets:
            sel = (bars >= lo) & (bars <= hi)
            n = int(sel.sum())
            label = f"{lo}-{hi if hi < 10**8 else '+'}"
            if n == 0:
                print(f"    {label:>9}{n:>6d}{'':>7}{'':>8}{'':>8}{'':>9}{'':>8}")
                continue
            sub = pp[sel]
            win_pct = float((sub > 0).mean() * 100)
            pfv = pf_from_array(sub)
            mean_pct = float(sub.mean())
            tot_pct = float(sub.sum())
            share = tot_pct / tot_all * 100 if tot_all != 0 else float("nan")
            bucket_rows.append((label, n, tot_pct, lo))
            print(f"    {label:>9}{n:>6d}{win_pct:>7.1f}{pfv:>8.3f}{mean_pct:>8.4f}{tot_pct:>9.2f}{share:>8.1f}")
        short_loss = sum(t for lbl, n, t, lo in bucket_rows if lo <= 10 and t < 0)
        long_loss = sum(t for lbl, n, t, lo in bucket_rows if lo > 10 and t < 0)
        verdict_a3 = ("短命(保有≤10本)の負けに集中" if abs(short_loss) > abs(long_loss)
                      else "長命(保有>10本)の負けが主要因")
        print(f"    [結論] 保有≤10本バケットの負け合計%={short_loss:.2f} vs 保有>10本の負け合計%={long_loss:.2f}"
              f" → {verdict_a3}")

        print("\n  -- A.4 MAE(負け)・MFE(勝ち) ATR倍数 --")
        dfa = exit_anatomy(central, r["df"], r["atr_arr"], tf)
        mae_loss = summ_stats(dfa.loc[~dfa["win"], "mae_atr"].abs())
        mfe_win = summ_stats(dfa.loc[dfa["win"], "mfe_atr"])
        print(f"    負けのMAE(ATR倍,絶対値): n={mae_loss['n']} mean={mae_loss['mean']:.3f} "
              f"median={mae_loss['median']:.3f} std={mae_loss['std']:.3f} "
              f"q1={mae_loss['q1']:.3f} q3={mae_loss['q3']:.3f}")
        print(f"    勝ちのMFE(ATR倍): n={mfe_win['n']} mean={mfe_win['mean']:.3f} "
              f"median={mfe_win['median']:.3f} std={mfe_win['std']:.3f} "
              f"q1={mfe_win['q1']:.3f} q3={mfe_win['q3']:.3f}")

    # =================== §B レンジ判定軸 ===================
    print("\n" + "=" * 100)
    print("§B レンジ判定軸（上位足の文脈変数、5分位）")
    print("=" * 100)

    b_names_all = ["B1_ADX14", "B2_ER20", "B3_daily_donch_atr", "B4_4h_donch_atr", "B5_VR5"]
    b_qtables = {}
    for tf in ("h1", "4h"):
        r = results[tf]
        central = r["central"]
        pp = np.array([t["pnl_pct"] for t in central])
        bars = np.array([t["bars_held"] for t in central], dtype=float)
        print(f"\n### TF={tf}")
        b_qtables[tf] = {}
        # B.4はTF=4hでは建て足の直近レンジ(C2)と定義上同一になるためスキップ（コーディネーター指示）
        b_names = b_names_all if tf == "h1" else [n for n in b_names_all if n != "B4_4h_donch_atr"]
        if tf == "4h":
            print("\n    -- B4_4h_donch_atr: 4hでは建て足のチャネル幅/ATR(C2)と定義上同一のため計算しない"
                  "（C2参照） --")
        for name in b_names:
            axis_vals = np.array([r["b_aligned"][name][t["entry_i"]] for t in central])
            rows, n_excl = analyze_axis(axis_vals, pp, bars, name)
            b_qtables[tf][name] = (axis_vals, rows, n_excl)

    # =================== §C signal強度軸 ===================
    print("\n" + "=" * 100)
    print("§C signal強度軸（建て時点、5分位）")
    print("=" * 100)

    c_names = ["C1_breakout_jump_atr", "C2_channel_width_atr", "C3_prior_slope_atr", "C4_initial_risk_pct"]
    c_qtables = {}
    for tf in ("h1", "4h"):
        r = results[tf]
        central = r["central"]
        pp = np.array([t["pnl_pct"] for t in central])
        bars = np.array([t["bars_held"] for t in central], dtype=float)
        cdf = r["c_axes_df"]
        print(f"\n### TF={tf}")
        c_qtables[tf] = {}
        for name in c_names:
            axis_vals = cdf[name].values
            rows, n_excl = analyze_axis(axis_vals, pp, bars, name)
            c_qtables[tf][name] = (axis_vals, rows, n_excl)

        # C4 のR建て版（全体 + 分位）
        risk_pct = cdf["C4_initial_risk_pct"].values
        valid = np.isfinite(risk_pct) & (risk_pct > 0)
        R = np.full(len(pp), np.nan)
        R[valid] = pp[valid] / risk_pct[valid]
        print(f"\n    -- C4 補助: 初期リスク幅で正規化したR建て（全体） --")
        Rv = R[np.isfinite(R)]
        print(f"    n={len(Rv)} meanR={Rv.mean():.4f} medianR={np.median(Rv):.4f} std={Rv.std(ddof=1):.4f} "
              f"PF(R)={pf_from_array(Rv):.3f}")
        rows_r, n_excl_r = quantile_table(risk_pct, R, bars, n_bins=5)
        if rows_r is not None:
            print(f"    -- C4 補助: 初期リスク幅5分位別のR建て統計（列'mean%'はmeanRの意味・'総%寄与'はtotRの意味）--")
            print_qtable(rows_r, n_excl_r, "C4_initial_risk_pct (R建て)")

    # =================== 判定手続き ===================
    print("\n" + "=" * 100)
    print("判定手続き（片側=90%ile / 両側=|ρ|≥0.8かつ両側帰無95%ile、コーディネーター指示2026-07-29）")
    print("=" * 100)

    passed_axes = {}
    for tf in ("h1", "4h"):
        print(f"\n### TF={tf}")
        passed_axes[tf] = []
        all_qtables = {**b_qtables[tf], **c_qtables[tf]}
        for name, (axis_vals, rows, n_excl) in all_qtables.items():
            disp = DISPLAY_NAME.get(name, name)
            hyp = HYPOTHESIS[name]
            kind_label = "片側(90%ile,符号+)" if hyp["kind"] == "one_sided" else "両側(95%ile,符号不問)"
            if rows is None:
                print(f"  {disp} [{kind_label}]: 分位テーブル無し(退化・上のフォールバック参照) → 不合格候補")
                continue
            rho, pval = spearman_bins(rows)
            monotone_ok = np.isfinite(rho) and abs(rho) >= 0.8
            if hyp["kind"] == "one_sided":
                sign_ok = np.isfinite(rho) and np.sign(rho) == np.sign(hyp["sign"])
                verdict = "単調でない(|ρ|<0.8)" if not monotone_ok else (
                    "符号が仮説と不一致" if not sign_ok else "候補")
            else:
                sign_ok = True  # 両側は符号不問
                verdict = "単調でない(|ρ|<0.8)" if not monotone_ok else "候補"
            print(f"  {disp} [{kind_label}]: ρ={rho:.3f} → {verdict}")
            if monotone_ok and sign_ok:
                passed_axes[tf].append(name)

        for name in passed_axes[tf]:
            disp = DISPLAY_NAME.get(name, name)
            hyp = HYPOTHESIS[name]
            axis_vals, rows, n_excl = (b_qtables[tf].get(name) or c_qtables[tf].get(name))
            central = results[tf]["central"]
            pp = np.array([t["pnl_pct"] for t in central])
            print(f"\n  -- {disp}: ラベルシャッフル帰無 (n={N_NULL_REPS}) --")
            ns = label_shuffle_null(axis_vals, pp, 5, N_NULL_REPS, rng)
            if ns is None:
                print("    分位不足でスキップ")
            else:
                if hyp["kind"] == "one_sided":
                    pctile = ns["one_sided_pctile"]
                    verdict_null = "合格(>=90%ile,片側)" if pctile >= 90 else "不合格(<90%ile,片側)"
                else:
                    pctile = ns["two_sided_pctile"]
                    verdict_null = "合格(>=95%ile,両側)" if pctile >= 95 else "不合格(<95%ile,両側)"
                print(f"    実測(上位2分位mean%-下位2分位mean%)={ns['actual']:.4f} "
                      f"帰無中央値={ns['null_median']:.4f} std={ns['null_std']:.4f} "
                      f"片側分位={ns['one_sided_pctile']:.1f}%ile 両側分位={ns['two_sided_pctile']:.1f}%ile "
                      f"→ {verdict_null}")

            print(f"  -- {disp}: 巡回ブロック・ブートストラップ（上位3分位のみ vs 全建て、年率/DD差） --")
            bs = top3_vs_all_bootstrap(axis_vals, central, results[tf]["df"], 5, BLOCK_MONTHS, N_NULL_REPS, rng)
            if bs is None:
                print("    分位不足またはトレード不足でスキップ")
            else:
                prev_frac = None
                monotone_up = True
                for bm in BLOCK_MONTHS:
                    res = bs[bm]
                    if res is None:
                        print(f"    ブロック{bm:>2d}か月: 有効サンプル無し")
                        continue
                    print(f"    ブロック{bm:>2d}か月: 差>0の割合={res['pos_frac']:5.1f}% "
                          f"中央値={res['median']:7.3f} std={res['std']:7.3f} (有効{res['n_eff']}/{N_NULL_REPS})")
                    if prev_frac is not None and res["pos_frac"] < prev_frac:
                        monotone_up = False
                    prev_frac = res["pos_frac"]
                print(f"    [結論] ブロックを長くするほど割合が単調に上がる={monotone_up}"
                      f"（上がらない場合は経路当てはめの疑い）")

    # =================== §D 交差 ===================
    print("\n" + "=" * 100)
    print("§D 交差")
    print("=" * 100)

    for tf in ("h1", "4h"):
        r = results[tf]
        central = r["central"]
        pp = np.array([t["pnl_pct"] for t in central])
        bars = np.array([t["bars_held"] for t in central], dtype=float)
        print(f"\n### TF={tf}")

        # Bの中で最も|ρ|が大きい軸を選ぶ
        best_name, best_rho = None, -1
        for name, (axis_vals, rows, n_excl) in b_qtables[tf].items():
            if rows is None:
                continue
            rho, _ = spearman_bins(rows)
            if np.isfinite(rho) and abs(rho) > best_rho:
                best_rho, best_name = abs(rho), name
        if best_name is None:
            print("  B軸が全て分位不足のためD章はスキップ")
            continue
        print(f"  最強軸(|ρ|基準) = {best_name} (|ρ|={best_rho:.3f})")

        axis_vals = b_qtables[tf][best_name][0]
        valid = np.isfinite(axis_vals)
        v = axis_vals[valid]
        bin_idx = pd.qcut(v, 5, labels=False, duplicates="drop")
        if np.all(pd.isna(bin_idx)):
            print(f"  {best_name} が退化（分位不可）のためD章クロス表はスキップ")
            continue
        k = int(np.nanmax(bin_idx)) + 1
        pp_v = pp[valid]
        bars_v = bars[valid]
        death_v = (bars_v <= 5) & (pp_v < 0)

        print(f"\n  -- {best_name} 分位 × 即死(保有≤5本かつ負け) クロス表 --")
        print(f"    {'分位':>4}{'即死n':>7}{'非即死n':>8}{'即死率%':>8}")
        for i in range(k):
            sel = bin_idx == i
            d_n = int((sel & death_v).sum())
            o_n = int((sel & ~death_v).sum())
            rate = float(d_n / (d_n + o_n) * 100) if (d_n + o_n) else float("nan")
            print(f"    {i+1:>4}{d_n:>7}{o_n:>8}{rate:>8.2f}")

        # 最下位2分位の総寄与
        bottom2 = bin_idx <= 1
        bottom2_tot = float(pp_v[bottom2].sum())
        total_loss = -float(pp_v[pp_v < 0].sum())
        bottom2_neg = -float(pp_v[bottom2 & (pp_v < 0)].sum())
        share_of_total_loss = bottom2_neg / total_loss * 100 if total_loss > 0 else float("nan")
        print(f"\n  -- 最下位2分位({best_name})の説明力 --")
        print(f"    最下位2分位トレードの総%寄与(符号込み)={bottom2_tot:.2f}")
        print(f"    全体の総損失(負けトレードの%合計の絶対値)={total_loss:.2f}")
        print(f"    最下位2分位内の負けトレードだけの合計={bottom2_neg:.2f} "
              f"→ 全体損失に占める割合={share_of_total_loss:.1f}%")

    print("\n" + "=" * 100)
    print("完了")
    print("=" * 100)


if __name__ == "__main__":
    main()
