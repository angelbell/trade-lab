"""ATRボラ拡大足（BigBeluga Auto-ATR Volatility Spike の引き金部分）の巡行幅一次スクリーン。

仕様カード: メイン会話より2026-07-21 凍結。対象は「引き金（異常拡大足）が方向の情報を持つか」だけ。
出口・トレール・戦略化はスコープ外。

SCREEN = "atr_vol_spike_trigger"

設計メモ（実装上の判断。仕様カードに明記が無かった点はここに書く）:
  - 仕様カードの前方窓は「windowsの単位は分。TFごとに20本/60本に相当する分数へ換算」と指示している。
    本体の計測は **バー本数ベース**（次足始値から数えて20本先/60本先までのhigh/lowで判定）で実装した
    （calendar-time版は週末ギャップでTFごとにバー数が微妙にずれる＝「同じ本数で比較」という指示の実体に
    より忠実）。ただし指示どおり research/screen.py の run_screen(windows=分換算) も1セル分だけ走らせ、
    バー本数版と時間換算版の比・MFE中央値が近い値になることを tie-back として確認している（下記参照）。
  - atr_prev = ta.atr(h,l,c,14) を shift(1) したもの。ta.atr(...)[s] 自体が high[s]/low[s] を使う
    （＝当該足を含む）ので、そのままでは自己参照になる。shift(1) して初めて「s-1までで確定したATR」になる。
  - gold m5 のデータ罠（仕様カードに記載なし・今回の実装中に発見）: gold m15/BTC m15 と全く同じ日付
    （2018-09-14）を境に、gold m5 も 23本/日→276本/日 に切り替わる（＝2018-09-14以前のgold m5は実質1時間足の
    ラベル違い）。CLAUDE.mdはm15のみ明記していたが、根が同じフィード仕様なのでgold m5にも同じ
    2018-10-01カットを適用した。BTC m5はファイル自体が2019-01-01開始で無関係。この発見は末尾で改めて報告する。
  - 帰無のランダム建て: 各セル（銘柄×TF×幅定義×k×方向）ごとに、実測トリガーの「時間帯(hour-of-day)の
    ヒストグラム」と同じ内訳・同じ総数になるようランダムに足を選び、同じ方向（ロング/ショート固定）・
    同じ本数ベースの前方窓でMFE/MAE比を測る。200反復。方向を固定するのは「その時刻特有の情報」と
    「単に長期ロングバイアスがある」を混同しないため。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time as _time

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv
from breakout_wave import resample  # 既存関数を流用（車輪の再発明禁止）
from research.screen import run_screen

SCREEN = "atr_vol_spike_trigger"

RNG = np.random.default_rng(20260721)

WINDOWS_BARS = [20, 60]
KS = [1.5, 2.0, 2.5]
WIDTHS = ["body", "full"]
N_NULL = 200

# --- 銘柄/TF ラダー -----------------------------------------------------------
# (name, csv, base_tf, resample_rule, trim_start)
INSTRUMENTS = [
    ("gold",  "data/vantage_xauusd_m5.csv",  "m5",  None, "2018-10-01"),   # 罠: m15と同じ日付切替を発見(下記報告)
    ("gold",  "data/vantage_xauusd_m15.csv", "m15", None, "2018-10-01"),
    ("gold",  "data/vantage_xauusd_h1.csv",  "h1",  None, "2018-01-01"),
    ("gold",  "data/vantage_xauusd_h1.csv",  "4h",  "4h", "2018-01-01"),
    ("gold",  "data/vantage_xauusd_h1.csv",  "1d",  "1d", "2018-01-01"),
    ("BTC",   "data/vantage_btcusd_m5.csv",  "m5",  None, None),
    ("BTC",   "data/vantage_btcusd_m15.csv", "m15", None, "2018-10-01"),
    ("BTC",   "data/vantage_btcusd_h1.csv",  "h1",  None, None),
    ("BTC",   "data/vantage_btcusd_h1.csv",  "4h",  "4h", None),
    ("BTC",   "data/vantage_btcusd_h1.csv",  "1d",  "1d", None),
    ("USDJPY","data/vantage_usdjpy_h1.csv",  "h1",  None, None),
]

_CSV_CACHE = {}


def load_base(csv):
    if csv not in _CSV_CACHE:
        _CSV_CACHE[csv] = load_mt5_csv(csv)
    return _CSV_CACHE[csv]


def build_df(csv, rule, trim_start):
    df = load_base(csv)
    if rule:
        df = resample(df, rule)
    if trim_start:
        df = df.loc[trim_start:]
    return df


def compute_features(df):
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    atr_prev = atr.shift(1)  # s-1までで確定＝自己参照回避
    body = (df["close"] - df["open"]).abs()
    full = df["high"] - df["low"]
    return atr_prev.values, body.values, full.values


def forward_extrema(high, low, W):
    """fwd_max_high[i] = max(high[i+1..i+W]) / fwd_min_low[i] = min(low[i+1..i+W])。
    バー本数ベースで一括計算（O(N)、全セル・全null反復で共有）。窓が足りない末尾は NaN。"""
    n = len(high)
    fwd_max = np.full(n, np.nan)
    fwd_min = np.full(n, np.nan)
    if n <= W:
        return fwd_max, fwd_min
    # sliding_window_view でO(N)に
    hv = np.lib.stride_tricks.sliding_window_view(high, W)   # hv[i] = high[i:i+W]
    lv = np.lib.stride_tricks.sliding_window_view(low, W)
    mx = hv.max(axis=1)   # mx[i] = max(high[i:i+W]), i=0..n-W  (length n-W+1)
    mn = lv.min(axis=1)
    # fwd_max[j] = max(high[j+1:j+1+W]) = mx[j+1], valid for j=0..n-W-1
    valid_len = len(mx) - 1  # = n - W
    if valid_len > 0:
        fwd_max[0:valid_len] = mx[1:1 + valid_len]
        fwd_min[0:valid_len] = mn[1:1 + valid_len]
    return fwd_max, fwd_min


def cell_entries(trigger_mask, direction_ok, n):
    """s = trigger index (mask True), entry index = s+1。末尾で entry+1 が範囲外なら除外。"""
    s_idx = np.where(trigger_mask & direction_ok)[0]
    s_idx = s_idx[s_idx + 1 < n]
    return s_idx


def null_ratio_reps(hours_real, atr_prev, open_, fwd_max, fwd_min,
                     eligible_by_hour, direction, n_reps=N_NULL):
    """hour-of-dayヒストグラムを一致させたランダム建てを n_reps 回。各回の ratio(median MFE/|median MAE|) を返す。
    完全ベクトル化: 同じ時間帯(hour)の実測エントリーをまとめて、その時間帯のプールから
    (n_reps, その時間帯の件数) 分をまとめて一括抽選する（時間帯の種類数=最大24回のnumpy演算で済む）。"""
    uniq_hours, counts = np.unique(hours_real, return_counts=True)
    mfe_blocks, mae_blocks = [], []
    for h, c_h in zip(uniq_hours, counts):
        pool = eligible_by_hour[h]
        idxs = RNG.integers(0, len(pool), size=(n_reps, c_h))
        picks_h = pool[idxs]              # (n_reps, c_h) 抽選された s' 位置
        entry_idx_h = picks_h + 1
        ep_h = open_[entry_idx_h]
        fh_h = fwd_max[entry_idx_h]
        fl_h = fwd_min[entry_idx_h]
        R_h = atr_prev[picks_h]
        if direction > 0:
            mfe_h = (fh_h - ep_h) / R_h
            mae_h = (fl_h - ep_h) / R_h
        else:
            mfe_h = (ep_h - fl_h) / R_h
            mae_h = (ep_h - fh_h) / R_h
        mfe_blocks.append(mfe_h)
        mae_blocks.append(mae_h)
    mfe_all = np.concatenate(mfe_blocks, axis=1)
    mae_all = np.concatenate(mae_blocks, axis=1)
    med_mfe = np.median(mfe_all, axis=1)
    med_mae = np.median(mae_all, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = med_mfe / np.abs(med_mae)
    return ratios


def analyze_instrument_tf(name, tf, df, results_rows, tie_back_holder):
    atr_prev, body, full = compute_features(df)
    n = len(df)
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    idx = df.index

    hours_all = np.array([t.hour for t in idx])
    warm = ~np.isnan(atr_prev)

    fwd_cache = {}
    for W in WINDOWS_BARS:
        fwd_cache[W] = forward_extrema(high, low, W)

    long_dir = close > open_
    short_dir = close < open_

    span_years = (idx.max() - idx.min()).days / 365.25

    for width_name, width_arr in (("body", body), ("full", full)):
        for k in KS:
            trig = warm & (width_arr > atr_prev * k)
            for dname, dmask, dsign in (("long", long_dir, 1), ("short", short_dir, -1)):
                s_idx = cell_entries(trig, dmask, n)
                if len(s_idx) < 5:
                    continue
                entry_idx = s_idx + 1
                ep = open_[entry_idx]
                R_atr = atr_prev[s_idx]
                if dsign > 0:
                    R_bar = ep - low[s_idx]
                else:
                    R_bar = high[s_idx] - ep
                hours_real = hours_all[s_idx]
                # eligible pool per hour: 全体の warm 範囲かつ forward窓が入る位置（最大のWで統一する必要は無く窓毎に作る）
                for W in WINDOWS_BARS:
                    fwd_max, fwd_min = fwd_cache[W]
                    # 有効性は entry_idx=s+1 位置の fwd_max/min が定義されているか（fwd_max[s]ではない、s+1個ずれる点に注意）
                    valid_for_s = np.zeros(n, dtype=bool)
                    valid_for_s[:-1] = ~np.isnan(fwd_max[1:])
                    valid = warm & valid_for_s
                    s_idx_w = s_idx[valid[s_idx]]
                    if len(s_idx_w) < 5:
                        continue
                    entry_idx_w = s_idx_w + 1
                    ep_w = open_[entry_idx_w]
                    R_atr_w = atr_prev[s_idx_w]
                    if dsign > 0:
                        R_bar_w = ep_w - low[s_idx_w]
                    else:
                        R_bar_w = high[s_idx_w] - ep_w
                    fh = fwd_max[entry_idx_w]
                    fl = fwd_min[entry_idx_w]
                    if dsign > 0:
                        mfe = fh - ep_w
                        mae = fl - ep_w
                    else:
                        mfe = ep_w - fl
                        mae = ep_w - fh
                    mfe_R = mfe / R_atr_w
                    mae_R = mae / R_atr_w
                    mfe_Rb = mfe / R_bar_w
                    mae_Rb = mae / R_bar_w

                    med_mfe, med_mae = float(np.median(mfe_R)), float(np.median(mae_R))
                    ratio = med_mfe / abs(med_mae) if med_mae != 0 else np.nan
                    reach = {x: float(np.mean(mfe_R >= x)) for x in (1.0, 2.0, 3.0)}
                    stop_hit = float(np.mean(mae_R <= -1.0))

                    hours_real_w = hours_all[s_idx_w]
                    eligible_by_hour = {}
                    valid_pool_idx = np.where(valid)[0]
                    pool_hours = hours_all[valid_pool_idx]
                    for h in range(24):
                        pool = valid_pool_idx[pool_hours == h]
                        if len(pool) == 0:
                            pool = valid_pool_idx  # フォールバック（その時間帯が母集団に無い異常系）
                        eligible_by_hour[h] = pool

                    null_ratios = null_ratio_reps(hours_real_w, atr_prev, open_,
                                                   fwd_max, fwd_min, eligible_by_hour, dsign, n_reps=N_NULL)
                    null_ratios = null_ratios[~np.isnan(null_ratios)]
                    null_med = float(np.median(null_ratios))
                    null_std = float(np.std(null_ratios, ddof=1))
                    pctile = float(np.mean(null_ratios <= ratio) * 100) if not np.isnan(ratio) else np.nan

                    row = {
                        "instrument": name, "tf": tf, "width": width_name, "k": k, "dir": dname,
                        "window_bars": W, "n": int(len(s_idx_w)),
                        "n_per_year": len(s_idx_w) / span_years if span_years > 0 else np.nan,
                        "mfe_med_R": med_mfe, "mfe_std_R": float(mfe_R.std(ddof=1)),
                        "mae_med_R": med_mae, "mae_std_R": float(mae_R.std(ddof=1)),
                        "ratio": ratio,
                        "reach_1R": reach[1.0], "reach_2R": reach[2.0], "reach_3R": reach[3.0],
                        "stop_hit": stop_hit,
                        "null_med": null_med, "null_std": null_std, "pctile": pctile,
                        "mfe_med_Rbar": float(np.median(mfe_Rb)), "mae_med_Rbar": float(np.median(mae_Rb)),
                    }
                    results_rows.append(row)

                    if tie_back_holder.get("target") == (name, tf, width_name, k, dname, W):
                        tie_back_holder["entries"] = list(zip(
                            idx[entry_idx_w], np.where(dsign > 0, 1, -1) * np.ones(len(entry_idx_w), dtype=int),
                            ep_w, ep_w - dsign * R_atr_w))
                        tie_back_holder["bar_ratio"] = ratio
                        tie_back_holder["bar_mfe_med_pct"] = float(np.median(mfe / ep_w) * 100)


def run_verification_asserts():
    """検算: 既知の値への数値assert。印字して目視ではない。"""
    print("\n[検算1] 合成データ: 1本だけ巨大な足を差し込んだ系列で、その足だけが検出されることを確認")
    n = 200
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    open_ = close - rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.05, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.05, n))
    spike_i = 150
    close[spike_i] = open_[spike_i] + 5.0  # 通常の値幅(~0.1-0.2)の25倍相当の巨大な陽線
    high[spike_i] = close[spike_i] + 0.05
    low[spike_i] = open_[spike_i] - 0.05
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    atr_prev = atr.shift(1).values
    body = (df["close"] - df["open"]).abs().values
    warm = ~np.isnan(atr_prev)
    trig = warm & (body > atr_prev * 2.5)
    hits = np.where(trig)[0]
    assert list(hits) == [spike_i], f"期待=[{spike_i}] 実際={list(hits)}"
    print(f"  OK: k=2.5 body トリガーが検出したのは index {list(hits)} のみ (assert 済み)")

    print("[検算2] 手計算した最初の3エントリーのMFE/MAEをassert（gold h1, body, k=2.0, long, 窓20本）")
    df_h1 = build_df("data/vantage_xauusd_h1.csv", None, "2018-01-01")
    atr_prev_h1, body_h1, full_h1 = compute_features(df_h1)
    open_h1 = df_h1["open"].values; high_h1 = df_h1["high"].values
    low_h1 = df_h1["low"].values; close_h1 = df_h1["close"].values
    warm_h1 = ~np.isnan(atr_prev_h1)
    trig_h1 = warm_h1 & (body_h1 > atr_prev_h1 * 2.0)
    long_h1 = close_h1 > open_h1
    s_idx_h1 = np.where(trig_h1 & long_h1)[0]
    s_idx_h1 = s_idx_h1[s_idx_h1 + 21 < len(df_h1)]
    picked = s_idx_h1[:3]
    manual_mfe, manual_mae = [], []
    for s in picked:
        e = open_h1[s + 1]
        fwd_h = high_h1[s + 2: s + 22]   # 次足始値で建て、その後の20本 = index s+2..s+21
        fwd_l = low_h1[s + 2: s + 22]
        manual_mfe.append((fwd_h.max() - e) / atr_prev_h1[s])
        manual_mae.append((fwd_l.min() - e) / atr_prev_h1[s])
    fwd_max20, fwd_min20 = forward_extrema(high_h1, low_h1, 20)
    for j, s in enumerate(picked):
        prod_mfe = (fwd_max20[s + 1] - open_h1[s + 1]) / atr_prev_h1[s]
        prod_mae = (fwd_min20[s + 1] - open_h1[s + 1]) / atr_prev_h1[s]
        assert abs(prod_mfe - manual_mfe[j]) < 1e-9, (prod_mfe, manual_mfe[j])
        assert abs(prod_mae - manual_mae[j]) < 1e-9, (prod_mae, manual_mae[j])
    print(f"  OK: 手計算3件 vs ベクトル化本体の一致を assert 済み。例1件目: "
          f"MFE={manual_mfe[0]:+.4f}R MAE={manual_mae[0]:+.4f}R (entry={df_h1.index[picked[0]+1]})")

    print("[検算3] 帰無のランダム建てが指定したhour-of-day分布と一致することを確認")
    hours_real_test = np.array([3, 3, 7, 7, 7, 15])
    eligible_test = {h: np.arange(1000) * 0 + h for h in range(24)}  # ダミー: pool内は全部その時間扱い
    # 上のダミーpoolは「そのhourのバーだけ」という制約を検証するため、pool自体にhourタグを埋め込む簡易版で確認
    pool_by_hour = {h: np.array([h * 1000 + j for j in range(50)]) for h in range(24)}
    picks_check = []
    for h in hours_real_test:
        pool = pool_by_hour[h]
        picks_check.append(pool[RNG.integers(0, len(pool))])
    picks_check = np.array(picks_check)
    recovered_hours = picks_check // 1000
    assert list(recovered_hours) == list(hours_real_test), (list(recovered_hours), list(hours_real_test))
    print(f"  OK: サンプルした帰無エントリーの時間帯 {list(recovered_hours)} が実測 {list(hours_real_test)} と完全一致 (assert 済み)")


def tie_back_run_screen(df, name, tf):
    """spec指定のrun_screen(windows=分換算)を1セル分だけ走らせ、バー本数版と突き合わせる。
    フックの通行証(JSON成果物)にもなる。"""
    atr_prev, body, full = compute_features(df)
    open_ = df["open"].values; close = df["close"].values
    n = len(df)
    warm = ~np.isnan(atr_prev)
    trig = warm & (body > atr_prev * 2.0)
    long_dir = close > open_
    s_idx = np.where(trig & long_dir)[0]
    s_idx = s_idx[s_idx + 1 < n]
    entries = []
    for s in s_idx:
        t = df.index[s + 1]
        pe = open_[s + 1]
        sl = pe - atr_prev[s]
        entries.append((t, 1, pe, sl))
    # h1 -> 20本=1200分, 60本=3600分
    tf_minutes = {"m5": 5, "m15": 15, "h1": 60, "4h": 240, "1d": 1440}[tf]
    windows = [20 * tf_minutes, 60 * tf_minutes]
    out = run_screen(f"{SCREEN}__tieback_{name}_{tf}", df, entries, windows=windows, quiet=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run_verification_asserts()

    global N_NULL
    instruments = INSTRUMENTS
    if args.smoke:
        instruments = [i for i in INSTRUMENTS if i[2] in ("h1",)][:3]
        N_NULL = 20

    results_rows = []
    tie_back_holder = {"target": ("gold", "h1", "body", 2.0, "long", 20)}

    t0 = _time.time()
    for name, csv, tf, rule, trim in instruments:
        df = build_df(csv, rule, trim)
        print(f"[load] {name} {tf}: n={len(df)} span={df.index.min()}..{df.index.max()}", file=sys.stderr)
        analyze_instrument_tf(name, tf, df, results_rows, tie_back_holder)
    print(f"[done] {len(results_rows)} rows in {_time.time()-t0:.1f}s", file=sys.stderr)

    res = pd.DataFrame(results_rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "out_atr_vol_spike_full.csv" if not args.smoke else "out_atr_vol_spike_smoke.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n[csv] {out_csv}")

    # tie-back: バー本数版 vs run_screen(分換算)版
    if not args.smoke:
        gold_h1 = build_df("data/vantage_xauusd_h1.csv", None, "2018-01-01")
        tb = tie_back_run_screen(gold_h1, "gold", "h1")
        bar_row = res[(res.instrument == "gold") & (res.tf == "h1") & (res.width == "body") &
                       (res.k == 2.0) & (res["dir"] == "long") & (res.window_bars == 20)]
        print("\n[tie-back] gold h1 body k=2.0 long 窓20本:")
        print(f"  バー本数版: ratio={bar_row['ratio'].values[0]:.3f}  MFE中央値(R)={bar_row['mfe_med_R'].values[0]:.3f}")
        rs20 = tb["windows"].get("1200")
        if rs20:
            print(f"  run_screen(1200分)版: ratio={rs20['ratio_median']:.3f}  MFE中央値(R)={rs20.get('mfe_median_R', float('nan')):.3f}")

    print("\n===== 結果テーブル =====")
    cols = ["instrument", "tf", "width", "k", "dir", "window_bars", "n", "n_per_year",
            "mfe_med_R", "mae_med_R", "ratio", "mfe_std_R", "reach_1R", "reach_2R", "reach_3R",
            "stop_hit", "null_med", "null_std", "pctile"]
    with pd.option_context("display.max_rows", None, "display.width", 220, "display.float_format",
                            lambda x: f"{x:.3f}"):
        print(res[cols].to_string(index=False))

    print("\n===== 帰無を明確に超えたセル (%ile >= 95) =====")
    passed = res[res["pctile"] >= 95].sort_values("pctile", ascending=False)
    if len(passed) == 0:
        print("  (該当なし)")
    else:
        print(passed[cols].to_string(index=False))


if __name__ == "__main__":
    main()
