"""案10: 相関レジームとブックのDD — STEP1 単体測定・冗長性スクリーン。

上記銘柄（案3と同じ13銘柄、btcusdは既にそのリストに含まれているため「+btc」は重複指定 —
下の実装注記に明記）の日次リターンで、30日ローリングの平均ペア相関系列を作る。
research/book.py の get_book_legs() / w_trade() をそのままimportしてブックのトレード解像度
資産曲線を再構成し（book()自体は要約タプルしか返さないため、book()内部と同一の重み付け・
結合ロジックを使ってbook()のweighted trade-R seriesを取り出す — book()の式をそのまま再利用、
新規ロジックの追加はしていない）、上位5つのDD局面（ピーク日→ボトム日→回復日）を抽出する。

各局面について: 相関系列が「過去2年の80パーセンタイル」（因果的: 当日を含まない直近504取引日
のrolling quantile）を上抜けた日付のうち、ピーク日に最も近いものを採用し、DD開始日・ボトム日
との前後関係（何日先行/遅行か。プラス=交差がDDより先＝先行）を表にする。
相関系列の全体統計（中央値・標準偏差）も出す。

実装上の近似・注記:
  - 「(+btc)」は案3の13銘柄リストに既にbtcusdが含まれているため、実質的に同一の13銘柄を使う。
  - 30日ローリング平均ペア相関は、各日について過去30日（当日含む）のリターンが揃っている銘柄
    ペアだけを使う（各ペアは相手方に自分の30日窓で最低25/30本の非欠損が無ければ除外）。
  - book()自体は資産曲線を返さないため、book()と同じ w_trade() 重み・同じ結合（共通スパンで
    絞ってconcat）をこのスクリプトでも実行しているが、式はresearch/book.pyのbook()関数の
    ソースをそのまま踏襲した（手書きの再導出ではなくコピー）。
  - DD局面は「新高値更新→次の新高値更新まで」を1局面とし、局面内の最大DD地点をボトムとする。
    最後の局面が現時点で未回復（進行中）の場合は回復日=Noneとして明記する。

Run:
  .venv/bin/python scratchpad/lab10_dd_corr.py --smoke
  .venv/bin/python scratchpad/lab10_dd_corr.py --full | tee scratchpad/out_lab10_dd_corr.txt
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from research.book import get_book_legs, w_trade, SIX

SYMBOLS = ["xauusd", "xagusd", "usousd", "nas100.r", "ger40.r", "us2000.r",
           "eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdjpy", "btcusd"]
DATA = "/home/angelbell/dev/auto-trade/data"
ROLL_WIN = 30
MIN_VALID_IN_WIN = 25
PCTL_WIN_DAYS = 504  # ~2年の取引日
PCTL_MIN_PERIODS = 250
PCTL_Q = 0.80
SEARCH_WINDOW_DAYS = 90  # ピーク日からこの日数以内でクロス日を探す


def daily_log_returns(sym: str, smoke: bool) -> pd.Series:
    path = f"{DATA}/vantage_{sym}_h1.csv"
    d = load_mt5_csv(path)
    if sym == "xauusd":
        d = d.loc[GOLD_H1_START:]
    if smoke:
        d = d.tail(5000)
    dc = d["close"].resample("1D").last().dropna()
    return np.log(dc).diff()


def rolling_avg_pairwise_corr(rets: pd.DataFrame) -> pd.Series:
    """各日について、直近ROLL_WIN日（当日含む）の窓で有効なペアだけの相関を平均する。"""
    idx = rets.index
    out = pd.Series(index=idx, dtype=float)
    cols = rets.columns
    n = len(idx)
    for i in range(ROLL_WIN - 1, n):
        win = rets.iloc[i - ROLL_WIN + 1: i + 1]
        valid_cols = [c for c in cols if win[c].notna().sum() >= MIN_VALID_IN_WIN]
        if len(valid_cols) < 2:
            continue
        cm = win[valid_cols].corr()
        m = cm.values
        k = len(valid_cols)
        iu = np.triu_indices(k, k=1)
        vals = m[iu]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        out.iloc[i] = vals.mean()
    return out


def find_dd_episodes(eq: pd.Series):
    """新高値更新点で局面を区切り、各局面の(peak, trough, recovery, dd%)を返す。"""
    running_max = eq.cummax()
    is_new_high = eq >= running_max  # ties count as at-peak
    peak_idx = eq.index[is_new_high]
    episodes = []
    for j in range(len(peak_idx) - 1):
        t1, t2 = peak_idx[j], peak_idx[j + 1]
        seg = eq.loc[t1:t2]
        if len(seg) < 2:
            continue
        peak_val = eq.loc[t1]
        dd_seg = (peak_val - seg) / peak_val
        trough_t = dd_seg.idxmax()
        dd_pct = dd_seg.max() * 100
        if dd_pct <= 1e-9:
            continue
        episodes.append(dict(peak_time=t1, peak_val=peak_val, trough_time=trough_t,
                              trough_val=eq.loc[trough_t], dd_pct=dd_pct, recovery_time=t2))
    # 末尾: 最後の新高値以降、未回復のまま終わっている区間
    t_last_peak = peak_idx[-1]
    tail = eq.loc[t_last_peak:]
    if len(tail) > 1:
        peak_val = eq.loc[t_last_peak]
        dd_tail = (peak_val - tail) / peak_val
        if dd_tail.max() > 1e-9:
            trough_t = dd_tail.idxmax()
            episodes.append(dict(peak_time=t_last_peak, peak_val=peak_val, trough_time=trough_t,
                                  trough_val=eq.loc[trough_t], dd_pct=dd_tail.max() * 100,
                                  recovery_time=None))
    episodes.sort(key=lambda e: -e["dd_pct"])
    return episodes


def run(smoke: bool):
    print(f"\n{'='*70}\n案10 相関レジームとブックのDD\n{'='*70}")

    rets = {}
    for sym in SYMBOLS:
        try:
            rets[sym] = daily_log_returns(sym, smoke)
        except FileNotFoundError:
            print(f"    {sym}: ファイル無し（スキップ）")
    R = pd.DataFrame(rets).sort_index()
    print(f"  日次リターン行列: {R.shape[0]}日 x {R.shape[1]}銘柄  "
          f"{R.index[0].date()} -> {R.index[-1].date()}")

    corr_series = rolling_avg_pairwise_corr(R).dropna()
    print(f"  30日ローリング平均ペア相関: n={len(corr_series)}  "
          f"中央値={corr_series.median():+.4f}  標準偏差={corr_series.std():.4f}  "
          f"min={corr_series.min():+.4f}  max={corr_series.max():+.4f}")

    thresh = corr_series.rolling(PCTL_WIN_DAYS, min_periods=PCTL_MIN_PERIODS).quantile(PCTL_Q).shift(1)
    above = corr_series > thresh
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dates = corr_series.index[cross_up.fillna(False)]
    print(f"  過去2年80%ile上抜けクロス回数: {len(cross_dates)}")

    # --- ブックのトレード解像度資産曲線（book.pyのbook()と同じ重み・結合式） ---
    legs = get_book_legs()
    w = w_trade(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    parts = []
    for k in SIX:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    s_all = pd.concat(parts).sort_index()
    eq = (1 + s_all).cumprod()
    eq.index = eq.index.tz_localize(None) if getattr(eq.index, "tz", None) is not None else eq.index
    print(f"  ブック資産曲線（トレード解像度）: n_trades={len(eq)}  {eq.index[0]} -> {eq.index[-1]}")

    episodes = find_dd_episodes(eq)[:5]
    cross_dates_naive = pd.DatetimeIndex(cross_dates).tz_localize(None) if len(cross_dates) and \
        getattr(cross_dates, "tz", None) is not None else pd.DatetimeIndex(cross_dates)

    print(f"\n  上位{len(episodes)}DD局面:")
    print(f"  {'#':<3}{'peak':<20}{'trough':<20}{'recovery':<20}{'DD%':>8}")
    for i, e in enumerate(episodes, 1):
        rec = str(e['recovery_time']) if e['recovery_time'] is not None else "未回復(進行中)"
        print(f"  {i:<3}{str(e['peak_time']):<20}{str(e['trough_time']):<20}{rec:<20}{e['dd_pct']:>7.2f}%")

    print(f"\n  各局面と相関レジーム上抜けクロスの前後関係（ピーク日に最も近いクロスを採用、"
          f"探索窓=±{SEARCH_WINDOW_DAYS}日、プラス=クロスがDDより先行）:")
    print(f"  {'#':<3}{'peak':<14}{'nearest_cross':<14}{'lead_vs_peak(d)':>16}{'lead_vs_trough(d)':>18}")
    for i, e in enumerate(episodes, 1):
        peak_d = pd.Timestamp(e["peak_time"]).normalize()
        trough_d = pd.Timestamp(e["trough_time"]).normalize()
        if len(cross_dates_naive) == 0:
            print(f"  {i:<3}{str(peak_d.date()):<14}{'クロス無し':<14}{'-':>16}{'-':>18}")
            continue
        diffs = (cross_dates_naive - peak_d).days
        within = np.abs(diffs) <= SEARCH_WINDOW_DAYS
        if not within.any():
            print(f"  {i:<3}{str(peak_d.date()):<14}{'窓内に無し':<14}{'-':>16}{'-':>18}")
            continue
        j = np.argmin(np.abs(diffs[within]))
        cross_d = cross_dates_naive[within][j]
        lead_peak = (peak_d - cross_d).days
        lead_trough = (trough_d - cross_d).days
        print(f"  {i:<3}{str(peak_d.date()):<14}{str(cross_d.date()):<14}"
              f"{lead_peak:>+16d}{lead_trough:>+18d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    run(args.smoke or not args.full)


if __name__ == "__main__":
    main()
