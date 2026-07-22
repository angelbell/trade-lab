"""共通ハーネス: STEP2（フィルター濃縮テスト、固定0.01ロット）— filt1/filt9/filt5 で共用。

契約:
  - サイズ写像はしない。トレードの R はレッグの正典 run() が返す (exit-entry)/stop_width
    そのまま（コスト控除込み、PDH-soft 等のサイズ乗数は一切かけない = 固定ベット）。
  - btc15m_L / gold15m は research/book.py get_book_legs() と同一の関数(breakout_wave.run)・
    同一の引数（ZigZag k=2.0 [BASE既定], pullback_frac, RR, ゲートTF, fill_win=200, ext_cap）
    で構築し、PDH-soft(WL)だけを外す（固定サイズのため）。コスト（gold $0.3/oz, BTC $15/BTC）は
    get_book_legs と同じネットコスト定義のまま残す（"実現R" である以上コストは残す）。
  - 0.01ロット換算: gold 1lot=100oz (0.01lot=1oz) ・ BTC 1lot=1BTC (0.01lot=0.01BTC)
    （根拠: CLAUDE.md "commission ≈ $3/lot/side flat (gold $0.06/oz RT" ⇒ $6/lot RT ÷ 100oz
    = $0.06/oz RT ⇒ 1lot=100oz。BTCは "$15 ≈ real" が1lot=1BTC前提の price-distance cost）。
    $PnL_trade = R_realized(price-move/stop, コスト控除後) × stop幅(価格) × 換算係数。
  - maxDD は R空間の加法的ドローダウン（research/edge_harness._card_stats と同じ定義:
    cum=R.cumsum(); dd=(cummax(cum)-cum).max()）。%複利のCAGR系maxDDとは別物 — 固定ロット
    比較の文脈では加法が正しい（複利はサイズが動く前提のため使わない）。
  - 先読み禁止: X はエントリー確定足（= 実際の約定/フィルバー、tを参照）までの情報のみで計算。
    ブレイク検出足(i)そのものではなく約定足(e_bar)を「確定足」として扱う近似 — pullback-limit
    執行では検出足と約定足が別バーになるが、run()が返す trade table は約定足しか保持しない
    （i は plan() 内部にしか残らず、既存 walk() を改変せず取り出す経路が無い）。この近似は
    出力冒頭に必ず明記する。
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/home/angelbell/dev/auto-trade"
sys.path.insert(0, ROOT)

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from src.engine.presets import BASE
from src.engine.gates import gate_sma
from src.engine.size import pdh_series, bar_idx
from research.edge_harness import _card_stats

import pandas_ta as ta


LEG_CSV = {
    "btc15m_L": f"{ROOT}/data/vantage_btcusd_m15.csv",
    "gold15m": f"{ROOT}/data/vantage_xauusd_m5.csv",
}
LEG_START = {"btc15m_L": "2018-10-01", "gold15m": "2018-09-14"}
LEG_COST_FLAT = {"btc15m_L": 15.0, "gold15m": 0.3}   # $/unit round-trip (price units)
LEG_CONV = {"btc15m_L": 0.01, "gold15m": 1.0}        # 0.01 lot -> units (0.01 BTC / 1 oz)
LEG_UNIT_NAME = {"btc15m_L": "BTC", "gold15m": "oz"}


def _leg_args(leg, fill_win=200):
    if leg == "btc15m_L":
        return {**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                "pullback_frac": 0.3, "rr": 4.5, "fill_win": fill_win}
    if leg == "gold15m":
        return {**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                "pullback_frac": 0.25, "fill_win": fill_win}
    raise ValueError(leg)


def load_leg(leg: str, smoke: bool = False):
    """Returns dict: d15 (OHLC, no size mapping), vol15 (aligned tick_volume),
    t (raw trade table from run()), R_fixed (net-of-cost R, NO size overlay),
    dollar_pnl (0.01-lot $ PnL per trade), ext_arr (gold only, else None),
    args (SimpleNamespace-compatible dict actually used -- printed for the record)."""
    from types import SimpleNamespace
    import io, contextlib

    csv = LEG_CSV[leg]
    raw = load_mt5_csv(csv).loc[LEG_START[leg]:]
    if smoke:
        raw = raw.tail(60000)   # ~625 days of 15m / ~208 days of 5m -- enough bars for
                                 # daily_sma(150)/gate_kama(14) warmup + a few hundred trades

    if leg == "btc15m_L":
        d15 = resample(raw, "15min")
        vol15 = raw["volume"].reindex(d15.index)
    else:  # gold15m: source is m5, OHLC resampled to 15min, volume SUMMED to 15min
        d15 = resample(raw, "15min")
        vol15 = raw["volume"].resample("15min").sum().reindex(d15.index)

    args_dict = _leg_args(leg)
    args = SimpleNamespace(**args_dict)
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        t = run(d15, args)
    if t is None or len(t) == 0:
        raise RuntimeError(f"{leg}: no trades (smoke window too short?)")

    cost_flat = LEG_COST_FLAT[leg]
    conv = LEG_CONV[leg]
    # t["R"] from run() is RAW (exit-entry)/risk -- BASE.cost=0.0, so no cost is baked in by
    # the engine itself. Net-of-realistic-flat-cost R (same definition as get_book_legs, minus
    # the PDH-soft size multiplier WL, which we deliberately do NOT apply -- fixed bet size):
    R_fixed = t["R"].values - cost_flat / t["risk"].values
    dollar_pnl = R_fixed * t["risk"].values * conv

    ext_arr = None
    if leg == "gold15m":
        _, ext_arr_full = gate_sma(d15, args)
        ii = bar_idx(d15, t)
        ext_arr = np.asarray(ext_arr_full)[ii]

    return dict(leg=leg, d15=d15, vol15=vol15, t=t, R_fixed=R_fixed, dollar_pnl=dollar_pnl,
                ext_arr=ext_arr, args_dict=args_dict, cost_flat=cost_flat, conv=conv,
                unit_name=LEG_UNIT_NAME[leg])


def redundancy_lines(leg_data, X):
    """Prints (returns as list of strings) the 1-line-per-pair Spearman redundancy checks."""
    lines = []
    t = leg_data["t"]; d15 = leg_data["d15"]
    a14 = ta.atr(d15["high"], d15["low"], d15["close"], length=14)
    ii = bar_idx(d15, t)
    atr_at_entry = a14.values[ii]
    stop_over_atr = t["risk"].values / atr_at_entry
    valid = np.isfinite(X) & np.isfinite(stop_over_atr)
    rho, p = spearmanr(X[valid], stop_over_atr[valid])
    lines.append(f"    X vs 損切り幅/ATR14: rho={rho:+.4f} p={p:.2e} n={valid.sum()}")

    pdh = pdh_series(d15)
    above_pdh = (t["e_px"].values > pdh[ii]).astype(float)
    valid = np.isfinite(X) & np.isfinite(above_pdh)
    rho, p = spearmanr(X[valid], above_pdh[valid])
    lines.append(f"    X vs PDHラベル(above=1): rho={rho:+.4f} p={p:.2e} n={valid.sum()}  "
                 f"(X>=中央値 のうち above_pdh 率={float(np.nanmean(above_pdh[X >= np.nanmedian(X)])):.3f}"
                 f" vs 全体 {float(np.nanmean(above_pdh)):.3f})")

    if leg_data["ext_arr"] is not None:
        ext = leg_data["ext_arr"]
        valid = np.isfinite(X) & np.isfinite(ext)
        rho, p = spearmanr(X[valid], ext[valid])
        lines.append(f"    X vs ext%（前日終値のSMA超過率）: rho={rho:+.4f} p={p:.2e} n={valid.sum()}")
    return lines


# --------------------------------------------------------------------- metrics


def r_metrics(R: np.ndarray) -> dict:
    n = len(R)
    if n == 0:
        return dict(n=0, win=np.nan, pf=np.nan, meanR=np.nan, totR=np.nan,
                    maxdd=np.nan, totr_dd=np.nan)
    pos = R[R > 0].sum()
    neg = abs(R[R <= 0].sum())
    pf = pos / neg if neg > 1e-12 else float("inf")
    cum = np.cumsum(R)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    totR = R.sum()
    totr_dd = totR / dd if dd > 1e-9 else float("inf")
    return dict(n=n, win=float((R > 0).mean() * 100), pf=float(pf), meanR=float(R.mean()),
                totR=float(totR), maxdd=dd, totr_dd=float(totr_dd))


def dollar_metrics(dollar: np.ndarray) -> dict:
    n = len(dollar)
    if n == 0:
        return dict(n=0, mean=np.nan, tot=np.nan)
    return dict(n=n, mean=float(np.mean(dollar)), tot=float(np.sum(dollar)))


def fmt_metrics_row(label: str, m: dict, dm: dict) -> str:
    return (f"    {label:<16}{m['n']:>6}{m['win']:>7.1f}%{m['pf']:>8.2f}{m['meanR']:>+9.3f}"
            f"{m['totR']:>+10.1f}{m['maxdd']:>9.2f}{m['totr_dd']:>+9.2f}"
            f"{dm['mean']:>+11.2f}{dm['tot']:>+13.1f}")


METRIC_HEADER = (f"    {'group':<16}{'n':>6}{'win%':>8}{'PF':>8}{'meanR':>9}"
                 f"{'totR':>10}{'maxDD(R)':>9}{'totR/DD':>9}{'$/trade':>11}{'$合計':>13}")


# --------------------------------------------------------------------- null1: random-drop


def random_drop_null(R_all: np.ndarray, n_keep: int, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_all = len(R_all)
    totRs = np.empty(n_boot)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n_all, size=n_keep, replace=False)
        m = r_metrics(R_all[idx])
        totRs[b] = m["totR"]
        ratios[b] = m["totr_dd"]
    return totRs, ratios


def percentile_of(value: float, dist: np.ndarray) -> float:
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0 or not np.isfinite(value):
        return float("nan")
    return float((dist < value).mean() * 100)


# --------------------------------------------------------------------- null2: circular month-block bootstrap


def _month_index(times: pd.DatetimeIndex) -> pd.PeriodIndex:
    idx = times.tz_convert(None) if getattr(times, "tz", None) is not None else times
    return idx.to_period("M")


def circular_block_bootstrap_P(times: pd.DatetimeIndex, R_all: np.ndarray, keep_mask: np.ndarray,
                                block_months: int, n_boot: int = 1000, seed: int = 0) -> float:
    """P(filtered totR/maxDD > baseline totR/maxDD) under a circular calendar-month
    block bootstrap. Both 'filtered' and 'baseline' are recomputed on the SAME resampled
    path each rep (filtered = subset of that path passing keep_mask)."""
    months_all = _month_index(times)
    uniq_months = pd.PeriodIndex(sorted(months_all.unique()))
    n_months = len(uniq_months)
    if n_months < block_months:
        return float("nan")
    month_to_pos = {m: i for i, m in enumerate(uniq_months)}
    trade_month_pos = np.array([month_to_pos[m] for m in months_all])
    # bucket trade array-indices by month position
    by_month = [np.where(trade_month_pos == p)[0] for p in range(n_months)]

    n_blocks_needed = int(np.ceil(n_months / block_months))
    rng = np.random.default_rng(seed)
    wins = 0
    valid_reps = 0
    for b in range(n_boot):
        starts = rng.integers(0, n_months, size=n_blocks_needed)
        idxs = []
        months_used = 0
        for s in starts:
            for k in range(block_months):
                pos = (s + k) % n_months
                idxs.append(by_month[pos])
                months_used += 1
                if months_used >= n_months:
                    break
            if months_used >= n_months:
                break
        if not idxs:
            continue
        path_idx = np.concatenate(idxs)
        if len(path_idx) == 0:
            continue
        R_path = R_all[path_idx]
        keep_path = keep_mask[path_idx]
        m_base = r_metrics(R_path)
        m_filt = r_metrics(R_path[keep_path])
        if not (np.isfinite(m_base["totr_dd"]) and np.isfinite(m_filt["totr_dd"])):
            continue
        valid_reps += 1
        if m_filt["totr_dd"] > m_base["totr_dd"]:
            wins += 1
    return float(wins / valid_reps * 100) if valid_reps else float("nan")


# --------------------------------------------------------------------- reporting


def quantile_thresholds(X: np.ndarray, qs=(40, 50, 60, 70, 80)):
    valid = X[np.isfinite(X)]
    return {q: float(np.percentile(valid, q)) for q in qs}


def per_year_totR(times: pd.DatetimeIndex, R: np.ndarray) -> dict:
    yrs = np.array([t.year for t in times])
    out = {}
    for y in sorted(set(yrs)):
        out[y] = float(R[yrs == y].sum())
    return out


def fmt_peryear(d: dict) -> str:
    return " ".join(f"{y}:{v:+.0f}" for y, v in d.items())
