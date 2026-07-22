"""fomc_event_study.py -- spec_fomc_event_study.md (仕様カード14)

FOMCイベント前後の価格挙動そのものを測る記述統計スタディ。レッグでもエントリールールでもない。
「変動率が低いと全戻し、高いと継続」仮説の検定 + 同時刻・非イベント日ランダムnullとの比較。

先読み禁止: P0 = リリース時刻 t0 直前に確定した足の終値（t0 を含む/跨ぐ足は見ない）。
方向 d と大きさ I はリリース後 1h 窓だけで確定し、以降の地平線(2h..24h)はそれを基準に retain/round_trip/
maxext/maxret を計測する。

データ品質の発見（本スクリプトで測定・CLAUDE.mdの gold h1 sparse-history trap と同型）:
  Vantage gold m15 と btc m15 は共に 2018-09-14 まで「日足ラベルのm15」（1本/日）で、そこから
  密なm15に切り替わる。2018-01-01 起点では初期のFOMCイベントが汚染される。
  ゆえに m15 は --start 既定を 2018-10-01（密化+ATR14ウォームアップ分の余裕）にしている。
  h1 も同型で 2018年1-2月は日足ラベル、3月が遷移月、4月から密。h1 ロバスト確認は 2018-04-01 起点。

実行:
  .venv/bin/python experiments/fomc_event_study.py --smoke   # 直近2年（FOMC ~16会合）
  .venv/bin/python experiments/fomc_event_study.py           # フル
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv, GOLD_H1_START

# ----------------------------------------------------------------------------
# Density-guard start dates (measured directly from the CSVs; see module docstring).
M15_START = "2018-10-01"   # gold m15 & btc m15: dense from 2018-09-14, +2wk ATR warmup
H1_START = "2018-04-01"    # gold h1 & btc h1: dense from ~April 2018 (Jan-Feb daily-labeled, Mar transitional)

HORIZONS_H = [1, 2, 4, 6, 8, 12, 18, 24]
NULL_DRAWS_TARGET = 3000
SEED = 42


def atr14(d, n=14):
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def realized_vol20(d, n=20):
    """std of close-to-close diffs over the n bars preceding (not including) the current bar."""
    return d["close"].diff().rolling(n).std()


def price_before(df, ts):
    """Close of the last bar with index strictly < ts (no lookahead). None if unavailable."""
    pos = df.index.searchsorted(ts, side="left")
    if pos == 0:
        return None
    return df["close"].iloc[pos - 1]


def window_bars(df, t0, h):
    """Bars covering the path (t0, t0+h] -- i.e. index in [t0, t0+h)."""
    mask = (df.index >= t0) & (df.index < t0 + pd.Timedelta(hours=h))
    return df.loc[mask]


def event_metrics(df, atr, rv20, t0, horizons=HORIZONS_H):
    """Compute all metrics for one instrument at one anchor time t0. Returns dict or None if unusable."""
    P0 = price_before(df, t0)
    if P0 is None or not np.isfinite(P0):
        return None
    pos0 = df.index.searchsorted(t0, side="left")
    if pos0 == 0:
        return None
    atr_val = atr.iloc[pos0 - 1]
    rv_val = rv20.iloc[pos0 - 1]
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None

    P_1h = price_before(df, t0 + pd.Timedelta(hours=1))
    if P_1h is None or not np.isfinite(P_1h):
        return None
    diff0 = P_1h - P0
    I = abs(diff0)
    if I <= 0:
        return None
    d = 1.0 if diff0 > 0 else -1.0

    out = {"t0": t0, "P0": P0, "I": I, "ATR14": atr_val, "I_over_atr": I / atr_val,
           "I_over_rv20": I / rv_val if np.isfinite(rv_val) and rv_val > 0 else np.nan, "d": d}

    max_h = max(horizons)
    full_win = window_bars(df, t0, max_h)
    if full_win.empty or len(full_win) < 4:  # need at least the 1h window to exist
        return None
    # need forward data actually reaching close to t0+max_h (else horizon is truncated by data edge)
    if full_win.index.max() < t0 + pd.Timedelta(hours=max_h) - pd.Timedelta(hours=2):
        return None

    hi = full_win["high"].to_numpy()
    lo = full_win["low"].to_numpy()
    idx = full_win.index
    s_fav = d * (hi if d > 0 else lo)
    s_bad = d * (lo if d > 0 else hi)
    run_peak = np.maximum.accumulate(s_fav)
    drawdown = run_peak - s_bad  # >=0 by construction

    for h in horizons:
        m = idx < (t0 + pd.Timedelta(hours=h))
        if not m.any():
            for k in ("retain", "round_trip", "maxext", "maxret"):
                out[f"{k}_{h}h"] = np.nan
            continue
        P_h = price_before(df, t0 + pd.Timedelta(hours=h))
        if P_h is None:
            P_h = full_win["close"].to_numpy()[m][-1]
        retain = d * (P_h - P0) / I
        round_trip = 1 - retain
        # run_peak/drawdown are raw d*price; anchor run_peak at P0 (d*P0) before normalizing by I.
        maxext = (run_peak[m].max() - d * P0) / I
        maxret = drawdown[m].max() / I
        out[f"retain_{h}h"] = retain
        out[f"round_trip_{h}h"] = round_trip
        out[f"maxext_{h}h"] = maxext
        out[f"maxret_{h}h"] = maxret

    return out


def build_events_table(df, events, label):
    atr = atr14(df)
    rv20 = realized_vol20(df)
    rows = []
    for t0 in events:
        r = event_metrics(df, atr, rv20, t0)
        if r is not None:
            rows.append(r)
    tbl = pd.DataFrame(rows)
    print(f"  [{label}] usable events: {len(tbl)} / {len(events)}", file=sys.stderr)
    return tbl


def candidate_dates(df, events, min_gap_days=2):
    """Weekdays within df's usable span, excluding +-min_gap_days around any FOMC date."""
    lo = df.index.min().normalize() + pd.Timedelta(days=20)
    hi = df.index.max().normalize() - pd.Timedelta(days=2)
    all_days = pd.bdate_range(lo, hi)
    blackout = set()
    for e in events:
        d0 = e.normalize()
        for k in range(-min_gap_days, min_gap_days + 1):
            blackout.add((d0 + pd.Timedelta(days=k)).date())
    cand = [d for d in all_days if d.date() not in blackout]
    return cand


def null_table(df, events, label, draws_target=NULL_DRAWS_TARGET, seed=SEED):
    atr = atr14(df)
    rv20 = realized_vol20(df)
    cand = candidate_dates(df, events)
    if not cand:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    per_event = max(1, int(np.ceil(draws_target / max(1, len(events)))))
    rows = []
    for e in events:
        hod = e.hour
        mod = e.minute
        picks = rng.choice(len(cand), size=min(per_event * 3, len(cand)), replace=False)
        n_ok = 0
        for pi in picks:
            if n_ok >= per_event:
                break
            day = cand[pi]
            t0 = pd.Timestamp(day.date(), tz="UTC") + pd.Timedelta(hours=hod, minutes=mod)
            r = event_metrics(df, atr, rv20, t0)
            if r is not None:
                rows.append(r)
                n_ok += 1
    tbl = pd.DataFrame(rows)
    print(f"  [{label} NULL] draws collected: {len(tbl)} (target ~{draws_target})", file=sys.stderr)
    return tbl


def summarize(tbl, cols, horizons=HORIZONS_H):
    """median/p25/p75/std/n table, rows=metric, cols=horizon."""
    stats = {}
    for h in horizons:
        for k in cols:
            col = f"{k}_{h}h"
            if col not in tbl.columns:
                continue
            s = tbl[col].dropna()
            stats[(k, h)] = {
                "median": s.median(), "p25": s.quantile(0.25), "p75": s.quantile(0.75),
                "std": s.std(), "n": len(s),
            }
    out = pd.DataFrame(stats).T
    out.index.names = ["metric", "h"]
    return out


def pct_of_real_in_null(real_tbl, null_tbl, cols, horizons=(6, 12, 24)):
    rows = []
    for k in cols:
        for h in horizons:
            col = f"{k}_{h}h"
            if col not in real_tbl.columns or col not in null_tbl.columns:
                continue
            real_med = real_tbl[col].dropna().median()
            nv = null_tbl[col].dropna().to_numpy()
            if len(nv) == 0 or not np.isfinite(real_med):
                continue
            pctile = (nv <= real_med).mean() * 100
            rows.append({"metric": k, "h": h, "real_median": real_med,
                         "null_median": np.median(nv), "real_pctile_in_null": pctile, "n_null": len(nv)})
    return pd.DataFrame(rows)


def bootstrap_null_median_pctile(real_tbl, null_tbl, cols, horizons=(1, 6, 12, 24), B=5000, seed=SEED):
    """Proper null test: resample n_real draws (with replacement) from the null pool B times,
    take the median each time -> distribution of the NULL MEDIAN under n=n_real sampling.
    Report where the real-FOMC median falls in that distribution (not just vs raw null draws,
    which is a weaker/noisier comparison)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in cols:
        for h in horizons:
            col = f"{k}_{h}h"
            if col not in real_tbl.columns or col not in null_tbl.columns:
                continue
            real_med = real_tbl[col].dropna().median()
            nv = null_tbl[col].dropna().to_numpy()
            n_real = real_tbl[col].dropna().shape[0]
            if len(nv) < 10 or n_real < 3 or not np.isfinite(real_med):
                continue
            boot_meds = np.array([np.median(rng.choice(nv, size=n_real, replace=True)) for _ in range(B)])
            pctile = (boot_meds <= real_med).mean() * 100
            rows.append({"metric": k, "h": h, "real_median": real_med,
                         "null_median_boot_mean": boot_meds.mean(),
                         "null_median_boot_p5": np.quantile(boot_meds, 0.05),
                         "null_median_boot_p95": np.quantile(boot_meds, 0.95),
                         "real_pctile_vs_null_median_dist": pctile})
    return pd.DataFrame(rows)


def vol_tercile_table(tbl, cols, h=24):
    t = tbl.dropna(subset=["I_over_atr"]).copy()
    t["tercile"] = pd.qcut(t["I_over_atr"], 3, labels=["low", "mid", "high"])
    rows = []
    for terc in ["low", "mid", "high"]:
        sub = t[t["tercile"] == terc]
        for k in cols:
            col = f"{k}_{h}h"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            rows.append({"tercile": terc, "metric": k, "median": s.median(), "p25": s.quantile(0.25),
                         "p75": s.quantile(0.75), "std": s.std(), "n": len(s)})
    return pd.DataFrame(rows), t


def time_development(tbl, tercile_series, horizons=HORIZONS_H):
    rows = []
    for terc in ["low", "high"]:
        idx = tercile_series[tercile_series == terc].index
        sub = tbl.loc[tbl.index.intersection(idx)]
        for h in horizons:
            col = f"retain_{h}h"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            rows.append({"tercile": terc, "h": h, "median_retain": s.median(), "n": len(s)})
    return pd.DataFrame(rows)


def run_instrument(name, csv_path, start, events, horizons=HORIZONS_H, draws_target=NULL_DRAWS_TARGET):
    df = load_mt5_csv(csv_path)
    df = df.loc[start:]
    print(f"\n{'='*78}\n{name}  (data from {start}, {len(df)} bars, span {df.index.min()} .. {df.index.max()})\n{'='*78}")

    real = build_events_table(df, events, name)
    if real.empty:
        print(f"  no usable events for {name}; skipping")
        return {}

    cols = ["retain", "round_trip", "maxext", "maxret"]
    print(f"\n--- {name}: real FOMC event stats (median/p25/p75/std/n) ---")
    print(summarize(real, cols, horizons).round(4).to_string())

    print(f"\n--- {name}: P(round_trip_h >= 1.0) ---")
    for h in [6, 12, 24]:
        col = f"round_trip_{h}h"
        if col in real.columns:
            s = real[col].dropna()
            p = (s >= 1.0).mean() * 100 if len(s) else np.nan
            print(f"  h={h:>2}h: P(full-round-trip)={p:.1f}%  (n={len(s)})")

    null = null_table(df, list(events), name, draws_target=draws_target)
    if not null.empty:
        print(f"\n--- {name}: NULL (same-clock-time non-FOMC weekday) stats ---")
        print(summarize(null, cols, horizons).round(4).to_string())
        print(f"\n--- {name}: real-FOMC median's percentile inside NULL pool (raw, weaker test) ---")
        pct = pct_of_real_in_null(real, null, cols, horizons=(1, 6, 12, 24))
        print(pct.round(3).to_string(index=False))

        print(f"\n--- {name}: real-FOMC median vs BOOTSTRAPPED null-median sampling distribution (n={len(real)}, B=5000) ---")
        bpct = bootstrap_null_median_pctile(real, null, ["retain", "round_trip"], horizons=(1, 6, 12, 24))
        print(bpct.round(3).to_string(index=False))

    if len(real) >= 6:
        vt, real_terc = vol_tercile_table(real, cols, h=24)
        print(f"\n--- {name}: volatility (I/ATR14) tercile table @ h=24h ---")
        print(vt.round(4).to_string(index=False))

        valid = real.dropna(subset=["I_over_atr", "retain_24h"])
        if len(valid) >= 6:
            rho, pval = spearmanr(valid["I_over_atr"], valid["retain_24h"])
            print(f"\n  Spearman(I/ATR14, retain_24h) = {rho:.4f}  p={pval:.4f}  n={len(valid)}")

        td = time_development(real, real_terc.set_index(real_terc.index)["tercile"], horizons)
        print(f"\n--- {name}: retain_h time development, low vs high I/ATR tercile ---")
        print(td.round(4).to_string(index=False))
    else:
        print(f"\n  [{name}] too few usable events (n={len(real)}) for tercile/Spearman -- skipped")

    return {"real": real, "null": null}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="last 2 years only (~16 FOMC meetings)")
    ap.add_argument("--draws", type=int, default=NULL_DRAWS_TARGET)
    args = ap.parse_args()

    ev = pd.read_csv("data/ext_fomc_dates.csv", parse_dates=["dt_utc", "dt_broker"])
    ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
    events_all = list(ev["dt_broker"].sort_values())

    if args.smoke:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=730)
        cutoff = pd.Timestamp(cutoff, tz="UTC")
        events_all = [e for e in events_all if e >= cutoff]
        print(f"[SMOKE] using {len(events_all)} FOMC events since {cutoff.date()}")

    print(f"Total FOMC events in candidate list: {len(events_all)}")

    results = {}
    results["gold_m15"] = run_instrument("GOLD m15", "data/vantage_xauusd_m15.csv", M15_START,
                                          events_all, draws_target=args.draws)
    results["btc_m15"] = run_instrument("BTC m15", "data/vantage_btcusd_m15.csv", M15_START,
                                         events_all, draws_target=args.draws)

    print(f"\n\n{'#'*78}\n# 1h ROBUSTNESS SIDE-CHECK (single pass, real events only, no null re-run)\n{'#'*78}")
    results["gold_h1"] = run_instrument("GOLD h1", "data/vantage_xauusd_h1.csv", H1_START,
                                         events_all, draws_target=min(args.draws, 1000))
    results["btc_h1"] = run_instrument("BTC h1", "data/vantage_btcusd_h1.csv", H1_START,
                                        events_all, draws_target=min(args.draws, 1000))

    print(f"\n\n{'='*78}\nDONE\n{'='*78}")


if __name__ == "__main__":
    main()
