"""event_scalp.py -- spec_event_scalp.md (仕様カード17)

FOMC発表直後の短期モメンタム・スキャルプ。カード14/24h継続とは別の賭け:
  「最初の w_c 分で方向確認 -> 確認バー終値で成行建て -> H 分後の終値で決済」
という数分ホライズンの反応スキャルプに、事象特有の短期モメンタム(自己相関)があるか、
それが往復コストを抜けるか、同時刻ランダムより強いかを測る。

流用（車輪の再発明禁止）: fomc_event_study.py から
  atr14, price_before, candidate_dates, M15_START
をそのまま import。ATRのanchor方式・価格取得(price_before=「t未満で最後に確定した足の終値」)・
同時刻非イベント平日の候補日抽出(candidate_dates)は一切書き直さない。
新規に書くのは「短期スキャルプの出口」だけ: scalp_metrics/build_scalp_table/null_scalp_table
(継続測定=retain/round_trip/maxext/maxretのロジックには一切触れない、それらは使わない)。

先読み禁止（すべて price_before(df, ts) = 「ts 未満で最後に確定した足の close」で統一）:
  P0      = price_before(df, t0)                    -- 発表直前確定足の終値
  P_entry = price_before(df, t0 + w_c分)             -- 確認バー終値。確認完了時点で既知＝先読み無し
  d       = sign(P_entry - P0)                       -- d=0 (無変化) はスキップ
  P_exit  = price_before(df, t0 + w_c分 + H分)        -- 決済バー終値（成行の想定）
  g       = d * (P_exit - P_entry)                    -- 建値からの値幅（$/oz gold, $ BTC）

データ密度ガード:
  gold m5: M15_START(2018-10-01)と同じ境界（event_kinetics.pyで実測済み: 2018-09-14まで
  日足ラベル1本/日、そこから密）。btc m5: ファイル自体が2019-01-01始まり（sparse行なし）。

実行:
  .venv/bin/python experiments/event_scalp.py --smoke   # 直近2年
  .venv/bin/python experiments/event_scalp.py           # フル
  .venv/bin/python experiments/event_scalp.py --events data/ext_cpi_dates.csv --draws 3000
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv  # noqa: E402
from fomc_event_study import atr14, price_before, candidate_dates, M15_START  # noqa: E402

GOLD_M5_START = M15_START     # "2018-10-01" -- same density boundary as m15 (verified in event_kinetics.py)
BTC_M5_START = "2019-01-01"   # btc m5 file itself starts here, no earlier sparse rows to slice away

HORIZONS_MIN = [5, 10, 15, 20, 30, 60]
PRIMARY_H = 5
W_C_MAIN = 5
W_C_ALT = 10                  # reference pass, single run, not swept
NULL_DRAWS_TARGET = 3000
BOOT_B_DEFAULT = 1500
SEED = 42

COST_ROUNDTRIP = {
    "GOLD": {"base": 0.30, "alt": [0.60]},          # $/oz round trip
    "BTC":  {"base": 15.0, "alt": [10.0, 25.0]},    # $ round trip
}
UNIT_LABEL = {"GOLD": "$/oz", "BTC": "$"}


# ----------------------------------------------------------------------------
# NEW: the short-horizon scalp outcome. Mirrors event_metrics'/en_metrics' no-lookahead
# pattern (price_before anchored strictly before each timestamp of interest) but computes
# ONLY the scalp entry/exit path -- it does not touch retain/round_trip/maxext/maxret.
# ----------------------------------------------------------------------------
def scalp_metrics(df, atr, t0, w_c, horizons):
    """One event -> dict, or None if unusable (missing P0/ATR/confirm bar, d=0, or exit
    horizon not yet covered by the data). horizons must be sorted ascending."""
    P0 = price_before(df, t0)
    if P0 is None or not np.isfinite(P0):
        return None
    pos0 = df.index.searchsorted(t0, side="left")
    if pos0 == 0:
        return None
    atr_val = atr.iloc[pos0 - 1]
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None

    t_entry = t0 + pd.Timedelta(minutes=w_c)
    P_entry = price_before(df, t_entry)
    if P_entry is None or not np.isfinite(P_entry):
        return None
    diff_c = P_entry - P0
    if diff_c == 0:
        return None  # d=0: no confirmed direction, skip per spec
    d = 1.0 if diff_c > 0 else -1.0

    max_h = max(horizons)
    if df.index.max() < t_entry + pd.Timedelta(minutes=max_h):
        return None  # exit horizon not yet reached by available data

    out = {"t0": t0, "P0": P0, "ATR": atr_val, "d": d,
           "confirm_move": abs(diff_c), "confirm_move_atr": abs(diff_c) / atr_val,
           "P_entry": P_entry}
    for h in horizons:
        t_exit = t_entry + pd.Timedelta(minutes=h)
        P_exit = price_before(df, t_exit)
        if P_exit is None or not np.isfinite(P_exit):
            out[f"g_{h}"] = np.nan
            out[f"gatr_{h}"] = np.nan
            continue
        g = d * (P_exit - P_entry)
        out[f"g_{h}"] = g
        out[f"gatr_{h}"] = g / atr_val
    return out


def build_scalp_table(df, events, w_c, horizons, label):
    atr = atr14(df)
    rows = []
    for t0 in events:
        r = scalp_metrics(df, atr, t0, w_c, horizons)
        if r is not None:
            rows.append(r)
    tbl = pd.DataFrame(rows)
    print(f"  [{label}] usable events: {len(tbl)} / {len(events)}  (w_c={w_c}min)", file=sys.stderr)
    return tbl


def null_scalp_table(df, events, w_c, horizons, label, draws_target=NULL_DRAWS_TARGET, seed=SEED):
    """Mirrors fomc_event_study.null_table's date-sampling procedure exactly (same
    candidate_dates() pool, same per-event same-clock-time draw loop) but evaluates
    scalp_metrics instead of event_metrics -- null_table is hardwired to the continuation
    metrics so this loop is the minimal re-wiring needed to point it at the scalp outcome."""
    atr = atr14(df)
    cand = candidate_dates(df, events)
    if not cand:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    per_event = max(1, int(np.ceil(draws_target / max(1, len(events)))))
    rows = []
    for e in events:
        hod, mod = e.hour, e.minute
        picks = rng.choice(len(cand), size=min(per_event * 3, len(cand)), replace=False)
        n_ok = 0
        for pi in picks:
            if n_ok >= per_event:
                break
            day = cand[pi]
            t0 = pd.Timestamp(day.date(), tz="UTC") + pd.Timedelta(hours=hod, minutes=mod)
            r = scalp_metrics(df, atr, t0, w_c, horizons)
            if r is not None:
                rows.append(r)
                n_ok += 1
    tbl = pd.DataFrame(rows)
    print(f"  [{label} NULL] draws collected: {len(tbl)} (target ~{draws_target})", file=sys.stderr)
    return tbl


# ----------------------------------------------------------------------------
# Report layer
# ----------------------------------------------------------------------------
def gross_net_table(tbl, horizons, cost_base, cost_alt, span_years):
    rows = []
    n_events = len(tbl)
    events_per_year = n_events / span_years if span_years > 0 else np.nan
    for h in horizons:
        gcol, gacol = f"g_{h}", f"gatr_{h}"
        if gcol not in tbl.columns:
            continue
        g = tbl[gcol].dropna()
        ga = tbl[gacol].dropna()
        if len(g) == 0:
            continue
        win = (g > 0).mean() * 100
        net_base = g - cost_base
        row = {
            "H_min": h, "n": len(g),
            "g_median": g.median(), "g_p25": g.quantile(0.25), "g_p75": g.quantile(0.75), "g_std": g.std(),
            "gatr_median": ga.median(), "gatr_std": ga.std(),
            "win_pct": win,
            "net_mean_base": net_base.mean(), "net_median_base": net_base.median(),
            "P_net_pos_base": (net_base > 0).mean() * 100,
            "annual_equiv_net_base": net_base.mean() * events_per_year,
        }
        for c in cost_alt:
            net_c = g - c
            row[f"net_mean_cost{c:g}"] = net_c.mean()
            row[f"P_net_pos_cost{c:g}"] = (net_c > 0).mean() * 100
        rows.append(row)
    out = pd.DataFrame(rows)
    return out, events_per_year


def follow_through_table(tbl, horizons):
    rows = []
    for h in horizons:
        gcol = f"g_{h}"
        if gcol not in tbl.columns:
            continue
        valid = tbl.dropna(subset=["confirm_move", gcol])
        if len(valid) >= 4 and valid["confirm_move"].std() > 0 and valid[gcol].std() > 0:
            rho, p = spearmanr(valid["confirm_move"], valid[gcol])
        else:
            rho, p = np.nan, np.nan
        rows.append({"H_min": h, "n": len(valid), "spearman_confirm_vs_g": rho, "p": p})
    return pd.DataFrame(rows)


def bootstrap_null_pctile(real_tbl, null_tbl, horizons, cost_base, B=BOOT_B_DEFAULT, seed=SEED):
    """For net-mean, win%, and follow-through Spearman at each H: resample n_real draws
    (with replacement) from the null pool B times, recompute the statistic each time, and
    report where the REAL value falls in that null sampling distribution (percentile).
    Mirrors fomc_event_study.bootstrap_null_median_pctile's logic (resample null pool at
    matched n, not raw-pool comparison) but applied to mean/win-rate/rho instead of median."""
    rng = np.random.default_rng(seed)
    n_real = len(real_tbl)
    rows = []
    if n_real < 3 or null_tbl.empty:
        return pd.DataFrame()
    for h in horizons:
        gcol = f"g_{h}"
        if gcol not in real_tbl.columns or gcol not in null_tbl.columns:
            continue
        real_g = real_tbl[gcol].dropna()
        if len(real_g) == 0:
            continue
        real_mean_net = (real_g - cost_base).mean()
        real_win = (real_g > 0).mean()
        rvalid = real_tbl.dropna(subset=["confirm_move", gcol])
        if len(rvalid) >= 4 and rvalid["confirm_move"].std() > 0 and rvalid[gcol].std() > 0:
            real_rho, _ = spearmanr(rvalid["confirm_move"], rvalid[gcol])
        else:
            real_rho = np.nan

        nmask = null_tbl[gcol].notna()
        null_g = null_tbl.loc[nmask, gcol].to_numpy()
        null_cm = null_tbl.loc[nmask, "confirm_move"].to_numpy()
        n_pool = len(null_g)
        if n_pool < 10:
            continue
        n_draw = min(n_real, n_pool)
        boot_mean_net = np.empty(B)
        boot_win = np.empty(B)
        boot_rho = np.full(B, np.nan)
        for b in range(B):
            idx = rng.integers(0, n_pool, size=n_draw)
            gb = null_g[idx]
            boot_mean_net[b] = (gb - cost_base).mean()
            boot_win[b] = (gb > 0).mean()
            cmb = null_cm[idx]
            if np.std(cmb) > 0 and np.std(gb) > 0:
                boot_rho[b], _ = spearmanr(cmb, gb)
        rho_valid = boot_rho[~np.isnan(boot_rho)]
        rows.append({
            "H_min": h,
            "real_net_mean": real_mean_net,
            "pctile_net_mean_in_null": (boot_mean_net <= real_mean_net).mean() * 100,
            "real_win_pct": real_win * 100,
            "pctile_win_in_null": (boot_win <= real_win).mean() * 100,
            "real_spearman": real_rho,
            "pctile_spearman_in_null": ((rho_valid <= real_rho).mean() * 100
                                         if len(rho_valid) and np.isfinite(real_rho) else np.nan),
            "n_real": n_real, "n_null_pool": n_pool,
        })
    return pd.DataFrame(rows)


def is_oos_table(tbl, horizons, cost_base):
    tbl_sorted = tbl.sort_values("t0").reset_index(drop=True)
    half = len(tbl_sorted) // 2
    is_tbl, oos_tbl = tbl_sorted.iloc[:half], tbl_sorted.iloc[half:]
    rows = []
    for seg_name, seg in [("IS", is_tbl), ("OOS", oos_tbl)]:
        for h in horizons:
            gcol = f"g_{h}"
            if gcol not in seg.columns:
                continue
            g = seg[gcol].dropna()
            if len(g) == 0:
                continue
            net = g - cost_base
            rows.append({"segment": seg_name, "H_min": h, "n": len(g),
                         "win_pct": (g > 0).mean() * 100,
                         "g_median": g.median(), "net_mean": net.mean(),
                         "P_net_pos": (net > 0).mean() * 100})
    span = (
        f"IS: {is_tbl['t0'].min()}..{is_tbl['t0'].max()} (n={len(is_tbl)})  |  "
        f"OOS: {oos_tbl['t0'].min()}..{oos_tbl['t0'].max()} (n={len(oos_tbl)})"
    )
    return pd.DataFrame(rows), span


def annual_table(tbl, h, cost_base):
    gcol = f"g_{h}"
    t = tbl.dropna(subset=[gcol]).copy()
    t["year"] = t["t0"].dt.year
    net = t[gcol] - cost_base
    t = t.assign(net=net)
    g = t.groupby("year").agg(n=("net", "size"), totNet=("net", "sum"), meanNet=("net", "mean"),
                               win_pct=(gcol, lambda s: (s > 0).mean() * 100))
    return g.reset_index()


def run_instrument_scalp(name, csv_path, start, events_all, draws_target, B):
    df = load_mt5_csv(csv_path)
    df = df.loc[start:]
    print(f"\n{'='*90}\n{name}  (m5, data from {start}, {len(df)} bars, span {df.index.min()} .. {df.index.max()})\n{'='*90}")

    cost_base = COST_ROUNDTRIP[name]["base"]
    cost_alt = COST_ROUNDTRIP[name]["alt"]
    unit = UNIT_LABEL[name]

    # -------- main pass: w_c = 5min --------
    real = build_scalp_table(df, events_all, W_C_MAIN, HORIZONS_MIN, name)
    if real.empty or len(real) < 5:
        print(f"  [{name}] too few usable events (n={len(real)}) -- skipping")
        return {}

    span_years = (real["t0"].max() - real["t0"].min()).days / 365.25
    print(f"\n--- {name} w_c={W_C_MAIN}min: usable events n={len(real)}, span {span_years:.2f}y "
          f"({len(real)/span_years:.2f} events/year) ---")

    gt, epy = gross_net_table(real, HORIZONS_MIN, cost_base, cost_alt, span_years)
    print(f"\n--- {name} w_c={W_C_MAIN}min: gross/net by H (unit={unit}; cost_base={cost_base}, cost_alt={cost_alt}) ---")
    print(gt.round(4).to_string(index=False))

    ft = follow_through_table(real, HORIZONS_MIN)
    print(f"\n--- {name} w_c={W_C_MAIN}min: follow-through Spearman(confirm_move, g_H) ---")
    print(ft.round(4).to_string(index=False))

    t_null0 = time.time()
    null = null_scalp_table(df, events_all, W_C_MAIN, HORIZONS_MIN, name, draws_target=draws_target)
    print(f"  [{name}] null pool build: {time.time()-t_null0:.1f}s", file=sys.stderr)

    if not null.empty:
        t_boot0 = time.time()
        bp = bootstrap_null_pctile(real, null, HORIZONS_MIN, cost_base, B=B)
        print(f"\n--- {name} w_c={W_C_MAIN}min: real vs bootstrapped null-sampling-dist (B={B}) ---")
        print(bp.round(3).to_string(index=False))
        print(f"  [{name}] bootstrap: {time.time()-t_boot0:.1f}s", file=sys.stderr)

    ist, span_desc = is_oos_table(real, HORIZONS_MIN, cost_base)
    print(f"\n--- {name} w_c={W_C_MAIN}min: IS/OOS split ({span_desc}) ---")
    print(ist.round(4).to_string(index=False))

    at = annual_table(real, PRIMARY_H, cost_base)
    print(f"\n--- {name} w_c={W_C_MAIN}min: annual breakdown @ H={PRIMARY_H}min (net, cost_base={cost_base}) ---")
    print(at.round(4).to_string(index=False))

    # -------- reference pass: w_c = 10min (single run, not swept) --------
    real10 = build_scalp_table(df, events_all, W_C_ALT, HORIZONS_MIN, f"{name} w_c=10 REF")
    result10 = {}
    if len(real10) >= 5:
        span_years10 = (real10["t0"].max() - real10["t0"].min()).days / 365.25
        gt10, _ = gross_net_table(real10, HORIZONS_MIN, cost_base, cost_alt, span_years10)
        print(f"\n--- {name} w_c={W_C_ALT}min (REFERENCE, single pass): gross/net by H ---")
        print(gt10.round(4).to_string(index=False))
        ft10 = follow_through_table(real10, HORIZONS_MIN)
        print(f"\n--- {name} w_c={W_C_ALT}min (REFERENCE): follow-through Spearman ---")
        print(ft10.round(4).to_string(index=False))
        null10 = null_scalp_table(df, events_all, W_C_ALT, HORIZONS_MIN, f"{name} w_c=10 REF",
                                   draws_target=draws_target)
        if not null10.empty:
            bp10 = bootstrap_null_pctile(real10, null10, HORIZONS_MIN, cost_base, B=B)
            print(f"\n--- {name} w_c={W_C_ALT}min (REFERENCE): real vs bootstrapped null-sampling-dist ---")
            print(bp10.round(3).to_string(index=False))
        result10 = {"real": real10, "null": null10}
    else:
        print(f"\n  [{name} w_c=10 REF] too few usable events (n={len(real10)}) -- skipped")

    return {"real": real, "null": null, "gross_net": gt, "follow_through": ft, "wc10": result10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/ext_fomc_dates.csv")
    ap.add_argument("--smoke", action="store_true", help="last 2 years only")
    ap.add_argument("--draws", type=int, default=NULL_DRAWS_TARGET)
    ap.add_argument("--boot", type=int, default=BOOT_B_DEFAULT)
    args = ap.parse_args()

    ev = pd.read_csv(args.events, parse_dates=["dt_utc", "dt_broker"])
    ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
    events_all = list(ev["dt_broker"].sort_values())

    if args.smoke:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=730)
        cutoff = pd.Timestamp(cutoff, tz="UTC")
        events_all = [e for e in events_all if e >= cutoff]
        print(f"[SMOKE] using {len(events_all)} events since {cutoff.date()}")

    print(f"Total events in candidate list ({args.events}): {len(events_all)}")

    run_instrument_scalp("GOLD", "data/vantage_xauusd_m5.csv", GOLD_M5_START, events_all,
                          draws_target=args.draws, B=args.boot)
    run_instrument_scalp("BTC", "data/vantage_btcusd_m5.csv", BTC_M5_START, events_all,
                          draws_target=args.draws, B=args.boot)

    print(f"\n\n{'='*90}\nDONE\n{'='*90}")


if __name__ == "__main__":
    main()
