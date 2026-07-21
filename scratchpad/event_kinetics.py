"""event_kinetics.py -- spec_event_kinetics.md (仕様カード15)

仕様カード14（fomc_event_study.py）の続き。変えるのは分類器だけ:
  14の「1h後の実現値I」（半分あと知恵）を、「入る瞬間に手に入る早い窓 E_N」
  （N=1/5/15/30分の値幅÷ATR）に置き換え、それで24h継続(retain_24h)を事前に当てられるかを測る。

流用（車輪の再発明禁止）: fomc_event_study.py から
  load_mt5_csv, atr14, price_before, event_metrics, build_events_table,
  candidate_dates, null_table, summarize, pct_of_real_in_null,
  bootstrap_null_median_pctile, vol_tercile_table
をそのまま import。retain/round_trip/maxext/maxret・同時刻ランダムnull・ブートストラップの
計算ロジックは一切書き直さない。新規に書くのは「早い窓の分類器」(en_metrics/build_en_table)
と、それをcard14の出力とmergeして相関/三分位/方向一致/飽和点を出すレポート層だけ。

データ罠（本スクリプトで実測・確認）:
  - gold m5: 2018-09-14 まで日足ラベル(1本/日)、そこから密。M15_START(2018-10-01)と同じ境界 -> 流用。
  - btc m5: ファイル自体が2019-01-01始まり（それ以前のsparse行は無い）。M15_STARTを適用しても
    no-opなので害はないが、density trapとしてではなく単純にファイルの短さとして扱う。
  - gold m1: ファイルが2025-12-01 03:12 .. 2026-06-25 21:29 の約束(200,000行=MT5輸出の行数上限とみられる)
    しか無い。**2007-の全履歴ではない**。この窓に入るFOMCイベントは5件のみ(2025-12-10/2026-01-28/
    03-18/04-29/06-17)。N=1(gold)の結果は n<=5 で統計的にほぼ無意味 -- 仕様カードが想定していた
    「n=59-65・薄い」を遥かに超える薄さ。レポートで最大級に警告する。
  - btcにm1は存在しない(仕様カード前提通り) -> btcはN∈{5,15,30}のみ、N=1は無し。

実行:
  .venv/bin/python scratchpad/event_kinetics.py --smoke   # 直近2年
  .venv/bin/python scratchpad/event_kinetics.py           # フル
  .venv/bin/python scratchpad/event_kinetics.py --events data/ext_cpi_dates.csv   # CPI/NFP等に流用
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
from src.data_loader import load_mt5_csv  # noqa: E402
from fomc_event_study import (  # noqa: E402
    M15_START, H1_START, HORIZONS_H, NULL_DRAWS_TARGET, SEED,
    atr14, price_before, event_metrics, build_events_table,
    candidate_dates, null_table, summarize, pct_of_real_in_null,
    bootstrap_null_median_pctile, vol_tercile_table,
)

# gold m1 file has no history before this -- it simply starts here (verified via bars/day scan).
# Not a density trap like m15/m5 (no earlier sparse rows to slice away); kept as a named constant
# purely for the report to state explicitly which window is covered.
GOLD_M1_ACTUAL_START = "2025-12-01"


# ----------------------------------------------------------------------------
# NEW: the "early window" classifier. Mirrors event_metrics' no-lookahead pattern
# (price_before / ATR-at-pos0-1) but only computes E_N / d_N -- it does NOT
# recompute retain/round_trip/maxext/maxret (those stay exclusively in fomc_event_study).
# ----------------------------------------------------------------------------
def en_metrics(df, atr, t0, minutes_list):
    """E_N = |P_{t0+N min} - P0| / ATR14(base tf), d_N = sign(P_{t0+N min} - P0).
    P0 and ATR are both anchored on df's OWN bars (base tf), no lookahead."""
    P0 = price_before(df, t0)
    if P0 is None or not np.isfinite(P0):
        return None
    pos0 = df.index.searchsorted(t0, side="left")
    if pos0 == 0:
        return None
    atr_val = atr.iloc[pos0 - 1]
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    out = {"t0": t0, "P0_hr": P0, "ATR_hr": atr_val}
    for n in minutes_list:
        Pn = price_before(df, t0 + pd.Timedelta(minutes=n))
        if Pn is None or not np.isfinite(Pn):
            out[f"E_{n}"] = np.nan
            out[f"d_{n}"] = np.nan
            continue
        diff = Pn - P0
        out[f"E_{n}"] = abs(diff) / atr_val
        out[f"d_{n}"] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
    return out


def build_en_table(df, t0_list, minutes_list, label):
    """Always emits exactly one row per element of t0_list, in order (NaN-filled on failure).
    This is deliberate: the null pool can contain duplicate t0 values (two different source
    events sharing the same clock-time hour:minute can independently draw the same calendar
    day), and a value-based merge on a duplicated key fans out combinatorially. Keeping
    positional 1:1 correspondence with the caller's t0_list lets the join happen by
    row position (see positional_join) instead of by t0 value, which sidesteps that blowup."""
    atr = atr14(df)
    rows = []
    n_ok = 0
    for t0 in t0_list:
        r = en_metrics(df, atr, t0, minutes_list)
        if r is None:
            r = {"t0": t0, "P0_hr": np.nan, "ATR_hr": np.nan}
            for n in minutes_list:
                r[f"E_{n}"] = np.nan
                r[f"d_{n}"] = np.nan
        else:
            n_ok += 1
        rows.append(r)
    tbl = pd.DataFrame(rows)
    print(f"  [{label}] usable E_N rows: {n_ok} / {len(t0_list)}  (table length {len(tbl)}, positionally aligned)",
          file=sys.stderr)
    return tbl


def positional_join(outcome_tbl, en_tbl):
    """en_tbl must have been built via build_en_table(df, list(outcome_tbl['t0']), ...) --
    i.e. same length and same row order as outcome_tbl. Join by row position, not by t0
    value, because t0 can repeat in the null pool (see build_en_table docstring); a
    pd.merge(on='t0') there would silently multiply rows for the repeated draws."""
    o = outcome_tbl.reset_index(drop=True)
    e = en_tbl.reset_index(drop=True)
    if len(o) != len(e):
        raise ValueError(f"positional_join length mismatch: outcome={len(o)} en={len(e)}")
    mism = int((o["t0"].to_numpy() != e["t0"].to_numpy()).sum())
    if mism:
        raise ValueError(f"positional_join: {mism} row(s) have mismatched t0 -- en_tbl was not "
                          f"built from list(outcome_tbl['t0']) in the same order")
    return pd.concat([o, e.drop(columns=["t0"])], axis=1)


# ----------------------------------------------------------------------------
# Report layer: merge E_N (fast classifier) with the outcome table (from fomc_event_study's
# own functions, unmodified) and compute the 5 "core" measurements from the spec card.
# ----------------------------------------------------------------------------
def analyze_pool(pool_label, outcome_tbl, en_tbl, minutes_list, is_null=False):
    """outcome_tbl: real or null table from build_events_table/null_table (has t0, I, I_over_atr, d, retain_24h, ...).
    en_tbl: from build_en_table (has t0, E_n, d_n per n).
    Returns (rho_table, tercile_tables dict, dirmatch dict)."""
    if outcome_tbl is None or outcome_tbl.empty or en_tbl is None or en_tbl.empty:
        print(f"  [{pool_label}] empty pool, skipped")
        return None, {}, {}

    merged = positional_join(outcome_tbl, en_tbl)
    n_merged = len(merged)
    print(f"\n  [{pool_label}] merged n={n_merged} (outcome n={len(outcome_tbl)}, E_N n={len(en_tbl)})")
    if n_merged < 3:
        print(f"  [{pool_label}] n<3 after merge -- too thin for any correlation, skipped")
        return None, {}, {}

    rho_rows = []
    tercile_tables = {}
    dirmatch = {}
    for n in minutes_list:
        ecol, dcol = f"E_{n}", f"d_{n}"
        if ecol not in merged.columns:
            continue
        valid = merged.dropna(subset=[ecol, "I_over_atr", "retain_24h"])
        row = {"N_min": n, "n": len(valid)}
        if len(valid) >= 4:
            rho_I, p_I = spearmanr(valid[ecol], valid["I"])
            rho_Ioa, p_Ioa = spearmanr(valid[ecol], valid["I_over_atr"])
            rho_ret, p_ret = spearmanr(valid[ecol], valid["retain_24h"])
            row.update({"rho_E_vs_I": rho_I, "p_E_vs_I": p_I,
                        "rho_E_vs_Ioveratr": rho_Ioa, "p_E_vs_Ioveratr": p_Ioa,
                        "rho_E_vs_retain24h": rho_ret, "p_E_vs_retain24h": p_ret})
        else:
            row.update({"rho_E_vs_I": np.nan, "p_E_vs_I": np.nan,
                        "rho_E_vs_Ioveratr": np.nan, "p_E_vs_Ioveratr": np.nan,
                        "rho_E_vs_retain24h": np.nan, "p_E_vs_retain24h": np.nan})

        dvalid = merged.dropna(subset=[dcol, "d"])
        if len(dvalid) > 0:
            match = (dvalid[dcol] == dvalid["d"]).mean()
            row["dir_match_rate"] = match
            row["n_dirmatch"] = len(dvalid)
            dirmatch[n] = (match, len(dvalid))
        else:
            row["dir_match_rate"] = np.nan
            row["n_dirmatch"] = 0
        rho_rows.append(row)

        # tercile table: reuse vol_tercile_table verbatim by presenting E_n under the
        # column name it expects ("I_over_atr"). No new bucketing logic written.
        if len(valid) >= 6:
            tmp = merged.dropna(subset=[ecol]).copy()
            tmp["I_over_atr"] = tmp[ecol]
            try:
                vt, _ = vol_tercile_table(tmp, ["retain", "round_trip"], h=24)
                tercile_tables[n] = vt
            except ValueError as e:
                print(f"    [{pool_label} N={n}] tercile failed ({e}) -- n too thin/degenerate")
        else:
            print(f"    [{pool_label} N={n}] n={len(valid)} < 6 -- tercile skipped (too thin)")

    # N=60 reference row = the original 1h I itself vs retain_24h (card14's own number, reprinted here
    # for the saturation comparison -- not recomputed, just pulled from outcome_tbl/merged).
    valid60 = merged.dropna(subset=["I_over_atr", "retain_24h"])
    if len(valid60) >= 4:
        rho60, p60 = spearmanr(valid60["I_over_atr"], valid60["retain_24h"])
        rho_rows.append({"N_min": 60, "n": len(valid60), "rho_E_vs_I": np.nan, "p_E_vs_I": np.nan,
                          "rho_E_vs_Ioveratr": np.nan, "p_E_vs_Ioveratr": np.nan,
                          "rho_E_vs_retain24h": rho60, "p_E_vs_retain24h": p60,
                          "dir_match_rate": 1.0, "n_dirmatch": len(valid60)})  # I's own direction = d by construction

    rho_tbl = pd.DataFrame(rho_rows)
    print(f"\n  --- [{pool_label}] saturation table: E_N predictive power vs N (60=I/ATR14 itself, card14) ---")
    print(rho_tbl.round(4).to_string(index=False))
    for n, vt in tercile_tables.items():
        print(f"\n  --- [{pool_label}] E_{n} tercile -> retain_24h/round_trip_24h @ h=24h ---")
        print(vt.round(4).to_string(index=False))

    return rho_tbl, tercile_tables, dirmatch


def run_asset(name, m15_path, h1_path, hires_specs, events, draws_target):
    """hires_specs: list of (tf_label, path, start_or_None, minutes_list)."""
    print(f"\n{'='*90}\n{name}\n{'='*90}")

    df_m15 = load_mt5_csv(m15_path).loc[M15_START:]
    real_m15 = build_events_table(df_m15, events, f"{name} m15")
    if real_m15.empty:
        print(f"  no usable m15 events for {name}; skipping asset")
        return

    cols = ["retain", "round_trip", "maxext", "maxret"]
    print(f"\n--- {name} m15: outcome stats (tie-back to card14, unmodified functions) ---")
    print(summarize(real_m15, cols, HORIZONS_H).round(4).to_string())

    null_m15 = null_table(df_m15, events, f"{name} m15", draws_target=draws_target)

    df_h1 = load_mt5_csv(h1_path).loc[H1_START:]
    real_h1 = build_events_table(df_h1, events, f"{name} h1")

    for tf_label, path, start, minutes_list in hires_specs:
        print(f"\n{'-'*90}\n{name} / early-window classifier on {tf_label}  (N={minutes_list} min)\n{'-'*90}")
        df_hr = load_mt5_csv(path)
        if start:
            df_hr = df_hr.loc[start:]
        print(f"  {tf_label} span used: {df_hr.index.min()} .. {df_hr.index.max()}  ({len(df_hr)} bars)")

        en_real = build_en_table(df_hr, list(real_m15["t0"]), minutes_list, f"{name} {tf_label} REAL")
        print(f"\n### {name} {tf_label}: REAL FOMC pool (outcome=m15) ###")
        analyze_pool(f"{name} {tf_label} REAL vs m15", real_m15, en_real, minutes_list)

        if not real_h1.empty:
            en_real_h1 = build_en_table(df_hr, list(real_h1["t0"]), minutes_list, f"{name} {tf_label} REAL(h1 idx)")
            print(f"\n### {name} {tf_label}: REAL FOMC pool (outcome=h1, side-check, single pass) ###")
            analyze_pool(f"{name} {tf_label} REAL vs h1", real_h1, en_real_h1, minutes_list)

        if not null_m15.empty:
            en_null = build_en_table(df_hr, list(null_m15["t0"]), minutes_list, f"{name} {tf_label} NULL")
            print(f"\n### {name} {tf_label}: NULL pool (same-clock-time non-FOMC weekdays, outcome=m15) ###")
            analyze_pool(f"{name} {tf_label} NULL vs m15", null_m15, en_null, minutes_list, is_null=True)

            # bootstrap: is E_N's rho/dirmatch on real FOMC different from what the null pool
            # would produce at the same n via resampling? Reuse bootstrap_null_median_pctile
            # on retain_24h/round_trip_24h split by E_N tercile membership is out of scope for
            # that function (built for scalar cols, not conditional split); instead we directly
            # reuse it on retain/round_trip themselves (already does real-vs-null-median-dist),
            # giving the same "is this FOMC-specific" evidence card14 used.
            print(f"\n  --- {name} {tf_label}: real-FOMC median vs bootstrapped null-median dist (retain/round_trip, reused from card14) ---")
            bpct = bootstrap_null_median_pctile(real_m15, null_m15, ["retain", "round_trip"], horizons=(1, 6, 12, 24))
            print(bpct.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/ext_fomc_dates.csv")
    ap.add_argument("--smoke", action="store_true", help="last 2 years only")
    ap.add_argument("--draws", type=int, default=NULL_DRAWS_TARGET)
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

    run_asset(
        "GOLD",
        m15_path="data/vantage_xauusd_m15.csv",
        h1_path="data/vantage_xauusd_h1.csv",
        hires_specs=[
            ("m5", "data/vantage_xauusd_m5.csv", M15_START, [5, 15, 30]),
            ("m1", "data/vantage_xauusd_m1.csv", None, [1]),
        ],
        events=events_all, draws_target=args.draws,
    )

    run_asset(
        "BTC",
        m15_path="data/vantage_btcusd_m15.csv",
        h1_path="data/vantage_btcusd_h1.csv",
        hires_specs=[
            ("m5", "data/vantage_btcusd_m5.csv", M15_START, [5, 15, 30]),
        ],
        events=events_all, draws_target=args.draws,
    )

    print(f"\n\n{'='*90}\nDONE\n{'='*90}")


if __name__ == "__main__":
    main()
