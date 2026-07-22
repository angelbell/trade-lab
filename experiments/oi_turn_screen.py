"""oi_turn_screen.py -- screen FREE Binance BTCUSDT futures positioning metrics
(data/ext_btc_oi_metrics.csv, built by experiments/binance_metrics_backfill.py) as a
trend-turn / regime detector for BTC. Pre-registered, no threshold tuning (quartiles only,
IS-fit veto applied OOS).

PRE-REGISTERED PREDICTION (printed verbatim below, do not edit post-hoc):
  "本命=KAMA冗長死(F&G前例 φ+0.43); OIは価格非導出の真の新情報という点だけがF&Gと違う;
   funding-fade死とは別問い(レジーム判定)"

TIMEZONE: Vantage broker time ~= UTC+2/+3 (DST-varying); metrics timestamps are UTC.
  - DAILY features (dOI_*, LS_ratio, taker_ratio, OIV/price trend) are built on daily
    UTC-close values, then shift(1) -- the ~2-3h broker/UTC day-boundary skew is absorbed
    by that full-day buffer.
  - INTRADAY joins (leg-bucket test, turn-detection race) add a CONSERVATIVE EXTRA 4-HOUR
    lag on top: the feature used at broker-time t is looked up as of (t - 4h) in UTC terms
    (see causal_join()), so even in the worst-case broker/UTC alignment the feature could
    not have seen same-day-or-later information.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.regime_adaptive import kama
from btc15m_L_anatomy import build_data, run_leg, pdh_weight, COST
from pdh_approach_fade import dedupe, race_matrix, side_win, hour_weighted_beta, build_control_pool

pd.set_option("display.width", 180)

METRICS_CSV = os.path.join(ROOT, "data", "ext_btc_oi_metrics.csv")
H1_CSV = os.path.join(ROOT, "data", "vantage_btcusd_h1.csv")

RNG = np.random.default_rng(7)
NDRAWS = 4000
RACE_K = 96          # 15m bars -> 24h race window (matches pdh_approach_fade's 1h*24 convention scaled to 15m: 96*15min=24h)
PASS_DELTA = 5.0
PASS_N = 300
EXTRA_LAG_H = 4       # conservative broker/UTC guard for intraday joins

BLEED1 = ("2022-03-01", "2023-01-31")
BLEED2 = ("2026-01-01", "2026-06-30")


# =============================================================== helpers
def totrdd(R):
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(eq) - eq).max()
    return R.sum() / max(dd, 1e-9)


def leg_line(label, R, n_indent=2):
    R = np.asarray(R, dtype=float)
    n = len(R)
    pad = " " * n_indent
    if n == 0:
        print(f"{pad}{label:<30} n=0"); return
    win = (R > 0).mean() * 100
    wins = R[R > 0].sum(); losses = -R[R <= 0].sum()
    pf = wins / losses if losses > 0 else np.inf
    eq = np.cumsum(R); dd = (np.maximum.accumulate(eq) - eq).max()
    print(f"{pad}{label:<30} n={n:5d}  win%={win:5.1f}  PF={pf:5.2f}  meanR={R.mean():+.3f}  "
          f"totR={R.sum():+8.1f}  maxDD(R)={dd:6.1f}  totR/DD={totrdd(R):6.2f}")


def med_std(x, label):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    print(f"  {label:<20} median={np.median(x):+10.4f}  std={np.std(x):9.4f}  n={len(x)}")


def causal_join(daily_index, daily_vals, target_times, extra_lag_h=EXTRA_LAG_H):
    """For each target timestamp (broker-time, tz-aware UTC-labelled), look up the last
    daily_vals entry whose daily_index <= (target_time - extra_lag_h). daily_index must be
    sorted ascending. Returns np.array (NaN where no prior value exists)."""
    daily_index = pd.DatetimeIndex(daily_index)
    lookup = pd.DatetimeIndex(target_times) - pd.Timedelta(hours=extra_lag_h)
    pos = daily_index.searchsorted(lookup, side="right") - 1
    out = np.full(len(lookup), np.nan)
    ok = pos >= 0
    out[ok] = np.asarray(daily_vals)[pos[ok]]
    return out


def phi_table(a_bool, b_bool, label_a, label_b):
    m = ~(np.isnan(a_bool.astype(float)) | np.isnan(b_bool.astype(float)))
    a = a_bool[m].astype(bool); b = b_bool[m].astype(bool)
    n11 = int((a & b).sum()); n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum()); n00 = int((~a & ~b).sum())
    phi = np.corrcoef(a.astype(int), b.astype(int))[0, 1] if m.sum() > 1 else np.nan
    print(f"  {label_a} x {label_b}  (n={m.sum()})")
    print(f"      {label_b}=True  {label_b}=False")
    print(f"    {label_a}=True   {n11:6d}      {n10:6d}")
    print(f"    {label_a}=False  {n01:6d}      {n00:6d}")
    print(f"    phi = {phi:+.3f}   (ref. F&G-vs-KAMA precedent phi=+0.43 = redundant-dead)")
    return phi


def quartile_bucket_report(feat_vals, R, times, span_yr, label):
    valid = ~np.isnan(feat_vals)
    if valid.sum() < 20:
        print(f"  ({label}) too few valid ({int(valid.sum())})"); return
    try:
        q, bins = pd.qcut(feat_vals[valid], 4, retbins=True, duplicates="drop")
    except Exception as ex:
        print(f"  ({label}) qcut failed: {ex}"); return
    print(f"  ({label})")
    d = pd.DataFrame({"q": q.astype(str), "R": R[valid], "t": pd.DatetimeIndex(times)[valid]})
    for k, g in d.sort_values("t").groupby("q", observed=True):
        r = g["R"].values
        win = (r > 0).mean() * 100
        wins = r[r > 0].sum(); losses = -r[r <= 0].sum()
        pf = wins / losses if losses > 0 else np.inf
        eqg = np.cumsum(r); ddg = (np.maximum.accumulate(eqg) - eqg).max()
        print(f"    {str(k):<28} n={len(r):4d}  N/yr={len(r)/span_yr:5.1f}  win%={win:5.1f}  "
              f"PF={pf:5.2f}  meanR={r.mean():+.3f}  totR={r.sum():+7.1f}  "
              f"maxDD(R)={ddg:5.1f}  totR/DD={r.sum()/max(ddg,1e-9):5.2f}")


def veto_test(feat_vals, R, times, label):
    valid = ~np.isnan(feat_vals)
    if valid.sum() < 60:
        print(f"  ({label}) too few valid ({int(valid.sum())}) -- skip"); return
    fv = feat_vals[valid]; Rv = R[valid]; tv = pd.DatetimeIndex(times)[valid]
    order = np.argsort(tv.values)
    fv, Rv, tv = fv[order], Rv[order], tv[order]
    med_date = tv[len(tv) // 2]
    is_mask = tv < med_date
    oos_mask = ~is_mask
    if is_mask.sum() < 25 or oos_mask.sum() < 25:
        print(f"  ({label}) half too small (IS={is_mask.sum()} OOS={oos_mask.sum()}) -- skip"); return

    try:
        is_codes, bins = pd.qcut(fv[is_mask], 4, labels=False, retbins=True, duplicates="drop")
    except Exception as ex:
        print(f"  ({label}) IS qcut failed: {ex}"); return
    bins2 = bins.copy(); bins2[0] = -np.inf; bins2[-1] = np.inf
    full_codes = pd.cut(fv, bins=bins2, labels=False, include_lowest=True)

    is_R = Rv[is_mask]
    is_code_vals = full_codes[is_mask]
    means = pd.Series(is_R).groupby(is_code_vals).mean()
    if means.empty:
        print(f"  ({label}) no IS quartile groups -- skip"); return
    worst_code = int(means.idxmin())
    print(f"  ({label})  IS-identified worst quartile code={worst_code}  "
          f"(IS quartile meanR: {dict(means.round(3))})")

    results = {}
    for half_name, half_mask in [("IS", is_mask), ("OOS", oos_mask)]:
        codes_h = full_codes[half_mask]
        R_h = Rv[half_mask]
        keep = codes_h != worst_code
        n_pool = len(R_h); n_keep = int(keep.sum())
        if n_pool < 20 or n_keep == n_pool or n_keep == 0:
            print(f"    {half_name}: degenerate (n_pool={n_pool} n_keep={n_keep}) -- skip")
            continue
        observed = totrdd(R_h[keep])
        base = totrdd(R_h)
        nl = np.array([totrdd(R_h[np.sort(RNG.choice(n_pool, n_keep, replace=False))])
                       for _ in range(NDRAWS)])
        pctile = (nl < observed).mean() * 100
        results[half_name] = (observed, base, pctile)
        print(f"    {half_name}: n_pool={n_pool:4d}  n_kept={n_keep:4d}  base totR/DD={base:6.2f}  "
              f"veto'd totR/DD={observed:6.2f}  null%ile={pctile:5.1f}")

    if len(results) == 2:
        both_pass = all(v[2] >= 90 for v in results.values())
        both_improve = all(v[0] > v[1] for v in results.values())
        verdict = "PASS" if (both_pass and both_improve) else ""
        print(f"    -> {label} veto verdict: {verdict if verdict else 'fail (needs >=90%ile AND '
              'improvement on BOTH halves)'}")


# =============================================================== main
def main():
    print("=" * 100)
    print("PRE-REGISTERED PREDICTION:")
    print("  本命=KAMA冗長死(F&G前例 φ+0.43); OIは価格非導出の真の新情報という点だけがF&Gと違う;")
    print("  funding-fade死とは別問い(レジーム判定)")
    print("=" * 100)

    if not os.path.exists(METRICS_CSV):
        print(f"FATAL: {METRICS_CSV} not found -- run experiments/binance_metrics_backfill.py first.")
        sys.exit(1)

    m = pd.read_csv(METRICS_CSV, index_col=0)
    m.index = pd.to_datetime(m.index, utc=True)
    m = m.sort_index()
    print(f"\nmetrics loaded: {len(m)} rows  {m.index[0]} -> {m.index[-1]}  cols={list(m.columns)}")

    h1 = load_mt5_csv(H1_CSV)
    dc = h1["close"].resample("1D").last().dropna()

    # ---------------------------------------------------------- daily feature construction
    daily_last = m.resample("1D").last()
    daily_mean = m.resample("1D").mean()

    oi = daily_last["sum_open_interest"]
    ls_count = daily_last["count_long_short_ratio"]
    taker_mean = daily_mean["sum_taker_long_short_vol_ratio"]
    oiv = daily_last["sum_open_interest_value"]
    price_d = dc.reindex(oiv.index, method="ffill")
    oiv_price_ratio = oiv / price_d

    feat_raw = {
        "dOI_1d":         oi.pct_change(1),
        "dOI_3d":         oi.pct_change(3),
        "dOI_7d":         oi.pct_change(7),
        "LS_level":       ls_count,
        "LS_chg3d":       ls_count.pct_change(3),
        "taker_1dmean":   taker_mean,
        "oiv_price_chg3d": oiv_price_ratio.pct_change(3),
    }
    SIGN_NATURAL = {"dOI_1d", "dOI_3d", "dOI_7d", "LS_chg3d", "oiv_price_chg3d"}
    # causal: shift(1) -- value known only at the START of the next day
    daily_feat = {k: v.shift(1) for k, v in feat_raw.items()}

    print("\n=== FEATURE MEDIAN +/- STD (causal, shift(1)'d daily values) ===")
    for k, v in daily_feat.items():
        med_std(v.values, k)

    # daily KAMA(14)-rising gate (same construction as research/portfolio_kama.py's kama_gate_btc,
    # collapsed to a direct h1->daily since the 4h intermediate step in portfolio_kama.py is
    # equivalent at daily-close resolution)
    km = kama(dc, 14)
    kama_rising = (km > km.shift(1)).shift(1)   # causal: known at start of the day

    common_idx = daily_feat["dOI_3d"].index
    kama_al = kama_rising.reindex(common_idx).values

    # =========================================================== TEST 1: redundancy vs KAMA
    print("\n" + "=" * 100)
    print("TEST 1: REDUNDANCY vs daily KAMA(14)-rising (phi coefficient + contingency table)")
    print("=" * 100)
    phis = {}
    for name in daily_feat:
        vals = daily_feat[name].reindex(common_idx).values
        med = np.nanmedian(vals)
        med_split = vals > med
        p = phi_table(kama_al, med_split, "KAMA-rising", f"{name}>median")
        phis[f"{name}_medsplit"] = p
        if name in SIGN_NATURAL:
            sign_split = vals > 0
            p2 = phi_table(kama_al, sign_split, "KAMA-rising", f"{name}>0")
            phis[f"{name}_sign"] = p2

    print("\n  -- summary (all phi vs KAMA-rising) --")
    for k, v in phis.items():
        tag = "REDUNDANT (>=0.35ish, F&G precedent 0.43)" if (not np.isnan(v) and abs(v) >= 0.35) else ""
        print(f"    {k:<24} phi={v:+.3f}  {tag}")

    # =========================================================== build btc15m_L leg (EXACT reuse)
    print("\n" + "=" * 100)
    print("Building btc15m_L leg (exact reuse of experiments/btc15m_L_anatomy.py construction)")
    print("=" * 100)
    d15 = build_data()
    tb = run_leg(d15, pullback_frac=0.3)
    Rn = tb["R"].values - COST / tb["risk"].values
    w, ab, ix_pdh = pdh_weight(d15, tb)
    Rw = Rn * w
    times_all = pd.DatetimeIndex(tb["time"])
    span_yr_full = max((times_all.max() - times_all.min()).days / 365.25, 1e-9)
    print(f"  full leg (unrestricted): n={len(Rw)}  span={times_all.min()} -> {times_all.max()}  "
          f"({span_yr_full:.2f}y)")
    leg_line("full leg (Rw)", Rw)

    metrics_start = m.index.min().normalize()
    cutoff = metrics_start + pd.Timedelta(days=8)   # buffer for dOI_7d + shift(1)
    keep_cov = times_all >= cutoff
    n_dropped = (~keep_cov).sum()
    print(f"\n  metrics coverage starts {metrics_start.date()} -> restricting leg entries to "
          f">= {cutoff.date()} (dropped {n_dropped} pre-coverage trades of {len(Rw)})")
    Rw_c = Rw[keep_cov]; times_c = times_all[keep_cov]; tb_c = tb[keep_cov].reset_index(drop=True)
    span_yr_c = max((times_c.max() - times_c.min()).days / 365.25, 1e-9)
    print(f"  restricted leg: n={len(Rw_c)}  span={times_c.min()} -> {times_c.max()}  ({span_yr_c:.2f}y)")
    leg_line("restricted leg (Rw, cov)", Rw_c)
    if len(Rw_c) < 200:
        print(f"  [FLAG: reduced span -- n={len(Rw_c)} well below the naive n~500 anchor in spec's prior estimate]")

    # entry-time OI features via the conservative +4h-lag causal join
    daily_idx_by_feat = {k: v.dropna().index for k, v in daily_feat.items()}
    feat_at_entry = {}
    for name, series in daily_feat.items():
        s = series.dropna()
        feat_at_entry[name] = causal_join(s.index, s.values, times_c)

    # =========================================================== TEST 2: leg buckets
    print("\n" + "=" * 100)
    print("TEST 2: LEG BUCKETS -- meanR/PF/n by quartile of each OI feature at entry (no threshold picking)")
    print("=" * 100)
    for name, vals in feat_at_entry.items():
        quartile_bucket_report(vals, Rw_c, times_c, span_yr_c, name)

    # =========================================================== TEST 3: veto test
    print("\n" + "=" * 100)
    print("TEST 3: VETO TEST -- remove IS-identified worst quartile, apply OOS, vs 4000-draw "
          "equal-keep random-drop null (both halves shown; PASS = >=90%ile AND improves BOTH halves)")
    print("=" * 100)
    for name, vals in feat_at_entry.items():
        veto_test(vals, Rw_c, times_c, name)

    # =========================================================== TEST 4: turn-detection race
    print("\n" + "=" * 100)
    print(f"TEST 4: TURN-DETECTION RACE (K={RACE_K} bars ~ {RACE_K*15/60:.0f}h, +-1 ATR barriers, "
          f"hour-matched beta; pass bar: delta>=+{PASS_DELTA:.0f}pt, n>={PASS_N}, year-sign majority)")
    print("=" * 100)
    atr15 = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1)
    atr_v = atr15.values
    high_v = d15["high"].values; low_v = d15["low"].values; close_v = d15["close"].values

    dh_daily = d15["high"].resample("1D").max().dropna()
    dl_daily = d15["low"].resample("1D").min().dropna()
    hi20 = dh_daily.rolling(20).max().shift(1).reindex(d15.index, method="ffill").values
    lo20 = dl_daily.rolling(20).min().shift(1).reindex(d15.index, method="ffill").values

    close_prev = pd.Series(close_v).shift(1).values
    raw_newhigh = (close_v > hi20) & (close_prev <= hi20) & ~np.isnan(hi20) & ~np.isnan(close_prev)
    raw_newlow = (close_v < lo20) & (close_prev >= lo20) & ~np.isnan(lo20) & ~np.isnan(close_prev)

    keep_newhigh = dedupe(raw_newhigh, RACE_K)
    keep_newlow = dedupe(raw_newlow, RACE_K)

    # dOI_3d at every 15m bar via the +4h-lag causal join (whole d15 index, not just leg entries)
    s3 = daily_feat["dOI_3d"].dropna()
    dOI3_15m = causal_join(s3.index, s3.values, d15.index)

    events = {
        "newhigh & dOI3d<0 -> DOWN-race (bearish divergence)":
            (np.flatnonzero(keep_newhigh)[np.isin(np.flatnonzero(keep_newhigh),
             np.flatnonzero(~np.isnan(dOI3_15m) & (dOI3_15m < 0)))], "short"),
        "newlow  & dOI3d<0 -> UP-race (short-covering bottom, mirror)":
            (np.flatnonzero(keep_newlow)[np.isin(np.flatnonzero(keep_newlow),
             np.flatnonzero(~np.isnan(dOI3_15m) & (dOI3_15m < 0)))], "long"),
        "[context] newhigh & dOI3d>0 -> UP-race (OI-confirm continuation)":
            (np.flatnonzero(keep_newhigh)[np.isin(np.flatnonzero(keep_newhigh),
             np.flatnonzero(~np.isnan(dOI3_15m) & (dOI3_15m > 0)))], "long"),
        "[context] newlow  & dOI3d>0 -> DOWN-race (OI-confirm continuation, mirror)":
            (np.flatnonzero(keep_newlow)[np.isin(np.flatnonzero(keep_newlow),
             np.flatnonzero(~np.isnan(dOI3_15m) & (dOI3_15m > 0)))], "short"),
    }

    span_years_d15 = (d15.index[-1] - d15.index[0]).days / 365.25
    valid_atr = ~np.isnan(atr_v)
    cand_pool = np.flatnonzero(valid_atr)
    cand_pool = cand_pool[cand_pool + RACE_K <= len(d15) - 1]
    ctrl_idx = build_control_pool(d15, atr_v, RACE_K, 3000, np.random.default_rng(7))
    ctrl_idx_v = ctrl_idx[ctrl_idx + RACE_K <= len(d15) - 1]
    _, ctrl_first_up, ctrl_first_dn, _, _, _, _ = race_matrix(
        high_v, low_v, ctrl_idx_v, atr_v, close_v, RACE_K, need_excursion=False)

    summary_rows = []
    for label, (event_idx, side) in events.items():
        idx_v, first_up, first_dn, _, _, atr_e, entry_e = race_matrix(
            high_v, low_v, event_idx, atr_v, close_v, RACE_K, need_excursion=False)
        n = len(idx_v)
        if n == 0:
            print(f"  {label:<58} n=0"); continue
        win = side_win(first_up, first_dn, side)
        win_pct = win.mean() * 100
        n_yr = n / span_years_d15

        ev_hours = d15.index.hour.values[idx_v]
        ev_years = d15.index.year.values[idx_v]
        ctrl_win_side = side_win(ctrl_first_up, ctrl_first_dn, side)
        ctrl_hours = d15.index.hour.values[ctrl_idx_v]
        ctrl_years = d15.index.year.values[ctrl_idx_v]
        ctrl_win_by_hour = pd.Series(ctrl_win_side, index=ctrl_hours).groupby(level=0).mean().to_dict()
        fallback = ctrl_win_side.mean()
        beta = hour_weighted_beta(ev_hours, ctrl_win_by_hour, fallback)
        delta = win_pct - beta * 100

        print(f"  {label:<58} n={n:5d}  N/yr={n_yr:6.1f}  win={win_pct:5.1f}%  "
              f"beta={beta*100:5.1f}%  delta={delta:+6.1f}pt")

        pass_sign = False
        yr_line = []
        uy = sorted(set(ev_years))
        if n >= 100:
            same = 0; tot_y = 0
            for y in uy:
                ym = ev_years == y
                if ym.sum() < 20:
                    continue
                tot_y += 1
                ycm = ctrl_years == y
                if ycm.any():
                    ycwbh = pd.Series(ctrl_win_side[ycm], index=ctrl_hours[ycm]).groupby(level=0).mean().to_dict()
                    yfb = ctrl_win_side[ycm].mean()
                else:
                    ycwbh, yfb = {}, fallback
                betay = hour_weighted_beta(ev_hours[ym], ycwbh, yfb)
                dy = (win[ym].mean() - betay) * 100
                yr_line.append(f"{y}:{dy:+.1f}")
                if np.sign(dy) == np.sign(delta):
                    same += 1
            if tot_y > 0:
                pass_sign = same > tot_y / 2
            print("    peryear: " + " ".join(yr_line))
        pass_n = n >= PASS_N
        pass_delta = delta >= PASS_DELTA
        verdict = "PASS" if (pass_n and pass_delta and pass_sign) else ""
        print(f"    -> n>={PASS_N}:{pass_n}  delta>=+{PASS_DELTA:.0f}pt:{pass_delta}  "
              f"year-sign-majority:{pass_sign}   VERDICT={verdict if verdict else '(fail)'}")
        summary_rows.append((label, n, n_yr, win_pct, beta * 100, delta, verdict))

    print("\n  -- TEST 4 summary --")
    print(f"  {'event':<58}{'n':>7}{'N/yr':>8}{'win%':>8}{'beta%':>8}{'delta':>9}  pass?")
    for label, n, n_yr, win_pct, beta_pct, delta, verdict in summary_rows:
        print(f"  {label:<58}{n:>7d}{n_yr:>8.1f}{win_pct:>7.1f}%{beta_pct:>7.1f}%{delta:>+8.1f}pt  {verdict}")

    # =========================================================== TEST 5: bleed-month descriptive
    print("\n" + "=" * 100)
    print("TEST 5: BLEED-MONTH DESCRIPTIVE (median +/- std of each OI feature; DESCRIPTIVE ONLY, "
          "not a gate/filter)")
    print("=" * 100)
    b1_start, b1_end = pd.Timestamp(BLEED1[0], tz="UTC"), pd.Timestamp(BLEED1[1], tz="UTC")
    b2_start, b2_end = pd.Timestamp(BLEED2[0], tz="UTC"), pd.Timestamp(BLEED2[1], tz="UTC")
    for name, series in daily_feat.items():
        s = series.dropna()
        in_b1 = (s.index >= b1_start) & (s.index <= b1_end)
        in_b2 = (s.index >= b2_start) & (s.index <= b2_end)
        bleed_mask = in_b1 | in_b2
        n_bleed = int(bleed_mask.sum()); n_other = int((~bleed_mask).sum())
        print(f"  ({name})")
        if n_bleed >= 5:
            v = s.values[bleed_mask]
            print(f"    bleed (2022-03->2023-01 + 2026-01->2026-06)  n={n_bleed:4d}  "
                  f"median={np.median(v):+.4f}  std={np.std(v):.4f}")
        else:
            print(f"    bleed  n={n_bleed} (too few / feature not yet covered by metrics span)")
        if n_other >= 5:
            v2 = s.values[~bleed_mask]
            print(f"    non-bleed (rest of covered span)              n={n_other:4d}  "
                  f"median={np.median(v2):+.4f}  std={np.std(v2):.4f}")

    print("\n" + "=" * 100)
    print("DONE. See PRE-REGISTERED PREDICTION at top for the falsifier this screen was built to test.")
    print("=" * 100)


if __name__ == "__main__":
    main()
