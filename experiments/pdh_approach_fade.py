"""
STEP-1 bounce-rate screen: PDH-fade (short) / PDL-bounce (long).

Question: does a REJECTION at the prior day's high/low predict direction,
vs an hour-of-day-matched baseline? No costs, no RR — pure directional
race between two ATR-scaled barriers.

Frozen spec: see task description. This script intentionally does not
redesign anything; the only deviation from the literal instructions is
documented at RESAMPLE_1H_FIX below (a real bug in the shared helper that
would otherwise silently break the "1h" TF cell).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv
from breakout_wave import resample as bw_resample

pd.set_option("display.width", 200)

RNG_SEED = 7
CAP_5M = 1000
CAP_DEFAULT = 3000

K_BY_TF = {"5m": 288, "15m": 96, "1h": 24, "2h": 12}
RULE_BY_TF = {"5m": "5min", "15m": "15min", "1h": "1h", "2h": "2h"}

PASS_DELTA = 5.0
PASS_N = 300


# ------------------------------------------------------------------ #
# RESAMPLE_1H_FIX
# breakout_wave.resample() special-cases rule in {"1h","h1",""} as an
# IDENTITY no-op, because everywhere else in this repo it's called on a
# frame that is ALREADY H1-native (the H1 csv loaded directly). Here the
# base frames are 5m (gold) / 15m (BTC), so calling bw_resample(df,"1h")
# would silently return the un-resampled base frame — the "1h" cell would
# just be a relabeled 5m/15m cell, not real hourly bars. Verified empirically
# before writing this: resample(gold_5m_df, "1h") returned 554764 rows,
# identical to the raw 5m frame. Patched narrowly here (real OHLC resample
# for 1h); every other rule (5min/15min/2h) still goes through the
# unmodified shared helper exactly as specified.
# ------------------------------------------------------------------ #
def resample_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = RULE_BY_TF[tf]
    if rule.lower() in ("1h", "h1"):
        return pd.DataFrame({
            "open": df["open"].resample("1h").first(),
            "high": df["high"].resample("1h").max(),
            "low": df["low"].resample("1h").min(),
            "close": df["close"].resample("1h").last(),
        }).dropna()
    return bw_resample(df, rule)


def build_levels(d: pd.DataFrame):
    pdh = d["high"].resample("1D").max().dropna().shift(1).reindex(d.index, method="ffill")
    pdl = d["low"].resample("1D").min().dropna().shift(1).reindex(d.index, method="ffill")
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1)
    return pdh, pdl, atr


def dedupe(raw_bool: np.ndarray, K: int) -> np.ndarray:
    """Greedy sequential dedupe: after an event at position p, the next
    event cannot fire until position > p + K (skip the next K bars)."""
    keep = np.zeros(len(raw_bool), dtype=bool)
    last_end = -1
    for i in np.flatnonzero(raw_bool):
        if i > last_end:
            keep[i] = True
            last_end = i + K
    return keep


def find_events(d: pd.DataFrame, pdh: pd.Series, pdl: pd.Series, atr: pd.Series, K: int):
    close = d["close"].values
    high = d["high"].values
    low = d["low"].values
    close_prev = d["close"].shift(1).values
    pdh_v = pdh.values
    pdl_v = pdl.values
    atr_v = atr.values

    ok_pdh = ~np.isnan(pdh_v) & ~np.isnan(atr_v) & ~np.isnan(close_prev)
    ok_pdl = ~np.isnan(pdl_v) & ~np.isnan(atr_v) & ~np.isnan(close_prev)

    raw_short = ok_pdh & (close_prev < pdh_v) & (high >= pdh_v - 0.1 * atr_v) & (close < pdh_v)
    raw_long = ok_pdl & (close_prev > pdl_v) & (low <= pdl_v + 0.1 * atr_v) & (close > pdl_v)

    keep_short = dedupe(raw_short, K)
    keep_long = dedupe(raw_long, K)
    return np.flatnonzero(keep_short), np.flatnonzero(keep_long)


def race_matrix(high, low, idx, atr_v, entry_v, K, need_excursion=False):
    """Vectorized race: for each position in idx, look forward K bars and
    find the first offset (0-indexed, sentinel=K meaning 'never') at which
    the up-barrier (entry+atr) and down-barrier (entry-atr) are touched.
    idx positions with fewer than K bars of future data are dropped
    (cannot fairly evaluate a K-bar race without K bars of data)."""
    idx = np.asarray(idx)
    n = len(high)
    idx_v = idx[idx + K <= n - 1]
    if len(idx_v) == 0:
        z = np.array([], dtype=int)
        return idx_v, z, z, None, None, np.array([]), np.array([])
    atr_e = atr_v[idx_v]
    entry_e = entry_v[idx_v]
    offsets = np.arange(1, K + 1)
    mat_idx = idx_v[:, None] + offsets[None, :]
    hi = high[mat_idx]
    lo = low[mat_idx]
    up_barrier = entry_e[:, None] + atr_e[:, None]
    dn_barrier = entry_e[:, None] - atr_e[:, None]
    up_hit = hi >= up_barrier
    dn_hit = lo <= dn_barrier
    first_up = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), K)
    first_dn = np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1), K)
    if need_excursion:
        return idx_v, first_up, first_dn, hi, lo, atr_e, entry_e
    return idx_v, first_up, first_dn, None, None, atr_e, entry_e


def side_win(first_up, first_dn, side):
    if side == "short":
        return first_dn < first_up
    return first_up < first_dn


def side_excursion(hi, lo, atr_e, entry_e, side):
    if side == "short":
        mfe = (entry_e[:, None] - lo) / atr_e[:, None]
        mae = (hi - entry_e[:, None]) / atr_e[:, None]
    else:
        mfe = (hi - entry_e[:, None]) / atr_e[:, None]
        mae = (entry_e[:, None] - lo) / atr_e[:, None]
    return mfe.max(axis=1), mae.max(axis=1)


def hour_weighted_beta(event_hours, ctrl_win_by_hour: dict, fallback: float) -> float:
    if len(event_hours) == 0:
        return float("nan")
    dist = pd.Series(event_hours).value_counts(normalize=True)
    beta = 0.0
    for h, share in dist.items():
        beta += share * ctrl_win_by_hour.get(h, fallback)
    return beta


def build_control_pool(d: pd.DataFrame, atr_v: np.ndarray, K: int, cap: int, rng):
    candidates = np.flatnonzero(~np.isnan(atr_v))
    candidates = candidates[candidates + K <= len(d) - 1]
    hours = d.index.hour.values[candidates]
    picked = []
    for h in range(24):
        h_idx = candidates[hours == h]
        if len(h_idx) > cap:
            h_idx = rng.choice(h_idx, size=cap, replace=False)
        picked.append(h_idx)
    return np.concatenate(picked) if picked else np.array([], dtype=int)


def fmt_year_line(years, deltas_by_year, ns_by_year):
    parts = []
    for y in years:
        n_y = ns_by_year.get(y, 0)
        if n_y < 20:
            parts.append(f"{y}:·")
        else:
            d_y = deltas_by_year[y]
            parts.append(f"{y}:{d_y:+.1f}")
    return " ".join(parts)


def process_cell(instr, tf, d, atr_v, high, low, close_v, event_idx, side, label,
                  ctrl_idx, ctrl_first_up, ctrl_first_dn, K, span_years, results):
    K_ = K
    idx_v, first_up, first_dn, hi_mat, lo_mat, atr_e, entry_e = race_matrix(
        high, low, event_idx, atr_v, close_v, K_, need_excursion=True
    )
    n = len(idx_v)
    win = side_win(first_up, first_dn, side) if n else np.array([], dtype=bool)
    win_pct = win.mean() * 100 if n else float("nan")
    n_yr = n / span_years if span_years > 0 else float("nan")

    ev_hours = d.index.hour.values[idx_v] if n else np.array([])
    ev_years = d.index.year.values[idx_v] if n else np.array([])

    ctrl_win_side = side_win(ctrl_first_up, ctrl_first_dn, side)
    ctrl_hours_full = d.index.hour.values[ctrl_idx]
    ctrl_years_full = d.index.year.values[ctrl_idx]

    ctrl_win_by_hour_overall = (
        pd.Series(ctrl_win_side, index=ctrl_hours_full).groupby(level=0).mean().to_dict()
    )
    overall_fallback = ctrl_win_side.mean() if len(ctrl_win_side) else float("nan")

    beta_overall = hour_weighted_beta(ev_hours, ctrl_win_by_hour_overall, overall_fallback)
    delta = win_pct - beta_overall * 100 if n else float("nan")

    header = (f"{instr:<4} {tf:<3} {label:<11} n={n:<6d} N/yr={n_yr:>7.1f}  "
              f"win={win_pct:>5.1f}%  beta={beta_overall*100:>5.1f}%  delta={delta:>+6.1f}pt")
    print(header)

    per_year_deltas = {}
    per_year_ns = {}
    years_span = sorted(pd.unique(d.index.year))

    if n >= 200:
        for y in years_span:
            y_ev_mask = ev_years == y
            n_y = int(y_ev_mask.sum())
            per_year_ns[y] = n_y
            if n_y < 20:
                continue
            y_ctrl_mask = ctrl_years_full == y
            if y_ctrl_mask.any():
                y_ctrl_win_by_hour = (
                    pd.Series(ctrl_win_side[y_ctrl_mask], index=ctrl_hours_full[y_ctrl_mask])
                    .groupby(level=0).mean().to_dict()
                )
                y_fallback = ctrl_win_side[y_ctrl_mask].mean()
            else:
                y_ctrl_win_by_hour = {}
                y_fallback = overall_fallback
            beta_y = hour_weighted_beta(ev_hours[y_ev_mask], y_ctrl_win_by_hour, y_fallback)
            win_y = win[y_ev_mask].mean()
            per_year_deltas[y] = (win_y - beta_y) * 100

        print("  peryear: " + fmt_year_line(years_span, per_year_deltas, per_year_ns))

    results.append(dict(
        instr=instr, tf=tf, side=side, label=label, n=n, n_yr=n_yr,
        win_pct=win_pct, beta_pct=beta_overall * 100, delta=delta,
        per_year_deltas=per_year_deltas, per_year_ns=per_year_ns,
        idx_v=idx_v, win=win, ev_hours=ev_hours, ev_years=ev_years,
        atr_e=atr_e, entry_e=entry_e, hi_mat=hi_mat, lo_mat=lo_mat,
        ctrl_win_by_hour_overall=ctrl_win_by_hour_overall, overall_fallback=overall_fallback,
        d=d,
    ))
    return idx_v, win, ev_hours, ev_years, atr_e, entry_e, hi_mat, lo_mat


def run_stratification(res, depth_full, K, side):
    n = res["n"]
    if n < 200:
        return
    d = res["d"]
    idx_v = res["idx_v"]
    win = res["win"]
    ev_hours = res["ev_hours"]
    ctrl_by_hour = res["ctrl_win_by_hour_overall"]
    fallback = res["overall_fallback"]

    # (a) poke-depth terciles
    depth_ev = depth_full[idx_v]
    try:
        terc = pd.qcut(depth_ev, 3, labels=False, duplicates="drop")
        n_bins = int(np.nanmax(terc)) + 1 if len(terc) else 0
    except (ValueError, IndexError):
        terc = None
        n_bins = 0

    if terc is not None and n_bins >= 2:
        parts = []
        names = ["T1(shallow)", "T2(mid)", "T3(deep)"] if n_bins == 3 else [f"T{i+1}" for i in range(n_bins)]
        for b in range(n_bins):
            mask = terc == b
            if mask.sum() == 0:
                continue
            w = win[mask].mean() * 100
            beta_b = hour_weighted_beta(ev_hours[mask], ctrl_by_hour, fallback) * 100
            parts.append(f"{names[b]} win={w:.1f}% beta={beta_b:.1f}% d={w-beta_b:+.1f}pt (n={int(mask.sum())})")
        print("  poke-tercile: " + " | ".join(parts))
    else:
        print("  poke-tercile: n/a (insufficient distinct depth values)")

    # (b) first-touch-of-day vs later
    dates = d.index.date[idx_v]
    order = np.argsort(idx_v)
    dates_sorted = dates[order]
    is_first = np.zeros(len(idx_v), dtype=bool)
    seen = set()
    for pos in order:
        dt = dates[pos]
        if dt not in seen:
            is_first[pos] = True
            seen.add(dt)
    for grp_name, mask in [("first", is_first), ("later", ~is_first)]:
        if mask.sum() == 0:
            print(f"  first-vs-later: {grp_name} n=0")
            continue
        w = win[mask].mean() * 100
        beta_g = hour_weighted_beta(ev_hours[mask], ctrl_by_hour, fallback) * 100
        print(f"  first-vs-later: {grp_name} win={w:.1f}% beta={beta_g:.1f}% "
              f"d={w-beta_g:+.1f}pt (n={int(mask.sum())})", end="  ")
    print()

    # MFE/MAE
    hi_mat, lo_mat, atr_e, entry_e = res["hi_mat"], res["lo_mat"], res["atr_e"], res["entry_e"]
    mfe, mae = side_excursion(hi_mat, lo_mat, atr_e, entry_e, side)
    print(f"  MFE med={np.median(mfe):.2f} sd={np.std(mfe):.2f}  "
          f"MAE med={np.median(mae):.2f} sd={np.std(mae):.2f}")


def print_mfe_mae_only(res, side):
    """For n<200 cells: still print MFE/MAE median/std (cheap, no gate)."""
    hi_mat, lo_mat, atr_e, entry_e = res["hi_mat"], res["lo_mat"], res["atr_e"], res["entry_e"]
    if hi_mat is None or len(atr_e) == 0:
        print("  MFE med=n/a sd=n/a  MAE med=n/a sd=n/a")
        return
    mfe, mae = side_excursion(hi_mat, lo_mat, atr_e, entry_e, side)
    print(f"  MFE med={np.median(mfe):.2f} sd={np.std(mfe):.2f}  "
          f"MAE med={np.median(mae):.2f} sd={np.std(mae):.2f}")


def process_instrument(instr, base_df, tfs, all_results):
    rng = np.random.default_rng(RNG_SEED)
    for tf in tfs:
        d = resample_tf(base_df, tf)
        K = K_BY_TF[tf]
        pdh, pdl, atr = build_levels(d)
        atr_v = atr.values
        high = d["high"].values
        low = d["low"].values
        close_v = d["close"].values

        short_idx, long_idx = find_events(d, pdh, pdl, atr, K)

        span_years = (d.index[-1] - d.index[0]).days / 365.25

        # Spec: default cap 3000/hour; only fall back to 1000/hour if a cell
        # proves too slow. Measured full-script runtime ~3s, so no cell needs
        # the harder cap -- use 3000/hour uniformly for better beta precision.
        cap = CAP_DEFAULT
        ctrl_idx = build_control_pool(d, atr_v, K, cap, rng)
        _, ctrl_first_up, ctrl_first_dn, _, _, _, _ = race_matrix(
            high, low, ctrl_idx, atr_v, close_v, K, need_excursion=False
        )
        # race_matrix may drop tail-invalid control positions; re-derive
        # the aligned ctrl_idx used for the returned first_up/first_dn
        ctrl_idx_v = ctrl_idx[ctrl_idx + K <= len(d) - 1]

        for side, event_idx, label, depth_fn in [
            ("short", short_idx, "PDHfadeS", lambda: (high - pdh.values) / atr_v),
            ("long", long_idx, "PDLbounceL", lambda: (pdl.values - low) / atr_v),
        ]:
            res_holder = []
            idx_v, win, ev_hours, ev_years, atr_e, entry_e, hi_mat, lo_mat = process_cell(
                instr, tf, d, atr_v, high, low, close_v, event_idx, side, label,
                ctrl_idx_v, ctrl_first_up, ctrl_first_dn, K, span_years, res_holder
            )
            res = res_holder[0]
            if res["n"] >= 200:
                run_stratification(res, depth_fn(), K, side)
            else:
                print_mfe_mae_only(res, side)
            all_results.append(res)
        print()


def main():
    all_results = []

    print("=" * 78)
    print("GOLD  (data/vantage_xauusd_h1... -> m5 base, restricted to 2018-09-14+)")
    print("=" * 78)
    gold = load_mt5_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "data", "vantage_xauusd_m5.csv"))
    gold = gold.loc["2018-09-14":]
    process_instrument("GOLD", gold, ["5m", "15m", "1h", "2h"], all_results)

    print("=" * 78)
    print("BTC   (data/vantage_btcusd_m15.csv, density-guarded)")
    print("=" * 78)
    btc = load_mt5_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "data", "vantage_btcusd_m15.csv"))
    cnt = btc.groupby(btc.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    btc = btc[btc.index.date >= okd.index[0]]
    process_instrument("BTC", btc, ["15m", "1h", "2h"], all_results)

    print("=" * 78)
    print(f"PASS BAR: delta >= +{PASS_DELTA:.0f}pt AND n>={PASS_N} AND majority of "
          f"(>=20-event) years share the overall delta's sign.")
    print("Pre-registered prediction: gold ~ 0 (fake-line precedent); BTC = open question.")
    print("=" * 78)

    summary = sorted(all_results, key=lambda r: (-r["delta"] if not np.isnan(r["delta"]) else 1e9))
    print(f"{'cell':<24}{'n':>8}{'N/yr':>9}{'win%':>8}{'beta%':>8}{'delta':>9}  {'pass?'}")
    for r in summary:
        cell = f"{r['instr']} {r['tf']} {r['label']}"
        n, delta = r["n"], r["delta"]
        pass_n = n >= PASS_N
        pass_delta = (not np.isnan(delta)) and delta >= PASS_DELTA
        pass_sign = False
        if r["per_year_deltas"]:
            signs = [np.sign(v) for v in r["per_year_deltas"].values()]
            overall_sign = np.sign(delta) if not np.isnan(delta) else 0
            same = sum(1 for s in signs if s == overall_sign)
            pass_sign = len(signs) > 0 and same > len(signs) / 2
        verdict = "PASS" if (pass_n and pass_delta and pass_sign) else ""
        print(f"{cell:<24}{n:>8d}{r['n_yr']:>9.1f}{r['win_pct']:>7.1f}%{r['beta_pct']:>7.1f}%"
              f"{delta:>+8.1f}pt  {verdict}")


if __name__ == "__main__":
    main()
