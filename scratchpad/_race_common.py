"""
Shared vectorized-race infrastructure for bandwalk_exit_bounce.py and
vwap_band_edge.py. Reuses the pattern established in pdh_approach_fade.py
(race_matrix / hour-weighted control beta / dedupe / per-year reporting)
so both new screens follow the same falsification discipline as the rest
of the toolkit.
"""
import numpy as np
import pandas as pd


RNG_SEED = 7
CAP_CTRL = 3000
PASS_DELTA = 5.0
PASS_N = 300


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Own OHLC(V) resampler (KNOWN TRAP: breakout_wave.resample() is a
    silent no-op for '1h'/'h1' rules elsewhere in this repo -- avoided by
    not reusing it here). open=first, high=max, low=min, close=last,
    volume=sum (if present)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule).agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def efficiency_ratio(close: pd.Series, W: int) -> pd.Series:
    """Trailing efficiency ratio over W bars, shift-safe (uses only closes
    <= i: close[i], close[i-W], and the |diff| path between them)."""
    num = (close - close.shift(W)).abs()
    den = close.diff().abs().rolling(W).sum()
    return num / den


def dedupe(idx_arr, K):
    """Greedy sequential dedupe identical to pdh_approach_fade.py: after an
    event at position p, the next event cannot fire until position > p+K."""
    idx_arr = np.asarray(idx_arr, dtype=int)
    if len(idx_arr) == 0:
        return idx_arr
    idx_sorted = np.sort(idx_arr)
    keep = []
    last_end = -1
    for i in idx_sorted:
        if i > last_end:
            keep.append(i)
            last_end = i + K
    return np.array(keep, dtype=int)


def next_true_at_or_after(bool_arr: np.ndarray) -> np.ndarray:
    """For each position i, the index of the next True at position >= i
    (or len(bool_arr) sentinel if none exists). Vectorized via a reverse
    running-min over a 'position-if-true else n' array."""
    n = len(bool_arr)
    idx_if_true = np.where(bool_arr, np.arange(n), n)
    return np.minimum.accumulate(idx_if_true[::-1])[::-1]


def race_matrix(high, low, idx, atr_v, entry_v, K):
    """Vectorized race: entry e=entry_v[idx], barriers e +/- 1*atr_v[idx],
    race over bars idx+1..idx+K. Positions without K full future bars are
    dropped (can't fairly evaluate a K-bar race)."""
    idx = np.asarray(idx, dtype=int)
    n = len(high)
    idx_v = idx[idx + K <= n - 1] if len(idx) else idx
    # Drop NaN-ATR rows (spec: "Drop NaN band/atr rows"). ATR is only NaN in
    # the first ~14 warm-up bars of the whole dataset, but an event landing
    # there would otherwise inject a single NaN into every downstream
    # median/std (numpy's median/std do not skip NaN), silently corrupting
    # an entire cell's MFE/MAE line from one stray warm-up-bar event.
    if len(idx_v):
        idx_v = idx_v[~np.isnan(atr_v[idx_v])]
    if len(idx_v) == 0:
        z = np.array([], dtype=int)
        return idx_v, z, z, np.array([]), np.array([]), None, None
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
    return idx_v, first_up, first_dn, atr_e, entry_e, hi, lo


def side_win(first_up, first_dn, side):
    """side='short' wins if down-barrier touched first; side='long' wins
    if up-barrier touched first. Same-bar double-touch or neither-hit is a
    loss for whichever side is scored (first_up==first_dn only when both
    are K, i.e. neither hit -> both False; a genuine same-bar double-touch
    has first_up==first_dn<K -> also both False, i.e. a loss for both
    sides, exactly as specified)."""
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
    """All valid bars as controls, subsampled <=cap/hour (rng seed 7)."""
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
            parts.append(f"{y}:.")
        else:
            parts.append(f"{y}:{deltas_by_year[y]:+.1f}")
    return " ".join(parts)


def process_cell(cell_id, d, atr_v, high, low, close_v, event_idx, side, K, span_years,
                  ctrl_idx_v, ctrl_first_up, ctrl_first_dn, print_line=True):
    """Race + hour-matched beta for one cell/side. Returns a result dict
    (always) and prints the one-line summary unless print_line=False."""
    idx_v, first_up, first_dn, atr_e, entry_e, hi_mat, lo_mat = race_matrix(
        high, low, event_idx, atr_v, close_v, K
    )
    n = len(idx_v)
    win = side_win(first_up, first_dn, side) if n else np.array([], dtype=bool)
    win_pct = win.mean() * 100 if n else float("nan")
    n_yr = n / span_years if span_years > 0 else float("nan")

    ev_hours = d.index.hour.values[idx_v] if n else np.array([])
    ev_years = d.index.year.values[idx_v] if n else np.array([])

    ctrl_win_side = side_win(ctrl_first_up, ctrl_first_dn, side)
    ctrl_hours_full = d.index.hour.values[ctrl_idx_v]
    ctrl_years_full = d.index.year.values[ctrl_idx_v]
    ctrl_win_by_hour = (
        pd.Series(ctrl_win_side, index=ctrl_hours_full).groupby(level=0).mean().to_dict()
        if len(ctrl_win_side) else {}
    )
    fallback = ctrl_win_side.mean() if len(ctrl_win_side) else float("nan")

    beta = hour_weighted_beta(ev_hours, ctrl_win_by_hour, fallback)
    delta = win_pct - beta * 100 if n else float("nan")

    if print_line:
        print(f"{cell_id:<52} n={n:<6d} N/yr={n_yr:>7.1f}  win={win_pct:>5.1f}%  "
              f"beta={beta*100:>5.1f}%  delta={delta:>+6.1f}pt")

    per_year_deltas, per_year_ns = {}, {}
    years_span = sorted(pd.unique(d.index.year))
    if n >= 200:
        for y in years_span:
            mask_y = ev_years == y
            n_y = int(mask_y.sum())
            per_year_ns[y] = n_y
            if n_y < 20:
                continue
            ymask_ctrl = ctrl_years_full == y
            if ymask_ctrl.any():
                y_ctrl_by_hour = (
                    pd.Series(ctrl_win_side[ymask_ctrl], index=ctrl_hours_full[ymask_ctrl])
                    .groupby(level=0).mean().to_dict()
                )
                y_fb = ctrl_win_side[ymask_ctrl].mean()
            else:
                y_ctrl_by_hour, y_fb = {}, fallback
            beta_y = hour_weighted_beta(ev_hours[mask_y], y_ctrl_by_hour, y_fb)
            win_y = win[mask_y].mean()
            per_year_deltas[y] = (win_y - beta_y) * 100

    return dict(
        cell_id=cell_id, n=n, n_yr=n_yr, win_pct=win_pct,
        beta_pct=beta * 100 if n else float("nan"), delta=delta,
        per_year_deltas=per_year_deltas, per_year_ns=per_year_ns,
        idx_v=idx_v, win=win, ev_hours=ev_hours, ev_years=ev_years,
        atr_e=atr_e, entry_e=entry_e, hi_mat=hi_mat, lo_mat=lo_mat,
        ctrl_win_by_hour=ctrl_win_by_hour, fallback=fallback, side=side,
        years_span=years_span,
    )


def print_peryear(res):
    if res["n"] < 200:
        return
    print("    peryear: " + fmt_year_line(res["years_span"], res["per_year_deltas"], res["per_year_ns"]))


def print_er_tercile(res, er_full):
    if res["n"] < 200:
        return
    idx_v = res["idx_v"]
    er_ev = er_full[idx_v]
    valid = ~np.isnan(er_ev)
    if valid.sum() < 30:
        print("    ER-tercile: n/a (insufficient ER data)")
        return
    valid_idx = np.flatnonzero(valid)
    try:
        terc = pd.qcut(er_ev[valid_idx], 3, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        print("    ER-tercile: n/a (degenerate ER distribution)")
        return
    nb = int(np.nanmax(terc)) + 1 if len(terc) else 0
    if nb < 2:
        print("    ER-tercile: n/a (degenerate ER distribution)")
        return
    win = res["win"]
    ev_hours = res["ev_hours"]
    ctrl_by_hour = res["ctrl_win_by_hour"]
    fallback = res["fallback"]
    names = ["T1(low-ER/chop)", "T2(mid)", "T3(high-ER/trend)"] if nb == 3 else [f"T{i+1}" for i in range(nb)]
    parts = []
    for b in range(nb):
        sel_global = valid_idx[terc == b]
        if len(sel_global) == 0:
            continue
        w = win[sel_global].mean() * 100
        beta_b = hour_weighted_beta(ev_hours[sel_global], ctrl_by_hour, fallback) * 100
        parts.append(f"{names[b]} win={w:.1f}% d={w - beta_b:+.1f}pt (n={len(sel_global)})")
    print("    ER-tercile: " + " | ".join(parts))


def print_mfe_mae(res):
    if res["n"] < 200:
        print("    MFE/MAE: n/a (n<200)")
        return
    hi_mat, lo_mat, atr_e, entry_e = res["hi_mat"], res["lo_mat"], res["atr_e"], res["entry_e"]
    if hi_mat is None or len(atr_e) == 0:
        print("    MFE/MAE: n/a")
        return
    mfe, mae = side_excursion(hi_mat, lo_mat, atr_e, entry_e, res["side"])
    print(f"    MFE med={np.median(mfe):.2f} sd={np.std(mfe):.2f}  "
          f"MAE med={np.median(mae):.2f} sd={np.std(mae):.2f}  (ATR units, fade side)")


def print_summary_table(all_results, title):
    print("=" * 100)
    print(title)
    print(f"PASS BAR: delta >= +{PASS_DELTA:.0f}pt AND n>={PASS_N} AND majority of "
          f"(>=20-event) years share the overall delta's sign.")
    print("=" * 100)
    summary = sorted(all_results, key=lambda r: (-r["delta"] if not np.isnan(r["delta"]) else 1e9))
    print(f"{'cell':<52}{'n':>8}{'N/yr':>9}{'win%':>8}{'beta%':>8}{'delta':>9}  {'pass?'}")
    n_pass = 0
    for r in summary:
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
        if verdict:
            n_pass += 1
        win_pct = r["win_pct"] if not np.isnan(r["win_pct"]) else 0.0
        beta_pct = r["beta_pct"] if not np.isnan(r["beta_pct"]) else 0.0
        delta_p = delta if not np.isnan(delta) else 0.0
        print(f"{r['cell_id']:<52}{n:>8d}{r['n_yr']:>9.1f}{win_pct:>7.1f}%{beta_pct:>7.1f}%"
              f"{delta_p:>+8.1f}pt  {verdict}")
    print(f"\n{n_pass} / {len(summary)} cells PASS.")
