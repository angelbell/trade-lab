"""
CLOSING AUDIT of the month-end LAST-BUSINESS-DAY, post-London-fix (16:00-16:30
London) cell found in scratchpad/fx_flow_edges.py (2026-07-11 finding: all 6
USD pairs move USD-positive in the 30min after the fix on month-end days,
mean +2.4 to +3.8 pips).

Reuses fx_flow_edges.py's loading / timezone (Athens-server -> target tz) /
fix-time / month-end-date conventions EXACTLY (imports the module, does not
reimplement them) so this audit cannot silently diverge from the finding it
is auditing.

Sections (see module docstring of each function):
  1. WINDOW PLATEAU  (15/30/45/60 min post-fix hold)
  2. DISTRIBUTION at 30min (mean/median/std/p10/p90)
  3. COST STRESS at 30min (net mean at 0.9/1.5/2.0 pip round-trip cost)
  4. GOTO-BI OVERLAP  (month-end vs goto-bi day membership)
  5. BASKET ANNUAL CONTRIBUTION (5-pair equal-weight, own-std-normalized, R terms)

Run:
    .venv/bin/python scratchpad/d1_close_audit.py --smoke   # 1yr smoke test
    .venv/bin/python scratchpad/d1_close_audit.py           # full run
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fx_flow_edges as ffe  # noqa: E402  -- reuse loading/tz/fix-time/month-end conventions exactly

pd.set_option("display.width", 160)

# USD-positive sign convention (matches the 2026-07-11 finding statement):
# EURUSD/GBPUSD/AUDUSD/NZDUSD are XXXUSD -> price DOWN = USD strength -> flip sign.
# USDJPY/USDCAD are USDXXX -> price UP = USD strength -> keep sign.
USD_SIGN = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1, "USDJPY": +1, "USDCAD": +1}
BASKET_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "NZDUSD"]  # AUDUSD excluded per spec
WINDOWS_MIN = [15, 30, 45, 60]
COSTS_PIPS = [0.9, 1.5, 2.0]
FIX_TZ = "Europe/London"
T_FIX = pd.Timestamp("16:00").time()


# ----------------------------------------------------------------------------
def trading_dates_london(df_athens):
    """Sorted list of unique calendar trading dates in London tz (Timestamps),
    exactly the convention run_month_end_day() uses for month-end determination."""
    dft = df_athens.tz_convert(FIX_TZ)
    dates = sorted(set(pd.DatetimeIndex(dft.index.date)))
    return dates


def post_fix_window_return(df_athens, minutes):
    """Return (ret, start_price) indexed by London calendar date: fractional
    price move from 16:00 fix to 16:00+minutes, London time. No lookahead:
    the window is strictly AFTER the fix timestamp."""
    dft = df_athens.tz_convert(FIX_TZ)
    t_end = (pd.Timestamp("2000-01-01 16:00") + pd.Timedelta(minutes=minutes)).time()
    p0 = ffe.price_at(dft, T_FIX)
    p1 = ffe.price_at(dft, t_end)
    ret = (p1 / p0 - 1.0).dropna()
    sp0 = p0.reindex(ret.index)
    return ret, sp0


def pips_usd_series(name, df_athens, minutes, me_dates=None):
    """USD-positive-signed pip move for the post-fix window, optionally
    filtered to a set of event dates (month-end dates)."""
    ret, sp0 = post_fix_window_return(df_athens, minutes)
    pip = ffe.PAIRS[name]["pip"]
    pips = (ret * sp0) / pip * USD_SIGN[name]
    pips = pips.dropna()
    if me_dates is not None:
        pips = pips[pd.Index(pips.index).isin(me_dates)]
    return pips.sort_index()


# ----------------------------------------------------------------------------
def section1_plateau(dfs, me_dates_by_pair):
    print("=" * 100)
    print("1. WINDOW PLATEAU -- post-fix hold of 15/30/45/60 min, month-end days only, USD-positive sign")
    print("=" * 100)
    rows = []
    for name in ffe.PAIRS:
        for m in WINDOWS_MIN:
            s = pips_usd_series(name, dfs[name], m, me_dates_by_pair[name])
            n = len(s)
            mean = s.mean() if n else float("nan")
            pctpos = (s > 0).mean() * 100 if n else float("nan")
            rows.append(dict(pair=name, min=m, n=n, mean_pips=mean, pct_pos=pctpos))
    df = pd.DataFrame(rows)
    for name in ffe.PAIRS:
        sub = df[df.pair == name]
        print(f"  {name}")
        for _, r in sub.iterrows():
            print(f"    hold={r['min']:>3.0f}min  n={r['n']:>4.0f}  mean={r['mean_pips']:+7.3f}p  %pos={r['pct_pos']:5.1f}%")
    print()
    return df


def section2_distribution(dfs, me_dates_by_pair):
    print("=" * 100)
    print("2. DISTRIBUTION at 30min post-fix, month-end days, USD-positive sign")
    print("=" * 100)
    rows = []
    for name in ffe.PAIRS:
        s = pips_usd_series(name, dfs[name], 30, me_dates_by_pair[name])
        n = len(s)
        if n == 0:
            rows.append(dict(pair=name, n=0))
            continue
        rows.append(dict(pair=name, n=n, mean=s.mean(), median=s.median(), std=s.std(),
                          p10=s.quantile(0.10), p90=s.quantile(0.90)))
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        if r["n"] == 0:
            print(f"  {r['pair']:<8s} n=0 (no data)")
        else:
            print(f"  {r['pair']:<8s} n={r['n']:>4.0f}  mean={r['mean']:+7.3f}  median={r['median']:+7.3f}  "
                  f"std={r['std']:6.3f}  p10={r['p10']:+7.3f}  p90={r['p90']:+7.3f}")
    print()
    return df


def section3_cost_stress(dfs, me_dates_by_pair):
    print("=" * 100)
    print("3. COST STRESS at 30min post-fix, month-end days, USD-positive sign (round-trip pip cost)")
    print("=" * 100)
    rows = []
    for name in ffe.PAIRS:
        s = pips_usd_series(name, dfs[name], 30, me_dates_by_pair[name])
        n = len(s)
        raw_mean = s.mean() if n else float("nan")
        row = dict(pair=name, n=n, raw_mean=raw_mean)
        for c in COSTS_PIPS:
            row[f"net@{c}"] = raw_mean - c if n else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        if r["n"] == 0:
            print(f"  {r['pair']:<8s} n=0 (no data)")
            continue
        parts = "  ".join(f"net@{c}={r[f'net@{c}']:+7.3f}" for c in COSTS_PIPS)
        print(f"  {r['pair']:<8s} n={r['n']:>4.0f}  raw_mean={r['raw_mean']:+7.3f}  {parts}")
    print()
    return df


def section4_gotobi_overlap(dfs, me_dates_by_pair):
    print("=" * 100)
    print("4. GOTO-BI OVERLAP -- month-end last-business-days that are ALSO goto-bi days (fx_flow_edges.gotobi_dates)")
    print("=" * 100)
    print("  NOTE: fx_flow_edges.gotobi_dates() unconditionally appends the month-end date itself")
    print("  as one of its targets (see line 'targets.append(month_end)') -- i.e. by the code's own")
    print("  definition, EVERY month-end IS a goto-bi day. Reporting the raw overlap AND, separately,")
    print("  a 'narrow' goto-bi set (5/10/15/20/25 anchors only, month-end NOT auto-included) so the")
    print("  non-overlap n is meaningful rather than trivially zero.")
    print()
    rows = []
    for name in ffe.PAIRS:
        tdates = trading_dates_london(dfs[name])
        me = sorted(me_dates_by_pair[name])
        gb_full = set(ffe.gotobi_dates(tdates))  # as coded: includes month-end by construction

        # narrow goto-bi: replicate gotobi_dates' day-5/10/15/20/25 anchor logic WITHOUT
        # the unconditional month-end append, so we can ask "does month-end coincide with
        # a 5/10/15/20/25 anchor independently of being month-end itself?"
        import bisect
        dates_arr = list(tdates)
        periods = sorted(set((d.year, d.month) for d in dates_arr))
        narrow = []
        for (y, m) in periods:
            month_dates = [d for d in dates_arr if d.year == y and d.month == m]
            for day in (5, 10, 15, 20, 25):
                want = pd.Timestamp(year=y, month=m, day=day)
                idx = bisect.bisect_right(dates_arr, want) - 1
                if idx >= 0:
                    narrow.append(dates_arr[idx])
        gb_narrow = set(narrow)

        n_me = len(me)
        n_overlap_full = sum(1 for d in me if d in gb_full)
        n_nonoverlap_full = n_me - n_overlap_full
        n_overlap_narrow = sum(1 for d in me if d in gb_narrow)
        n_nonoverlap_narrow = n_me - n_overlap_narrow

        rows.append(dict(pair=name, n_me=n_me,
                          overlap_full=n_overlap_full, nonoverlap_full=n_nonoverlap_full,
                          overlap_narrow=n_overlap_narrow, nonoverlap_narrow=n_nonoverlap_narrow))

        me_set_narrow_overlap = set(d for d in me if d in gb_narrow)
        me_set_narrow_nonoverlap = set(d for d in me if d not in gb_narrow)

        s30 = pips_usd_series(name, dfs[name], 30, None)  # unfiltered, filter below
        s_overlap = s30[pd.Index(s30.index).isin(me_set_narrow_overlap)]
        s_nonoverlap = s30[pd.Index(s30.index).isin(me_set_narrow_nonoverlap)]

        print(f"  {name:<8s} n_month_end={n_me}  "
              f"[full goto-bi def] overlap={n_overlap_full} nonoverlap={n_nonoverlap_full}  |  "
              f"[narrow 5/10/15/20/25-only def] overlap={n_overlap_narrow} nonoverlap={n_nonoverlap_narrow}")
        print(f"           narrow-overlap fix-effect:    n={len(s_overlap):>3d}  mean={s_overlap.mean() if len(s_overlap) else float('nan'):+7.3f}p")
        print(f"           narrow-non-overlap fix-effect: n={len(s_nonoverlap):>3d}  mean={s_nonoverlap.mean() if len(s_nonoverlap) else float('nan'):+7.3f}p")
    print()
    return pd.DataFrame(rows)


def section5_basket_annual(dfs, me_dates_by_pair, years_span):
    print("=" * 100)
    print("5. BASKET ANNUAL CONTRIBUTION -- equal-weight USD basket, 5 pairs (excl. AUDUSD), 30min post-fix")
    print("=" * 100)
    per_pair = {}
    for name in BASKET_PAIRS:
        s = pips_usd_series(name, dfs[name], 30, me_dates_by_pair[name])
        per_pair[name] = s

    common_idx = None
    for name, s in per_pair.items():
        idx = pd.Index(s.index)
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_events = len(common_idx)

    z_frame = {}
    for name, s in per_pair.items():
        s_common = s.reindex(common_idx)
        own_std = s.std()  # event-level std computed over the pair's OWN full month-end sample (not just common_idx)
        z_frame[name] = s_common / own_std
    z_df = pd.DataFrame(z_frame)
    basket = z_df.mean(axis=1)  # equal-weight average of own-std-normalized moves, per event

    basket_mean = basket.mean()
    basket_std = basket.std()
    events_per_yr = n_events / years_span if years_span > 0 else float("nan")
    per_event_R = basket_mean / basket_std if basket_std > 0 else float("nan")
    annual_R = per_event_R * events_per_yr

    print(f"  pairs in basket: {BASKET_PAIRS}")
    print(f"  common events (all 5 pairs have data): n={n_events}  span={years_span:.2f}yr  events/yr={events_per_yr:.2f}")
    print(f"  per-pair own-std used for normalization (pips, full month-end sample, not just common events):")
    for name, s in per_pair.items():
        print(f"    {name:<8s} own_std={s.std():7.3f}p  n_own={len(s)}")
    print(f"  basket per-event: mean={basket_mean:+.4f}  std={basket_std:.4f}  (units: own-std-normalized pips)")
    print(f"  per-event R (mean/std of the basket series itself) = {per_event_R:+.4f}")
    print(f"  naive annual sum, expressed in R (1R = one basket-event std) = per_event_R * events/yr = {annual_R:+.4f}R/yr")
    print()
    return dict(n_events=n_events, events_per_yr=events_per_yr, basket_mean=basket_mean,
                basket_std=basket_std, per_event_R=per_event_R, annual_R=annual_R)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1-year smoke test (last full year only)")
    args = ap.parse_args()

    start = "2025-01-01" if args.smoke else "2016-01-01"

    print("=" * 100)
    print("D1 CLOSE AUDIT -- month-end last-business-day, post-London-fix (16:00-16:30) cell")
    print(f"Window: {'SMOKE 1yr (2025)' if args.smoke else 'FULL 2016-01-01 -> present'}")
    print("=" * 100)
    print()

    dfs = {}
    for name in ffe.PAIRS:
        dfs[name] = ffe.load_intraday(name, start=start)

    me_dates_by_pair = {}
    for name in ffe.PAIRS:
        tdates = trading_dates_london(dfs[name])
        me_dates_by_pair[name] = set(ffe.month_end_dates(tdates))

    # years span for section 5 (use USDJPY's date range as reference)
    ref_dates = trading_dates_london(dfs["USDJPY"])
    years_span = (ref_dates[-1] - ref_dates[0]).days / 365.25

    section1_plateau(dfs, me_dates_by_pair)
    section2_distribution(dfs, me_dates_by_pair)
    section3_cost_stress(dfs, me_dates_by_pair)
    section4_gotobi_overlap(dfs, me_dates_by_pair)
    section5_basket_annual(dfs, me_dates_by_pair, years_span)

    print("=" * 100)
    print("END OF AUDIT")
    print("=" * 100)


if __name__ == "__main__":
    main()
