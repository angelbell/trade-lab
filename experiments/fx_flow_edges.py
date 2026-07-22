"""
FROZEN SPEC: screen FLOW/CONSTRAINT edges (not information edges) in FX around
fixing times. Companion in spirit to the goto-bi (5-10-15-20-25 Japanese
corporate-flow day) finding, which is used here as the positive control.

Rationale (printed verbatim below):
情報エッジは全滅、生き残った暦系＝ゴトー日はフロー・エッジ（実需が価格非感応で強制執行）。同族を探す。
ゴトー日=陽性対照。

Run:
    .venv/bin/python experiments/fx_flow_edges.py --smoke   # 1yr smoke test
    .venv/bin/python experiments/fx_flow_edges.py           # full 10yr run
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_mt5_csv

pd.set_option("display.width", 160)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- pair config: prefer m5, fall back to m15 for pairs whose m5 export timed out
PAIRS = {
    "EURUSD": dict(file="vantage_eurusd_m5.csv",  tf="m5",  pip=0.0001),
    "GBPUSD": dict(file="vantage_gbpusd_m5.csv",  tf="m5",  pip=0.0001),
    "USDJPY": dict(file="vantage_usdjpy_m5.csv",  tf="m5",  pip=0.01),
    "AUDUSD": dict(file="vantage_audusd_m15.csv", tf="m15", pip=0.0001),
    "NZDUSD": dict(file="vantage_nzdusd_m15.csv", tf="m15", pip=0.0001),
    "USDCAD": dict(file="vantage_usdcad_m15.csv", tf="m15", pip=0.0001),
}

COST_PIPS_LO, COST_PIPS_HI = 1.0, 1.2
COST_PIPS = COST_PIPS_HI  # conservative bar for pass/fail

RATIONALE = (
    "情報エッジは全滅、生き残った暦系＝ゴトー日はフロー・エッジ"
    "（実需が価格非感応で強制執行）。同族を探す。ゴトー日=陽性対照。"
)

PREREGISTERED = (
    "London fix月末=文献既知の実在候補だが2013スキャンダル後の減衰が本命リスク; "
    "NYカットのピンは弱いか消滅; ゴトー日は陽性に出ねばならない"
)


# ----------------------------------------------------------------------------
# loading + timezone plumbing
# ----------------------------------------------------------------------------
def load_intraday(name, start="2016-01-01"):
    cfg = PAIRS[name]
    df = load_mt5_csv(DATA_DIR / cfg["file"])
    df = df.loc[start:]
    # index is tz-labeled UTC but is actually broker server time (EET/EEST).
    naive = df.index.tz_localize(None)
    athens = naive.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT")
    df = df.set_axis(athens)
    df = df[df.index.notna()]
    df = df.sort_index()
    return df


def load_daily_atr(name, start="2016-01-01", n=14):
    path = DATA_DIR / f"vantage_{name.lower()}_d1.csv"
    df = load_mt5_csv(path)
    df = df.loc[start:]
    naive = df.index.tz_localize(None)
    dates = naive.normalize()
    df = df.set_axis(dates)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    # shift(1): ATR known as of PRIOR day's close -> no lookahead into "today"
    atr_prev = atr.shift(1)
    out = pd.DataFrame({"date": atr_prev.index, "atr": atr_prev.values}).dropna()
    out = out.sort_values("date")
    return out


def attach_atr(dates_index, atr_df):
    """merge_asof (backward) the known-as-of-prior-day ATR onto arbitrary event dates."""
    ev = pd.DataFrame({"date": pd.to_datetime(pd.Index(dates_index))}).sort_values("date")
    merged = pd.merge_asof(ev, atr_df, on="date", direction="backward")
    merged = merged.set_index(ev["date"])
    return merged["atr"]


# ----------------------------------------------------------------------------
# self-check: modal weekday+NY-hour of each pair's weekly LAST bar == Fri 16-17 NY
# ----------------------------------------------------------------------------
def self_check(name, df_athens):
    iso = df_athens.index.isocalendar()
    key = iso["year"].astype(str) + "-W" + iso["week"].astype(str)
    last_idx = df_athens.groupby(key.values).apply(lambda g: g.index.max())
    ny = last_idx.dt.tz_convert("America/New_York")
    wk = ny.dt.day_name()
    hr = ny.dt.hour
    combo = wk + " " + hr.astype(str) + "h"
    mode = combo.mode()
    modal = mode.iloc[0] if len(mode) else "N/A"
    frac = (combo == modal).mean()
    ok = modal.startswith("Friday 16") or modal.startswith("Friday 17")
    return dict(modal=modal, frac=frac, ok=ok, n_weeks=len(combo))


# ----------------------------------------------------------------------------
# vectorized "price at wall-clock time T, per calendar day" extractor
# ----------------------------------------------------------------------------
def price_at(df_tz, t):
    sub = df_tz[df_tz.index.time == t]
    s = sub["open"].copy()
    s.index = pd.DatetimeIndex(sub.index.date)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def window_pair_returns(df_athens, tz, t_start, t_mid, t_end):
    """pre = [t_start,t_mid), post = [t_mid,t_end) fractional returns, indexed by
    the calendar date (in `tz`) of the window. Fully vectorized (boolean time
    filter + Series division), no per-event python loop."""
    dft = df_athens.tz_convert(tz)
    p1, p2, p3 = price_at(dft, t_start), price_at(dft, t_mid), price_at(dft, t_end)
    pre = (p2 / p1 - 1.0).dropna()
    post = (p3 / p2 - 1.0).dropna()
    start_price_pre = p1.reindex(pre.index)
    start_price_post = p2.reindex(post.index)
    return pre, post, start_price_pre, start_price_post


def month_end_dates(date_index):
    """last TRADED date per (year,month), derived from the actual data (so
    real holidays are respected, not just a calendar business-day-end guess)."""
    di = pd.DatetimeIndex(sorted(set(date_index)))
    per = di.to_period("M")
    s = pd.Series(di, index=per)
    last = s.groupby(level=0).max()
    return set(last.values)


def stats_block(ret, start_price, atr_prev, pip):
    """ret: fractional return series indexed by date. Returns dict of pip/ATR stats."""
    abs_chg = ret * start_price
    pips = abs_chg / pip
    atru = abs_chg / atr_prev.reindex(abs_chg.index)
    pips = pips.dropna()
    atru = atru.dropna()
    n = len(pips)
    if n == 0:
        return dict(n=0)
    return dict(
        n=n,
        pips_median=pips.median(), pips_std=pips.std(), pips_mean=pips.mean(),
        pct_pos=(pips > 0).mean() * 100,
        atr_median=atru.median(), atr_std=atru.std(), atr_mean=atru.mean(),
        pips_series=pips, atru_series=atru,
    )


def fmt_block(label, b):
    if b.get("n", 0) == 0:
        return f"    {label:<26s} n=0 (no data)"
    return (f"    {label:<26s} n={b['n']:>5d}  "
            f"pips: med={b['pips_median']:+7.3f} std={b['pips_std']:6.3f} "
            f"mean={b['pips_mean']:+7.3f}  %pos={b['pct_pos']:5.1f}%  "
            f"| ATR-units: med={b['atr_median']:+6.4f} std={b['atr_std']:6.4f} "
            f"mean={b['atr_mean']:+6.4f}")


def pass_fail(b):
    if b.get("n", 0) == 0:
        return "NO-DATA"
    return "PASS" if abs(b["pips_mean"]) >= 2 * COST_PIPS else "fail(<2xcost)"


def half_sign_stable(pips_series, split_date):
    s = pips_series.sort_index()
    h1 = s[s.index < split_date]
    h2 = s[s.index >= split_date]
    if len(h1) == 0 or len(h2) == 0:
        return None, None, None
    m1, m2 = h1.mean(), h2.mean()
    stable = np.sign(m1) == np.sign(m2) and np.sign(m1) != 0
    return m1, m2, stable


def per_year_signs(pips_series):
    s = pips_series.sort_index()
    yearly = s.groupby(s.index.year).mean()
    signs = np.sign(yearly)
    pos = (signs > 0).sum()
    neg = (signs < 0).sum()
    tot = len(yearly)
    return yearly, pos, neg, tot


# ----------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------
def run_london_fix(name, df_athens, atr_df, split_date):
    tz = "Europe/London"
    t_start, t_mid, t_end = pd.Timestamp("15:30").time(), pd.Timestamp("16:00").time(), pd.Timestamp("16:30").time()
    pre, post, sp_pre, sp_post = window_pair_returns(df_athens, tz, t_start, t_mid, t_end)
    atr_pre = attach_atr(pre.index, atr_df)
    atr_post = attach_atr(post.index, atr_df)
    pip = PAIRS[name]["pip"]

    me = month_end_dates(pre.index.union(post.index))
    is_me_pre = pd.Index(pre.index).isin(me)
    is_me_post = pd.Index(post.index).isin(me)

    out = {}
    out["pre_all"] = stats_block(pre, sp_pre, atr_pre, pip)
    out["post_all"] = stats_block(post, sp_post, atr_post, pip)
    out["pre_me"] = stats_block(pre[is_me_pre], sp_pre[is_me_pre], atr_pre, pip)
    out["pre_nonme"] = stats_block(pre[~is_me_pre], sp_pre[~is_me_pre], atr_pre, pip)
    out["post_me"] = stats_block(post[is_me_post], sp_post[is_me_post], atr_post, pip)
    out["post_nonme"] = stats_block(post[~is_me_post], sp_post[~is_me_post], atr_post, pip)

    # matched control: same 30-min length, same clock +/-2h
    ctrl_series = []
    for sign in (-1, 1):
        off = pd.Timedelta(hours=2 * sign)
        cs = (pd.Timestamp("2000-01-01") + pd.Timedelta(hours=t_start.hour, minutes=t_start.minute) + off).time()
        cm = (pd.Timestamp("2000-01-01") + pd.Timedelta(hours=t_mid.hour, minutes=t_mid.minute) + off).time()
        ce = (pd.Timestamp("2000-01-01") + pd.Timedelta(hours=t_end.hour, minutes=t_end.minute) + off).time()
        cpre, cpost, csp_pre, csp_post = window_pair_returns(df_athens, tz, cs, cm, ce)
        catr_pre = attach_atr(cpre.index, atr_df)
        catr_post = attach_atr(cpost.index, atr_df)
        ctrl_series.append(stats_block(cpre, csp_pre, catr_pre, pip))
        ctrl_series.append(stats_block(cpost, csp_post, catr_post, pip))
    valid_ctrl = [c for c in ctrl_series if c.get("n", 0) > 0]
    if valid_ctrl:
        all_ctrl_pips = pd.concat([c["pips_series"] for c in valid_ctrl])
        all_ctrl_atru = pd.concat([c["atru_series"] for c in valid_ctrl])
        out["control"] = dict(n=len(all_ctrl_pips), pips_mean=all_ctrl_pips.mean(),
                               pips_median=all_ctrl_pips.median(), pips_std=all_ctrl_pips.std(),
                               pct_pos=(all_ctrl_pips > 0).mean() * 100,
                               atr_mean=all_ctrl_atru.mean(), atr_median=all_ctrl_atru.median(),
                               atr_std=all_ctrl_atru.std(),
                               pips_series=all_ctrl_pips, atru_series=all_ctrl_atru)
    else:
        out["control"] = dict(n=0)

    out["pre_series"] = pre.rename(name); out["post_series"] = post.rename(name)
    return out


def run_ny_cut(name, df_athens, atr_df, split_date):
    tz = "America/New_York"
    t_start, t_mid, t_end = pd.Timestamp("09:30").time(), pd.Timestamp("10:00").time(), pd.Timestamp("10:30").time()
    pre, post, sp_pre, sp_post = window_pair_returns(df_athens, tz, t_start, t_mid, t_end)
    atr_pre = attach_atr(pre.index, atr_df)
    atr_post = attach_atr(post.index, atr_df)
    pip = PAIRS[name]["pip"]

    me = month_end_dates(pre.index.union(post.index))
    is_me_pre = pd.Index(pre.index).isin(me)
    is_me_post = pd.Index(post.index).isin(me)

    out = {}
    out["pre_all"] = stats_block(pre, sp_pre, atr_pre, pip)
    out["post_all"] = stats_block(post, sp_post, atr_post, pip)
    out["pre_me"] = stats_block(pre[is_me_pre], sp_pre[is_me_pre], atr_pre, pip)
    out["pre_nonme"] = stats_block(pre[~is_me_pre], sp_pre[~is_me_pre], atr_pre, pip)
    out["post_me"] = stats_block(post[is_me_post], sp_post[is_me_post], atr_post, pip)
    out["post_nonme"] = stats_block(post[~is_me_post], sp_post[~is_me_post], atr_post, pip)

    ctrl_series = []
    for sign in (-1, 1):
        off = pd.Timedelta(hours=2 * sign)
        cs = (pd.Timestamp("2000-01-01") + pd.Timedelta(hours=t_start.hour, minutes=t_start.minute) + off).time()
        cm = (pd.Timestamp("2000-01-01") + pd.Timedelta(hours=t_mid.hour, minutes=t_mid.minute) + off).time()
        ce = (pd.Timestamp("2000-01-01") + pd.Timedelta(hours=t_end.hour, minutes=t_end.minute) + off).time()
        cpre, cpost, csp_pre, csp_post = window_pair_returns(df_athens, tz, cs, cm, ce)
        catr_pre = attach_atr(cpre.index, atr_df)
        catr_post = attach_atr(cpost.index, atr_df)
        ctrl_series.append(stats_block(cpre, csp_pre, catr_pre, pip))
        ctrl_series.append(stats_block(cpost, csp_post, catr_post, pip))
    valid_ctrl = [c for c in ctrl_series if c.get("n", 0) > 0]
    if valid_ctrl:
        all_ctrl_pips = pd.concat([c["pips_series"] for c in valid_ctrl])
        all_ctrl_atru = pd.concat([c["atru_series"] for c in valid_ctrl])
        out["control"] = dict(n=len(all_ctrl_pips), pips_mean=all_ctrl_pips.mean(),
                               pips_median=all_ctrl_pips.median(), pips_std=all_ctrl_pips.std(),
                               pct_pos=(all_ctrl_pips > 0).mean() * 100,
                               atr_mean=all_ctrl_atru.mean(), atr_median=all_ctrl_atru.median(),
                               atr_std=all_ctrl_atru.std(),
                               pips_series=all_ctrl_pips, atru_series=all_ctrl_atru)
    else:
        out["control"] = dict(n=0)

    out["pre_series"] = pre.rename(name); out["post_series"] = post.rename(name)
    return out


def run_month_end_day(name, df_athens, atr_df):
    tz = "Europe/London"
    dft = df_athens.tz_convert(tz)
    g = dft.groupby(dft.index.date)
    o = g["open"].first()
    c = g["close"].last()
    o.index = pd.DatetimeIndex(o.index)
    c.index = pd.DatetimeIndex(c.index)
    ret = (c / o - 1.0).dropna()
    atr_prev = attach_atr(ret.index, atr_df)
    pip = PAIRS[name]["pip"]

    me = month_end_dates(ret.index)
    is_me = pd.Index(ret.index).isin(me)

    out = {}
    out["me"] = stats_block(ret[is_me], o.reindex(ret.index)[is_me], atr_prev, pip)
    out["nonme"] = stats_block(ret[~is_me], o.reindex(ret.index)[~is_me], atr_prev, pip)
    out["series_me"] = ret[is_me]
    out["series_nonme"] = ret[~is_me]
    return out


def gotobi_dates(trading_dates_sorted):
    import bisect
    dates_arr = list(trading_dates_sorted)
    periods = sorted(set((d.year, d.month) for d in dates_arr))
    targets = []
    for (y, m) in periods:
        month_dates = [d for d in dates_arr if d.year == y and d.month == m]
        month_end = max(month_dates)
        for day in (5, 10, 15, 20, 25):
            want = pd.Timestamp(year=y, month=m, day=day)
            idx = bisect.bisect_right(dates_arr, want) - 1
            if idx >= 0 and dates_arr[idx].year == y and dates_arr[idx].month == m:
                targets.append(dates_arr[idx])
            elif idx >= 0:
                targets.append(dates_arr[idx])  # rolled into prior month, still valid preceding trading day
        targets.append(month_end)
    return sorted(set(targets))


def run_gotobi(df_athens, atr_df):
    tz = "Asia/Tokyo"
    dft = df_athens.tz_convert(tz)
    t_start, t_end = pd.Timestamp("09:00").time(), pd.Timestamp("09:55").time()
    p1 = price_at(dft, t_start)
    p2 = price_at(dft, t_end)
    ret = (p2 / p1 - 1.0).dropna()
    sp = p1.reindex(ret.index)
    atr_prev = attach_atr(ret.index, atr_df)
    pip = PAIRS["USDJPY"]["pip"]

    all_dates = sorted(set(ret.index))
    gb = set(gotobi_dates(all_dates))
    is_gb = pd.Index(ret.index).isin(gb)

    out = {}
    out["gotobi"] = stats_block(ret[is_gb], sp[is_gb], atr_prev, pip)
    out["other"] = stats_block(ret[~is_gb], sp[~is_gb], atr_prev, pip)
    out["series_gotobi"] = ret[is_gb]
    out["series_other"] = ret[~is_gb]
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1-year smoke test (last full year only)")
    args = ap.parse_args()

    start = "2025-01-01" if args.smoke else "2016-01-01"
    split_date = pd.Timestamp("2025-06-01") if args.smoke else pd.Timestamp("2021-01-01")

    print("=" * 100)
    print("FX FLOW/CONSTRAINT EDGE SCREEN (fixing-time flow, not information)")
    print("=" * 100)
    print("Rationale:", RATIONALE)
    print("Pre-registered predictions:", PREREGISTERED)
    print(f"Cost bar (round-trip majors): {COST_PIPS_LO}-{COST_PIPS_HI} pips. "
          f"Pass = |mean effect| >= 2x{COST_PIPS} = {2*COST_PIPS} pips AND half-stable sign.")
    print(f"Window: {'SMOKE 1yr (2025)' if args.smoke else 'FULL 2016-01-01 -> present (10yr)'}")
    print()

    dfs = {}
    atrs = {}
    for name, cfg in PAIRS.items():
        dfs[name] = load_intraday(name, start=start)
        atrs[name] = load_daily_atr(name, start=start)

    # ---- SELF-CHECK ----
    print("-" * 100)
    print("SELF-CHECK: modal weekday+NY-hour of each pair's weekly LAST bar (expect Friday 16-17h NY)")
    print("-" * 100)
    any_fail = False
    for name, df in dfs.items():
        r = self_check(name, df)
        flag = "OK" if r["ok"] else "*** FAIL ***"
        if not r["ok"]:
            any_fail = True
        print(f"  {name:<8s} modal={r['modal']:<14s} frac={r['frac']*100:5.1f}% "
              f"n_weeks={r['n_weeks']:<5d} [{flag}]")
    if any_fail:
        print("  WARNING: at least one pair failed the self-check -> tz assumption suspect for that pair;")
        print("  results for the failing pair(s) below should be treated as UNRELIABLE (flagged inline).")
    print()

    # ============================================================ EVENT 1: LONDON FIX
    print("=" * 100)
    print("EVENT 1: LONDON FIX (15:30-16:00 pre / 16:00-16:30 post London time)")
    print("=" * 100)
    london_results = {}
    for name, df in dfs.items():
        ok = self_check(name, df)["ok"]
        r = run_london_fix(name, df, atrs[name], split_date)
        london_results[name] = r
        tag = "" if ok else "  [TZ-SUSPECT]"
        print(f"{name}{tag}")
        print(fmt_block("pre (all days)", r["pre_all"]), pass_fail(r["pre_all"]))
        print(fmt_block("post (all days)", r["post_all"]), pass_fail(r["post_all"]))
        print(fmt_block("pre (month-end)", r["pre_me"]), pass_fail(r["pre_me"]))
        print(fmt_block("pre (non-month-end)", r["pre_nonme"]), pass_fail(r["pre_nonme"]))
        print(fmt_block("post (month-end)", r["post_me"]), pass_fail(r["post_me"]))
        print(fmt_block("post (non-month-end)", r["post_nonme"]), pass_fail(r["post_nonme"]))
        print(fmt_block("MATCHED CONTROL (+/-2h)", r["control"]))
        print()

    # ============================================================ EVENT 2: NY OPTION CUT
    print("=" * 100)
    print("EVENT 2: NY OPTION CUT (09:30-10:00 pre / 10:00-10:30 post NY time)")
    print("=" * 100)
    ny_results = {}
    for name, df in dfs.items():
        ok = self_check(name, df)["ok"]
        r = run_ny_cut(name, df, atrs[name], split_date)
        ny_results[name] = r
        tag = "" if ok else "  [TZ-SUSPECT]"
        print(f"{name}{tag}")
        print(fmt_block("pre (all days)", r["pre_all"]), pass_fail(r["pre_all"]))
        print(fmt_block("post (all days)", r["post_all"]), pass_fail(r["post_all"]))
        print(fmt_block("pre (month-end)", r["pre_me"]), pass_fail(r["pre_me"]))
        print(fmt_block("pre (non-month-end)", r["pre_nonme"]), pass_fail(r["pre_nonme"]))
        print(fmt_block("post (month-end)", r["post_me"]), pass_fail(r["post_me"]))
        print(fmt_block("post (non-month-end)", r["post_nonme"]), pass_fail(r["post_nonme"]))
        print(fmt_block("MATCHED CONTROL (+/-2h)", r["control"]))
        print()

    # ============================================================ EVENT 3: MONTH-END DAY
    print("=" * 100)
    print("EVENT 3: MONTH-END DAY (full-day return, 00:00->24:00 London calendar day)")
    print("=" * 100)
    me_results = {}
    for name, df in dfs.items():
        r = run_month_end_day(name, df, atrs[name])
        me_results[name] = r
        print(name)
        print(fmt_block("last trading day of month", r["me"]), pass_fail(r["me"]))
        print(fmt_block("ordinary day", r["nonme"]), pass_fail(r["nonme"]))
        print()

    # ============================================================ EVENT 4: GOTOBI (positive control)
    print("=" * 100)
    print("EVENT 4: GOTOBI positive control (USDJPY only, 09:00-09:55 JST)")
    print("=" * 100)
    gb = run_gotobi(dfs["USDJPY"], atrs["USDJPY"])
    print(fmt_block("gotobi day", gb["gotobi"]), pass_fail(gb["gotobi"]))
    print(fmt_block("other day", gb["other"]), pass_fail(gb["other"]))
    if gb["gotobi"].get("n", 0) > 0 and gb["gotobi"]["pips_mean"] > 0 and pass_fail(gb["gotobi"]) == "PASS":
        print("  -> HARNESS VALIDATED: gotobi shows the expected positive drift.")
    else:
        print("  -> *** HARNESS SUSPECT ***: gotobi (a known-real edge) did not show the expected "
              "positive/significant drift. Treat ALL other results in this run with caution -- "
              "this likely reflects a methodology bug, not that gotobi is dead.")
    print()

    # ============================================================ ROBUSTNESS: halves + per-year sign
    print("=" * 100)
    print(f"ROBUSTNESS: 2 halves (early < {split_date.date()} vs late >= {split_date.date()}) "
          f"+ per-year sign consistency, for anything clearing the pass bar above")
    print("=" * 100)

    def robustness_report(tag, pips_series):
        m1, m2, stable = half_sign_stable(pips_series, split_date)
        if m1 is None:
            print(f"  {tag:<40s} insufficient data for half-split")
            return
        yearly, pos, neg, tot = per_year_signs(pips_series)
        print(f"  {tag:<40s} half1_mean={m1:+7.3f}p  half2_mean={m2:+7.3f}p  "
              f"half-stable={'YES' if stable else 'no'}  "
              f"years: +{pos}/-{neg} of {tot}")

    candidates = []
    for name, r in london_results.items():
        for k in ("pre_all", "post_all", "pre_me", "pre_nonme", "post_me", "post_nonme"):
            b = r[k]
            if b.get("n", 0) > 0 and pass_fail(b) == "PASS":
                candidates.append((f"LondonFix {name} {k}", b["pips_series"]))
    for name, r in ny_results.items():
        for k in ("pre_all", "post_all", "pre_me", "pre_nonme", "post_me", "post_nonme"):
            b = r[k]
            if b.get("n", 0) > 0 and pass_fail(b) == "PASS":
                candidates.append((f"NYcut {name} {k}", b["pips_series"]))
    for name, r in me_results.items():
        for k in ("me", "nonme"):
            b = r[k]
            if b.get("n", 0) > 0 and pass_fail(b) == "PASS":
                candidates.append((f"MonthEndDay {name} {k}", b["pips_series"]))
    if gb["gotobi"].get("n", 0) > 0 and pass_fail(gb["gotobi"]) == "PASS":
        candidates.append(("Gotobi USDJPY gotobi", gb["gotobi"]["pips_series"]))
    if gb["other"].get("n", 0) > 0 and pass_fail(gb["other"]) == "PASS":
        candidates.append(("Gotobi USDJPY other", gb["other"]["pips_series"]))

    if not candidates:
        print("  (nothing cleared the 2x-cost pass bar -- no robustness table to show)")
    else:
        for tag, series in candidates:
            robustness_report(tag, series)
    print()
    print("=" * 100)
    print("END OF RUN")
    print("=" * 100)


if __name__ == "__main__":
    main()
