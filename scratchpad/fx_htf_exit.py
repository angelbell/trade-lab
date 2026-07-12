"""fx_htf_exit.py -- test the user's hypothesis that the USDJPY 1h Pattern-B breakout's thin
body (scratchpad/usdjpy_1h_funnel.py: n=861, meanR net +0.074, totR/yr +2.39, maxDD 43.1R) is
the FIXED RR3 exit's fault, not the entry. Swaps ONLY the exit engine -- entries and the initial
structural stop stay bit-identical to the funnel base (Pattern B, zigzag zz-k 2.0, trend-ema 80,
bo-window 20, fwd 500, cost 0.9pip net) -- and rides winners to the nearest HTF support/resistance
level instead, bailing on a structure-break (reversal) instead of at a fixed R-multiple.

ENTRY SETS (both re-use breakout_wave.run() verbatim; the exit-engine columns are re-walked here
with a hand-rolled forward walker so hold-bars / level-touch / reversal-touch can all be tracked,
then cross-checked per-trade against run()'s own RR3 numbers as the tie-back proof):
  PRIMARY   USDJPY 1h  LONG   (data/vantage_usdjpy_h1.csv, full history)
  SECONDARY GBPUSD 4h  SHORT  (mirror control via price inversion, data/vantage_gbpusd_h1.csv
                                resampled 4h; D2 base n=228 meanR(net) +0.129)

HTF LEVEL INVENTORY (causal, no lookahead):
  - 4h-resample ZigZag swing highs (same zz-k 2.0 x ATR machine as the entry skeleton)
  - daily-resample ZigZag swing highs (same machine)
  - prior COMPLETED day's high (resample 1D, shift(1), ffill)
  - prior COMPLETED week's high (resample 1W, shift(1), ffill)
  A ZigZag pivot becomes usable starting at the NEXT bar of its own TF after the bar that
  confirmed it (matches breakout_wave.run()'s own convention: entries search from cL2+1, one
  bar after the confirming swing). Day/week highs use resample(label='left')+shift(1)+ffill,
  the same idiom as every other daily gate in this repo (usdjpy_1h_funnel.gate_daily_sma etc).
  For a LONG the target = nearest level ABOVE entry that is >= 0.5R away; if none within 20R,
  no level target for that trade.
  SHORT (GBPUSD) runs the identical machinery on price-inverted bars (p' = C - p, C = 2*max(raw
  1h high), inverted ONCE at the finest 1h grain then resampled -- algebraically identical to
  inverting after resampling, see short_mirror_15m.py / d2_fx_reexam.py), so "swing highs
  above entry" in inverted space are the mirror of "swing lows below entry" in real short space.
  No special-casing needed anywhere downstream.

REVERSAL EXIT (structure flip, causal): the entry-TF's own confirmed ZigZag swing LOWS are
tracked bar by bar (same avail-next-bar convention). If a bar CLOSES below the most recently
confirmed swing low, the position exits at the FOLLOWING bar's OPEN. The initial structural
stop stays live throughout; same-bar conflicts resolve stop-first (conservative), then level-TP,
then reversal-signal-for-next-bar -- exactly mirroring breakout_wave.run()'s own stop->target->
regime-exit priority order.

EXIT COLUMNS (same entries/stop/cost=0.9pip net throughout):
  E0 baseline  fixed RR3 (re-walked here AND cross-checked per-trade against run()'s own output)
  E1           TP @ nearest HTF level + reversal exit + stop (the user's full design; no-level
               trades fall back to reversal-only)
  E2           reversal exit ONLY, no TP (pure ride-until-flip)
  E3           TP @ HTF level ONLY, no reversal (decomposition column; no-level trades just ride
               to the stop or the time cap)

Run:
  .venv/bin/python scratchpad/fx_htf_exit.py --smoke   (USDJPY 2018-> subset, E0+E1 only)
  .venv/bin/python scratchpad/fx_htf_exit.py            (full history, both entry sets, all cols)
Tee to scratchpad/out_fx_htf_exit.txt by the caller.
"""
import os, sys, io, contextlib, warnings, argparse, time as _time
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, swings_zigzag

np.random.seed(20260712)

FWD = 500
ZZ_K = 2.0
ATR_LEN = 14
LEVEL_MIN_R = 0.5
LEVEL_MAX_R = 20.0

BASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, sl_b="swinglow", sl_b_k=1.5,
            swing="zigzag", zz_k=ZZ_K, pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26,
            trend_ema=80, bo_window=20, tp_mode="rr", rr=3.0, atr=ATR_LEN, cost=0.0, swap_pct=0.0,
            fwd=FWD, peryear=False, start=None, end=None, daily_sma=0, daily_slope_k=0,
            gate_tf="1D", risk=0.01, gate_kama=0, gate_kama_tf="1D", gate_kama_tf2="",
            ext_cap=0.0, retest=0, retest_tol=0.10, pullback_frac=0.0, max_pos=1, exec_split=0,
            exit_kama=0, exit_kama_tf="1D", tp1_frac=0.0, tp1_rr=1.0, tp1_be=1,
            wave="all", dump_trades=False, tf="", csv="")

ERAS = [("<=2008", None, "2008-12-31"), ("2009-2017", "2009-01-01", "2017-12-31"),
        ("2018-", "2018-01-01", None)]


def era_bounds(y):
    if y <= 2008:
        return "<=2008"
    if y <= 2017:
        return "2009-2017"
    return "2018-"


# ------------------------------------------------------------------ data / inversion
def invert(d):
    C = 2 * d["high"].max()
    return pd.DataFrame({"open": C - d["open"], "high": C - d["low"],
                          "low": C - d["high"], "close": C - d["close"]}, index=d.index)


def load_csv(name, start=None):
    with contextlib.redirect_stderr(io.StringIO()):
        d = load_mt5_csv(os.path.join(ROOT, "data", f"vantage_{name}_h1.csv"))
    if start is not None:
        d = d[d.index >= start]
    return d


# ------------------------------------------------------------------ ZigZag pivots -> causal availability
def zigzag_avail(bars, zz_k=ZZ_K, atr_len=ATR_LEN):
    """Run swings_zigzag on `bars`. Return DataFrame [avail_time, price, kind] where avail_time
    is the START of the NEXT bar in bars' own index after the confirming bar -- the earliest
    instant this pivot is usable without lookahead (matches breakout_wave.run()'s cL2+1 convention)."""
    h, l, c = bars["high"].values, bars["low"].values, bars["close"].values
    a = ta.atr(bars["high"], bars["low"], bars["close"], length=atr_len).values
    sw = swings_zigzag(h, l, a, zz_k)
    idx = bars.index
    n = len(idx)
    rows = []
    for (ci, pi, price, kind) in sw:
        j = ci + 1
        avail = idx[j] if j < n else idx[-1] + (idx[-1] - idx[-2] if n >= 2 else pd.Timedelta(0))
        rows.append((avail, price, kind))
    if not rows:
        return pd.DataFrame(columns=["avail_time", "price", "kind"])
    out = pd.DataFrame(rows, columns=["avail_time", "price", "kind"])
    return out.sort_values("avail_time").reset_index(drop=True)


def recent_low_array(d_entry, zz_avail):
    """For every bar of d_entry, the most-recently-available confirmed swing LOW price (NaN if
    none yet). Vectorized via merge_asof (causal: 'backward' = last avail_time <= bar time)."""
    lows = zz_avail[zz_avail["kind"] == -1][["avail_time", "price"]].sort_values("avail_time")
    bar_times = pd.DataFrame({"time": d_entry.index})
    if lows.empty:
        return np.full(len(d_entry), np.nan)
    m = pd.merge_asof(bar_times, lows, left_on="time", right_on="avail_time", direction="backward")
    return m["price"].values


# ------------------------------------------------------------------ prior-day / prior-week high
def prior_period_high(d1h, rule, entry_index):
    h = d1h["high"].resample(rule, label="left", closed="left").max().dropna()
    h_avail = h.shift(1)
    return h_avail.reindex(entry_index, method="ffill").values


# ------------------------------------------------------------------ nearest-level-above-entry
def nearest_levels(t, cand_4h, cand_1d, day_high_arr, week_high_arr):
    """cand_4h / cand_1d: DataFrame[avail_time, price] (kind==+1 only, already filtered).
    day_high_arr / week_high_arr: arrays aligned with t (already looked up at each entry's bar).
    Returns level_px array (NaN = no level within [0.5R, 20R])."""
    c4t = cand_4h["avail_time"].values; c4p = cand_4h["price"].values
    c1t = cand_1d["avail_time"].values; c1p = cand_1d["price"].values
    out = np.full(len(t), np.nan)
    times = t["time"].values
    e_px = t["e_px"].values
    risk = t["risk"].values
    for i in range(len(t)):
        et, e, r = times[i], e_px[i], risk[i]
        lo = e + LEVEL_MIN_R * r
        hi = e + LEVEL_MAX_R * r
        cands = []
        m4 = c4t <= et
        if m4.any():
            cands.append(c4p[m4])
        m1 = c1t <= et
        if m1.any():
            cands.append(c1p[m1])
        dh, wh = day_high_arr[i], week_high_arr[i]
        extra = [x for x in (dh, wh) if np.isfinite(x)]
        if extra:
            cands.append(np.array(extra))
        if not cands:
            continue
        allc = np.concatenate(cands)
        allc = allc[(allc >= lo) & (allc <= hi)]
        if len(allc):
            out[i] = allc.min()
    return out


# ------------------------------------------------------------------ generic forward walker
def walk_exit(entry_i, e_px, stop_px, level_px, use_level, use_reversal,
              o, h, l, c, recent_low, fwd=FWD):
    """One trade, one exit-engine config. Priority per bar: stop (intrabar) -> level TP
    (intrabar, if use_level & finite) -> reversal signal on CLOSE, executed at the FOLLOWING
    bar's open. Time cap = close of last bar in window. Returns (R_gross, exit_bar_i, reason)."""
    n = len(c)
    risk = e_px - stop_px
    last_j = min(entry_i + fwd, n - 1)
    pending = False
    for j in range(entry_i + 1, last_j + 1):
        if pending:
            return (o[j] - e_px) / risk, j, "reversal"
        if l[j] <= stop_px:
            return (stop_px - e_px) / risk, j, "stop"
        if use_level and np.isfinite(level_px) and h[j] >= level_px:
            return (level_px - e_px) / risk, j, "level_tp"
        if use_reversal and np.isfinite(recent_low[j]) and c[j] < recent_low[j]:
            if j == last_j:
                return (c[j] - e_px) / risk, j, "reversal_capped"
            pending = True
    return (c[last_j] - e_px) / risk, last_j, "timecap"


def reach_before_stop(entry_i, e_px, stop_px, level_px, h, l, fwd=FWD):
    """Diagnostic: does price touch level_px BEFORE stop_px (same-bar tie -> stop wins, same
    conservative convention as the exit engines)? Only meaningful when level_px is finite."""
    if not np.isfinite(level_px):
        return np.nan
    n = len(h)
    last_j = min(entry_i + fwd, n - 1)
    for j in range(entry_i + 1, last_j + 1):
        if l[j] <= stop_px:
            return False
        if h[j] >= level_px:
            return True
    return False


# ------------------------------------------------------------------ build one entry-set (all engines)
def build_entry_set(tag, d1h_raw, tf, invert_first, pip, ref=None):
    print(f"=== {tag}: building entry set ===")
    d1h = invert(d1h_raw) if invert_first else d1h_raw
    d_entry = resample(d1h, tf) if tf != "1h" else d1h

    args = SimpleNamespace(**BASE)
    with contextlib.redirect_stdout(io.StringIO()):
        t = run(d_entry, args)
    if t is None or len(t) < 5:
        print("  ERROR: base cell produced <5 trades"); return None
    t = t.copy()
    t["time"] = pd.DatetimeIndex(t["time"])
    t = t.sort_values("time").reset_index(drop=True)
    t["stop_px"] = t["e_px"] - t["risk"]
    entry_i = d_entry.index.get_indexer(t["time"])
    assert (entry_i >= 0).all(), f"{tag}: some entry timestamps not found in resampled index"
    t["entry_i"] = entry_i

    rt_cost = 0.9 * pip
    t["Rg_run"] = t["R"].values                       # run()'s own gross RR3 R (cost=0 in BASE)
    t["Rn_run"] = t["Rg_run"] - rt_cost / t["risk"].values

    span_yr = max((d1h.index[-1] - d1h.index[0]).days / 365.25, 0.25)
    print(f"  underlying 1h span: {d1h.index[0]} -> {d1h.index[-1]}  ({span_yr:.1f}yr)  "
          f"n_entries={len(t)}")

    # tie-back check
    if ref is not None:
        flags = []
        if abs(len(t) - ref["n"]) > 5:
            flags.append(f"n {len(t)} vs ref {ref['n']}")
        if abs(t['Rn_run'].mean() - ref["meanR"]) > 0.01:
            flags.append(f"meanR(net) {t['Rn_run'].mean():.3f} vs ref {ref['meanR']:.3f}")
        if flags:
            print(f"  !! TIE-BACK DIVERGENCE: {'; '.join(flags)}")
        else:
            print(f"  TIE-BACK MATCH: n={len(t)} meanR(net)={t['Rn_run'].mean():+.3f} "
                  f"(ref n={ref['n']} meanR={ref['meanR']:+.3f})")

    # ---- HTF level inventory ----
    zz_4h = zigzag_avail(resample(d1h, "4h"))
    zz_1d = zigzag_avail(resample(d1h, "1D"))
    cand_4h = zz_4h[zz_4h["kind"] == 1][["avail_time", "price"]]
    cand_1d = zz_1d[zz_1d["kind"] == 1][["avail_time", "price"]]
    day_high_full = prior_period_high(d1h, "1D", d_entry.index)
    week_high_full = prior_period_high(d1h, "1W", d_entry.index)
    day_high_arr = day_high_full[t["entry_i"].values]
    week_high_arr = week_high_full[t["entry_i"].values]
    t["level_px"] = nearest_levels(t, cand_4h, cand_1d, day_high_arr, week_high_arr)
    t["level_R"] = (t["level_px"] - t["e_px"]) / t["risk"]

    # ---- reversal-exit swing-low series on the entry TF ----
    zz_entry = zigzag_avail(d_entry)
    rlow = recent_low_array(d_entry, zz_entry)

    o = d_entry["open"].values; h = d_entry["high"].values
    l = d_entry["low"].values; c = d_entry["close"].values

    # ---- reach-rate diagnostic ----
    reach = []
    for i in range(len(t)):
        reach.append(reach_before_stop(int(t["entry_i"].iloc[i]), t["e_px"].iloc[i],
                                        t["stop_px"].iloc[i], t["level_px"].iloc[i], h, l))
    t["reach"] = reach

    # ---- E0: my own walker, RR3 fixed target -- cross-check against run()'s own R ----
    e0 = []
    for i in range(len(t)):
        ei = int(t["entry_i"].iloc[i]); e_px = t["e_px"].iloc[i]; stop_px = t["stop_px"].iloc[i]
        tgt = e_px + args.rr * t["risk"].iloc[i]
        Rg, xj, reason = walk_exit(ei, e_px, stop_px, tgt, True, False, o, h, l, c, rlow)
        e0.append((Rg, xj - ei, reason))
    t["E0_Rg"], t["E0_hold"], t["E0_reason"] = zip(*e0)
    mismatch = (t["E0_Rg"] - t["Rg_run"]).abs() > 1e-6
    if mismatch.any():
        print(f"  !! E0 WALKER MISMATCH vs run(): {mismatch.sum()}/{len(t)} trades differ "
              f"(max |diff|={ (t['E0_Rg']-t['Rg_run']).abs().max():.4f}) -- INVESTIGATE")
    else:
        print(f"  E0 walker cross-check: bit-identical to run()'s own RR3 R on all {len(t)} trades.")

    # ---- E1/E2/E3 ----
    for tagcol, use_level, use_reversal in (("E1", True, True), ("E2", False, True), ("E3", True, False)):
        rows = []
        for i in range(len(t)):
            ei = int(t["entry_i"].iloc[i]); e_px = t["e_px"].iloc[i]; stop_px = t["stop_px"].iloc[i]
            lvl = t["level_px"].iloc[i]
            Rg, xj, reason = walk_exit(ei, e_px, stop_px, lvl, use_level, use_reversal, o, h, l, c, rlow)
            rows.append((Rg, xj - ei, reason))
        t[f"{tagcol}_Rg"], t[f"{tagcol}_hold"], t[f"{tagcol}_reason"] = zip(*rows)

    for col in ("E0", "E1", "E2", "E3"):
        t[f"{col}_Rn"] = t[f"{col}_Rg"] - rt_cost / t["risk"]

    return dict(t=t, span_yr=span_yr, d_entry=d_entry)


# ------------------------------------------------------------------ stats
def cell_stats(t, rcol, span_yr):
    n = len(t)
    if n < 5:
        return None
    r = t[rcol].values
    ts = t["time"]
    win = (r > 0).mean() * 100
    pos = r[r > 0].sum(); neg = abs(r[r <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(eq) - eq).max() if n else 0.0
    yr = ts.dt.year.values
    ys = np.unique(yr)
    green = sum(r[yr == y].sum() > 0 for y in ys)
    era_tot = {}
    for tag in ("<=2008", "2009-2017", "2018-"):
        m = np.array([era_bounds(y) == tag for y in yr])
        era_tot[tag] = r[m].sum() if m.any() else 0.0
    totR_yr = r.sum() / span_yr
    cdd = totR_yr / dd if dd > 0 else np.inf
    half = n // 2
    is_mean = r[:half].mean() if half >= 3 else np.nan
    oos_mean = r[half:].mean() if (n - half) >= 3 else np.nan
    holdcol = rcol.split("_")[0] + "_hold" if "_" in rcol else None
    hold_med = t[holdcol].median() if holdcol and holdcol in t.columns else np.nan
    return dict(n=n, n_yr=n / span_yr, win=win, pf=pf, meanR=r.mean(), medianR=np.median(r),
                stdR=r.std(), totR=r.sum(), totR_yr=totR_yr, maxDD=dd, cdd=cdd,
                green_frac=green / len(ys), n_years=len(ys), era_tot=era_tot,
                is_mean=is_mean, oos_mean=oos_mean, hold_med=hold_med)


def fmt_stats(tag, s):
    if s is None:
        return f"  {tag:<10} n<5"
    e = s["era_tot"]
    is_oos = f"IS={s['is_mean']:+.3f}/OOS={s['oos_mean']:+.3f}" if not np.isnan(s['is_mean']) else "IS/OOS n/a"
    return (f"  {tag:<10} n={s['n']:>4} n/yr={s['n_yr']:>5.1f} win={s['win']:>4.0f}% "
            f"PF={s['pf']:>5.2f} meanR={s['meanR']:>+.3f}(med{s['medianR']:>+.3f}/sd{s['stdR']:.2f}) "
            f"totR/yr={s['totR_yr']:>+6.2f} maxDD={s['maxDD']:>6.1f}R CAGR/DD={s['cdd']:>+5.2f} "
            f"hold(bar)med={s['hold_med']:>5.1f} grnYr={s['green_frac']*100:>4.0f}%({s['n_years']}) "
            f"era[<=08/09-17/18-]=[{e['<=2008']:+.1f}/{e['2009-2017']:+.1f}/{e['2018-']:+.1f}] {is_oos}")


def report_entry_set(tag, res, cols):
    t, span_yr = res["t"], res["span_yr"]
    print(f"\n--- {tag}: DIAGNOSTICS (nearest HTF-level distance in R at entry) ---")
    with_lvl = t["level_R"].dropna()
    pct_none = 100 * t["level_px"].isna().mean()
    if len(with_lvl):
        print(f"  n_with_level={len(with_lvl)}/{len(t)}  median={with_lvl.median():.2f}R "
              f"std={with_lvl.std():.2f}R  p10={with_lvl.quantile(.10):.2f}R "
              f"p90={with_lvl.quantile(.90):.2f}R")
    print(f"  % trades with NO level within {LEVEL_MAX_R:.0f}R: {pct_none:.1f}%")
    reach_sub = t["reach"].dropna()
    reach_rate = 100 * reach_sub.mean() if len(reach_sub) else float("nan")
    print(f"  reach-rate (MFE touches level before stop, among trades WITH a level): "
          f"{reach_rate:.1f}%  (n={len(reach_sub)})")

    print(f"\n--- {tag}: RESULTS ---")
    for col in cols:
        s = cell_stats(t, f"{col}_Rn", span_yr)
        print(fmt_stats(col, s))
    return {col: cell_stats(t, f"{col}_Rn", span_yr) for col in cols}, reach_rate


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = _time.time()

    # ---- PRIMARY: USDJPY 1h LONG ----
    start = "2018-01-01" if args.smoke else None
    usdjpy = load_csv("usdjpy", start=start)
    ref_primary = dict(n=861, meanR=0.074) if not args.smoke else None
    res_p = build_entry_set("PRIMARY USDJPY 1h LONG", usdjpy, "1h", invert_first=False,
                             pip=0.01, ref=ref_primary)
    if res_p is None:
        print("PRIMARY entry set failed -- aborting"); return
    cols = ["E0", "E1"] if args.smoke else ["E0", "E1", "E2", "E3"]
    stats_p, reach_p = report_entry_set("PRIMARY USDJPY 1h LONG", res_p, cols)

    if args.smoke:
        print(f"\n(smoke test only -- {_time.time()-t0:.0f}s -- run without --smoke for the full test)")
        return

    # ---- SECONDARY: GBPUSD 4h SHORT (mirror control) ----
    gbpusd = load_csv("gbpusd")
    ref_secondary = dict(n=228, meanR=0.129)
    res_s = build_entry_set("SECONDARY GBPUSD 4h SHORT (mirror)", gbpusd, "4h",
                             invert_first=True, pip=0.0001, ref=ref_secondary)
    stats_s, reach_s = (None, None)
    if res_s is not None:
        stats_s, reach_s = report_entry_set("SECONDARY GBPUSD 4h SHORT (mirror)", res_s, cols)

    # ---- pre-registered verdict ----
    print("\n=== PRE-REGISTERED PASS/KILL CHECK (PRIMARY set is the decider) ===")
    print("  PASS: some exit column beats E0 on BOTH totR/yr AND (totR/yr / maxDD) with "
          "meanR>=+0.15 and IS/OOS same sign (on PRIMARY; SECONDARY agreeing in direction = bonus)")
    print("  KILL: every exit column <= E0 on totR/yr (fixed-RR law extends to FX), "
          "OR reach-rate < 35% (levels too far -- mechanism premise fails)\n")

    e0 = stats_p["E0"]
    print(f"  PRIMARY E0 baseline: totR/yr={e0['totR_yr']:+.2f}  CAGR/DD={e0['cdd']:+.2f}")
    any_pass = False
    for col in ("E1", "E2", "E3"):
        s = stats_p[col]
        if s is None:
            print(f"  PRIMARY {col}: n<5, skipped"); continue
        beats_totR = s["totR_yr"] > e0["totR_yr"]
        beats_cdd = s["cdd"] > e0["cdd"]
        meanR_ok = s["meanR"] >= 0.15
        is_oos_ok = (not np.isnan(s["is_mean"])) and (not np.isnan(s["oos_mean"])) and \
                    np.sign(s["is_mean"]) == np.sign(s["oos_mean"])
        passed = beats_totR and beats_cdd and meanR_ok and is_oos_ok
        any_pass = any_pass or passed
        print(f"  PRIMARY {col}: totR/yr={s['totR_yr']:+.2f}(beat E0={beats_totR}) "
              f"CAGR/DD={s['cdd']:+.2f}(beat E0={beats_cdd}) meanR={s['meanR']:+.3f}(>=0.15={meanR_ok}) "
              f"IS/OOS-agree={is_oos_ok} -> PASS_this_col={passed}")

    all_le = all(stats_p[col]["totR_yr"] <= e0["totR_yr"] for col in ("E1", "E2", "E3") if stats_p[col])
    print(f"\n  reach-rate PRIMARY = {reach_p:.1f}%  ({'>=35% OK' if reach_p >= 35 else '<35% KILL trigger'})")
    if stats_s is not None:
        print(f"  reach-rate SECONDARY = {reach_s:.1f}%")
        print(f"  SECONDARY E0 totR/yr={stats_s['E0']['totR_yr']:+.2f}")
        for col in ("E1", "E2", "E3"):
            s = stats_s[col]
            if s is None:
                continue
            print(f"  SECONDARY {col}: totR/yr={s['totR_yr']:+.2f} "
                  f"(agrees w/ PRIMARY direction={ (s['totR_yr']>stats_s['E0']['totR_yr']) == (stats_p[col]['totR_yr']>e0['totR_yr']) if stats_p[col] else 'n/a'})")

    print(f"\n  any exit column clears the full PASS bar on PRIMARY: {any_pass}")
    print(f"  ALL exit columns <= E0 on totR/yr (PRIMARY): {all_le}")
    kill = all_le or (reach_p < 35)
    print(f"  KILL condition met: {kill}")
    print(f"\n(done in {_time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
