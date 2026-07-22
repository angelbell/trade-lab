"""usdjpy_1h_funnel.py -- funnel test for the ONE tension cell found by experiments/d2_fx_reexam.py:
USDJPY 1h LONG breakout (Pattern B, zigzag zz-k 2.0, trend-ema 80, bo-window 20, tp-mode rr, rr=3,
fwd=500, NO gate, net cost 0.9 pip) = n=861, N/yr 32.5, win 28% (breakeven 25%), PF 1.10,
meanR net +0.074, era totR (<=2008/09-17/18-26) = +8/+16/+40, maxDD 43R.

Decides whether this cell is worth a full overfit_audit.py pass. Steps:
  STEP 0  tie-back: re-run the exact base cell, flag if it diverges from the numbers above.
  STEP 1  zz-k sweep {1.5, 2.0, 2.5} on the base (plateau vs spike?).
  STEP 2  three regime gates on the zz=2.0 base, post-filtered on the UNGATED trade set
          (faithful: gate only vetoes/keeps entries, does not change which bar is the entry --
          see breakout_wave.run()'s Pattern-B loop, gate checks are plain `continue`s before any
          entry/stop/target math -- so post-filtering the ungated trade-set by the causal gate
          mask at each entry's timestamp reproduces bit-identical results to gating inside run()).
            a. daily SMA150 + slope10 rising   (gold_bo style; breakout_wave reg logic)
            b. daily KAMA(14) rising            (breakout_wave kreg logic)
            c. weekly SMA(30) level              price ABOVE / BELOW prior completed weekly SMA30
                                                  (btc_pull cycle-gate style, computed by hand)
          +-1 neighbor grids for each.
  STEP 3  random-drop null (1000 draws, same kept-%, uniform random trade subset, REPLAYED IN
          TIME ORDER for the equity curve) for any gate cell that looks promising: percentile of
          the gated cell's (totR/maxDD) AND meanR vs the null.

Run:
  .venv/bin/python experiments/usdjpy_1h_funnel.py --smoke     (2018-> subset, gate (a) only)
  .venv/bin/python experiments/usdjpy_1h_funnel.py             (full history, all steps)
Tee to experiments/out_usdjpy_funnel.txt (done by the caller / via `| tee`).
"""
import os, sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_adaptive import kama as kama_adaptive
from research.regime_gate_lab import at

np.random.seed(20260622)

PAIR = "usdjpy"
PIP = 0.01                       # JPY pair
RT_COST = 0.9 * PIP              # net round-trip cost, price units
CSV = os.path.join(ROOT, "data", "vantage_usdjpy_h1.csv")

BASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, sl_b="swinglow", sl_b_k=1.5,
            swing="zigzag", zz_k=2.0, pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26,
            trend_ema=80, bo_window=20, tp_mode="rr", rr=3.0, atr=14, cost=0.0, swap_pct=0.0,
            fwd=500, peryear=False, start=None, end=None, daily_sma=0, daily_slope_k=0,
            gate_tf="1D", risk=0.01, gate_kama=0, gate_kama_tf="1D", gate_kama_tf2="",
            ext_cap=0.0, retest=0, retest_tol=0.10, pullback_frac=0.0, max_pos=1, exec_split=0,
            exit_kama=0, exit_kama_tf="1D", tp1_frac=0.0, tp1_rr=1.0, tp1_be=1,
            wave="all", dump_trades=False, tf="1h", csv="")

ERAS = [("<=2008", None, "2008-12-31"), ("2009-2017", "2009-01-01", "2017-12-31"),
        ("2018-", "2018-01-01", None)]


def era_bounds(y):
    if y <= 2008:
        return "<=2008"
    if y <= 2017:
        return "2009-2017"
    return "2018-"


# ------------------------------------------------------------------ data / base run
def load_1h(start=None):
    with contextlib.redirect_stderr(io.StringIO()):
        d = load_mt5_csv(CSV)
    d = resample(d, "1h")
    if start is not None:
        d = d[d.index >= start]
    return d


def run_base(d, zz_k=2.0):
    """Return the raw (ungated) trades DataFrame for the given zz_k, with gross/net R attached."""
    args = SimpleNamespace(**{**BASE, "zz_k": zz_k})
    with contextlib.redirect_stdout(io.StringIO()):
        t = run(d, args)
    if t is None or len(t) < 5:
        return None
    t = t.copy()
    t["Rg"] = t["R"].values                       # gross (BASE cost=0.0 so R==gross already)
    t["Rn"] = t["Rg"] - RT_COST / t["risk"].values  # net of 0.9-pip round-trip
    t["time"] = pd.DatetimeIndex(t["time"])
    return t.sort_values("time").reset_index(drop=True)


# ------------------------------------------------------------------ stats
def cell_stats(sub, span_yr, rcol="Rn"):
    """sub = trades DataFrame (already filtered). Stats on column `rcol`."""
    n = len(sub)
    if n < 5:
        return None
    r = sub[rcol].values
    ts = sub["time"]
    win = (r > 0).mean() * 100
    pos = r[r > 0].sum()
    neg = abs(r[r <= 0].sum())
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
    return dict(n=n, n_yr=n / span_yr, win=win, pf=pf, meanR=r.mean(), medianR=np.median(r),
                stdR=r.std(), totR=r.sum(), totR_yr=totR_yr, maxDD=dd, cdd=cdd,
                green_frac=green / len(ys), n_years=len(ys), era_tot=era_tot,
                is_mean=is_mean, oos_mean=oos_mean)


def fmt_stats(tag, s):
    if s is None:
        return f"  {tag:<34} n<5"
    e = s["era_tot"]
    is_oos = f"IS={s['is_mean']:+.3f}/OOS={s['oos_mean']:+.3f}" if not np.isnan(s['is_mean']) else "IS/OOS n/a"
    return (f"  {tag:<34} n={s['n']:>4} n/yr={s['n_yr']:>5.1f} win={s['win']:>4.0f}% "
            f"PF={s['pf']:>5.2f} meanR={s['meanR']:>+.3f} totR/yr={s['totR_yr']:>+6.2f} "
            f"maxDD={s['maxDD']:>6.1f}R CAGR/DD={s['cdd']:>+5.2f} "
            f"grnYr={s['green_frac']*100:>4.0f}%({s['n_years']}) "
            f"era[<=08/09-17/18-]=[{e['<=2008']:+.1f}/{e['2009-2017']:+.1f}/{e['2018-']:+.1f}] {is_oos}")


# ------------------------------------------------------------------ gate masks (causal, shift(1)+ffill)
def gate_daily_sma(d, n, slope_k):
    dc = d["close"].resample("1D").last().dropna()
    sma = dc.rolling(n).mean()
    up = dc > sma
    if slope_k > 0:
        up = up & (sma > sma.shift(slope_k))
    return up.shift(1)


def gate_daily_kama(d, n):
    dc = d["close"].resample("1D").last().dropna()
    km = kama_adaptive(dc, n)
    rising = km > km.shift(1)
    return rising.shift(1)


def gate_weekly_sma(d, n):
    """Returns (above, below) causal boolean series at weekly freq, shift(1) applied."""
    w = d["close"].resample("1W").last().dropna()
    w30 = w.rolling(n).mean()
    above = (w > w30).shift(1)
    below = (w < w30).shift(1)
    return above, below


def apply_mask(t, mask_series):
    m = at(mask_series, t["time"])
    return t[m].reset_index(drop=True)


# ------------------------------------------------------------------ random-drop null
def random_drop_null(base_t, gated_t, span_yr, ndraw=1000, rcol="Rn"):
    n_base = len(base_t)
    k = len(gated_t)
    if k < 5 or k >= n_base:
        return None
    r_all = base_t[rcol].values
    ts_order = np.argsort(base_t["time"].values)   # base_t already time-sorted, but be safe
    r_sorted = r_all[ts_order]
    cdd_null, mean_null = [], []
    for _ in range(ndraw):
        idx = np.sort(np.random.choice(n_base, size=k, replace=False))
        r = r_sorted[idx]
        eq = np.cumsum(r)
        dd = (np.maximum.accumulate(eq) - eq).max()
        totR_yr = r.sum() / span_yr
        cdd_null.append(totR_yr / dd if dd > 0 else np.nan)
        mean_null.append(r.mean())
    cdd_null = np.array(cdd_null); mean_null = np.array(mean_null)
    actual_cdd = (gated_t[rcol].sum() / span_yr) / max(
        (np.maximum.accumulate(np.cumsum(gated_t[rcol].values)) - np.cumsum(gated_t[rcol].values)).max(), 1e-9)
    actual_mean = gated_t[rcol].mean()
    pct_cdd = (np.nan_to_num(cdd_null, nan=-1e9) < actual_cdd).mean() * 100
    pct_mean = (mean_null < actual_mean).mean() * 100
    return dict(k=k, n_base=n_base, kept_pct=100 * k / n_base,
                cdd_null_median=np.nanmedian(cdd_null), cdd_null_std=np.nanstd(cdd_null),
                mean_null_median=np.median(mean_null), mean_null_std=np.std(mean_null),
                actual_cdd=actual_cdd, actual_mean=actual_mean,
                pct_cdd=pct_cdd, pct_mean=pct_mean)


def fmt_null(tag, r):
    if r is None:
        return f"  {tag:<28} n/a (too few / too many kept)"
    return (f"  {tag:<28} kept={r['k']}/{r['n_base']} ({r['kept_pct']:.0f}%) | "
            f"actual CAGR/DD={r['actual_cdd']:+.2f} vs null median={r['cdd_null_median']:+.2f} "
            f"std={r['cdd_null_std']:.2f} -> pct={r['pct_cdd']:.1f}%ile | "
            f"actual meanR={r['actual_mean']:+.3f} vs null median={r['mean_null_median']:+.3f} "
            f"std={r['mean_null_std']:.3f} -> pct={r['pct_mean']:.1f}%ile")


# ------------------------------------------------------------------ steps
def step0_tieback(d, span_yr):
    print("=== STEP 0: TIE-BACK (re-run the exact base cell reported by d2_fx_reexam.py) ===")
    t = run_base(d, zz_k=2.0)
    if t is None:
        print("  ERROR: base cell produced <5 trades -- cannot proceed"); sys.exit(1)
    s = cell_stats(t, span_yr, rcol="Rn")
    print(fmt_stats("zz=2.0 (tie-back)", s))
    ref = dict(n=861, n_yr=32.5, win=28, pf=1.10, meanR=0.074,
               era=dict(**{"<=2008": 8, "2009-2017": 16, "2018-": 40}), maxDD=43)
    flags = []
    if abs(s["n"] - ref["n"]) > 5:
        flags.append(f"n {s['n']} vs ref {ref['n']}")
    if abs(s["meanR"] - ref["meanR"]) > 0.01:
        flags.append(f"meanR {s['meanR']:.3f} vs ref {ref['meanR']:.3f}")
    if abs(s["maxDD"] - ref["maxDD"]) > 3:
        flags.append(f"maxDD {s['maxDD']:.1f} vs ref {ref['maxDD']}")
    if flags:
        print(f"  !! DIVERGENCE FLAGGED: {'; '.join(flags)}")
    else:
        print("  MATCH: tie-back numbers agree with the context cell (within tolerance).")
    print()
    return t, s


def step1_sweep(d, span_yr):
    print("=== STEP 1: zz-k SWEEP {1.5, 2.0, 2.5} (all else fixed; the base cell = zz=2.0) ===")
    out = {}
    for k in (1.5, 2.0, 2.5):
        t = run_base(d, zz_k=k)
        s = cell_stats(t, span_yr, rcol="Rn") if t is not None else None
        out[k] = (t, s)
        print(fmt_stats(f"zz={k}", s))
    m15, m20, m25 = out[1.5][1], out[2.0][1], out[2.5][1]
    if all(x is not None for x in (m15, m20, m25)):
        agree = (m15["meanR"] > 0) == (m20["meanR"] > 0) == (m25["meanR"] > 0)
        spread = max(m15["meanR"], m20["meanR"], m25["meanR"]) - min(m15["meanR"], m20["meanR"], m25["meanR"])
        print(f"  read: sign-agree across neighbors = {agree}; meanR spread = {spread:.3f} "
              f"({'looks like a PLATEAU' if agree and spread < 0.06 else 'WIDE spread -- check for a lone spike'})")
    print()
    return out


def neighbor_grid_daily_sma(d, base_t, span_yr):
    print("  -- +-1 neighbor grid: daily SMA {120,150,180} x slope {5,10,15} --")
    for n in (120, 150, 180):
        for sk in (5, 10, 15):
            gate = gate_daily_sma(d, n, sk)
            g = apply_mask(base_t, gate)
            s = cell_stats(g, span_yr, rcol="Rn") if g is not None and len(g) >= 5 else None
            tag = f"SMA{n}/slope{sk}" + ("  <-- base" if (n, sk) == (150, 10) else "")
            print(fmt_stats(f"    {tag}", s))
    print()


def neighbor_grid_kama(d, base_t, span_yr):
    print("  -- +-1 neighbor grid: daily KAMA {10,14,20} --")
    for n in (10, 14, 20):
        gate = gate_daily_kama(d, n)
        g = apply_mask(base_t, gate)
        s = cell_stats(g, span_yr, rcol="Rn") if g is not None and len(g) >= 5 else None
        tag = f"KAMA{n}" + ("  <-- base" if n == 14 else "")
        print(fmt_stats(f"    {tag}", s))
    print()


def neighbor_grid_weekly(d, base_t, span_yr, side):
    print(f"  -- +-1 neighbor grid: weekly SMA {{25,30,35}} ({side}) --")
    for n in (25, 30, 35):
        above, below = gate_weekly_sma(d, n)
        gate = above if side == "above" else below
        g = apply_mask(base_t, gate)
        s = cell_stats(g, span_yr, rcol="Rn") if g is not None and len(g) >= 5 else None
        tag = f"wkSMA{n}({side})" + ("  <-- base" if n == 30 else "")
        print(fmt_stats(f"    {tag}", s))
    print()


def step2_gates(d, base_t, span_yr, smoke=False):
    print("=== STEP 2: REGIME GATES on the zz=2.0 base (post-filter, bit-identical to gating inside run()) ===")
    base_s = cell_stats(base_t, span_yr, rcol="Rn")
    print(fmt_stats("UNGATED (zz=2.0 base)", base_s))
    print()

    results = {}

    print("-- (a) daily SMA150 + slope10 rising (gold_bo style) --")
    gate_a = gate_daily_sma(d, 150, 10)
    ta = apply_mask(base_t, gate_a)
    sa = cell_stats(ta, span_yr, rcol="Rn") if len(ta) >= 5 else None
    print(fmt_stats("gate(a) SMA150/slope10", sa))
    results["a"] = (ta, sa)
    if not smoke:
        neighbor_grid_daily_sma(d, base_t, span_yr)

    print("-- (b) daily KAMA(14) rising --")
    gate_b = gate_daily_kama(d, 14)
    tb = apply_mask(base_t, gate_b)
    sb = cell_stats(tb, span_yr, rcol="Rn") if len(tb) >= 5 else None
    print(fmt_stats("gate(b) KAMA14", sb))
    results["b"] = (tb, sb)
    if not smoke:
        neighbor_grid_kama(d, base_t, span_yr)

    print("-- (c) weekly SMA(30) level: price ABOVE prior completed weekly SMA30 (trend side) --")
    above, below = gate_weekly_sma(d, 30)
    tc_above = apply_mask(base_t, above)
    sc_above = cell_stats(tc_above, span_yr, rcol="Rn") if len(tc_above) >= 5 else None
    print(fmt_stats("gate(c) wkSMA30 ABOVE", sc_above))
    results["c_above"] = (tc_above, sc_above)

    print("-- (c') weekly SMA(30) level: price BELOW prior completed weekly SMA30 (for completeness) --")
    tc_below = apply_mask(base_t, below)
    sc_below = cell_stats(tc_below, span_yr, rcol="Rn") if len(tc_below) >= 5 else None
    print(fmt_stats("gate(c') wkSMA30 BELOW", sc_below))
    results["c_below"] = (tc_below, sc_below)
    if not smoke:
        neighbor_grid_weekly(d, base_t, span_yr, "above")
        neighbor_grid_weekly(d, base_t, span_yr, "below")

    print()
    return base_s, results


def step3_null(base_t, results, span_yr):
    print("=== STEP 3: RANDOM-DROP NULL (1000 draws, same kept-%, replayed in time order) ===")
    print("    run for every gate cell with >=5 kept trades (house lesson: check totR/maxDD null, not just meanR)\n")
    for tag, (t, s) in results.items():
        if s is None:
            print(f"  {tag:<28} skipped (n<5)"); continue
        nr = random_drop_null(base_t, t, span_yr)
        print(fmt_null(tag, nr))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2018-> subset, gate (a) only, skip neighbor grids")
    args = ap.parse_args()

    start = "2018-01-01" if args.smoke else None
    d = load_1h(start=start)
    span_yr = max((d.index[-1] - d.index[0]).days / 365.25, 0.25)
    print(f"=== DATA: {CSV} ({'SMOKE 2018-> subset' if args.smoke else 'FULL HISTORY'}) ===")
    print(f"  n_bars={len(d)}  {d.index[0]} -> {d.index[-1]}  span={span_yr:.1f}yr\n")

    base_t, base_s = step0_tieback(d, span_yr)

    if args.smoke:
        print("=== SMOKE: gate (a) only on the 2018-> subset ===")
        gate_a = gate_daily_sma(d, 150, 10)
        ta = apply_mask(base_t, gate_a)
        sa = cell_stats(ta, span_yr, rcol="Rn") if len(ta) >= 5 else None
        print(fmt_stats("gate(a) SMA150/slope10 [smoke]", sa))
        if sa is not None:
            nr = random_drop_null(base_t, ta, span_yr)
            print(fmt_null("gate(a) [smoke]", nr))
        print("\n(smoke test only -- run without --smoke for the full funnel)")
        return

    sweep = step1_sweep(d, span_yr)
    base_s2, results = step2_gates(d, base_t, span_yr, smoke=False)
    step3_null(base_t, results, span_yr)

    # ------------------------------------------------------------------ pre-registered verdict
    print("=== PRE-REGISTERED PASS/KILL CHECK ===")
    print("  PASS bar: gated meanR>=+0.15 AND green-years>=60% AND (totR/DD) null pct>=90 AND "
          "IS/OOS same sign, on a gate whose +-1 neighbors agree (plateau)")
    print("  KILL bar: gate removes winners (gated totR/yr < ungated totR/yr) across ALL three gates; "
          "OR meanR stays <+0.10 everywhere regardless of gate\n")
    ungated_totR_yr = base_s2["totR_yr"]
    print(f"  ungated totR/yr = {ungated_totR_yr:+.2f}")
    any_pass = False
    all_removed_winners = True
    all_thin = True
    for tag, (t, s) in results.items():
        if s is None:
            continue
        removed = s["totR_yr"] < ungated_totR_yr
        all_removed_winners = all_removed_winners and removed
        if s["meanR"] >= 0.10:
            all_thin = False
        passed = (s["meanR"] >= 0.15 and s["green_frac"] >= 0.60 and
                  (not np.isnan(s["is_mean"])) and (not np.isnan(s["oos_mean"])) and
                  np.sign(s["is_mean"]) == np.sign(s["oos_mean"]))
        any_pass = any_pass or passed
        print(f"  {tag:<10} meanR={s['meanR']:+.3f} green%={s['green_frac']*100:.0f} "
              f"IS/OOS-sign-agree={np.sign(s['is_mean'])==np.sign(s['oos_mean']) if not np.isnan(s['is_mean']) else 'n/a'} "
              f"totR/yr-removed-winners={removed} -> gate-meanR-bar-passed={passed}")
    print(f"\n  any gate clears the meanR/green/IS-OOS bar (pre-null-check): {any_pass}")
    print(f"  ALL gates remove winners vs ungated: {all_removed_winners}")
    print(f"  ALL gates leave meanR<+0.10: {all_thin}")


if __name__ == "__main__":
    main()
