"""D2 (docs/proposals.md): re-examine "FX = trend-dead" -- that verdict rests on USDJPY alone
(n=1, the most managed currency). Now testing ~26yr x 6 USD pairs (EUR/GBP/AUD/NZD/CAD/JPY vs USD).

STAGE 1 -- trend-CHARACTER screen (Hurst / variance-ratio / efficiency-ratio / trend%), reusing the
exact metric functions from research/instrument_screen.py (research.regime_gate_lab.er +
research.regime_statedet.hurst_rs / variance_ratio), split into 3 eras (<=2008 / 2009-2017 / 2018-).

STAGE 2 -- the REAL gold_bo machine (breakout_wave.run), canon Pattern-B config (zigzag k=2,
trend-ema 80, bo-window 20, tp-mode rr, rr=3, fwd=500), NO daily-sma gate (all-signals base).
Grid: 6 pairs x TF{15min,1h,4h,1d} x side{long,short}. Short = price inversion (C=2*high.max()),
exactly as scratchpad/short_mirror_15m.py. Cost: net round-trip 0.9 pips, subtracted per-trade via
R_net = R - rt_cost/risk (rt_cost in price units: 0.0001*0.9 non-JPY, 0.01*0.9 JPY pairs).

Run: .venv/bin/python scratchpad/d2_fx_reexam.py            (full grid)
     .venv/bin/python scratchpad/d2_fx_reexam.py --smoke     (1 pair x 1 tf x 1 side)
Output tee'd to scratchpad/out_d2_fx.txt by the caller.
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
from research.regime_gate_lab import er
from research.regime_statedet import hurst_rs, variance_ratio

PAIRS = ["eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdjpy"]
JPY_PAIRS = {"usdjpy"}
ERAS = [("<=2008", None, "2008-12-31"), ("2009-2017", "2009-01-01", "2017-12-31"),
        ("2018-", "2018-01-01", None)]

BASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, sl_b="swinglow", sl_b_k=1.5,
            swing="zigzag", zz_k=2.0, pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26,
            trend_ema=80, bo_window=20, tp_mode="rr", rr=3.0, atr=14, cost=0.0, swap_pct=0.0,
            fwd=500, peryear=False, start=None, end=None, daily_sma=0, daily_slope_k=0,
            gate_tf="1D", risk=0.01, gate_kama=0, gate_kama_tf="1D", gate_kama_tf2="",
            ext_cap=0.0, retest=0, retest_tol=0.10, pullback_frac=0.0, max_pos=1, exec_split=0,
            exit_kama=0, exit_kama_tf="1D", tp1_frac=0.0, tp1_rr=1.0, tp1_be=1,
            wave="all", dump_trades=False, tf="", csv="")


# ------------------------------------------------------------------ span check
def print_spans():
    print("=== FILE SPANS (actual, verify before trusting the '~26yr x 6 pairs' claim) ===")
    for p in PAIRS:
        for tf in ("m15", "h1", "d1"):
            path = os.path.join(ROOT, "data", f"vantage_{p}_{tf}.csv")
            if not os.path.exists(path):
                print(f"  {p:<8}{tf:<5} MISSING"); continue
            with contextlib.redirect_stderr(io.StringIO()):
                d = load_mt5_csv(path)
            span_yr = (d.index[-1] - d.index[0]).days / 365.25
            print(f"  {p:<8}{tf:<5} n={len(d):>8}  {d.index[0].date()} -> {d.index[-1].date()}  ({span_yr:.1f}yr)")
    print()


# ------------------------------------------------------------------ stage 1: character
def era_slice(s, lo, hi):
    return s.loc[lo:hi]


def char_metrics(close):
    close = close.dropna()
    if len(close) < 60:
        return None
    r = np.log(close).diff().dropna().values
    H = hurst_rs(r)
    VR10 = variance_ratio(r, 10)
    VR20 = variance_ratio(r, 20)
    ER = er(close, 20).mean()
    s200 = close.rolling(200).mean()
    trend = ((close > s200) & (s200 > s200.shift(20))).mean()
    return dict(n=len(close), H=H, VR10=VR10, VR20=VR20, ER=ER, trend=trend)


def stage1(density_verbose=False):
    print("=== STAGE 1: trend-CHARACTER screen (daily bars; USDJPY = reference row) ===")
    print("    metric functions REUSED verbatim from research/instrument_screen.py "
          "(research.regime_gate_lab.er + research.regime_statedet.hurst_rs/variance_ratio)\n")
    header = f"  {'pair':<8}{'era':<11}{'n':>6}{'Hurst':>7}{'VR10':>6}{'VR20':>6}{'ER':>6}{'trend%':>7}"
    print(header)
    rows = []
    for p in PAIRS:
        path = os.path.join(ROOT, "data", f"vantage_{p}_d1.csv")
        with contextlib.redirect_stderr(io.StringIO()):
            d = load_mt5_csv(path)
        close = d["close"]
        for era_tag, lo, hi in ERAS:
            m = char_metrics(era_slice(close, lo, hi))
            if m is None:
                print(f"  {p:<8}{era_tag:<11}  too few bars"); continue
            tag = f"{p.upper():<8}{era_tag:<11}"
            ref = " <-- REF" if p == "usdjpy" else ""
            print(f"  {tag}{m['n']:>6}{m['H']:>7.2f}{m['VR10']:>6.2f}{m['VR20']:>6.2f}"
                  f"{m['ER']:>6.2f}{m['trend']*100:>6.0f}%{ref}")
            rows.append(dict(pair=p, era=era_tag, **m))
        # full-history row too
        m = char_metrics(close)
        if m is not None:
            tag = f"{p.upper():<8}{'FULL':<11}"
            print(f"  {tag}{m['n']:>6}{m['H']:>7.2f}{m['VR10']:>6.2f}{m['VR20']:>6.2f}"
                  f"{m['ER']:>6.2f}{m['trend']*100:>6.0f}%")
        print()
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stage 2: real method
def invert(d):
    C = 2 * d["high"].max()
    return pd.DataFrame({"open": C - d["open"], "high": C - d["low"],
                          "low": C - d["high"], "close": C - d["close"]}, index=d.index)


def density_guard_m15(d, min_bars_per_day=60):
    """m15 files have sparse early years; keep only from the first date where the trailing
    30-day median bar-count clears the threshold (mirrors scratchpad/short_mirror_15m.py)."""
    cnt = d.groupby(d.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= min_bars_per_day]
    if len(ok) == 0:
        return d
    start = ok.index[0]
    return d[d.index.date >= start]


def load_tf(pair, tf):
    """Return the OHLC frame at the requested TF, per the task's routing rule:
    15min <- m15 file (density-guarded); 1h/4h/1d <- h1 file."""
    if tf == "15min":
        path = os.path.join(ROOT, "data", f"vantage_{pair}_m15.csv")
        with contextlib.redirect_stderr(io.StringIO()):
            d = load_mt5_csv(path)
        d = density_guard_m15(d)
        eff_start = d.index[0]
        d = resample(d, "15min")
    else:
        path = os.path.join(ROOT, "data", f"vantage_{pair}_h1.csv")
        with contextlib.redirect_stderr(io.StringIO()):
            d = load_mt5_csv(path)
        eff_start = d.index[0]
        d = resample(d, tf)
    return d, eff_start


def era_bounds(y):
    if y <= 2008:
        return "<=2008"
    if y <= 2017:
        return "2009-2017"
    return "2018-"


def cell_stats(t_net, gross, ts, span_yr):
    """t_net = net R array, gross = gross R array (pre-cost), ts = entry timestamps."""
    n = len(t_net)
    if n < 5:
        return None
    win = (t_net > 0).mean() * 100
    pos = t_net[t_net > 0].sum()
    neg = abs(t_net[t_net <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    eq = np.cumsum(t_net)
    dd = (np.maximum.accumulate(eq) - eq).max() if n else 0.0
    yr = ts.year.values
    ys = np.unique(yr)
    green = sum(t_net[yr == y].sum() > 0 for y in ys)
    era_tot = {}
    for tag in ("<=2008", "2009-2017", "2018-"):
        m = np.array([era_bounds(y) == tag for y in yr])
        era_tot[tag] = t_net[m].sum() if m.any() else 0.0
    return dict(n=n, n_yr=n / span_yr, win=win, pf=pf, meanR_net=t_net.mean(),
                meanR_gross=gross.mean(), totR_yr=t_net.sum() / span_yr, maxDD=dd,
                green_frac=green / len(ys), n_years=len(ys), era_tot=era_tot)


def run_cell(pair, tf, side, verbose_errors=True):
    d, eff_start = load_tf(pair, tf)
    if side == "short":
        d = invert(d)
    span_yr = max((d.index[-1] - d.index[0]).days / 365.25, 0.25)
    args = SimpleNamespace(**{**BASE, "tf": tf})
    with contextlib.redirect_stdout(io.StringIO()):
        t = run(d, args)
    if t is None or len(t) < 5:
        return None, eff_start, span_yr
    pip = 0.01 if pair in JPY_PAIRS else 0.0001
    rt_cost = 0.9 * pip
    gross = t["R"].values
    net = gross - rt_cost / t["risk"].values
    ts = pd.DatetimeIndex(t["time"])
    stats = cell_stats(net, gross, ts, span_yr)
    return stats, eff_start, span_yr


def fmt_cell(pair, tf, side, stats, eff_start):
    if stats is None:
        return (f"  {pair.upper():<8}{tf:<7}{side:<6} n<5 or no entries "
                f"(eff_start={eff_start.date()})")
    e = stats["era_tot"]
    return (f"  {pair.upper():<8}{tf:<7}{side:<6} "
            f"n={stats['n']:>4} n/yr={stats['n_yr']:>5.1f} win={stats['win']:>4.0f}% "
            f"PF={stats['pf']:>5.2f} meanR(net)={stats['meanR_net']:>+.3f} "
            f"meanR(gross)={stats['meanR_gross']:>+.3f} totR/yr={stats['totR_yr']:>+6.1f} "
            f"maxDD={stats['maxDD']:>6.1f}R grnYr={stats['green_frac']*100:>4.0f}%"
            f"({stats['n_years']}) era[<=08/09-17/18-]=[{e['<=2008']:+.0f}/{e['2009-2017']:+.0f}/{e['2018-']:+.0f}]"
            f" eff_start={eff_start.date()}")


def stage2(smoke=False):
    print("=== STAGE 2: real gold_bo machine (Pattern-B, zigzag k=2, trend-ema80, bo-win20, "
          "rr=3, fwd=500, NO daily-sma gate = all-signals base) ===")
    print("    cost: net round-trip 0.9 pips (0.01 JPY / 0.0001 non-JPY), R_net = R - rt_cost/risk\n")
    pairs = PAIRS[:1] if smoke else PAIRS
    tfs = ["15min"] if smoke else ["15min", "1h", "4h", "1d"]
    sides = ["long"] if smoke else ["long", "short"]
    results = {}
    for pair in pairs:
        for tf in tfs:
            for side in sides:
                try:
                    stats, eff_start, span_yr = run_cell(pair, tf, side)
                except Exception as ex:
                    print(f"  {pair.upper():<8}{tf:<7}{side:<6} ERROR: {ex}")
                    continue
                results[(pair, tf, side)] = stats
                print(fmt_cell(pair, tf, side, stats, eff_start))
        print()
    return results


def summarize(results):
    print("=== SUMMARY ===")
    rows = []
    for (pair, tf, side), s in results.items():
        if s is None:
            continue
        rows.append(dict(pair=pair, tf=tf, side=side, **s))
    if not rows:
        print("  no cells with >=5 trades"); return
    df = pd.DataFrame(rows)

    print("\n-- top 10 cells by meanR(net) --")
    top = df.sort_values("meanR_net", ascending=False).head(10)
    for _, r in top.iterrows():
        e = r["era_tot"]
        print(f"  {r['pair'].upper():<8}{r['tf']:<7}{r['side']:<6} n={r['n']:>4} n/yr={r['n_yr']:>5.1f} "
              f"win={r['win']:>4.0f}% PF={r['pf']:>5.2f} meanR(net)={r['meanR_net']:>+.3f} "
              f"meanR(gross)={r['meanR_gross']:>+.3f} totR/yr={r['totR_yr']:>+6.1f} maxDD={r['maxDD']:>6.1f}R "
              f"grnYr={r['green_frac']*100:>4.0f}%({r['n_years']}) "
              f"era[<=08/09-17/18-]=[{e['<=2008']:+.0f}/{e['2009-2017']:+.0f}/{e['2018-']:+.0f}]")

    print("\n-- pair x TF: net-positive cell count (long vs short, meanR(net) > 0) --")
    piv = df.pivot_table(index="pair", columns=["tf", "side"],
                          values="meanR_net", aggfunc=lambda x: (x > 0).sum())
    print(piv.to_string())

    print("\n-- net-positive fraction by side --")
    for side in ("long", "short"):
        sub = df[df.side == side]
        pos = (sub.meanR_net > 0).sum()
        print(f"  {side}: {pos}/{len(sub)} cells net-positive")

    print("\n-- per-pair: best cell (any tf/side) by meanR(net), + is it USDJPY-shaped? --")
    for pair in PAIRS:
        sub = df[df.pair == pair]
        if len(sub) == 0:
            continue
        best = sub.sort_values("meanR_net", ascending=False).iloc[0]
        e = best["era_tot"]
        eras_nonzero = sum(1 for v in e.values() if abs(v) > 1e-9)
        concentrated = max(e.values(), key=abs)
        conc_frac = abs(concentrated) / (abs(e["<=2008"]) + abs(e["2009-2017"]) + abs(e["2018-"]) + 1e-9)
        print(f"  {pair.upper():<8} best={best['tf']}/{best['side']} meanR(net)={best['meanR_net']:>+.3f} "
              f"PF={best['pf']:.2f} n/yr={best['n_yr']:.1f} grnYr={best['green_frac']*100:.0f}% "
              f"era-conc={conc_frac*100:.0f}% (era[<=08/09-17/18-]=[{e['<=2008']:+.0f}/{e['2009-2017']:+.0f}/{e['2018-']:+.0f}])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-stage1", action="store_true")
    args = ap.parse_args()

    print_spans()
    if not args.skip_stage1:
        stage1()
    results = stage2(smoke=args.smoke)
    if not args.smoke:
        summarize(results)
    else:
        print("(smoke test -- run without --smoke for the full grid)")


if __name__ == "__main__":
    main()
