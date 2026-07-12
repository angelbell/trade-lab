"""btc15m_L_exit_giveback.py -- STAGE 2: post-+2R give-back measurement + PRE-REGISTERED
exit variants on the btc15m_L leg (same construction as scratchpad/btc15m_L_anatomy.py,
which itself copies scratchpad/pwh_adoption.py lines ~123-132 exactly).

The variant grid below is FROZEN (declared before seeing the stage-1 distribution):
  R1..R6 : ratchet = trigger {+1.5R,+2.0R,+2.5R} x floor {BE,+1.0R}. Once the trigger
           level is TOUCHED (bar high), the stop moves to the floor and never trails
           further. Target unchanged.
  S1..S2 : stall-cut at +16 / +32 bars after entry: if mark-to-market close progress
           < 0R at exactly that bar, exit at that bar's close. No other change.
  C1x    : combo = R(trigger +2.0R, floor +1.0R) AND stall-cut at +32 bars.
Pre-declared default for the book test = R trigger+2.0R / floor+1.0R (R4), NOT the
empirical best.

Replay discipline (conservative, causal):
  - Each trade's real 15m path is walked bar-by-bar from entry+1 to the trade's REAL
    exit bar (reconstructed from tb['hold'], validated exact in stage 1 -- no
    re-simulation of the base stop/target, so pullback-fill target geometry cannot
    be reconstructed wrongly).
  - Within a bar the order is: (1) current stop vs low FIRST -- if both the (possibly
    ratcheted) stop and further favourable progress conflict in one bar, the STOP is
    counted as hit first; (2) stall-cut close check; (3) ratchet ARM on the bar's high
    -- the raised stop takes effect from the NEXT bar (a same-bar touch-trigger-then-
    collapse-through-the-floor does NOT save the trade at the floor). All conservative
    against the variants.
  - Variants can only exit EARLIER than base, never later; if the walk reaches the real
    exit bar without a variant exit, the trade books its real base R.
  - NO RE-ARM modeling: early exits would free the single position slot and re-arm new
    signals in live operation; that is NOT modeled here, which UNDERSTATES the early-exit
    variants' live totR. Flagged, not corrected.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from research.portfolio_kama import get_legs
from radar_gate_race import BASE
from btc15m_L_anatomy import build_data, run_leg, pdh_weight, summary, FWD, COST


def invert(d):
    """Price-mirror (copied verbatim from short_mirror_15m.invert -- importing that module
    would execute its module-level screens, so the 3-liner is inlined instead)."""
    C = 2 * d["high"].max()
    return pd.DataFrame({"open": C - d["open"], "high": C - d["low"],
                         "low": C - d["high"], "close": C - d["close"]}, index=d.index)

rng = np.random.default_rng(7)


# --------------------------------------------------------------------------- helpers
def cdd(x):
    """CAGR/DD on a monthly fractional-return stream (copied from pwh_adoption.py)."""
    eq = np.cumprod(1 + x)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    yrs = len(x) / 12
    return (eq[-1] ** (1 / yrs) - 1) / max(dd, 1e-9)


def classify(bars_held, R_gross, exit_is_last_bar):
    """stop / target / timeout from the base outcome (no target px needed)."""
    if abs(R_gross + 1.0) < 1e-9:
        return "stop"
    if bars_held >= FWD or exit_is_last_bar:
        return "timeout"
    return "target" if R_gross > 0 else "timeout"


def per_year_row(Rw, times):
    yr = times.year.values
    return " ".join(f"{y}:{Rw[yr == y].sum():+.0f}" for y in np.unique(yr))


def stream_metrics(Rw, times, label):
    summary(Rw, times, label)
    print(f"      per-year totR: {per_year_row(Rw, times)}")


# --------------------------------------------------------------------------- stage 2a
def touch_stats(h, l, trades, touch, verbose=False):
    """For trades whose path touches entry+touch*risk: worst drawdown-from-peak (R)
    AFTER the touch and before final exit, plus the post-touch minimum low (R).
    Exit-bar conventions: stop-exit bar's low capped at -1R (the realized trough is
    the stop); target-exit bar's high capped at the realized R (no peak beyond the
    realized exit) but its RAW low included (intrabar order unknown -> counts as
    breathing = conservative-deep); timeout bar raw. Same-bar highs update the peak
    BEFORE that bar's low is scored (counts a same-bar collapse as give-back)."""
    rows = []
    for tr in trades:
        i, j_exit, e_px, risk, R_base, kind = tr
        peak = None
        ddmax = 0.0
        minlow = np.inf
        for j in range(i + 1, j_exit + 1):
            rh = (h[j] - e_px) / risk
            rl = (l[j] - e_px) / risk
            if j == j_exit:
                if kind == "stop":
                    rl = max(rl, -1.0)
                elif kind == "target":
                    rh = min(rh, R_base)
            if peak is None:
                if rh >= touch:
                    peak = rh
                    ddmax = max(ddmax, peak - rl)
                    minlow = min(minlow, rl)
            else:
                peak = max(peak, rh)
                ddmax = max(ddmax, peak - rl)
                minlow = min(minlow, rl)
        if peak is not None:
            rows.append((kind, ddmax, minlow))
    df = pd.DataFrame(rows, columns=["kind", "dd", "minlow"])
    if verbose and len(df):
        print(f"  touch level = +{touch:.1f}R : n touchers = {len(df)} "
              f"({(df['kind']=='target').sum()} target-winners, {(df['kind']=='stop').sum()} stop-losers, "
              f"{(df['kind']=='timeout').sum()} timeout)")
        for kind in ("target", "stop", "timeout"):
            g = df[df["kind"] == kind]["dd"].values
            if len(g) == 0:
                continue
            q1, q2, q3 = np.percentile(g, [25, 50, 75])
            print(f"    {kind:<8} n={len(g):4d}  give-back from peak: median={q2:5.2f}R std={np.std(g):5.2f}R  "
                  f"quartiles p25={q1:5.2f} p50={q2:5.2f} p75={q3:5.2f}")
        for th, tag in [(1.0, "<=+1R"), (0.0, "<=BE"), (-0.5, "<=-0.5R")]:
            frac_all = (df["minlow"] <= th).mean() * 100
            fw = (df[df["kind"] == "target"]["minlow"] <= th).mean() * 100 if (df["kind"] == "target").any() else float("nan")
            fl = (df[df["kind"] == "stop"]["minlow"] <= th).mean() * 100 if (df["kind"] == "stop").any() else float("nan")
            print(f"    pulled back {tag:<8} before exit: all={frac_all:5.1f}%  winners={fw:5.1f}%  losers={fl:5.1f}%")
    return df


# --------------------------------------------------------------------------- stage 2b
def variant_replay(h, l, c, trades, trig=None, floor=None, stall=None):
    """Replay the FIXED trade set under an exit variant. Returns gross R array.
    Bar order (conservative): stop-first, then stall close-check, then ratchet arm
    (raised stop effective NEXT bar). Reaching the real exit bar without a variant
    exit books the real base R."""
    out = np.empty(len(trades))
    for k, tr in enumerate(trades):
        i, j_exit, e_px, risk, R_base, kind = tr
        cur_stop = e_px - risk
        trig_px = e_px + trig * risk if trig is not None else None
        floor_px = e_px + floor * risk if trig is not None else None
        R = None
        for j in range(i + 1, j_exit + 1):
            if l[j] <= cur_stop:                       # 1. stop first (conservative)
                R = (cur_stop - e_px) / risk
                break
            if j == j_exit:                            # base exit bar reached un-stopped
                R = R_base
                break
            if stall is not None and (j - i) == stall: # 2. stall-cut at the close
                prog = (c[j] - e_px) / risk
                if prog < 0:
                    R = prog
                    break
            if trig_px is not None and h[j] >= trig_px:  # 3. arm ratchet (next-bar effect)
                cur_stop = max(cur_stop, floor_px)
        if R is None:                                  # defensive (loop always sets R)
            R = R_base
        out[k] = R
    return out


def main():
    print("=" * 100)
    print("KNOWN CONTRADICTING PRIOR (printed before any numbers):")
    print('  "BE-move=runner-cut killed on other legs; the difference here is activation only')
    print('   after +2R -- the question is whether post-2R retraces through the floor are')
    print('   deaths (save) or breathers (runner-cut)". Stage 2a\'s winner-breath distribution')
    print("  directly informs the read.")
    print("  NOTE: all early-exit variants are replayed on the FIXED trade set -- freed slots")
    print("  would re-arm new signals in live operation; NOT modeled -> conservative for them.")
    print("=" * 100)

    d15 = build_data()
    tb = run_leg(d15, pullback_frac=0.3)
    Rg = tb["R"].values                                   # gross R
    Rn = Rg - COST / tb["risk"].values                    # net $15
    w, ab, ix = pdh_weight(d15, tb)
    Rw = Rn * w                                           # LIVE weighted stream
    times = pd.DatetimeIndex(tb["time"])

    h, l, c = d15["high"].values, d15["low"].values, d15["close"].values
    n_bars = len(c)
    idx = d15.index
    pos = idx.get_indexer(tb["time"])
    exit_time = tb["time"] + pd.to_timedelta(tb["hold"].values, unit="D")
    exit_pos = idx.get_indexer(exit_time, method="nearest")
    snap_err = np.abs((idx[exit_pos] - exit_time).values.astype("timedelta64[s]").astype(float))
    print(f"\n[exit-bar reconstruction: {int((snap_err > 1.0).sum())}/{len(tb)} snapped >1s off "
          f"(max {snap_err.max():.3f}s) -- same hold-based method validated in stage 1]")

    trades = []
    viol = 0
    for k in range(len(tb)):
        i = int(pos[k]); j_exit = int(min(max(exit_pos[k], i), n_bars - 1))
        e_px = float(tb["e_px"].values[k]); risk = float(tb["risk"].values[k])
        R_base = float(Rg[k])
        kind = classify(j_exit - i, R_base, j_exit == n_bars - 1)
        # sanity: the ORIGINAL stop must never be pierced before the real exit bar
        stop0 = e_px - risk
        if any(l[j] <= stop0 for j in range(i + 1, j_exit)):
            viol += 1
        trades.append((i, j_exit, e_px, risk, R_base, kind))
    kinds = pd.Series([t[5] for t in trades])
    print(f"[path sanity: {viol}/{len(trades)} trades pierce the original stop before their real exit bar "
          f"(must be 0)]  exits: target={int((kinds=='target').sum())} stop={int((kinds=='stop').sum())} "
          f"timeout={int((kinds=='timeout').sum())}")

    # ------------------------------------------------------------- 2a MEASUREMENT
    print("\n" + "=" * 100)
    print("STAGE 2a -- POST-TOUCH GIVE-BACK (R units, real 15m path, gross R basis)")
    print("=" * 100)
    touch_stats(h, l, trades, 2.0, verbose=True)
    print("\n  mirror one-liners (same stats, other touch levels):")
    for tch in (1.5, 2.5):
        df = touch_stats(h, l, trades, tch)
        if len(df) == 0:
            print(f"  +{tch:.1f}R: no touchers")
            continue
        gw = df[df["kind"] == "target"]["dd"].values
        gl = df[df["kind"] == "stop"]["dd"].values
        f1 = (df["minlow"] <= 1.0).mean() * 100
        f0 = (df["minlow"] <= 0.0).mean() * 100
        fm = (df["minlow"] <= -0.5).mean() * 100
        print(f"  +{tch:.1f}R: n={len(df)}  give-back median/std winners={np.median(gw):.2f}/{np.std(gw):.2f}R "
              f"losers={np.median(gl):.2f}/{np.std(gl):.2f}R  |  pullback<=+1R {f1:.1f}%  <=BE {f0:.1f}%  <=-0.5R {fm:.1f}%")

    # ------------------------------------------------------------- 2b VARIANTS
    print("\n" + "=" * 100)
    print("STAGE 2b -- PRE-REGISTERED EXIT VARIANTS (FROZEN grid, replay on the FIXED trade set,")
    print("            LIVE weighted stream: PDH soft 0.5, net $15. Early-exit variants are")
    print("            CONSERVATIVE here (no re-arm of freed slots modeled).")
    print("=" * 100)
    grid = [("base", None, None, None)]
    kR = 1
    for trig in (1.5, 2.0, 2.5):
        for floor in (0.0, 1.0):
            grid.append((f"R{kR} t+{trig:.1f}/f{'BE' if floor == 0 else '+1R'}", trig, floor, None))
            kR += 1
    grid.append(("S1 stall16", None, None, 16))
    grid.append(("S2 stall32", None, None, 32))
    grid.append(("C1x t2/f1+st32", 2.0, 1.0, 32))

    risk_arr = tb["risk"].values
    mon_idx = times.to_period("M")
    streams = {}
    print(f"\n  {'variant':<18} summary")
    for label, trig, floor, stall in grid:
        if label == "base":
            Rw_v = Rw
        else:
            Rg_v = variant_replay(h, l, c, trades, trig=trig, floor=floor, stall=stall)
            Rw_v = (Rg_v - COST / risk_arr) * w
        streams[label] = Rw_v
        stream_metrics(Rw_v, times, label)

    # judgment layer 1: paired monthly block bootstrap, P(variant > base on CAGR/DD @1% risk)
    print("\n  -- layer 1: paired monthly bootstrap (2000 joint draws), P(variant beats base on CAGR/DD @1% risk) --")
    mon_base = pd.Series(streams["base"] * 0.01, index=times).groupby(mon_idx).sum()
    A = mon_base.values
    for label, *_ in grid:
        if label == "base":
            continue
        mon_v = pd.Series(streams[label] * 0.01, index=times).groupby(mon_idx).sum()
        B = mon_v.reindex(mon_base.index, fill_value=0.0).values
        wins = 0
        for _ in range(2000):
            bidx = rng.integers(0, len(A), len(A))
            wins += cdd(B[bidx]) > cdd(A[bidx])
        dmed = np.median(B - A) * 100
        print(f"    {label:<18} P(beat base) = {wins/2000*100:4.0f}%   monthly delta med={dmed:+.3f}%pt sd={np.std(B-A)*100:.3f}%pt")

    # ------------------------------------------------------------- layer 2: BOOK test (R4 only)
    print("\n  -- layer 2: BOOK test, pre-declared default R4 (trigger +2.0R / floor +1.0R) ONLY --")
    print("     (6-leg book as pwh_adoption.py section E: get_legs + gold15m + btc15m_L + btc15m_S,")
    print("      total 3% inv-vol; gold_bo = raw get_legs stream, NO PWH-soft tilt -- the axis under")
    print("      test here is btc15m_L only)")
    legs = {}
    for k2, tt in get_legs().items():
        legs[k2] = pd.Series(tt.R.values, index=pd.DatetimeIndex(tt.time))
    g = resample(load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    tg = run_bo(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
    legs["gold15m"] = pd.Series(tg["R"].values - 0.3 / tg["risk"].values,
                                index=pd.DatetimeIndex(tg["time"]))
    inv = invert(d15)
    ts_ = run_bo(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
    Rs = ts_["R"].values - COST / ts_["risk"].values
    pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
    Cinv = 2 * d15["high"].max()
    mS = (Cinv - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
    legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

    for tag, key in [("btc15m_L base", "base"), ("btc15m_L R4(t2/f1)", "R4 t+2.0/f+1R")]:
        L = dict(legs)
        L["btc15m_L"] = pd.Series(streams[key], index=times)
        mon = {k3: s.groupby(s.index.to_period("M")).sum() for k3, s in L.items()}
        start = max(s.index.min() for s in mon.values())
        end = min(s.index.max() for s in mon.values())
        midx2 = pd.period_range(start, end, freq="M")
        M = pd.DataFrame({k3: v.reindex(midx2, fill_value=0.0) for k3, v in mon.items()})
        wgt = 1.0 / M.std(); wgt = wgt / wgt.sum() * 0.03
        port = (M * wgt).sum(axis=1).values
        mult = np.array([np.prod(1 + port[rng.integers(0, len(port), 12)]) for _ in range(4000)])
        eq = np.cumprod(1 + port)
        ddp = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
        cagr = (eq[-1] ** (12 / len(port)) - 1) * 100
        print(f"    {tag:<20} CAGR={cagr:5.1f}%  maxDD={ddp:4.1f}%  CAGR/DD={cagr/ddp:5.2f}  | "
              f"1yr-mult med={np.median(mult):.2f} sd={mult.std():.2f} "
              f"p10={np.percentile(mult,10):.2f} p90={np.percentile(mult,90):.2f}")


if __name__ == "__main__":
    main()
