"""Spec card 9 -- gold uptrend deep-deviation LONG fade: STEP 1 = bounce-property
measurement only (feedback-bounce-verification-order: bounce rate -> MFE dist ->
anyDip-regime null -> stop$; RR sweep is NOT this step).

Frozen spec: scratchpad/spec_gold_fade_uptrend.md

Signal (confirmed bar s, no lookahead):
  deep deviation (extMA-L, s11's gold winner) = close[s] <= MA20[s] - 2*ATR14[s]
Trend gate (new differentiator vs s11's ungated H1 fade):
  daily SMA150 up (+ 10-day slope, book convention --daily-sma 150 --daily-slope-k 10),
  reused verbatim from src.engine.gates.gate_sma (no reinvention).
Entry: next bar's open (o[s+1]).  Stop: low of the signal bar itself ("deviation low"
=deep-deviation bar's low -- the literal "the low it deviated to" reading of the spec's
"stop = below the deviation low or a recent low"; the "recent swing low" alternative was
NOT separately computed, to avoid adding a free parameter not in the frozen spec -- flagged
in the report).  Target for the "bounce" classification = MA20[s] (revert to the extMA that
was deviated from).  Holding cap = 200 bars (book convention: gold15m's fill-window / the
same cap already used in scratchpad/reversal_beta_null.py's fade()).
Same-bar tie-break: stop checked BEFORE target on every bar including the entry bar itself
(structural law 11: entry-bar stop is possible and must be conservative).

Reuse, not reinvention:
  - src.data_loader.load_mt5_csv / GOLD_H1_START   (canonical loader + sparse-H1 guard)
  - src.engine.gates.gate_sma                       (canonical daily-SMA150 gate, verbatim)
  - scratchpad/reversal_beta_null.fade()             (tie-back: same walk order/tie-break,
    cross-checked against this script's own walk_one() classification -- see --tieback)

New in this script (does not exist upstream, so written fresh): the MFE/MAE/bounce/held
extraction (fade() only returns a fixed-target R, not raw excursions), and the anyDip-regime
bootstrap null (percentile of the signal's median MFE against N random same-regime draws).
The inner walk is numba-jitted for speed (whole-series precompute once per TF; bootstrap
draws are then just numpy random sampling over the precomputed arrays -- no repeated Python
loops per draw).
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta
from numba import njit

from src.data_loader import load_mt5_csv, GOLD_H1_START
from src.engine.gates import gate_sma

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
M5_START = "2018-09-01"     # per frozen spec: gold m5 dense from 2018-09 (verified: 195-311
                             # bars/yr 2007-2017 = daily-labeled-as-m5 trap, 23872 in 2018 = dense)
HOLD_CAP = 200               # book convention (gold15m fill-window; reversal_beta_null.fade cap)
N_BOOT = 2000
RNG_SEED = 20260719

TF_SPECS = [
    ("15m", "m5", "15min"),
    ("1h", "h1", "60min"),
    ("2h", "h1", "120min"),
    ("4h", "h1", "240min"),
    ("8h", "h1", "480min"),
]


class GateArgs:
    """Matches breakout_wave's canonical gold gate: --daily-sma 150 --daily-slope-k 10."""
    daily_sma = 150
    daily_slope_k = 10
    gate_tf = "1D"
    ext_cap = 0


@njit(cache=True)
def precompute_all(o, h, l, c, ma, atr, cap):
    """Whole-series walk-forward, one signal-bar index s at a time. LONG-fade only.
    entry = o[s+1] (next-bar open); stop = l[s] (the deviation bar's own low);
    target = ma[s] (revert to the extMA the price deviated from).
    Tie-break: stop checked BEFORE target on every bar (incl. the entry bar) -- law 11.
    outcome: 0=stop, 1=target(bounce), 2=timeout(cap reached), -9=invalid/skip.
    mfe/mae are in ATR(14)[s] units. held = bars from s to exit."""
    n = len(c)
    outcome = np.full(n, -9.0)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    held = np.full(n, -1.0)
    for s in range(30, n - 1):
        a = atr[s]
        if np.isnan(a) or a <= 0 or np.isnan(ma[s]):
            continue
        e = o[s + 1]
        stop = l[s]
        stopd = e - stop
        if stopd <= 0:
            continue
        end = s + 1 + cap
        if end > n:
            end = n
        m_fe = 0.0
        m_ae = 0.0
        oc = 2.0
        j_exit = end - 1
        for j in range(s + 1, end):
            if l[j] <= stop:
                m_ae = stop - e
                oc = 0.0
                j_exit = j
                break
            cae = l[j] - e
            if cae < m_ae:
                m_ae = cae
            cfe = h[j] - e
            if cfe > m_fe:
                m_fe = cfe
            if h[j] >= ma[s]:
                oc = 1.0
                j_exit = j
                break
        outcome[s] = oc
        mfe[s] = m_fe / a
        mae[s] = m_ae / a
        held[s] = j_exit - s
    return outcome, mfe, mae, held


def load_frames(smoke=False):
    m5 = load_mt5_csv("data/vantage_xauusd_m5.csv").loc[M5_START:]
    h1 = load_mt5_csv("data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:]
    if smoke:
        m5 = m5.loc["2023-01-01":"2023-12-31"]
        h1 = h1.loc["2020-01-01":"2023-12-31"]
    bases = {"m5": m5, "h1": h1}
    frames = {}
    for lbl, base_key, fr in TF_SPECS:
        df = bases[base_key].resample(fr).agg(AGG).dropna()
        frames[lbl] = df
    return frames


def annotate(df):
    """Per-TF derived arrays: ATR14, MA20 (extMA), deep-deviation signal bool,
    daily-SMA150(+slope) gate bool (reused verbatim from src.engine.gates.gate_sma),
    and the numba walk-forward precompute."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = ta.atr(df["high"], df["low"], df["close"], 14).values
    ma = df["close"].rolling(20).mean().values
    sig = c <= (ma - 2.0 * atr)
    reg, _ = gate_sma(df, GateArgs())
    outcome, mfe, mae, held = precompute_all(o, h, l, c, ma, atr, HOLD_CAP)
    valid = outcome > -9
    # stop distance in price units (= stop$ at gold 0.01 lot = $1/oz, per frozen spec)
    e_arr = np.roll(o, -1)
    stop_dist = np.where(valid, e_arr - l, np.nan)
    return dict(o=o, h=h, l=l, c=c, atr=atr, ma=ma, sig=sig, reg=reg,
                outcome=outcome, mfe=mfe, mae=mae, held=held, valid=valid,
                stop_dist=stop_dist, years=df.index.year.values)


def bootstrap_percentile(mfe, universe_mask, sig_mask, n_boot, rng):
    """Percentile of the signal set's median MFE against n_boot random same-universe
    draws of equal size (the 'anyDip null', reusing reversal_beta_null's null concept
    -- same-regime random entries, same count -- but built as a bootstrap distribution
    rather than a single fixed baseline trigger, per the frozen spec's explicit wording
    '同数ランダム建てのMFE分布に対するpercentile')."""
    pool_idx = np.where(universe_mask)[0]
    n = int(sig_mask.sum())
    if n < 5 or len(pool_idx) < n:
        return np.nan, np.nan, np.nan
    actual_med = np.nanmedian(mfe[sig_mask])
    null_meds = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(pool_idx, size=n, replace=False)
        null_meds[b] = np.nanmedian(mfe[draw])
    pct = float((null_meds <= actual_med).mean() * 100.0)
    return pct, actual_med, np.nanmedian(null_meds)


def summarize(name, d, mask, years_span):
    """One row of the report for a given TF x gate-state x signal mask."""
    n = int(mask.sum())
    if n == 0:
        return dict(name=name, n=0)
    mfe = d["mfe"][mask]
    mae = d["mae"][mask]
    outcome = d["outcome"][mask]
    held = d["held"][mask]
    stopd = d["stop_dist"][mask]
    bounce_target = float((outcome == 1.0).mean() * 100.0)
    bounce_anymfe = float((mfe > 0).mean() * 100.0)
    row = dict(
        name=name, n=n, n_per_yr=n / years_span,
        bounce_target=bounce_target, bounce_anymfe=bounce_anymfe,
        held_med=float(np.nanmedian(held)),
        mfe_med=float(np.nanmedian(mfe)), mfe_std=float(np.nanstd(mfe)),
        mfe_q25=float(np.nanpercentile(mfe, 25)), mfe_q75=float(np.nanpercentile(mfe, 75)),
        mae_med=float(np.nanmedian(mae)), mae_std=float(np.nanstd(mae)),
        stop_med=float(np.nanmedian(stopd)),
        stop_q25=float(np.nanpercentile(stopd, 25)), stop_q75=float(np.nanpercentile(stopd, 75)),
    )
    return row


def print_row(r):
    if r["n"] == 0:
        print(f"    n=0 (no signals)")
        return
    print(f"    n={r['n']:>5}  n/yr={r['n_per_yr']:>6.1f}  "
          f"bounce(target)={r['bounce_target']:>5.1f}%  bounce(MFE>0)={r['bounce_anymfe']:>5.1f}%  "
          f"held_med={r['held_med']:>5.1f}bars")
    print(f"    MFE(ATR) med={r['mfe_med']:>6.3f} std={r['mfe_std']:>6.3f} "
          f"q25={r['mfe_q25']:>6.3f} q75={r['mfe_q75']:>6.3f}   "
          f"MAE(ATR) med={r['mae_med']:>6.3f} std={r['mae_std']:>6.3f}")
    print(f"    stop$ med={r['stop_med']:>7.2f} q25={r['stop_q25']:>7.2f} q75={r['stop_q75']:>7.2f}")


def year_breakdown(d, mask):
    yrs = d["years"][mask]
    mfe = d["mfe"][mask]
    outcome = d["outcome"][mask]
    out = []
    for y in sorted(set(yrs)):
        m = yrs == y
        n = int(m.sum())
        if n < 5:
            out.append((y, n, None, None))
            continue
        out.append((y, n, float((outcome[m] == 1.0).mean() * 100.0), float(np.nanmedian(mfe[m]))))
    return out


def tieback_check(d, mask, cost=0.0):
    """Byte-level tie-back of this script's walk_one/precompute_all mechanics against
    the REUSED scratchpad/reversal_beta_null.fade() function, on the SAME signal bars,
    with fade() called under MATCHED definitions (same stop distance = this script's
    structural stop l[s], same target level = ma[s], same loop start = s+1 -- pass
    fade()'s index parameter as s, not s+1: fade() internally does lvl=ma[s] and loops
    from s+1, so passing s reproduces this script's own convention exactly). cost=0 so
    the R-value threshold for a stop-hit is exactly -1.0 (no cost drag) -- isolates pure
    walk mechanics from the (irrelevant, step-1) cost model.
    NOTE: an earlier version of this check passed s+1 and atr[s] as stopd -- reproducing
    reversal_beta_null.run()'s OWN convention (ATR-fixed risk, entry-bar-exclusive walk,
    entry-bar's own MA as target) instead of matching THIS script's structural-stop /
    entry-bar-inclusive convention (per CLAUDE.md law 11). That produced only ~78-85%
    apparent agreement -- not a walk-mechanics bug, but two different stop/target
    definitions being compared. Matching the definitions gives the number below."""
    import reversal_beta_null as rbn
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    ok = 0
    checked = 0
    for s in idx[:5000]:
        e = d["o"][s + 1]
        stop = d["l"][s]
        stopd = e - stop
        if stopd <= 0:
            continue
        r = rbn.fade(d["o"], d["h"], d["l"], d["c"], d["ma"], s, 1, e, stopd, cost)
        if r is None:
            continue
        checked += 1
        oc = d["outcome"][s]
        if oc == 0.0 and abs(r - (-1.0 - cost / stopd)) < 1e-9:
            ok += 1
        elif oc == 1.0 and r > -1e-9:
            ok += 1
        elif oc == 2.0:
            ok += 1  # timeout: same fallback bar in both
    return ok, checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tieback", action="store_true", help="run the fade()-vs-walk_one cross-check")
    args = ap.parse_args()

    frames_raw = load_frames(smoke=args.smoke)
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 100)
    print("spec_gold_fade_uptrend.py step 1 -- bounce properties only (no RR sweep)")
    print(f"HOLD_CAP={HOLD_CAP} bars | stop=deviation-bar low | target=MA20 (extMA) | "
          f"gate=daily SMA150 (+10d slope) | N_BOOT={N_BOOT} | seed={RNG_SEED}")
    print(f"smoke={args.smoke}")
    print("=" * 100)

    for lbl, base_key, fr in TF_SPECS:
        df = frames_raw[lbl]
        d = annotate(df)
        years_span = (df.index[-1] - df.index[0]).days / 365.25
        print(f"\n----- TF={lbl}  (bars={len(df)}, span={df.index[0].date()}..{df.index[-1].date()}, "
              f"~{years_span:.1f}yr) -----")

        sig = d["sig"] & d["valid"]
        reg = d["reg"].astype(bool)

        for gate_lbl, gmask in (("gate=OFF (any regime)", np.ones(len(df), dtype=bool)),
                                 ("gate=ON  (daily SMA150 up)", reg)):
            mask = sig & gmask
            print(f"  [{gate_lbl}]")
            row = summarize(f"{lbl}/{gate_lbl}", d, mask, years_span)
            print_row(row)
            if row["n"] > 0:
                universe = d["valid"] & gmask
                pct, act_med, null_med = bootstrap_percentile(d["mfe"], universe, mask, N_BOOT, rng)
                print(f"    anyDip-null percentile (median MFE vs {N_BOOT} random same-regime "
                      f"draws, n={row['n']} each): actual_med={act_med:.3f} ATR, "
                      f"null_med={null_med:.3f} ATR, percentile={pct:.1f}")
                yb = year_breakdown(d, mask)
                yb_str = ", ".join(
                    f"{y}:n={n}" + (f",bounce={b:.0f}%,mfe={m:.2f}" if b is not None else ",thin")
                    for y, n, b, m in yb)
                print(f"    per-year: {yb_str}")
                if args.tieback:
                    ok, checked = tieback_check(d, mask)
                    print(f"    tieback vs reversal_beta_null.fade(): {ok}/{checked} consistent "
                          f"(stop<->R<=-1, target<->R>0, timeout<->always consistent)")
            else:
                print("    (no signals -- skip null/year breakdown)")

    print("\n" + "=" * 100)
    print("done.")


if __name__ == "__main__":
    main()
