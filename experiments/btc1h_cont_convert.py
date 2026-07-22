"""btc1h_cont_convert.py -- convert the DOWN-side "BTC 1h band-edge continuation"
family (C1/C2/C4 of btc1h_cont_redund.py, i.e. bandwalk_exit_bounce.find_walk_eventA
DOWN cells) into an actual SHORT strategy shape with a designed stop/target/time-cap,
instead of the redundancy screen's symmetric +-1ATR race. Question: does a designed
exit give this family a tradeable body once the $15 BTC round-trip is paid?

PRE-REGISTERED PASS BAR: net meanR>0 AND totR/yr>=10R AND a plateau across the RR
ladder (neighbors agree -- not a lone spike).
PRE-REGISTERED KILL SIGNATURE: at RR>=2 the win% falls below breakeven
(1/(1+RR)) with meanR<=0, i.e. the continuation exhausts near 1x ATR and can't
support a wide target.

CELLS (all SHORT, reusing bandwalk_exit_bounce.find_walk_eventA / compute_bb
VERBATIM by import, exactly as in btc1h_cont_redund.py):
  C1 = bandwalk DN 2sigma walk, M=3, reentry at 1sigma  (the primary cell)
  C2 = bandwalk DN 1sigma walk, M=5, reentry at 1sigma
  C4 = bandwalk DN 1sigma walk, M=3, reentry at 1sigma

TRADE MECHANIZATION (causal, no lookahead) -- see the module docstring sections
below for the exact rules; IMPLEMENTATION DECISIONS are flagged inline where the
brief was ambiguous (same discipline as bandwalk_exit_bounce.py's own header).

IMPLEMENTATION DECISIONS (brief was ambiguous on these; flagged rather than
guessed silently):
  1. STOP is a FIXED PRICE LEVEL computed once from the MARKET entry_ref (next-bar
     open), for BOTH stop designs:
       structural = max(high) over the 12 bars ending at (and including) the
         signal bar itself -- "at signal time", i.e. independent of entry_ref.
       k*ATR       = entry_ref + k*ATR14[signal_bar] (ATR pre-shifted by 1, same
         convention bandwalk_exit_bounce.py/vwap_band_edge.py use) -- anchored to
         "entry" per the brief's own wording (structural says "at signal time",
         the ATR design says "above entry" -- read as two different anchors).
  2. TARGET is likewise a FIXED PRICE LEVEL: tgt = entry_ref - RR*(stop-entry_ref),
     computed once from the market entry. This copies breakout_wave.py's
     pullback-limit state machine literally: stop/target stay at the MARKET
     levels even when the limit column fills at a better (realized) price --
     realized risk shrinks, so a filled limit trade's REALIZED RR against the
     SAME fixed target exceeds the nominal RR ("effective RR rises", breakout_
     wave.py's own comment). This is simpler than re-deriving a target from the
     realized entry and is the literal reading of "copy breakout_wave's state
     machine" for the limit column.
  3. TIME CAP = 48 bars (2 days), matching the existing 24-bars=1-day convention
     used throughout (_race_common.py's K=24). For MARKET entries the 48-bar
     check window is bars [entry_bar .. entry_bar+47] INCLUDING the entry bar
     itself (spec: "the entry bar itself after the open fill"). For LIMIT fills
     the window is [fill_bar+1 .. fill_bar+48], i.e. the fill bar's own
     remainder is NOT checked -- this exactly copies breakout_wave.py's
     pullback-limit walk, which restarts at e_bar+1 after a limit fill.
  4. Same-bar double-touch (both stop and target trade on one bar) scores the
     STOP (conservative), per spec, checked in that order every bar.
  5. LIMIT column computed only for the PRIMARY cell (C1) x both stop designs
     (structural, 1xATR) x the full RR ladder, "to bound the grid" per the
     brief ("limit column for the best few configs only"). Limit window = 24
     bars (spec: "live for 24 bars"), cancelled if TARGET (the fixed market
     level) trades first -- low[j]<=tgt before high[j]>=lim.
  6. COST: net_R = gross_R - 15/risk_realized (risk_realized = stop-entry_ref for
     market trades, stop-lim for filled limit trades) -- same convention as
     btc1h_cont_redund.py's BTC_COST/atr_e, generalized to the trade's own risk
     distance (which equals k*ATR exactly for the ATR stop design, but differs
     for the structural design).
  7. BUSY-UNTIL: one open position per cell at a time, tracked by the ACTUAL
     realized exit bar of each accepted trade (breakout_wave.py's open_x
     pattern), not a fixed-K dedupe -- "like the book legs" per the brief.

Usage:
  .venv/bin/python experiments/btc1h_cont_convert.py            # smoke(2023) + full grid
  .venv/bin/python experiments/btc1h_cont_convert.py > experiments/out_btc1h_convert.txt 2>&1
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))   # project root
sys.path.insert(0, ROOT)                    # experiments/

from src.data_loader import load_mt5_csv
import bandwalk_exit_bounce as bw

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 25)

DATA_DIR = os.path.join(os.path.dirname(ROOT), "data")
BTC_COST = 15.0          # flat round-trip $, canon per CLAUDE.md
K_CAP = 48                # time-cap bars (2 days)
K_LIMIT = 24               # limit-order live window (bars)
PF_LIMIT = 0.3              # rally-limit fraction of (stop-entry_ref)
STRUCT_LOOKBACK = 12
RR_LADDER = [1.0, 1.5, 2.0, 3.0]
STOP_DESIGNS = [("structural", None), ("atr1.0", 1.0), ("atr1.5", 1.5), ("atr2.0", 2.0)]

CELL_DEFS = [
    # name, walk_band_kind ('dn2'/'dn1'), reentry_kind('dn1'), M
    ("C1 bandwalk DN 2sig_M3 A-cont", "dn2", "dn1", 3),
    ("C2 bandwalk DN 1sig_M5 A-cont", "dn1", "dn1", 5),
    ("C4 bandwalk DN 1sig_M3 A-cont", "dn1", "dn1", 3),
]


# ============================================================================
def build_frame(smoke_year=None):
    d = load_mt5_csv(os.path.join(DATA_DIR, "vantage_btcusd_h1.csv"))
    if smoke_year is not None:
        d = d[d.index.year == smoke_year]
    return d


def build_bands(d):
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1)
    mid, std = bw.compute_bb(d["close"], 20)
    bands = {
        "dn1": (mid - std).values,
        "dn2": (mid - 2 * std).values,
    }
    return atr.values, bands


def build_events(close_v, bands):
    """Raw (undeduped) DOWN event-A indices for each of C1/C2/C4, reusing
    bw.find_walk_eventA verbatim. No fixed-K dedupe here -- overlap is
    resolved later by the actual busy-until trade builder (decision #7)."""
    events = {}
    for name, walk_kind, reentry_kind, M in CELL_DEFS:
        walk_band = bands[walk_kind]
        reentry_band = bands[reentry_kind]
        events[name] = bw.find_walk_eventA(close_v, walk_band, M, reentry_band, "down")
    return events


# ============================================================================
def _stop_price(kind, k, high_v, entry_ref, atr_e, sig_i):
    if kind == "structural":
        lo = max(0, sig_i - STRUCT_LOOKBACK + 1)
        return float(np.max(high_v[lo:sig_i + 1]))
    if np.isnan(atr_e):
        return np.nan
    return entry_ref + k * atr_e


def simulate(d, high_v, low_v, open_v, close_v, atr_v, event_idx, stop_kind, k,
             rr, use_limit=False):
    """Build the busy-until trade set for one (cell-events, stop_design, rr,
    market|limit) config. Returns a DataFrame of accepted trades."""
    n = len(close_v)
    cand = []
    for i in event_idx:
        entry_bar = i + 1
        if entry_bar >= n:
            continue
        entry_ref = open_v[entry_bar]
        atr_e = atr_v[i]
        stop = _stop_price(stop_kind, k, high_v, entry_ref, atr_e, i)
        if np.isnan(stop):
            continue
        risk = stop - entry_ref
        if risk <= 0 or np.isnan(risk):
            continue
        tgt = entry_ref - rr * risk
        cand.append((i, entry_bar, entry_ref, stop, tgt, risk))
    cand.sort(key=lambda c: c[1])

    trades = []
    open_x = []
    n_attempted = 0            # busy-until-eligible candidates (denominator for fill-rate)
    for (i, entry_bar, entry_ref, stop, tgt, risk) in cand:
        open_x = [x for x in open_x if x >= entry_bar]
        if len(open_x) >= 1:
            continue
        n_attempted += 1

        filled = 1.0
        if use_limit:
            lim = entry_ref + PF_LIMIT * (stop - entry_ref)
            end_lim = min(entry_bar + K_LIMIT, n - 1)
            fj = None
            for j in range(entry_bar, end_lim + 1):
                if low_v[j] <= tgt:          # target trades before fill -> cancelled
                    break
                if high_v[j] >= lim:          # rally touches the limit -> filled
                    fj = j
                    break
            if fj is None:
                continue   # missed (cancelled or window expired) -- no trade
            e_px, e_bar = lim, fj
            risk_r = stop - lim
            if risk_r <= 0:
                continue
            reward = lim - tgt
            start = e_bar + 1     # copy breakout_wave: fill bar's own move not re-checked
        else:
            e_px, e_bar = entry_ref, entry_bar
            risk_r = risk
            reward = entry_ref - tgt
            start = e_bar          # entry bar itself checked, post open-fill

        end = min(start + K_CAP - 1, n - 1)
        R, exit_bar, timeout = None, end, False
        for j in range(start, end + 1):
            if high_v[j] >= stop:      # conservative: stop wins same-bar double touch
                R, exit_bar = -1.0, j
                break
            if low_v[j] <= tgt:
                R, exit_bar = reward / risk_r, j
                break
        if R is None:
            R = (e_px - close_v[end]) / risk_r
            exit_bar, timeout = end, True

        cost_R = BTC_COST / risk_r
        trades.append((d.index[entry_bar], d.index[e_bar], R, R - cost_R, risk_r, filled, timeout))
        open_x.append(exit_bar)

    cols = ["sig_time", "fill_time", "R_gross", "R_net", "risk", "filled", "timeout"]
    return pd.DataFrame(trades, columns=cols), n_attempted


# ============================================================================
def summarize(t, rr, n_attempted=None):
    if len(t) == 0:
        return None
    n = len(t)
    span_yr = max((t["fill_time"].iloc[-1] - t["fill_time"].iloc[0]).days / 365.25, 0.1)
    win_pct = (t["R_gross"] > 0).mean() * 100
    breakeven = 1.0 / (1.0 + rr) * 100
    Rg, Rn = t["R_gross"].values, t["R_net"].values

    def pf_of(R):
        pos, neg = R[R > 0].sum(), abs(R[R <= 0].sum())
        return pos / neg if neg > 0 else np.nan

    t = t.copy()
    t["year"] = t["fill_time"].dt.year
    yrs = sorted(t["year"].unique())
    half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    is_r = t[t["year"] < half]["R_net"] if half else t["R_net"]
    oos_r = t[t["year"] >= half]["R_net"] if half else t["R_net"]

    cum = t["R_net"].cumsum()
    maxdd = (cum.cummax() - cum).max()

    green = sum(1 for _, g in t.groupby("year") if g["R_net"].sum() > 0)
    total_yr = t["year"].nunique()

    return dict(
        n=n, n_yr=n / span_yr, win_pct=win_pct, breakeven=breakeven,
        pf_gross=pf_of(Rg), pf_net=pf_of(Rn),
        meanR_gross=Rg.mean(), medR_gross=np.median(Rg), sdR_gross=Rg.std(),
        meanR_net=Rn.mean(), medR_net=np.median(Rn), sdR_net=Rn.std(),
        totR_yr_net=Rn.sum() / span_yr, maxdd=maxdd,
        is_net=is_r.mean() if len(is_r) else np.nan,
        oos_net=oos_r.mean() if len(oos_r) else np.nan,
        green=green, total_yr=total_yr,
        timeout_pct=t["timeout"].mean() * 100,
        fill_pct=(n / n_attempted * 100) if n_attempted else np.nan,
    )


def fmt_row(label, s, rr):
    if s is None:
        return f"{label:<46} NO TRADES"
    pass_bar = (s["meanR_net"] > 0) and (s["totR_yr_net"] >= 10.0)
    kill_sig = (rr >= 2.0) and (s["win_pct"] < s["breakeven"]) and (s["meanR_net"] <= 0)
    tag = "PASS-BAR" if pass_bar else ("KILL-SIG" if kill_sig else "")
    return (f"{label:<46} n={s['n']:<5d} n/yr={s['n_yr']:>5.1f}  "
            f"win={s['win_pct']:>5.1f}% (be={s['breakeven']:>5.1f}%)  "
            f"PF g/n={s['pf_gross']:>5.2f}/{s['pf_net']:>5.2f}  "
            f"meanR g={s['meanR_gross']:+.3f} n={s['meanR_net']:+.3f} "
            f"(med={s['medR_net']:+.3f} sd={s['sdR_net']:.3f})  "
            f"totR/yr(net)={s['totR_yr_net']:+6.2f}  maxDD(R)={s['maxdd']:>5.1f}  "
            f"IS/OOS(net)={s['is_net']:+.2f}/{s['oos_net']:+.2f}  "
            f"grn={s['green']}/{s['total_yr']}  timeout={s['timeout_pct']:>4.1f}%  "
            + (f"fill-rate={s['fill_pct']:>5.1f}%  " if not np.isnan(s['fill_pct']) else "")
            + tag)


# ============================================================================
def run_grid(d, tag):
    high_v, low_v, open_v, close_v = (d["high"].values, d["low"].values,
                                       d["open"].values, d["close"].values)
    atr_v, bands = build_bands(d)
    events = build_events(close_v, bands)

    print(f"\n{'='*112}\n{tag}  (n_bars={len(d)}, span={d.index[0]}..{d.index[-1]})\n{'='*112}")

    for name, walk_kind, reentry_kind, M in CELL_DEFS:
        ev = events[name]
        print(f"\n--- {name}  (raw event-A count={len(ev)}) ---")
        for stop_kind, k in STOP_DESIGNS:
            for rr in RR_LADDER:
                t, n_att = simulate(d, high_v, low_v, open_v, close_v, atr_v, ev,
                                     stop_kind, k, rr, use_limit=False)
                s = summarize(t, rr, n_att)
                label = f"  {stop_kind:<11} RR={rr:<4}"
                print(fmt_row(label, s, rr))

    # limit column: primary cell (C1) only, structural + 1xATR stop, full RR ladder
    print("\n--- LIMIT column (C1 only, structural & 1xATR stop, bounding the grid) ---")
    print("  fill-rate% = filled trades / busy-until-eligible signals (the rest were either")
    print("  cancelled -- target traded before the rally reached the limit -- or the 24h window expired)")
    name = CELL_DEFS[0][0]
    ev = events[name]
    for stop_kind, k in [("structural", None), ("atr1.0", 1.0)]:
        for rr in RR_LADDER:
            t, n_att = simulate(d, high_v, low_v, open_v, close_v, atr_v, ev,
                                 stop_kind, k, rr, use_limit=True)
            s = summarize(t, rr, n_att)
            label = f"  LIMIT {stop_kind:<11} RR={rr:<4}"
            print(fmt_row(label, s, rr))


def tie_back(d):
    high_v, low_v, open_v, close_v = (d["high"].values, d["low"].values,
                                       d["open"].values, d["close"].values)
    atr_v, bands = build_bands(d)
    events = build_events(close_v, bands)
    name = CELL_DEFS[0][0]
    ev = events[name]
    print("\n--- TIE-BACK: C1, stop=1xATR, RR=1, MARKET entry ---")
    print("  (redundancy-run reference for this cell/config: win~=57%, meanR(gross)~=+0.15;")
    print("   our K_CAP=48 vs the redund screen's K=24, and entry=next-bar-open vs the ")
    print("   redund screen's entry=event-bar's-own-close, so an exact match is NOT expected --")
    print("   flagging only a directional/magnitude check, per the pre-registered brief.)")
    t48, na48 = simulate(d, high_v, low_v, open_v, close_v, atr_v, ev, "atr1.0", 1.0, 1.0, use_limit=False)
    s48 = summarize(t48, 1.0, na48)
    print(fmt_row("  [K_CAP=48, this script's official config]", s48, 1.0))
    # also print a K=24 variant restricted to match the redund race window more closely
    global K_CAP
    orig = K_CAP
    K_CAP = 24
    t24, na24 = simulate(d, high_v, low_v, open_v, close_v, atr_v, ev, "atr1.0", 1.0, 1.0, use_limit=False)
    K_CAP = orig
    s24 = summarize(t24, 1.0, na24)
    print(fmt_row("  [K_CAP=24, closer to the redund race window]", s24, 1.0))
    return s48, s24


# ============================================================================
def main():
    print("=" * 112)
    print("btc1h_cont_convert.py -- DOWN-side band-edge continuation -> designed SHORT strategy")
    print("PRE-REGISTERED PASS BAR: net meanR>0 AND totR/yr>=10R AND RR-ladder plateau")
    print("PRE-REGISTERED KILL: at RR>=2, win% < breakeven(1/(1+RR)) AND meanR<=0")
    print("=" * 112)

    d_smoke = build_frame(2023)
    print(f"\nSMOKE (2023 only): n_bars={len(d_smoke)}")
    run_grid(d_smoke, "SMOKE 2023")

    d_full = build_frame(None)
    tie_back(d_full)
    run_grid(d_full, "FULL HISTORY")

    print("\n" + "=" * 112)
    print("DONE.")
    print("=" * 112)


if __name__ == "__main__":
    main()
