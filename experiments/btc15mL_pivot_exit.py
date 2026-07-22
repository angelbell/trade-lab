"""btc15m_L: structural (fractal-pivot) exit overlay on the ADOPTED leg.

Spec card source: user's discretionary observation while holding a btc15m_L long --
"entry -> spike up -> can't exceed the spike high -> lower high -> support (last swing
low) breaks -> should have exited there, instead the machine rode it to the fixed stop."
User is explicit: this is a STRUCTURE-BREAK exit, not an R-level ratchet (that is a
DIFFERENT, already-running experiment in btc15mL_stop_repair.py -- untouched here), and
the structure must be read with 15m FRACTAL PIVOTS (not ZigZag -- user's instruction),
even though the lab already found ZigZag > pivot as an ENTRY detector (s01_entries.md --
that finding is about entries; this is about reading in-trade structure, a different job).

ADOPTED LEG (unchanged -- entry/stop/target untouched; this script only asks "would an
early structural exit, bolted on top, have helped?"):
  BTC 15m . ZigZag(2.0xATR) Pattern-B . trend_ema80 . 4h-KAMA(14) gate . RR4.5 .
  pullback-limit 0.30 . fill_win=200 . fwd=500 . cost $15/risk . start 2018-10-01
Tie-back (breakout_wave.run, --cost 0): n=759 win=23% meanR=+0.59 medRR=6.86 -- reproduced
below via run() itself (no re-derivation of the entry/fill logic -- see NO-REINVENTION
note below), then a NEW bar-by-bar exit walker adds the structural-exit overlay on top.

NO-REINVENTION: breakout_wave.run() does not expose the raw (entry_bar, entry_px, stop,
tgt) tuples it evaluates internally, only the resolved per-trade R/hold/risk/e_px. Rather
than re-implement (and risk mis-transcribing) the Pattern-B skeleton-matching + KAMA-gate +
pullback-limit-fill logic, this script calls run() itself (t = run(d15, args), the exact
config above) and ANALYTICALLY reconstructs stop/tgt from its output columns:
    stop = e_px - risk                              (risk is defined that way in run())
    tgt  = e_px + risk * (rr + pf) / (1 - pf)        (pullback-limit moves entry down by
           pf*risk_market but leaves the target price fixed at the ORIGINAL market entry's
           rr*risk_market; algebra below; verified against run()'s own medRR=6.86 print,
           which is exactly (4.5+0.30)/(1-0.30) = 6.857142857..., a constant across all
           trades since rr and pf are fixed -- this IS the tie-back check for the formula.)
A per-trade "market_walk" reproduction of run()'s own post-fill stop/target/timeout loop
(with the structural condition forced OFF) is then asserted to reproduce t["R"] and
t["hold"] EXACTLY for all 759 trades before anything else runs -- see verify_tieback().

NO-LOOKAHEAD for the pivot structure (this is the part the spec explicitly warns about):
  A fractal pivot high/low at bar c (strict local extremum over window [c-L, c+R]) is only
  KNOWN at bar c+R. swings_pivot_lr() returns (confirm_idx=c+R, pivot_idx=c, price, kind).
  The exit walker NEVER looks at a pivot with confirm_idx > current bar j -- see
  last_confirmed() which does a right-side searchsorted on confirm_idx <= j, and
  assert_no_lookahead() explicitly checks, for a sample of (trade, bar) uses, that the
  pivot used has confirm_idx <= j AND confirm_idx - pivot_idx == R (the exact confirmation
  lag), i.e. the same discipline as the 2026-07-12 flow-exit lookahead post-mortem.
  swings_pivot_lr(h, l, L, R) generalizes breakout_wave.swings_pivot(h, l, n) (which is
  symmetric-only, L=R=n) to asymmetric L!=R; equivalence to swings_pivot is asserted for
  L=R=5 in verify_tieback() so the generalization isn't silently wrong.

EXECUTION / no-lookahead for the EXIT itself: the structural condition is evaluated on a
CONFIRMED bar CLOSE; if true, the position is flagged "armed" and closed at the NEXT bar's
OPEN (never the same bar). If in that next bar the STOP is also hit intrabar, stop wins
(spec: same-bar conflict -> stop, conservative) -- i.e. every bar's low is checked against
the stop BEFORE an armed structural exit is allowed to fire, exactly mirroring run()'s own
stop-before-target priority.

Run (full):    .venv/bin/python experiments/btc15mL_pivot_exit.py 2>&1 | tee experiments/out_btc15mL_pivot_exit.txt
Run (smoke):   .venv/bin/python experiments/btc15mL_pivot_exit.py --smoke 2>&1 | tee experiments/out_btc15mL_pivot_exit_smoke.txt
"""
import argparse
import contextlib
import io
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, swings_pivot

ROOT = "/home/angelbell/dev/auto-trade"
RR = 4.5
PF = 0.30
FILL_WIN = 200
FWD = 500
COST = 15.0            # $/risk, live-cost convention used throughout the book (CLAUDE.md)
RISK_PCT = 0.01         # per-trade risk fraction for the leg CAGR/DD curve

WIDTHS = [(3, 3), (5, 5), (8, 8), (12, 12), (8, 3)]
ARMS = ["A1_support", "A2_lh_support", "A3_lh_only", "A4_spike_fade"]


# --------------------------------------------------------------------------- pivots
def swings_pivot_lr(h, l, L, R):
    """Generalized N-bar fractal pivot (breakout_wave.swings_pivot is symmetric-only,
    L=R=n). Same rule: bar c is a pivot high iff h[c] is the STRICT max over the window
    [c-L, c+R]; confirmed (knowable) only at bar c+R. Returns two SEPARATE sorted-by-
    confirm-index arrays (confirm_idx, pivot_idx, price) for highs and lows -- kept apart
    (unlike breakout_wave's single interleaved list) because the exit logic needs fast
    per-bar "last confirmed X" lookups via np.searchsorted.
    """
    n = len(h)
    hi, lo = [], []
    for c in range(L, n - R):
        wh = h[c - L:c + R + 1]
        wl = l[c - L:c + R + 1]
        if h[c] == wh.max() and (wh == h[c]).sum() == 1:
            hi.append((c + R, c, h[c]))
        if l[c] == wl.min() and (wl == l[c]).sum() == 1:
            lo.append((c + R, c, l[c]))
    hi.sort(key=lambda t: t[0])
    lo.sort(key=lambda t: t[0])
    return (np.array([x[0] for x in hi], dtype=np.int64), np.array([x[1] for x in hi], dtype=np.int64), np.array([x[2] for x in hi])), \
           (np.array([x[0] for x in lo], dtype=np.int64), np.array([x[1] for x in lo], dtype=np.int64), np.array([x[2] for x in lo]))


def check_pivot_equivalence(h, l, n=5):
    """swings_pivot_lr(h,l,n,n) must reproduce breakout_wave.swings_pivot(h,l,n) exactly
    (kind-split). This is the generalization's own tie-back."""
    ref = swings_pivot(h, l, n)
    ref_hi = sorted((ci, ii, p) for (ci, ii, p, k) in ref if k == +1)
    ref_lo = sorted((ci, ii, p) for (ci, ii, p, k) in ref if k == -1)
    (hci, hii, hp), (lci, lii, lp) = swings_pivot_lr(h, l, n, n)
    got_hi = sorted(zip(hci.tolist(), hii.tolist(), hp.tolist()))
    got_lo = sorted(zip(lci.tolist(), lii.tolist(), lp.tolist()))
    assert ref_hi == got_hi, "pivot-high generalization mismatch vs swings_pivot"
    assert ref_lo == got_lo, "pivot-low generalization mismatch vs swings_pivot"


def last_le(confirm_idx, j):
    """rightmost position with confirm_idx[pos] <= j, or -1."""
    return int(np.searchsorted(confirm_idx, j, side="right")) - 1


# --------------------------------------------------------------------------- base leg
def get_base_trades():
    with contextlib.redirect_stderr(io.StringIO()):
        from radar_gate_race import BASE
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        args = SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                  "pullback_frac": PF, "rr": RR, "fill_win": FILL_WIN, "cost": 0.0})
        t = run(d15, args)
    return d15, t


def market_walk(entry_bar, entry_px, stop, tgt, fwd, l, h, c):
    """Pure reproduction of run()'s post-fill stop/target/timeout loop (structural OFF).
    Returns (exit_bar, R, reason)."""
    risk = entry_px - stop
    reward = tgt - entry_px
    n = len(c)
    last = min(entry_bar + fwd, n - 1)
    for j in range(entry_bar + 1, last + 1):
        if l[j] <= stop:
            return j, -1.0, "stop"
        if h[j] >= tgt:
            return j, reward / risk, "tgt"
    return last, (c[last] - entry_px) / risk, "timeout"


def verify_tieback(d15, t):
    h, l, c = d15["high"].values, d15["low"].values, d15["close"].values
    check_pivot_equivalence(h, l, 5)
    ei = d15.index.get_indexer(t["time"])
    e_px = t["e_px"].values
    risk = t["risk"].values
    stop = e_px - risk
    tgt = e_px + risk * (RR + PF) / (1 - PF)
    n_bad = 0
    max_r_err, max_h_err = 0.0, 0.0
    natural_exit_bar = np.empty(len(t), dtype=np.int64)
    for k in range(len(t)):
        xb, R, _ = market_walk(int(ei[k]), e_px[k], stop[k], tgt[k], FWD, l, h, c)
        natural_exit_bar[k] = xb
        hold = (d15.index[xb] - d15.index[ei[k]]).total_seconds() / 86400.0
        r_err = abs(R - t["R"].iloc[k])
        h_err = abs(hold - t["hold"].iloc[k])
        max_r_err = max(max_r_err, r_err)
        max_h_err = max(max_h_err, h_err)
        if r_err > 1e-6 or h_err > 1e-6:
            n_bad += 1
    print(f"  [tie-back self-check] n={len(t)} market_walk vs run(): "
          f"mismatches={n_bad}  max|dR|={max_r_err:.2e}  max|dHold|={max_h_err:.2e}")
    assert n_bad == 0, "market_walk does not reproduce run() -- stop/tgt reconstruction is wrong"
    return ei.astype(np.int64), e_px, risk, stop, tgt, natural_exit_bar


# --------------------------------------------------------------------------- structural exit
def structural_walk(entry_bar, entry_px, stop, tgt, risk, fwd, l, h, c, o,
                     hi_confirm, hi_price, lo_confirm, lo_price, arm, profit_only):
    """Bar-by-bar walk with the structural-exit overlay bolted on top of the market
    stop/target/timeout (identical priority/timeout to market_walk when the structural
    condition never fires -- this IS why the per-cell leg reduces exactly to the baseline
    for un-fired trades). NO-LOOKAHEAD: the structural condition at bar j only reads
    pivots with confirm_idx <= j (last_le on the confirm-index array). EXECUTION: the
    condition is evaluated on bar j's CONFIRMED close; if true, "armed" carries into bar
    j+1 and the exit fills at bar j+1's OPEN -- but bar j+1's stop (checked first, every
    bar, armed or not) wins on a same-bar conflict (conservative, per spec)."""
    n = len(c)
    last = min(entry_bar + fwd, n - 1)
    armed = False
    peak_before = -np.inf
    reached_1r = False
    fired, fire_bar = False, None
    for j in range(entry_bar + 1, last + 1):
        if l[j] <= stop:                                  # stop ALWAYS checked first
            return j, -1.0, "stop", fired, fire_bar
        if armed:                                          # pending structural exit -> fill at this open
            return j, (o[j] - entry_px) / risk, "structural", fired, fire_bar
        if h[j] >= tgt:
            return j, (tgt - entry_px) / risk, "tgt", fired, fire_bar

        lph_pos = last_le(hi_confirm, j)
        lpl_pos = last_le(lo_confirm, j)
        lpl = lo_price[lpl_pos] if lpl_pos >= 0 else None
        lower_high = False
        if lph_pos >= 1:
            lower_high = hi_price[lph_pos] < hi_price[lph_pos - 1]

        if arm == "A1_support":
            cond = lpl is not None and c[j] < lpl
        elif arm == "A2_lh_support":
            cond = lower_high and (lpl is not None and c[j] < lpl)
        elif arm == "A3_lh_only":
            cond = lower_high
        elif arm == "A4_spike_fade":
            made_new_high = h[j] > peak_before
            if made_new_high:
                peak_before = h[j]
            if peak_before >= entry_px + risk:
                reached_1r = True
            cond = reached_1r and (not made_new_high) and (lpl is not None and c[j] < lpl)
        else:
            raise ValueError(arm)

        if profit_only:
            cond = cond and (c[j] > entry_px)
        if cond:
            armed = True
            if not fired:
                fired, fire_bar = True, j
    return last, (c[last] - entry_px) / risk, "timeout", fired, fire_bar


def assert_no_lookahead(hi_confirm, hi_pivot, lo_confirm, lo_pivot, R_hi, R_lo, sample_bars):
    """For a sample of bars, confirm: the last-confirmed pivot used has confirm_idx <= bar,
    and confirm_idx - pivot_idx == R exactly (the declared confirmation lag)."""
    for j in sample_bars:
        p = last_le(hi_confirm, j)
        if p >= 0:
            assert hi_confirm[p] <= j
            assert hi_confirm[p] - hi_pivot[p] == R_hi
        p = last_le(lo_confirm, j)
        if p >= 0:
            assert lo_confirm[p] <= j
            assert lo_confirm[p] - lo_pivot[p] == R_lo
    return True


def cdd(vals, days):
    """CAGR/maxDD off a per-trade R*risk-fraction series (reused verbatim from
    experiments/book_leave_one_out.py -- do not re-derive)."""
    eq = np.cumprod(1 + vals)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    if dd <= 0:
        return np.nan, np.nan, np.nan
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd


def leg_metrics(R_cost, times, hold_days):
    n = len(R_cost)
    win = (R_cost > 0).mean() * 100
    meanR = R_cost.mean()
    pos = R_cost[R_cost > 0].sum()
    neg = -R_cost[R_cost <= 0].sum()
    pf = pos / neg if neg > 0 else np.nan
    order = np.argsort(times.values)
    days = max((times.iloc[order[-1]] - times.iloc[order[0]]).days, 1)
    cagr, dd, cdd_ratio = cdd(RISK_PCT * R_cost[order], days)
    return dict(n=n, win=win, meanR=meanR, pf=pf, cagr=cagr, dd=dd, cdd=cdd_ratio,
                hold_med=np.median(hold_days), hold_mean=np.mean(hold_days))


def run_cell(d15, t, ei, e_px, risk, stop, tgt, hi_confirm, hi_pivot, hi_price,
             lo_confirm, lo_pivot, lo_price, arm, profit_only, fwd=FWD):
    h, l, c, o = d15["high"].values, d15["low"].values, d15["close"].values, d15["open"].values
    n = len(t)
    R_struct = np.empty(n)
    exit_bar = np.empty(n, dtype=np.int64)
    fired = np.zeros(n, dtype=bool)
    fire_bar = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        xb, R, reason, f, fb = structural_walk(
            int(ei[k]), e_px[k], stop[k], tgt[k], risk[k], fwd, l, h, c, o,
            hi_confirm, hi_price, lo_confirm, lo_price, arm, profit_only)
        R_struct[k] = R
        exit_bar[k] = xb
        fired[k] = f
        fire_bar[k] = fb if fb is not None else -1
    hold_days = (d15.index[exit_bar].values - d15.index[ei].values) / np.timedelta64(1, "D")
    R_struct_cost = R_struct - COST / risk
    R_base_cost = t["R"].values - COST / risk
    return dict(R_struct=R_struct, R_struct_cost=R_struct_cost, R_base_cost=R_base_cost,
                exit_bar=exit_bar, fired=fired, fire_bar=fire_bar, hold_days=hold_days)


def counterfactual(R_base_cost, R_struct_cost, fired):
    """The decisive decomposition (spec item 2): among FIRED trades, split into
    SAVED (baseline would have been a real loss, structural made it less bad / a win)
    vs CUT (baseline would have been a real win, structural cut it short)."""
    b = R_base_cost[fired]
    s = R_struct_cost[fired]
    saved_mask = (b <= 0) & (s > b)
    cut_mask = (b > 0) & (s < b)
    saved_n, saved_r = int(saved_mask.sum()), float((s[saved_mask] - b[saved_mask]).sum())
    cut_n, cut_r = int(cut_mask.sum()), float((b[cut_mask] - s[cut_mask]).sum())
    return dict(fired_n=int(fired.sum()), saved_n=saved_n, saved_r=saved_r,
                cut_n=cut_n, cut_r=cut_r, net_r=saved_r - cut_r)


# --------------------------------------------------------------------------- random-drop null
def random_exit_null(d15, ei, e_px, risk, stop, tgt, natural_exit_bar, fired, ndraw=1000, seed=20260713):
    """Spec item 4: for the SAME fired trades (same count), replace the structural exit
    point with a UNIFORMLY RANDOM bar between entry+1 and the trade's NATURAL resolution
    bar (stop/tgt/timeout -- guaranteed neither has fired yet in that open window, since
    natural_exit_bar is BY DEFINITION the first bar either fires), executed the same way
    (next bar's open) -- tests whether "exit structurally" beats plain "exit early,
    randomly" at the same frequency. Returns the null distribution of leg meanR (cost-adj)
    over the fired subset, to rank the actual structural meanR against."""
    o = d15["open"].values
    idx = np.where(fired)[0]
    if len(idx) == 0:
        return None
    rng = np.random.default_rng(seed)
    eb, exb = ei[idx], natural_exit_bar[idx]
    epx, rk = e_px[idx], risk[idx]
    n = len(idx)
    # random "fire" bar in [entry+1, natural_exit-1] (guaranteed stop/tgt not yet hit
    # there, by definition of natural_exit_bar); executes at THAT bar's next open, i.e.
    # in [entry+2, natural_exit] -- never later than the real natural resolution.
    lo = eb + 1
    hi = np.maximum(exb - 1, lo)
    null_meanR = np.empty(ndraw)
    for d in range(ndraw):
        fbar = lo + (rng.integers(0, 1_000_000, n) % (hi - lo + 1))
        xbar = fbar + 1                            # execute at next bar's open (<= exb, by construction)
        R = (o[xbar] - epx) / rk
        R_cost = R - COST / rk
        null_meanR[d] = R_cost.mean()
    return null_meanR


# --------------------------------------------------------------------------- book + bootstrap
def build_book_legs(gold_start="2018-01-01", gold15m_skip=False, s_rr=4.5, btc15mL_override=None):
    """Reuses experiments/book_spec_fix.build() VERBATIM (the M1+M2 spec-fixed 6-leg
    machine, tie-back target 8.48/7.49%) with ONE deliberate one-line patch: btc15m_S's
    rr is not parameterised in that file (it silently used BASE's default rr=4.0), so it
    is re-run here at rr=4.5 -- the CURRENTLY ADOPTED short leg -- per this spec card's
    explicit instruction. If btc15mL_override is given (a pd.Series indexed like the
    original btc15m_L leg) it REPLACES that leg (structural-exit substitution)."""
    from types import SimpleNamespace as SNS
    with contextlib.redirect_stderr(io.StringIO()):
        from research.regime_gate_lab import CFG
        from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
        from ema_pullback import run as run_pb
        from short_mirror_15m import invert
        from radar_gate_race import BASE

        g1h = load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv")
        if gold_start:
            g1h = g1h.loc[gold_start:]
        gb = run(resample(g1h, "1h"), SNS(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0,
                                             "fwd": 500, "daily_sma": 150, "daily_slope_k": 10}))
        legs = {"gold_bo": pd.Series(gb["R"].values, index=pd.DatetimeIndex(gb["time"]))}
        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        bo = run(b4, SNS(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))[["time", "R"]]
        bk = kama_gate_btc(bo)
        legs["btc_bo_kama"] = pd.Series(bk.R.values, index=pd.DatetimeIndex(bk.time))
        pb = cycle_gate_pull(run_pb(b4, "long", SNS(**{**PB, "csv": "x", "tf": "4h"}), 0.0)[["time", "R"]])
        legs["btc_pull"] = pd.Series(pb.R.values, index=pd.DatetimeIndex(pb.time))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        tg = run(g15, SNS(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                             "ext_cap": 8.0, "pullback_frac": 0.25}))
        Rg = tg["R"].values - 0.3 / tg["risk"].values
        idxg = pd.DatetimeIndex(tg["time"])
        if gold15m_skip:
            keep = ~idxg.hour.isin(range(9, 15))
            Rg, idxg = Rg[keep], idxg[keep]
        legs["gold15m"] = pd.Series(Rg, index=idxg)

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        tL = run(d15, SNS(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                             "pullback_frac": 0.3, "rr": 4.5}))
        RL = tL["R"].values - 15.0 / tL["risk"].values
        eiL = d15.index.get_indexer(tL["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        w_pdh = np.where(tL["e_px"].values > pdh[eiL], 1.0, 0.5)
        if btc15mL_override is None:
            legs["btc15m_L"] = pd.Series(RL * w_pdh, index=pd.DatetimeIndex(tL["time"]))
        else:
            legs["btc15m_L"] = btc15mL_override

        inv = invert(d15)
        C = 2 * d15["high"].max()
        ts = run(inv, SNS(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": s_rr}))
        Rs = ts["R"].values - 15.0 / ts["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts["e_px"].values) < pdl[d15.index.get_indexer(ts["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts["time"])[mS])
    return legs


NEW_LEGS = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]


def w_trade(legs, basket, budget=0.03):
    sig = pd.Series({k: legs[k].std() for k in basket})
    w = 1.0 / sig
    return w / w.sum() * budget


def book(legs, basket=NEW_LEGS, budget=0.03):
    w = w_trade(legs, basket, budget)
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = []
    for k in basket:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    s = pd.concat(parts).sort_index()
    return cdd(s.values, (s.index[-1] - s.index[0]).days) + (len(s),)


def book_monthly(legs, basket=NEW_LEGS, budget=0.03):
    """Reused machinery from experiments/book_bootstrap_arbiter.py: the book's MONTHLY
    return series at inv-vol weights, for the circular block bootstrap (single-path
    maxDD is noisy -- CLAUDE.md law 7)."""
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in legs.items() if k in basket}
    st = max(v.index.min() for v in mon.values())
    en = min(v.index.max() for v in mon.values())
    midx = pd.period_range(st, en, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    sig = M.std()
    w = (1.0 / sig[basket])
    w = w / w.sum() * budget
    return (M[basket] * w).sum(axis=1)


def cdd_monthly(port_vals, months):
    eq = np.cumprod(1 + port_vals)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max()
    if dd <= 0:
        return np.nan
    return ((eq[-1] ** (12 / months) - 1)) / dd


def block_bootstrap(monthly_a, monthly_b, ndraw=2000, seed=20260713):
    """Circular block bootstrap (1/3/6/12mo) of the BOOK's monthly returns, PAIRED
    (same resampled months for base and candidate) -- reused pattern from
    experiments/book_bootstrap_arbiter.py. Returns {block: (median_cagr_dd_b, P(b>a))}."""
    idxm = monthly_a.index.union(monthly_b.index)
    a = monthly_a.reindex(idxm, fill_value=0.0).values
    b = monthly_b.reindex(idxm, fill_value=0.0).values
    m = len(idxm)
    rng = np.random.default_rng(seed)
    out = {}
    for blk in (1, 3, 6, 12):
        nb = int(np.ceil(m / blk))
        da, db = [], []
        for _ in range(ndraw):
            st = rng.integers(0, m, nb)
            k_ = np.concatenate([(np.arange(s, s + blk) % m) for s in st])[:m]
            da.append(cdd_monthly(a[k_], m))
            db.append(cdd_monthly(b[k_], m))
        da, db = np.array(da), np.array(db)
        win = np.nanmean(db > da) * 100
        out[blk] = (np.nanmedian(da), np.nanmedian(db), win)
    return out, m


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="last ~700 trades only, for a quick correctness pass")
    ap.add_argument("--ndraw-null", type=int, default=1000)
    ap.add_argument("--ndraw-boot", type=int, default=2000)
    args = ap.parse_args()

    print("=" * 100)
    print("btc15m_L structural (fractal-pivot) exit overlay -- spec-card measurement")
    print("=" * 100)

    d15, t = get_base_trades()
    ei, e_px, risk, stop, tgt, natural_exit_bar = verify_tieback(d15, t)
    h, l, c = d15["high"].values, d15["low"].values, d15["close"].values

    if args.smoke:
        keep = np.arange(len(t) - 150, len(t))
        t = t.iloc[keep].reset_index(drop=True)
        ei, e_px, risk, stop, tgt, natural_exit_bar = (a[keep] for a in
            (ei, e_px, risk, stop, tgt, natural_exit_bar))
        print(f"  [SMOKE] subset to last {len(t)} trades\n")

    widths = WIDTHS[:2] if args.smoke else WIDTHS
    ndraw_null = 100 if args.smoke else args.ndraw_null
    ndraw_boot = 200 if args.smoke else args.ndraw_boot

    pivots = {}
    for (L, R) in widths:
        (hci, hpi, hpr), (lci, lpi, lpr) = swings_pivot_lr(h, l, L, R)
        pivots[(L, R)] = (hci, hpi, hpr, lci, lpi, lpr)
        sample = np.random.default_rng(1).integers(0, len(c), 2000)
        assert_no_lookahead(hci, hpi, lci, lpi, R, R, sample)
    print(f"  [no-lookahead check] {len(widths)} pivot widths, 2000-bar sample each: "
          f"confirm_idx<=bar AND confirm_idx-pivot_idx==R -- all asserts passed\n")

    rows = []
    cells = {}
    for (L, R) in widths:
        hci, hpi, hpr, lci, lpi, lpr = pivots[(L, R)]
        for arm in ARMS:
            for profit_only in (False, True):
                key = (arm, L, R, profit_only)
                res = run_cell(d15, t, ei, e_px, risk, stop, tgt, hci, hpi, hpr, lci, lpi, lpr,
                                arm, profit_only)
                lm = leg_metrics(res["R_struct_cost"], t["time"], res["hold_days"])
                cf = counterfactual(res["R_base_cost"], res["R_struct_cost"], res["fired"])
                rows.append(dict(arm=arm, L=L, R=R, profit_only=profit_only, **lm, **cf,
                                  fire_rate=res["fired"].mean() * 100))
                cells[key] = res

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    cols = ["arm", "L", "R", "profit_only", "n", "win", "meanR", "pf", "cagr", "dd", "cdd",
            "hold_med", "fire_rate", "fired_n", "saved_n", "saved_r", "cut_n", "cut_r", "net_r"]
    print("=" * 100)
    print("STAGE 1+2: leg metrics + counterfactual decomposition, all 40 cells")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:8.3f}"):
        print(df[cols].to_string(index=False))

    survivors = df[df["net_r"] > 0]
    print(f"\n  Pre-registered death test (net_r = saved_r - cut_r > 0): "
          f"{len(survivors)}/{len(df)} cells survive")
    if len(survivors):
        print(survivors[cols].to_string(index=False))

    # ---- baseline book tie-back (item 3, first half) ----
    print("\n" + "=" * 100)
    print("STAGE 3: 6-leg book tie-back (book_spec_fix.build('2018-01-01', False), btc15m_S->RR4.5)")
    print("=" * 100)
    base_legs = build_book_legs(gold_start="2018-01-01", gold15m_skip=False, s_rr=4.5)
    base_book = book(base_legs)
    print(f"  6-leg book: CAGR/DD={base_book[2]:.2f}  maxDD={base_book[1]:.2f}%  "
          f"CAGR={base_book[0]:.2f}%  n={base_book[3]}   [tie-back target: 8.48 / 7.49%]")

    # decide which cells to push through stages 3-5: survivors of the counterfactual test;
    # if none survive, still run ONE diagnostic cell (least-bad net_r) purely to illustrate
    # the mechanism -- clearly labelled, not a validation attempt.
    if len(survivors):
        to_run = list(survivors.itertuples(index=False))
        diag_only = False
    else:
        best = df.loc[df["net_r"].idxmax()]
        to_run = [best]
        diag_only = True
        print(f"\n  NO cell survives the counterfactual death test. Running the single "
              f"LEAST-BAD cell as a mechanism diagnostic only (NOT a candidate):")
        print(f"  {best['arm']} L={best['L']} R={best['R']} profit_only={best['profit_only']} "
              f"net_r={best['net_r']:.2f}")

    print("\n" + "=" * 100)
    print(f"STAGE 4+5: random-exit null (n={ndraw_null}) + book substitution + block bootstrap "
          f"(n={ndraw_boot}) for {'survivors' if not diag_only else 'the diagnostic cell'}")
    print("=" * 100)

    base_monthly = book_monthly(base_legs)
    for row in to_run:
        arm, L, R, po = row.arm, row.L, row.R, row.profit_only
        key = (arm, L, R, po)
        res = cells[key]
        print(f"\n  --- {arm}  L={L} R={R}  profit_only={po} "
              f"(fired={res['fired'].sum()}, fire_rate={res['fired'].mean()*100:.1f}%) ---")

        # random-exit null
        nullR = random_exit_null(d15, ei, e_px, risk, stop, tgt, natural_exit_bar,
                                  res["fired"], ndraw=ndraw_null)
        struct_meanR_fired = res["R_struct_cost"][res["fired"]].mean()
        if nullR is not None:
            pct = (nullR < struct_meanR_fired).mean() * 100
            print(f"    random-exit null (same fire count): structural fired-trade meanR="
                  f"{struct_meanR_fired:+.3f}  null median={np.median(nullR):+.3f}  "
                  f"null std={nullR.std():.3f}  percentile of structural within null={pct:.0f}%")
        else:
            print("    no fired trades -- null skipped")

        # book substitution
        eiL_all = d15.index.get_indexer(t["time"])
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        w_pdh = np.where(t["e_px"].values > pdh[eiL_all], 1.0, 0.5)
        new_leg = pd.Series(res["R_struct_cost"] * w_pdh, index=pd.DatetimeIndex(t["time"]))
        arm_legs = dict(base_legs)
        arm_legs["btc15m_L"] = new_leg
        arm_book = book(arm_legs)
        print(f"    6-leg book WITH this arm: CAGR/DD={arm_book[2]:.2f}  maxDD={arm_book[1]:.2f}%  "
              f"CAGR={arm_book[0]:.2f}%   (baseline {base_book[2]:.2f} / {base_book[1]:.2f}%)")

        # block bootstrap of the book's monthly returns
        arm_monthly = book_monthly(arm_legs)
        boot, m = block_bootstrap(base_monthly, arm_monthly, ndraw=ndraw_boot)
        print(f"    circular block bootstrap of book monthly returns (m={m} months, paired):")
        for blk, (mda, mdb, win) in boot.items():
            print(f"      block={blk:>2}mo   base med C/DD={mda:.2f}   arm med C/DD={mdb:.2f}   "
                  f"P(arm>base)={win:.0f}%")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)


if __name__ == "__main__":
    main()
