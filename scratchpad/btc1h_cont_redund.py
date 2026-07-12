"""btc1h_cont_redund.py -- REDUNDANCY check: are the 5 passing BTC-1h continuation cells
(bandwalk_exit_bounce.py / vwap_band_edge.py, screened 2026-07-10) independent of the
existing BTC book legs, or just a re-description of them (the "every trendline detector
turns out +0.7-0.9 corr with the breakout leg" pattern)?

STEP 1 rebuilds the 5 passing cells on data/vantage_btcusd_h1.csv, reusing the source
scripts' own functions VERBATIM (bandwalk_exit_bounce.compute_bb/find_walk_eventA,
vwap_band_edge.compute_vwap_sigma/find_vwap_events, _race_common.dedupe/race_matrix) --
not reimplemented. Outcome rule matches the screen (entry = event-bar CLOSE -- the bar
whose close decided the signal; race the next K=24 bars to +-1*ATR(entry); no lookahead,
since ATR is already shift(1)'d before use and the decision uses only that bar's own
close). Timeouts (neither barrier hit in 24h) are NOT scored as a flat loss (which is
what the original screen's boolean win% does): here they're separately marked and
mark-to-market at the bar+K close, so meanR/PF are not distorted by the screen's
win-oriented convention. A same-bar double-touch (both barriers hit on the same forward
bar) is scored as a loss for both sides, matching _race_common.side_win's documented
convention.

STEP 2 rebuilds the 3 BTC book legs at their literal canon configs:
  btc_bo_kama : research.portfolio_kama.get_legs()["btc_bo_kama"]  (4h, zz2/ema80/rr2/fwd300,
                daily-KAMA14-rising gate, cost=0.001 frac -- the ACTUAL adopted leg, reused
                by direct import, not re-derived)
  btc15m_L    : radar_gate_race.BASE + gate_kama=14/gate_kama_tf=240min/pullback_frac=0.3,
                m15 density-guarded, cost $15/risk post-hoc (exact construction copied from
                scratchpad/btc15m_L_anatomy.py lines ~30-48)
  btc15m_S    : short_mirror_15m.invert() + BASE + gate_kama=14 (1D gate) + pullback_frac=0.3,
                cost $15/risk post-hoc, PDL new-low intersection (exact construction copied
                from scratchpad/short_mirror_15m.py)

STEP 3: for each of the 5 cells x each of the 3 legs (+ the 3-leg book sum): monthly and
annual summed-R Pearson corr, same-calendar-day entry overlap %, and (per cell only) the
% of the cell's own signals firing while daily KAMA(14) is rising (up cells) / falling
(down cells) -- causal, prior completed day, using breakout_wave.kama_adaptive (the same
implementation the book legs' own KAMA gates use).

Usage:
  .venv/bin/python scratchpad/btc1h_cont_redund.py --smoke 2023   # 1yr sanity check first
  .venv/bin/python scratchpad/btc1h_cont_redund.py                # full history
"""
import argparse
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))   # project root
sys.path.insert(0, ROOT)                    # scratchpad/

from src.data_loader import load_mt5_csv
import _race_common as rc
import bandwalk_exit_bounce as bw
import vwap_band_edge as vw
from breakout_wave import run as run_bo, resample, kama_adaptive
from radar_gate_race import BASE
from short_mirror_15m import invert as short_invert
from research.regime_gate_lab import CFG
from research.portfolio_kama import get_legs as get_book_legs

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 25)

BTC_COST = 15.0     # flat round-trip $, canon per CLAUDE.md ("BTC canon $15 ~ real")
DATA_DIR = os.path.join(os.path.dirname(ROOT), "data")


# ============================================================================
# STEP 1: the 5 continuation cells, rebuilt on data/vantage_btcusd_h1.csv
# ============================================================================
def build_btc_h1(smoke_year=None):
    d = load_mt5_csv(os.path.join(DATA_DIR, "vantage_btcusd_h1.csv"))
    if smoke_year is not None:
        d = d[d.index.year == smoke_year]
    d = rc.resample_ohlcv(d, "1h")
    return d


def evaluate_cell(d, atr_v, high, low, close_v, event_idx, side, K, span_years):
    """Race outcome for one signal set, side='long'/'short'. Reuses
    _race_common.race_matrix verbatim (K-bar barrier race off the event bar's
    OWN close, no lookahead -- ATR is already shift(1)'d, and the race window
    is bars idx+1..idx+K, strictly after the decision bar).
    Timeout (neither barrier hit) is mark-to-market at bar idx+K's close
    (NOT scored as a flat loss, unlike the screen's win%-only convention)."""
    idx_v, first_up, first_dn, atr_e, entry_e, hi_mat, lo_mat = rc.race_matrix(
        high, low, event_idx, atr_v, close_v, K
    )
    n = len(idx_v)
    if n == 0:
        return None
    own = first_up if side == "long" else first_dn
    other = first_dn if side == "long" else first_up
    win_mask = own < other
    timeout_mask = (first_up == K) & (first_dn == K)

    exit_close = close_v[idx_v + K]
    mtm = (exit_close - entry_e) if side == "long" else (entry_e - exit_close)
    mtm_R = mtm / atr_e

    raw_R = np.where(win_mask, 1.0, np.where(timeout_mask, mtm_R, -1.0))
    cost_R_per = BTC_COST / atr_e
    net_R = raw_R - cost_R_per

    times = d.index[idx_v]
    win_pct = win_mask.mean() * 100
    n_yr = n / span_years

    def pf_of(R):
        pos = R[R > 0].sum()
        neg = abs(R[R < 0].sum())
        return pos / neg if neg > 0 else np.nan

    stats = dict(
        n=n, n_yr=n_yr, win_pct=win_pct, timeout_pct=timeout_mask.mean() * 100,
        pf_raw=pf_of(raw_R), pf_net=pf_of(net_R),
        meanR_raw=raw_R.mean(), medR_raw=np.median(raw_R), sdR_raw=raw_R.std(),
        meanR_net=net_R.mean(), medR_net=np.median(net_R), sdR_net=net_R.std(),
    )
    trades = pd.DataFrame({"time": times, "raw_R": raw_R, "net_R": net_R})
    return dict(stats=stats, trades=trades, side=side)


def build_cells(d, smoke=False):
    K = 24
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1)
    atr_v = atr.values
    mid, std = bw.compute_bb(d["close"], 20)
    up1, dn1 = (mid + std).values, (mid - std).values
    up2, dn2 = (mid + 2 * std).values, (mid - 2 * std).values
    high, low, close_v = d["high"].values, d["low"].values, d["close"].values
    n = len(d)
    span_years = (d.index[-1] - d.index[0]).days / 365.25 if n > 1 else 1.0

    bandwalk_defs = [
        # name, direction(mid_cond_kind), walk_band, reentry_band, M, side
        ("C1 bandwalk DN 2sig_M3 A-cont", "down", dn2, dn1, 3, "short"),
        ("C2 bandwalk DN 1sig_M5 A-cont", "down", dn1, dn1, 5, "short"),
        ("C3 bandwalk UP 2sig_M3 A-cont", "up",   up2, up1, 3, "long"),
        ("C4 bandwalk DN 1sig_M3 A-cont", "down", dn1, dn1, 3, "short"),
    ]

    cells = {}
    for name, direction, walk_band, reentry_band, M, side in bandwalk_defs:
        eventA = bw.find_walk_eventA(close_v, walk_band, M, reentry_band, direction)
        eventA = rc.dedupe(eventA, K)
        res = evaluate_cell(d, atr_v, high, low, close_v, eventA, side, K, span_years)
        cells[name] = res

    # C5: VWAP k=2 LOWER breakout(=downside-continuation), WIN-B window (server hour>=10)
    vwap_v, sigma_v = vw.compute_vwap_sigma(d, [])
    lower = vwap_v - 2 * sigma_v
    upper = vwap_v + 2 * sigma_v
    elapsed = (d.index - d.index.normalize()).values
    mask = elapsed >= np.timedelta64(10, "h")
    up_idx, lo_idx = vw.find_vwap_events(close_v, high, low, upper, lower, mask)
    ev5 = rc.dedupe(lo_idx, K)
    cells["C5 vwap k2 LOWER breakout WIN-B"] = evaluate_cell(
        d, atr_v, high, low, close_v, ev5, "short", K, span_years
    )
    return cells


CELL_DIRECTION = {   # for the KAMA-state check: which state SHOULD align with the cell
    "C1 bandwalk DN 2sig_M3 A-cont": "falling",
    "C2 bandwalk DN 1sig_M5 A-cont": "falling",
    "C3 bandwalk UP 2sig_M3 A-cont": "rising",
    "C4 bandwalk DN 1sig_M3 A-cont": "falling",
    "C5 vwap k2 LOWER breakout WIN-B": "falling",
}


# ============================================================================
# STEP 2: the 3 BTC book legs, canon configs
# ============================================================================
def build_book_legs(smoke_year=None):
    legs = {}

    # --- a. btc_bo_kama: the ACTUAL adopted leg, reused by direct import ---
    book = get_book_legs()
    btc_bo_kama = book["btc_bo_kama"].copy()
    if smoke_year is not None:
        btc_bo_kama = btc_bo_kama[btc_bo_kama.time.dt.year == smoke_year]
    legs["btc_bo_kama"] = btc_bo_kama.reset_index(drop=True)

    # shared m15 density-guarded 15-min BTC frame (short_mirror_15m.py / btc15m_L_anatomy.py)
    b15 = load_mt5_csv(os.path.join(DATA_DIR, "vantage_btcusd_m15.csv"))
    cnt = b15.groupby(b15.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    b15 = b15[b15.index.date >= okd.index[0]]
    if smoke_year is not None:
        b15 = b15[b15.index.year == smoke_year]
    d15 = resample(b15, "15min")

    # --- b. btc15m_L: BASE + KAMA-4h gate + pullback_frac 0.3, cost $15/risk post-hoc ---
    kwL = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3}
    tL = run_bo(d15, SimpleNamespace(**kwL))
    if tL is not None and len(tL):
        RnL = tL["R"].values - BTC_COST / tL["risk"].values
        legs["btc15m_L"] = pd.DataFrame({"time": tL["time"].values, "R": RnL})
    else:
        legs["btc15m_L"] = pd.DataFrame({"time": pd.Series([], dtype="datetime64[ns, UTC]"), "R": []})

    # --- c. btc15m_S: short mirror (inversion), BASE + KAMA-1D gate + pullback_frac 0.3,
    #        cost $15/risk post-hoc, intersected with the PDL new-low filter -----------
    inv = invert_wrap(d15)
    kwS = {**BASE, "gate_kama": 14, "pullback_frac": 0.3}   # gate_kama_tf defaults "1D"
    tS = run_bo(inv, SimpleNamespace(**kwS))
    if tS is not None and len(tS):
        RnS = tS["R"].values - BTC_COST / tS["risk"].values
        C = 2 * d15["high"].max()
        e_real = C - tS["e_px"].values
        pdl = (d15["low"].resample("1D").min().dropna().shift(1)
               .reindex(d15.index, method="ffill").values)
        idxr = d15.index.get_indexer(tS["time"])
        m = e_real < pdl[idxr]
        legs["btc15m_S"] = pd.DataFrame({"time": tS["time"].values[m], "R": RnS[m]})
    else:
        legs["btc15m_S"] = pd.DataFrame({"time": pd.Series([], dtype="datetime64[ns, UTC]"), "R": []})

    return legs, d15


def invert_wrap(d15):
    return short_invert(d15)


# ============================================================================
# KAMA(14) daily state, causal (prior completed day) -- same impl the book legs' own
# KAMA gates use (breakout_wave.kama_adaptive)
# ============================================================================
def build_daily_kama_state(smoke_year=None):
    h1 = load_mt5_csv(os.path.join(DATA_DIR, "vantage_btcusd_h1.csv"))
    if smoke_year is not None:
        h1 = h1[h1.index.year == smoke_year]
    dc = h1["close"].resample("1D").last().dropna()
    km = kama_adaptive(dc, 14)
    rising = (km > km.shift(1)).shift(1)
    falling = (km < km.shift(1)).shift(1)
    out = pd.DataFrame({"rising": rising, "falling": falling}, index=dc.index)
    out.index = out.index.tz_localize(None)   # match the naive times ".values" produces elsewhere
    return out


def kama_state_pct(times, daily_state, direction):
    if len(times) == 0:
        return float("nan")
    days = pd.DatetimeIndex(times)
    if days.tz is not None:
        days = days.tz_convert(None)
    days = days.normalize()
    s = daily_state[direction].reindex(days, method="ffill")
    return float(s.fillna(False).mean() * 100)


# ============================================================================
# STEP 3: redundancy metrics
# ============================================================================
def monthly_sum(times, R):
    if len(times) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(np.asarray(R), index=pd.DatetimeIndex(times))
    return s.resample("ME").sum()


def annual_sum(times, R):
    if len(times) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(np.asarray(R), index=pd.DatetimeIndex(times))
    return s.resample("YE").sum()


def clip_overlap(t1, t2):
    if len(t1) == 0 or len(t2) == 0:
        return None, None
    lo = max(pd.DatetimeIndex(t1).min(), pd.DatetimeIndex(t2).min())
    hi = min(pd.DatetimeIndex(t1).max(), pd.DatetimeIndex(t2).max())
    if lo >= hi:
        return None, None
    return lo, hi


def corr_pair(cell_times, cell_R, leg_times, leg_R, freq_fn):
    lo, hi = clip_overlap(cell_times, leg_times)
    if lo is None:
        return float("nan"), 0
    c_mask = (pd.DatetimeIndex(cell_times) >= lo) & (pd.DatetimeIndex(cell_times) <= hi)
    l_mask = (pd.DatetimeIndex(leg_times) >= lo) & (pd.DatetimeIndex(leg_times) <= hi)
    cs = freq_fn(np.asarray(cell_times)[c_mask], np.asarray(cell_R)[c_mask])
    ls = freq_fn(np.asarray(leg_times)[l_mask], np.asarray(leg_R)[l_mask])
    if len(cs) == 0 or len(ls) == 0:
        return float("nan"), 0
    idx = cs.index.union(ls.index)
    cs = cs.reindex(idx, fill_value=0.0)
    ls = ls.reindex(idx, fill_value=0.0)
    if cs.std() == 0 or ls.std() == 0:
        return float("nan"), len(idx)
    return float(cs.corr(ls)), len(idx)


def same_day_overlap_pct(cell_times, leg_times):
    lo, hi = clip_overlap(cell_times, leg_times)
    if lo is None:
        return float("nan"), 0
    cd = pd.DatetimeIndex(cell_times)
    ld = pd.DatetimeIndex(leg_times)
    cd = cd[(cd >= lo) & (cd <= hi)]
    ld = ld[(ld >= lo) & (ld <= hi)]
    cdays = set(cd.normalize())
    ldays = set(ld.normalize())
    if len(cdays) == 0:
        return float("nan"), 0
    return len(cdays & ldays) / len(cdays) * 100, len(cdays)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="restrict to a single calendar year first")
    args = ap.parse_args()

    smoke_year = args.smoke
    print("=" * 110)
    print(f"btc1h_cont_redund.py {'SMOKE year=' + str(smoke_year) if smoke_year else 'FULL HISTORY'}")
    print("=" * 110)

    d = build_btc_h1(smoke_year)
    print(f"BTC h1 (native vantage_btcusd_h1.csv): n_bars={len(d)}  "
          f"span={d.index[0]}..{d.index[-1]}  ({(d.index[-1]-d.index[0]).days/365.25:.2f}yr)")
    cells = build_cells(d, smoke=bool(smoke_year))

    print("\n--- STEP 1: continuation cells rebuilt on vantage_btcusd_h1.csv ---")
    for name, res in cells.items():
        if res is None:
            print(f"{name:<38} NO EVENTS")
            continue
        s = res["stats"]
        print(f"{name:<38} n={s['n']:<6d} N/yr={s['n_yr']:>6.1f}  win={s['win_pct']:>5.1f}%  "
              f"timeout={s['timeout_pct']:>5.1f}%  "
              f"PF(cost0)={s['pf_raw']:>5.2f} PF(net)={s['pf_net']:>5.2f}  "
              f"meanR(cost0)={s['meanR_raw']:+.3f} med={s['medR_raw']:+.3f} sd={s['sdR_raw']:.3f}  "
              f"meanR(net)={s['meanR_net']:+.3f} med={s['medR_net']:+.3f} sd={s['sdR_net']:.3f}")

    legs, d15 = build_book_legs(smoke_year)
    print("\n--- STEP 2: BTC book legs, canon configs ---")
    for name, t in legs.items():
        if len(t) == 0:
            print(f"{name:<14} NO TRADES")
            continue
        Rv = t["R"].values
        span = max((t.time.max() - t.time.min()).days / 365.25, 0.1)
        pf = Rv[Rv > 0].sum() / abs(Rv[Rv <= 0].sum()) if (Rv <= 0).any() else np.nan
        print(f"{name:<14} n={len(t):<6d} N/yr={len(t)/span:>6.1f}  win={(Rv>0).mean()*100:>5.1f}%  "
              f"PF={pf:>5.2f}  meanR={Rv.mean():+.3f} med={np.median(Rv):+.3f} sd={Rv.std():.3f}  "
              f"span={t.time.min()}..{t.time.max()}")

    # book-sum monthly/annual R series (for the "vs the 3-leg book" comparison)
    book_times = np.concatenate([legs[k]["time"].values for k in legs if len(legs[k])])
    book_R = np.concatenate([legs[k]["R"].values for k in legs if len(legs[k])])

    daily_state = build_daily_kama_state(smoke_year)

    print("\n--- STEP 3: redundancy metrics (monthly/annual R corr, same-day overlap%, KAMA-state) ---")
    for cname, res in cells.items():
        if res is None:
            continue
        tr = res["trades"]
        ctimes, cR = tr["time"].values, tr["net_R"].values
        direction = CELL_DIRECTION[cname]
        cell_kama_pct = kama_state_pct(ctimes, daily_state, direction)
        print(f"\n{cname}  (KAMA-{direction} at signal time: {cell_kama_pct:.1f}% of n={len(tr)})")
        for lname, t in list(legs.items()) + [("BOOK-SUM(3legs)", pd.DataFrame({"time": book_times, "R": book_R}))]:
            if len(t) == 0:
                print(f"    vs {lname:<16} NO DATA")
                continue
            ltimes, lR = t["time"].values, t["R"].values
            mcorr, nmo = corr_pair(ctimes, cR, ltimes, lR, monthly_sum)
            acorr, nyr = corr_pair(ctimes, cR, ltimes, lR, annual_sum)
            ovl_pct, novl = same_day_overlap_pct(ctimes, ltimes)
            leg_kama_pct = (kama_state_pct(ltimes, daily_state, direction)
                             if lname != "BOOK-SUM(3legs)" else float("nan"))
            print(f"    vs {lname:<16} monthly-corr={mcorr:+.2f} (n_mo={nmo:>3d})  "
                  f"annual-corr={acorr:+.2f} (n_yr={nyr:>2d})  "
                  f"same-day-overlap={ovl_pct:>5.1f}% (n_days={novl:>4d})  "
                  f"leg's own KAMA-{direction}%={leg_kama_pct:.1f}%" if not np.isnan(leg_kama_pct)
                  else f"    vs {lname:<16} monthly-corr={mcorr:+.2f} (n_mo={nmo:>3d})  "
                       f"annual-corr={acorr:+.2f} (n_yr={nyr:>2d})  "
                       f"same-day-overlap={ovl_pct:>5.1f}% (n_days={novl:>4d})")

    print("\n" + "=" * 110)
    print("DONE.")
    print("=" * 110)


if __name__ == "__main__":
    main()
