"""scalp_lab.py -- disciplined research harness for 5-min scalp hypotheses.

THE WHOLE POINT is anti-overfit discipline, not finding a winner fast. Cranking
hundreds of variants at one dataset just finds false positives faster. So:

  * DATA IS SPLIT and the TEST slice is SEALED:
        IS   (develop) : 2018-06 .. 2022-12   ~4.5yr
        VAL  (confirm) : 2023-01 .. 2024-12   ~2yr
        TEST (sealed)  : 2025-01 .. 2026-06   ~1.5yr   <- one look, EVER
    A hypothesis is judged on IS first. VAL is only looked at if IS passes.
    TEST is touched only via --unseal and is logged permanently when used.

  * PASS CRITERIA ARE PRE-REGISTERED (in docs/scalp_research_log.md) BEFORE the
    result is seen. We also report a +/-1 parameter sweep: a real edge is a
    PLATEAU (neighbours agree), an overfit is a SPIKE (lone peak).

  * EVERY hypothesis is logged -- including failures -- so multiple-testing is
    visible. The more we try, the higher the bar.

Strategies expose:  signal_fn(d, p) -> (dir, sl_px, tp_px)   (numpy arrays)
    dir[i] in {0,+1,-1}: take an entry at bar i (filled at NEXT bar open).
    sl_px[i]/tp_px[i]   : absolute price levels for that entry (NaN if no entry).
The backtester is generic: next-bar-open fill, intrabar SL/TP, forced flat at
`force_exit_h` UTC (intraday only), one position at a time, one entry/day, cost
charged once per round trip.

Gold pip = 0.1 USD.  cost default 1.4 pips (round trip).

  .venv/bin/python research/scalp_lab.py orb --csv data/vantage_xauusd_m5.csv --split is --byyear
  .venv/bin/python research/scalp_lab.py orb --csv data/vantage_xauusd_m5.csv --split is --sweep
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.regime_adaptive import kama   # validated KAMA (ER-modulated adaptive MA)

PIP = 0.1

SPLITS = {
    "is":   ("2018-06-01", "2022-12-31"),
    "val":  ("2023-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-12-31"),   # SEALED -- only via --unseal
}


# ---------------------------------------------------------------- strategies --
def orb_signals(d: pd.DataFrame, p) -> tuple:
    """Asian-range -> London/NY-open breakout (session IS the mechanism).

    Per day: range = [min low, max high] during the Asian window. The first 5m
    bar inside the breakout window whose CLOSE clears the range edge (by a buffer)
    fires a breakout. One entry/day (first break wins). Optional 'tight range'
    filter: skip days whose range is wider than max_range_atr * ATR.
    SL = opposite range edge (+/- buffer); TP = rr * range width (measured move).
    """
    hi, lo, cl = d["high"].values, d["low"].values, d["close"].values
    op = d["open"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    rsi = ta.rsi(d["close"], 14).values                    # exhaustion filter (--rsi-max)
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values
    n = len(cl)

    asia = (minute >= p.asia_start_h * 60) & (minute < p.asia_end_h * 60)
    # GUARD (no-lookahead): the box (rh/rl) is computed over the WHOLE asia window up front, so a breakout
    # bar that sits INSIDE that window would be tested against a box containing its own future bars =
    # lookahead. Force breakout bars to start at/after asia_end regardless of bo_start_h.
    if p.bo_start_h < p.asia_end_h:
        print(f"[orb] WARN: bo_start_h({p.bo_start_h}) < asia_end_h({p.asia_end_h}) -> breakouts before "
              f"asia_end are SKIPPED to avoid lookahead (box peeks at future bars).", file=sys.stderr)
    bo = (minute >= max(p.bo_start_h, p.asia_end_h) * 60) & (minute < p.bo_end_h * 60)

    dir_ = np.zeros(n, np.int8)
    sl_px = np.full(n, np.nan)
    tp_px = np.full(n, np.nan)

    # group bar indices by day
    uniq, start = np.unique(day, return_index=True)
    bounds = list(start) + [n]
    for di in range(len(uniq)):
        a, b = bounds[di], bounds[di + 1]
        seg = slice(a, b)
        amask = asia[seg]
        if not amask.any():
            continue
        rh = hi[seg][amask].max()
        rl = lo[seg][amask].min()
        rw = rh - rl
        if rw <= 0:
            continue
        # box DIRECTIONALITY: net move across the Asian window / range width. ~0 = a true
        # range/coil; ->+1 = trended UP the whole window (not a range -> the user's 'kept
        # making new highs' case); ->-1 = trended down. Used by --box-trend-max to skip
        # breakouts in the direction the box ALREADY ran (late, extended entry).
        a_idx = a + np.where(amask)[0]
        box_dir = (cl[a_idx[-1]] - op[a_idx[0]]) / rw
        ref_atr = atr[a + np.where(amask)[0][-1]]            # ATR at end of Asian window
        if not np.isnan(ref_atr):
            if p.max_range_atr > 0 and rw > p.max_range_atr * ref_atr:
                continue
            if p.min_range_atr > 0 and rw < p.min_range_atr * ref_atr:
                continue
        buf = p.buf_atr * (ref_atr if not np.isnan(ref_atr) else 0.0)
        slb = p.sl_buf_atr * (ref_atr if not np.isnan(ref_atr) else 0.0)
        # first breakout bar inside the breakout window
        for j in range(a, b):
            if not bo[j]:
                continue
            up_break = cl[j] > rh + buf
            dn_break = cl[j] < rl - buf
            if not (up_break or dn_break):
                continue
            # exhaustion filter: if the 'box' was actually a TREND (price already extended),
            # the breakout is a late entry that tends to revert. Skip overbought longs /
            # oversold shorts. rsi_max=100 -> off.  (the user's RSI>80 idea, made testable)
            if p.rsi_max < 100 and not np.isnan(rsi[j]):
                if up_break and rsi[j] >= p.rsi_max:
                    up_break = False
                if dn_break and rsi[j] <= 100 - p.rsi_max:
                    dn_break = False
                if not (up_break or dn_break):
                    continue
            # box-directionality filter: skip a breakout in the direction the box ALREADY
            # ran (box_dir>=thr for longs / <=-thr for shorts = 'not a range, a trend').
            if p.box_trend_max < 1:
                if up_break and box_dir >= p.box_trend_max:
                    up_break = False
                if dn_break and box_dir <= -p.box_trend_max:
                    dn_break = False
                if not (up_break or dn_break):
                    continue
            if not p.fade:
                # breakout continuation. SL = sl_frac of the range BELOW the broken edge:
                # sl_frac=1.0 -> opposite range edge (original); <1.0 -> tighter ABORT stop
                # that exits a failed breakout that falls back INTO the range (the user's
                # 'fakeout bled to EOD' loser) instead of riding it to the session close.
                if up_break and p.dir in ("both", "long"):
                    dir_[j] = 1; sl_px[j] = (rh + buf) - p.sl_frac * rw - slb
                    tp_px[j] = np.nan if p.no_tp else (rh + buf) + p.rr * rw
                elif dn_break and p.dir in ("both", "short"):
                    dir_[j] = -1; sl_px[j] = (rl - buf) + p.sl_frac * rw + slb
                    tp_px[j] = np.nan if p.no_tp else (rl - buf) - p.rr * rw
            else:
                # FADE: bet the break is a fakeout, revert into the range
                if up_break and p.dir in ("both", "short"):
                    dir_[j] = -1; sl_px[j] = (rh + buf) + slb + p.sl_buf_atr * ref_atr
                    tp_px[j] = (rh + buf) - p.rr * rw
                elif dn_break and p.dir in ("both", "long"):
                    dir_[j] = 1; sl_px[j] = (rl - buf) - slb - p.sl_buf_atr * ref_atr
                    tp_px[j] = (rl - buf) + p.rr * rw
            if dir_[j] != 0:
                break
    return dir_, sl_px, tp_px


def squeeze_signals(d: pd.DataFrame, p) -> tuple:
    """Volatility CONTRACTION -> EXPANSION breakout (the user's actual setup):
    recently the market moved big (wide boxes), then it COILS (a short window's
    range shrinks to a fraction of that recent max), then price breaks the coil.
    Event-driven, any time of day. Small coil box => small stop => high RR if it
    runs. One entry/day (first qualifying break).

      box (last sq_win bars) : box_hi/box_lo/box_rng
      recent expansion       : recent_max = max box_rng over last sq_look bars
      squeeze ON             : box_rng <= sq_ratio * recent_max          (coiled)
      (optional) prior move  : recent_max >= exp_atr * ATR               (was big)
      entry                  : while squeezed, close breaks the PRIOR coil box edge
      SL                     : opposite coil-box edge (+/- buffer);  TP: rr*box or none
    """
    hi, lo, cl = d["high"], d["low"], d["close"]
    atr = ta.atr(hi, lo, cl, length=14).values
    hv, lv, cv = hi.values, lo.values, cl.values
    n = len(cv)
    box_hi = hi.rolling(p.sq_win).max()
    box_lo = lo.rolling(p.sq_win).min()
    box_rng = (box_hi - box_lo)
    recent_max = box_rng.rolling(p.sq_look).max()
    squeeze_on = (box_rng <= p.sq_ratio * recent_max).values
    bh, bl, rng_p = box_hi.shift(1).values, box_lo.shift(1).values, box_rng.shift(1).values
    rmax_p, sq_p = recent_max.shift(1).values, pd.Series(squeeze_on).shift(1).fillna(False).values
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values

    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur_day = None; done = False
    for i in range(n):
        if day[i] != cur_day:
            cur_day, done = day[i], False
        if done or not sq_p[i] or np.isnan(bh[i]) or np.isnan(atr[i]):
            continue
        if p.exp_atr > 0 and not (rmax_p[i] >= p.exp_atr * atr[i]):
            continue
        if p.force_exit_h and minute[i] >= p.force_exit_h * 60:   # no fresh entries past cutoff
            continue
        buf = p.buf_atr * atr[i]; slb = p.sl_buf_atr * atr[i]
        up = cv[i] > bh[i] + buf
        dn = cv[i] < bl[i] - buf
        if p.cont and i - p.sq_look - 1 >= 0:           # continuation: break must follow prior move
            prior = cv[i - 1] - cv[i - 1 - p.sq_look]
            if prior > 0:
                dn = False
            elif prior < 0:
                up = False
        if not p.fade:
            if up and p.dir in ("both", "long"):
                dir_[i] = 1; sl_px[i] = bl[i] - slb
                tp_px[i] = np.nan if p.no_tp else (bh[i] + buf) + p.rr * rng_p[i]
            elif dn and p.dir in ("both", "short"):
                dir_[i] = -1; sl_px[i] = bh[i] + slb
                tp_px[i] = np.nan if p.no_tp else (bl[i] - buf) - p.rr * rng_p[i]
        else:
            fsl = slb if slb > 0 else atr[i]            # real stop beyond the break (>=1 ATR)
            if up and p.dir in ("both", "short"):
                dir_[i] = -1; sl_px[i] = (bh[i] + buf) + fsl
                tp_px[i] = (bh[i] + buf) - p.rr * rng_p[i]
            elif dn and p.dir in ("both", "long"):
                dir_[i] = 1; sl_px[i] = (bl[i] - buf) - fsl
                tp_px[i] = (bl[i] - buf) + p.rr * rng_p[i]
        if dir_[i] != 0:
            done = True
    return dir_, sl_px, tp_px


def htf_pivots(d: pd.DataFrame, rule: str, k: int):
    """Higher-TF fractal pivot lows/highs = PROVEN S/R (price reacted there before).
    A pivot at HTF bar j is only KNOWN k bars later -> we map that confirm time to the
    M5 position from which it may be used (no lookahead). Returns sorted arrays of
    (m5_pos_when_known, price) for support (lows) and resistance (highs)."""
    h = d["high"].resample(rule, label="left", closed="left").max().dropna()
    lo = d["low"].resample(rule, label="left", closed="left").min().dropna()
    idx = h.index; hv = h.values; lv = lo.values; m = len(idx)
    st, sp, rt, rp = [], [], [], []
    for j in range(k, m - k):
        if lv[j] == lv[j - k:j + k + 1].min():
            st.append(idx[j + k]); sp.append(lv[j])
        if hv[j] == hv[j - k:j + k + 1].max():
            rt.append(idx[j + k]); rp.append(hv[j])
    m5 = d.index
    spos = m5.searchsorted(pd.DatetimeIndex(st)) if st else np.array([], int)
    rpos = m5.searchsorted(pd.DatetimeIndex(rt)) if rt else np.array([], int)
    return np.asarray(spos), np.asarray(sp), np.asarray(rpos), np.asarray(rp)


def bounce_signals(d: pd.DataFrame, p) -> tuple:
    """HTF support/resistance bounce (the user's setup): price falls back to a
    higher-TF level, shows a reversal on the current TF, ride it, TP = recent
    swing high/low.  Mean-reversion (aligns with 'gold 5m fades breaks').

      level (proxy for HTF S/R) : support = lowest low over level_lb bars (shifted);
                                  resistance = highest high over level_lb (shifted).
      tag                       : price dips to within tol*ATR of the level.
      reversal confirm          : bullish bar that holds back above support
                                  (mirror at resistance).
      --double-bottom           : require a PRIOR tag of the same zone within db_win
                                  bars (so this is the 2nd touch, not the 1st).
      entry next open. SL = recent reversal low - buf.  TP = recent swing high
      (highest high over tp_lb bars). One entry/day. Skip if TP not beyond entry.
    """
    hi, lo, cl, op = d["high"], d["low"], d["close"], d["open"]
    atr = ta.atr(hi, lo, cl, length=14).values
    hv, lv, cv, ov = hi.values, lo.values, cl.values, op.values
    n = len(cv)
    sup = lo.rolling(p.level_lb).min().shift(1).values
    res = hi.rolling(p.level_lb).max().shift(1).values
    swing_hi = hi.rolling(p.tp_lb).max().shift(1).values     # recent swing high (TP for longs)
    swing_lo = lo.rolling(p.tp_lb).min().shift(1).values
    sl_lo = lo.rolling(p.sl_lb).min().values                 # reversal low for SL
    sl_hi = hi.rolling(p.sl_lb).max().values
    # confirmation inputs: local high/low (structure break) + RSI (momentum)
    hh = hi.rolling(p.cf_lb).max().shift(1).values           # recent local high (break = HH)
    ll = lo.rolling(p.cf_lb).min().shift(1).values
    rsi = ta.rsi(cl, 14).values
    confirms = set(c.strip() for c in p.confirm.split(",") if c.strip())
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values

    # HTF pivot levels (15m + 1h ...) for S/R zone, trendline, and htf-level core.
    use_piv = p.htf_sr or p.trendline or p.htf_level
    spos = sp = rpos = rp = np.array([])
    if use_piv:
        sps, sxs, rps, rxs = [], [], [], []
        for rule in (t.strip() for t in p.piv_tf.split(",") if t.strip()):
            a, b, c2, d2 = htf_pivots(d, rule, p.piv_k)
            sps.append(a); sxs.append(b); rps.append(c2); rxs.append(d2)
        spos = np.concatenate(sps) if sps else np.array([], int)
        sp = np.concatenate(sxs) if sxs else np.array([])
        rpos = np.concatenate(rps) if rps else np.array([], int)
        rp = np.concatenate(rxs) if rxs else np.array([])
        o1 = np.argsort(spos, kind="stable"); spos, sp = spos[o1], sp[o1]
        o2 = np.argsort(rpos, kind="stable"); rpos, rp = rpos[o2], rp[o2]
    cur_sup = []; cur_res = []; ip_s = 0; ip_r = 0      # (pos, price) confirmed so far

    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur_day = None; done = False
    arm_l = arm_s = False; ab_l = ab_s = 0
    alow_l = asup_l = ahigh_s = ares_s = 0.0
    for i in range(p.level_lb + 1, n):
        if use_piv:                                      # advance confirmed-pivot lists to bar i
            while ip_s < len(spos) and spos[ip_s] <= i:
                cur_sup.append((spos[ip_s], sp[ip_s])); ip_s += 1
                if len(cur_sup) > p.sr_keep: cur_sup.pop(0)
            while ip_r < len(rpos) and rpos[ip_r] <= i:
                cur_res.append((rpos[ip_r], rp[ip_r])); ip_r += 1
                if len(cur_res) > p.sr_keep: cur_res.pop(0)
        if day[i] != cur_day:
            cur_day, done = day[i], False
        if done or np.isnan(sup[i]) or np.isnan(atr[i]):
            continue
        if p.force_exit_h and minute[i] >= p.force_exit_h * 60:
            continue
        tol = p.tol_atr * atr[i]; buf = p.sl_buf_atr * atr[i]

        # --- selection filters (the user's discretionary cues, made explicit) ---
        def sr_near(levels, price):                      # near a PROVEN HTF level
            return any(abs(price - lv) <= tol for _, lv in levels)

        def tl_ok(levels, price, want_up):               # touch a projected trendline
            if len(levels) < 2:
                return False
            (x1, y1), (x2, y2) = levels[-2], levels[-1]
            if x2 == x1:
                return False
            slope = (y2 - y1) / (x2 - x1)
            if (want_up and slope <= 0) or (not want_up and slope >= 0):
                return False
            proj = y2 + slope * (i - x2)
            return abs(price - proj) <= tol

        def htf_ref(levels, price, up):                  # nearest HTF pivot level on the trade side
            cands = [lv for _, lv in levels if (lv <= price + tol if up else lv >= price - tol)]
            if not cands:
                return None
            return max(cands) if up else min(cands)

        def db_ok(arr, level, up):
            if not p.double_bottom:
                return True
            lo0, hi0 = max(p.level_lb + 1, i - p.db_win), i - p.db_gap
            if hi0 <= lo0:
                return False
            return np.any(arr[lo0:hi0] <= level + tol) if up else np.any(arr[lo0:hi0] >= level - tol)

        def confirmed(up):
            """Did the reversal actually TAKE HOLD? require ALL chosen confirmations."""
            for c in confirms:
                if c == "simple":
                    ok = (cv[i] > ov[i] and cv[i] > asup_l) if up else (cv[i] < ov[i] and cv[i] < ares_s)
                elif c == "struct":              # break a recent local high/low (higher high / lower low)
                    ok = (not np.isnan(hh[i]) and cv[i] > hh[i]) if up else (not np.isnan(ll[i]) and cv[i] < ll[i])
                elif c == "twobar":              # close beyond the PRIOR bar's extreme (follow-through)
                    ok = cv[i] > hv[i - 1] if up else cv[i] < lv[i - 1]
                elif c == "mom":                 # RSI turning in the trade direction
                    ok = (not np.isnan(rsi[i]) and rsi[i] > rsi[i - 1] and rsi[i] >= p.mom_th) if up \
                        else (not np.isnan(rsi[i]) and rsi[i] < rsi[i - 1] and rsi[i] <= 100 - p.mom_th)
                elif c == "stall":               # "下落が止まった": wick-rejection at the level
                    rng = hv[i] - lv[i]
                    ok = rng > 0 and ((cv[i] - lv[i]) >= p.rej * rng if up else (hv[i] - cv[i]) >= p.rej * rng)
                else:
                    ok = True
                if not ok:
                    return False
            return True

        # ---- LONG at support: tag -> wait (cf_win bars) -> confirm -> enter ----
        if p.dir in ("both", "long"):
            if arm_l and (i - ab_l > p.cf_win or cv[i] < asup_l - tol):
                arm_l = False                                # timeout / lost the level
            if not arm_l:
                lvl = htf_ref(cur_sup, cv[i], True) if p.htf_level else sup[i]   # HTF pivot or M5 rolling
                tag = lvl is not None and lv[i] <= lvl + tol and db_ok(lv, lvl, True)
                if p.htf_sr and not sr_near(cur_sup, lv[i]):
                    tag = False
                if p.trendline and not tl_ok(cur_sup, lv[i], want_up=True):
                    tag = False
                if tag:
                    arm_l, ab_l, alow_l, asup_l = True, i, lv[i], lvl
            if arm_l:
                alow_l = min(alow_l, lv[i])                  # track the real reversal low (for SL)
                if confirmed(True):
                    tp = swing_hi[i]; sl = alow_l - buf
                    if tp > cv[i] > sl and (tp - cv[i]) >= p.min_rr * (cv[i] - sl):
                        dir_[i] = 1; sl_px[i] = sl; tp_px[i] = tp; arm_l = False
        # ---- SHORT at resistance ----
        if dir_[i] == 0 and p.dir in ("both", "short"):
            if arm_s and (i - ab_s > p.cf_win or cv[i] > ares_s + tol):
                arm_s = False
            if not arm_s:
                lvl = htf_ref(cur_res, cv[i], False) if p.htf_level else res[i]
                tag = lvl is not None and hv[i] >= lvl - tol and db_ok(hv, lvl, False)
                if p.htf_sr and not sr_near(cur_res, hv[i]):
                    tag = False
                if p.trendline and not tl_ok(cur_res, hv[i], want_up=False):
                    tag = False
                if tag:
                    arm_s, ab_s, ahigh_s, ares_s = True, i, hv[i], lvl
            if arm_s:
                ahigh_s = max(ahigh_s, hv[i])
                if confirmed(False):
                    tp = swing_lo[i]; sl = ahigh_s + buf
                    if tp < cv[i] < sl and (cv[i] - tp) >= p.min_rr * (sl - cv[i]):
                        dir_[i] = -1; sl_px[i] = sl; tp_px[i] = tp; arm_s = False
        if dir_[i] != 0:
            done = True
    return dir_, sl_px, tp_px


def streak_signals(d: pd.DataFrame, p) -> tuple:
    """H10 -- consecutive-bar EXHAUSTION fade (keys on SEQUENCE/velocity, not a price level).
    Distinct from S/R bounce (fades at a fixed price) and break-fade (fades a range edge):
    the trigger is a run COUNTER. On the close of the Nth consecutive same-colour M5 bar
    whose total run-range >= stretch*ATR (only fade STRETCHED runs), enter the OPPOSITE
    direction next open. SL beyond the run extreme (+buf); TP = rr * risk. Confirmed-close,
    one entry/day. The wall-test: does exhaustion (how FAST we got here) beat MFE~MAE where
    location did not?"""
    hi, lo, cl, op = d["high"], d["low"], d["close"], d["open"]
    atr = ta.atr(hi, lo, cl, length=14).values
    hv, lv, cv, ov = hi.values, lo.values, cl.values, op.values
    n = len(cv)
    color = np.sign(cv - ov).astype(int)            # +1 green, -1 red, 0 doji
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values

    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur_day = None; done = False
    run_dir = 0; run_len = 0; run_start = 0
    for i in range(n):
        ci = color[i]                               # update the streak counter first
        if ci == 0:
            run_dir, run_len = 0, 0
        elif ci == run_dir:
            run_len += 1
        else:
            run_dir, run_len, run_start = ci, 1, i
        if day[i] != cur_day:
            cur_day, done = day[i], False
        if done or np.isnan(atr[i]):
            continue
        if p.force_exit_h and minute[i] >= p.force_exit_h * 60:
            continue
        if run_len < p.streak_n or run_dir == 0:
            continue
        seg_hi = hv[run_start:i + 1].max(); seg_lo = lv[run_start:i + 1].min()
        if p.streak_stretch > 0 and (seg_hi - seg_lo) < p.streak_stretch * atr[i]:
            continue
        buf = p.sl_buf_atr * atr[i]
        ref = cv[i]
        if run_dir > 0 and p.dir in ("both", "short"):          # green run -> fade short
            sl = seg_hi + buf; risk = sl - ref
            if risk > 0:
                dir_[i] = -1; sl_px[i] = sl; tp_px[i] = ref - p.rr * risk
        elif run_dir < 0 and p.dir in ("both", "long"):         # red run -> fade long
            sl = seg_lo - buf; risk = ref - sl
            if risk > 0:
                dir_[i] = 1; sl_px[i] = sl; tp_px[i] = ref + p.rr * risk
        if dir_[i] != 0:
            done = True
    return dir_, sl_px, tp_px


def vwap_signals(d: pd.DataFrame, p) -> tuple:
    """H11 -- session-anchored VWAP reversion band. VWAP is a COORDINATED, volume-weighted
    reference (real participants trade around it) -- categorically different from the static
    rolling-extrema 'HTF S/R' of the bounce family. Price closes >= k*sigma beyond session
    VWAP -> fade TOWARD VWAP (confirmed-close beyond the band, NOT intrabar touch).
    SL = (k+1)*sigma extension; TP = VWAP. One entry/day. Wall-test: does a coordinated
    reference make the reaction asymmetric where arbitrary fractal levels did not?"""
    h, l, c, v = d["high"], d["low"], d["close"], d["volume"]
    tpv = ((h + l + c) / 3.0).values
    vol = v.values.astype(float)
    cv = c.values
    n = len(cv)
    sess = (d.index - pd.Timedelta(hours=p.vwap_anchor_h)).normalize().values
    vwap = np.full(n, np.nan)
    cum_pv = cum_v = 0.0
    cur = None
    for i in range(n):
        if sess[i] != cur:
            cur, cum_pv, cum_v = sess[i], 0.0, 0.0
        cum_pv += tpv[i] * vol[i]; cum_v += vol[i]
        if cum_v > 0:
            vwap[i] = cum_pv / cum_v
    sigma = pd.Series(cv - vwap).rolling(p.vwap_lb).std().values
    minute = (d.index.hour * 60 + d.index.minute).values
    cday = d.index.normalize().values

    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur_day = None; done = False
    for i in range(n):
        if cday[i] != cur_day:
            cur_day, done = cday[i], False
        if done or np.isnan(vwap[i]) or np.isnan(sigma[i]) or sigma[i] <= 0:
            continue
        if p.force_exit_h and minute[i] >= p.force_exit_h * 60:
            continue
        band = p.vwap_k * sigma[i]
        if cv[i] >= vwap[i] + band and p.dir in ("both", "short"):      # stretched high -> fade short
            sl = vwap[i] + (p.vwap_k + 1.0) * sigma[i]; tp = vwap[i]
            if sl > cv[i] > tp:
                dir_[i] = -1; sl_px[i] = sl; tp_px[i] = tp; done = True
        elif cv[i] <= vwap[i] - band and p.dir in ("both", "long"):     # stretched low -> fade long
            sl = vwap[i] - (p.vwap_k + 1.0) * sigma[i]; tp = vwap[i]
            if sl < cv[i] < tp:
                dir_[i] = 1; sl_px[i] = sl; tp_px[i] = tp; done = True
    return dir_, sl_px, tp_px


def nr7_signals(d: pd.DataFrame, p) -> tuple:
    """H14 -- NR7/inside-bar COMPRESSION as a TIMING clock, with DIRECTION imported from the
    HTF trend (1h EMA slope, shift(1) = prev completed hour, no lookahead). Distinct from the
    dead squeeze-continuation (H5 bet the BREAK direction = a coinflip); here an INDEPENDENT
    variable picks direction and compression only picks WHEN. Confirmed-close break of the NR
    bar's range, trend-aligned only (unless --nr-bothdir). SL = opposite side of NR bar; TP = rr*risk."""
    h, l, c = d["high"], d["low"], d["close"]
    atr = ta.atr(h, l, c, length=14).values
    hv, lv, cv = h.values, l.values, c.values
    rng = (h - l).values
    n = len(cv)
    rollmin = pd.Series(rng).rolling(p.nr_lb).min().values
    is_nr = (rng <= rollmin) & ~np.isnan(rollmin)
    inside = np.zeros(n, bool)
    inside[1:] = (hv[1:] < hv[:-1]) & (lv[1:] > lv[:-1])
    compress = is_nr | inside
    # HTF 1h EMA slope, no-lookahead: value known only at hour end -> shift(1) = prev completed hour
    c1h = c.resample("1h", label="left", closed="left").last().dropna()
    ema = c1h.ewm(span=p.ema_len, adjust=False).mean()
    slope = np.sign(ema.diff(p.slope_k)).shift(1)
    slope_m5 = slope.reindex(d.index, method="ffill").values
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values

    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur_day = None; done = False
    for i in range(1, n):
        if day[i] != cur_day:
            cur_day, done = day[i], False
        if done or np.isnan(atr[i]) or not compress[i - 1]:
            continue
        if p.force_exit_h and minute[i] >= p.force_exit_h * 60:
            continue
        sd = slope_m5[i]
        if np.isnan(sd) or sd == 0:
            continue
        nh, nl = hv[i - 1], lv[i - 1]
        up = cv[i] > nh; dn = cv[i] < nl
        if not getattr(p, "nr_bothdir", False):         # trend-aligned only
            if sd > 0:
                dn = False
            else:
                up = False
        buf = p.sl_buf_atr * atr[i]; ref = cv[i]
        if up and p.dir in ("both", "long"):
            sl = nl - buf; risk = ref - sl
            if risk > 0:
                dir_[i] = 1; sl_px[i] = sl; tp_px[i] = ref + p.rr * risk; done = True
        elif dn and p.dir in ("both", "short"):
            sl = nh + buf; risk = sl - ref
            if risk > 0:
                dir_[i] = -1; sl_px[i] = sl; tp_px[i] = ref - p.rr * risk; done = True
    return dir_, sl_px, tp_px


def session_signals(d: pd.DataFrame, p) -> tuple:
    """H18 -- box-LESS London-open trend ride. Each day, at the FIRST bar at/after
    bo_start_h (and before bo_end_h), enter in the HTF (1H default) trend direction
    -- close vs EMA(htf_ema), shift(1) = prev completed HTF bar, no lookahead --
    and ride to force_exit_h. NO range, NO breakout trigger, NO box-SL. This isolates
    the three suspected real levers (gate + session-timing + ride-to-close) from the
    decorative Asian-range box. Optional --sl-atr adds an ATR stop (0 = ride-only)."""
    c = d["close"]
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    cv = c.values
    n = len(cv)
    chtf = c.resample(p.htf_tf or "1h", label="left", closed="left").last().dropna()
    ema = chtf.ewm(span=p.htf_ema, adjust=False).mean()
    if p.htf_slope_k > 0:
        updn = np.sign(ema.diff(p.htf_slope_k))
    else:
        updn = np.sign(chtf - ema)
    trend = updn.shift(1).reindex(d.index, method="ffill").values
    minute = (d.index.hour * 60 + d.index.minute).values
    day = d.index.normalize().values
    bo = (minute >= p.bo_start_h * 60) & (minute < p.bo_end_h * 60)

    dir_ = np.zeros(n, np.int8); sl_px = np.full(n, np.nan); tp_px = np.full(n, np.nan)
    cur_day = None; done = False
    for i in range(n):
        if day[i] != cur_day:
            cur_day, done = day[i], False
        if done or not bo[i] or np.isnan(atr[i]) or np.isnan(trend[i]) or trend[i] == 0:
            continue
        sd = int(trend[i]); slb = p.sl_atr * atr[i]
        if sd > 0 and p.dir in ("both", "long"):
            dir_[i] = 1; sl_px[i] = (cv[i] - slb) if p.sl_atr > 0 else np.nan
        elif sd < 0 and p.dir in ("both", "short"):
            dir_[i] = -1; sl_px[i] = (cv[i] + slb) if p.sl_atr > 0 else np.nan
        if dir_[i] != 0:
            done = True                          # one entry/day; tp stays NaN = ride to close
    return dir_, sl_px, tp_px


STRATS = {"orb": orb_signals, "squeeze": squeeze_signals, "bounce": bounce_signals,
          "streak": streak_signals, "vwap": vwap_signals, "nr7": nr7_signals,
          "session": session_signals}


# ---------------------------------------------------------------- backtester --
def backtest(d: pd.DataFrame, dir_, sl_px, tp_px, p) -> pd.DataFrame:
    op, hi, lo = d["open"].values, d["high"].values, d["low"].values
    minute = (d.index.hour * 60 + d.index.minute).values
    idx = d.index
    n = len(op)
    cost = p.cost * PIP
    trades = []
    pos = 0; ei = 0; e_px = stop = tp = 0.0

    def close(j, px):
        nonlocal pos
        g = (px - e_px) if pos > 0 else (e_px - px)
        trades.append((idx[ei], pos, g / PIP - cost / PIP, j - ei))
        pos = 0

    for i in range(n - 1):
        if pos != 0:
            # forced intraday flat at/after force_exit_h
            if minute[i] >= p.force_exit_h * 60:
                close(i, op[i]); continue
            # stop_slip models a headline candle GAPPING THROUGH the stop: the SL fills
            # stop_slip pips WORSE than the stop price (the backtest's blind spot). TP
            # fills stay exact. Applied to every stop = conservative worst-case envelope.
            sl_slip = p.stop_slip * PIP
            if pos > 0:
                if lo[i] <= stop:
                    close(i, stop - sl_slip)
                elif hi[i] >= tp:
                    close(i, tp)
            else:
                if hi[i] >= stop:
                    close(i, stop + sl_slip)
                elif lo[i] <= tp:
                    close(i, tp)
        if pos != 0:
            continue
        if dir_[i] == 0:
            continue
        e_px = op[i + 1]; ei = i + 1; pos = int(dir_[i])
        stop = sl_px[i]; tp = tp_px[i]
    return pd.DataFrame(trades, columns=["t_in", "dir", "pips", "bars"])


# ---------------------------------------------------------------- vol gate ----
def vol_gate(d: pd.DataFrame, dir_, sl_px, tp_px, p):
    """H12 meta-gate: zero out entries whose ATR(14) is outside a TRAILING percentile
    band (q33/q66 over the last atr_win bars -> no lookahead). Tests whether H8's
    post-entry symmetry is a POOLING artifact of opposite vol regimes."""
    if getattr(p, "vol_band", "all") == "all":
        return dir_, sl_px, tp_px
    atr = ta.atr(d["high"], d["low"], d["close"], length=14)
    q33 = atr.rolling(p.atr_win).quantile(0.33).values
    q66 = atr.rolling(p.atr_win).quantile(0.66).values
    a = atr.values
    if p.vol_band == "low":
        m = a <= q33
    elif p.vol_band == "mid":
        m = (a > q33) & (a <= q66)
    else:
        m = a > q66
    m = np.where(np.isnan(q33) | np.isnan(q66), False, m)
    dir_ = dir_.copy()
    dir_[~m] = 0
    return dir_, sl_px, tp_px


# ---------------------------------------------------------------- daily gate --
def daily_gate(d: pd.DataFrame, dir_, sl_px, tp_px, p):
    """Regime gate (the project's validated gold lever): longs only when the PRIOR completed
    daily close > daily SMA(N) [and SMA rising if slope_k>0]; shorts mirror. Gap-days dropped +
    shift(1)+ffill = no lookahead (matches breakout_wave.py / iMA on real daily bars). Removes
    the chop-year bleed that pins ungated gold breakout at breakeven."""
    if getattr(p, "daily_sma", 0) <= 0:
        return dir_, sl_px, tp_px
    dc = d["close"].resample("1D").last().dropna()
    sma = dc.rolling(p.daily_sma).mean()
    above = dc > sma
    if p.daily_slope_k > 0:
        rising = sma > sma.shift(p.daily_slope_k)
        long_d, short_d = above & rising, (~above) & (sma < sma.shift(p.daily_slope_k))
    else:
        long_d, short_d = above, ~above
    long_ok = long_d.shift(1).reindex(d.index, method="ffill").fillna(False).values
    short_ok = short_d.shift(1).reindex(d.index, method="ffill").fillna(False).values
    dir_ = dir_.copy()
    dir_[(dir_ > 0) & ~long_ok] = 0
    dir_[(dir_ < 0) & ~short_ok] = 0
    return dir_, sl_px, tp_px


# ------------------------------------------------------------- htf-trend gate -
def kama_gate(d: pd.DataFrame, dir_, sl_px, tp_px, p):
    """KAMA-rising regime gate at a CONFIGURABLE TF (--kama-tf: 1h/2h/4h/1D), so the gate
    granularity can match a 15m execution TF instead of being forced to daily. ER-modulated
    KAMA self-adapts to vol (the one adaptive gate that transferred across the breakout family).
    rising = kama > kama.shift(slope_k) on the gate TF; longs only when rising, shorts when
    falling. KAMA known only at its bar close -> shift(1)+ffill = no lookahead."""
    ktf = getattr(p, "kama_tf", "")
    if not ktf:
        return dir_, sl_px, tp_px
    ck = d["close"].resample(ktf, label="left", closed="left").last().dropna()
    km = kama(ck, getattr(p, "kama_n", 14))
    k = getattr(p, "kama_slope_k", 1)
    up_d, down_d = km > km.shift(k), km < km.shift(k)
    long_ok = up_d.shift(1).reindex(d.index, method="ffill").fillna(False).values
    short_ok = down_d.shift(1).reindex(d.index, method="ffill").fillna(False).values
    dir_ = dir_.copy()
    dir_[(dir_ > 0) & ~long_ok] = 0
    dir_[(dir_ < 0) & ~short_ok] = 0
    return dir_, sl_px, tp_px


def htf_trend_gate(d: pd.DataFrame, dir_, sl_px, tp_px, p):
    """Proportionate trend gate for a low execution TF: align entries with the 1H/4H trend
    (EMA slope if htf_slope_k>0, else price-vs-EMA). HTF value known only at its bar close ->
    shift(1)+ffill = no lookahead. Lighter than the daily gate for an M15 execution TF."""
    if not getattr(p, "htf_tf", ""):
        return dir_, sl_px, tp_px
    chtf = d["close"].resample(p.htf_tf, label="left", closed="left").last().dropna()
    ema = chtf.ewm(span=p.htf_ema, adjust=False).mean()
    if p.htf_slope_k > 0:
        up_d, down_d = ema > ema.shift(p.htf_slope_k), ema < ema.shift(p.htf_slope_k)
    else:
        up_d, down_d = chtf > ema, chtf < ema
    long_ok = up_d.shift(1).reindex(d.index, method="ffill").fillna(False).values
    short_ok = down_d.shift(1).reindex(d.index, method="ffill").fillna(False).values
    dir_ = dir_.copy()
    dir_[(dir_ > 0) & ~long_ok] = 0
    dir_[(dir_ < 0) & ~short_ok] = 0
    return dir_, sl_px, tp_px


# ---------------------------------------------------------------- reporting ---
def metrics(t: pd.DataFrame):
    if len(t) == 0:
        return None
    wins = t[t.pips > 0].pips; loss = t[t.pips < 0].pips
    pf = wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() else float("inf")
    one_r = abs(loss.mean()) if len(loss) else 1.0
    rets = (t.pips.values / one_r) * 0.01     # 1% risk per 1R-ish
    eqc = np.cumprod(1 + rets); peak = np.maximum.accumulate(eqc)
    dd = ((peak - eqc) / peak).max() * 100
    return dict(n=len(t), net=t.pips.sum(), win=(t.pips > 0).mean() * 100, pf=pf,
                aw=wins.mean() if len(wins) else 0, al=loss.mean() if len(loss) else 0, dd=dd)


def show(t: pd.DataFrame, label: str, byyear=False):
    m = metrics(t)
    if m is None:
        print(f"  {label:<10} no trades"); return
    print(f"  {label:<10} trades={m['n']:>4}  net={m['net']:+8.0f}p  win={m['win']:>3.0f}%  "
          f"PF={m['pf']:4.2f}  avgW={m['aw']:+5.1f} avgL={m['al']:+6.1f}  maxDD={m['dd']:4.1f}%")
    if byyear:
        ty = t.copy(); ty["y"] = ty.t_in.dt.tz_localize(None).dt.year
        print("      by year: " + "  ".join(
            f"{int(y)}:{g.pips.sum():+.0f}({len(g)})" for y, g in ty.groupby("y")))


# ---------------------------------------------------------------- CLI ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strat", choices=list(STRATS))
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="is", choices=["is", "val"])
    ap.add_argument("--unseal", action="store_true", help="run the SEALED test slice (logged)")
    ap.add_argument("--byyear", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="+/-1 robustness sweep on key params")
    # orb params
    ap.add_argument("--asia-start-h", type=int, default=0)
    ap.add_argument("--asia-end-h", type=int, default=7)
    ap.add_argument("--bo-start-h", type=int, default=7)
    ap.add_argument("--bo-end-h", type=int, default=11)
    ap.add_argument("--force-exit-h", type=int, default=20)
    ap.add_argument("--rr", type=float, default=1.0)
    ap.add_argument("--buf-atr", type=float, default=0.0)
    ap.add_argument("--sl-buf-atr", type=float, default=0.0)
    ap.add_argument("--max-range-atr", type=float, default=0.0, help="skip days range>this*ATR (0=off)")
    ap.add_argument("--min-range-atr", type=float, default=0.0)
    ap.add_argument("--sl-frac", type=float, default=1.0, help="continuation SL depth as frac of range below broken edge (1.0=opposite edge; <1=tight abort stop on fall-back into range)")
    ap.add_argument("--rsi-max", type=float, default=100.0, help="skip longs if RSI(14)>=this at breakout (shorts mirror at 100-this); 100=off (exhaustion filter)")
    ap.add_argument("--box-trend-max", type=float, default=1.0, help="skip breakout if Asian box net-move/range >= this in the break dir (1=off; e.g. 0.5 = box trended >half its range = not a range)")
    ap.add_argument("--no-tp", action="store_true", help="let it run: no TP, exit on SL or session close")
    ap.add_argument("--fade", action="store_true", help="fade the break (mean-revert into the range)")
    ap.add_argument("--dir", default="both", choices=["both", "long", "short"])
    # squeeze params
    ap.add_argument("--sq-win", type=int, default=12, help="coil box length (bars)")
    ap.add_argument("--sq-look", type=int, default=72, help="lookback for recent expansion (bars)")
    ap.add_argument("--sq-ratio", type=float, default=0.5, help="coil if box_rng <= this * recent_max")
    ap.add_argument("--exp-atr", type=float, default=0.0, help="require recent_max >= this*ATR (prior big move; 0=off)")
    # bounce params
    ap.add_argument("--level-lb", type=int, default=288, help="bars to define HTF S/R level (288=1 day on M5)")
    ap.add_argument("--tol-atr", type=float, default=0.5, help="'tag' the level if within this*ATR")
    ap.add_argument("--tp-lb", type=int, default=48, help="recent swing-high/low lookback for TP")
    ap.add_argument("--sl-lb", type=int, default=12, help="recent swing low/high lookback for SL")
    ap.add_argument("--min-rr", type=float, default=1.0, help="only enter if (TP-entry)/(entry-SL) >= this")
    ap.add_argument("--double-bottom", action="store_true", help="require a prior tag of the zone (2nd touch)")
    ap.add_argument("--db-win", type=int, default=48, help="lookback for the prior touch (double bottom)")
    ap.add_argument("--db-gap", type=int, default=3, help="min bars between the two touches")
    # selection filters (the user's discretionary cues, made explicit & testable)
    ap.add_argument("--htf-sr", action="store_true", help="only enter near a PROVEN higher-TF pivot S/R level")
    ap.add_argument("--trendline", action="store_true", help="only enter on a touch of a projected HTF trendline")
    ap.add_argument("--htf-level", action="store_true",
                    help="use 15m+1h pivot S/R as the bounce level (the user's actual method)")
    ap.add_argument("--piv-tf", default="15min,1h", help="higher TFs for pivots (comma-sep)")
    ap.add_argument("--piv-k", type=int, default=3, help="fractal pivot half-width (bars each side)")
    ap.add_argument("--sr-keep", type=int, default=20, help="how many recent HTF pivots to keep")
    ap.add_argument("--rej", type=float, default=0.6, help="'stall' confirm: close within top/bottom (1-rej) of bar range")
    # reversal CONFIRMATION (did the bounce actually take hold?) -- comma-sep = AND combo
    ap.add_argument("--confirm", default="simple",
                    help="simple|struct|twobar|mom (comma-sep requires ALL, e.g. struct,mom)")
    ap.add_argument("--cf-win", type=int, default=6, help="bars to wait after the tag for confirmation")
    ap.add_argument("--cf-lb", type=int, default=6, help="local high/low lookback for struct break")
    ap.add_argument("--mom-th", type=float, default=45.0, help="RSI threshold for the 'mom' confirmation")
    ap.add_argument("--cont", action="store_true", help="continuation: only break in the direction of the prior move")
    # streak params (H10)
    ap.add_argument("--streak-n", type=int, default=4, help="fade after this many consecutive same-colour bars")
    ap.add_argument("--streak-stretch", type=float, default=1.5, help="only fade runs whose range >= this*ATR (0=off)")
    # vwap params (H11)
    ap.add_argument("--vwap-k", type=float, default=2.0, help="fade when close is this many sigma beyond VWAP")
    ap.add_argument("--vwap-lb", type=int, default=48, help="rolling lookback for sigma of (close-VWAP)")
    ap.add_argument("--vwap-anchor-h", type=int, default=0, help="session anchor hour for VWAP reset (UTC)")
    # nr7 params (H14)
    ap.add_argument("--nr-lb", type=int, default=7, help="NR-N: range is min of last N (7=NR7)")
    ap.add_argument("--ema-len", type=int, default=50, help="1h EMA length for HTF trend direction")
    ap.add_argument("--slope-k", type=int, default=3, help="bars for the 1h EMA slope (diff)")
    ap.add_argument("--nr-bothdir", action="store_true", help="allow both break directions (drop the HTF-trend filter)")
    # vol-regime META-GATE (H12): keep only entries whose ATR is in a trailing percentile band
    ap.add_argument("--vol-band", default="all", choices=["all", "low", "mid", "high"],
                    help="gate entries by trailing ATR regime (low<=q33, mid, high>q66)")
    ap.add_argument("--atr-win", type=int, default=8000, help="trailing window (bars) for the ATR quantiles")
    # daily-trend regime gate (H-suffix): the validated gold lever
    ap.add_argument("--daily-sma", type=int, default=0, help="daily SMA regime gate (0=off): longs only above, shorts below")
    ap.add_argument("--daily-slope-k", type=int, default=0, help="also require daily SMA rising/falling over k days (0=off)")
    # proportionate HTF-trend gate (1h/4h) -- lighter than daily for a low exec TF
    ap.add_argument("--htf-tf", default="", help="align entries with this HTF trend (e.g. 1h/4h); '' = off")
    ap.add_argument("--htf-ema", type=int, default=50, help="EMA length on the HTF for trend direction")
    ap.add_argument("--htf-slope-k", type=int, default=0, help="use EMA slope over k HTF bars (0 = price-vs-EMA level)")
    ap.add_argument("--kama-tf", default="", help="KAMA-rising gate TF (e.g. 1h/2h/4h/1D); '' = off")
    ap.add_argument("--kama-n", type=int, default=14, help="KAMA length on the gate TF")
    ap.add_argument("--kama-slope-k", type=int, default=1, help="KAMA rising = km > km.shift(k) on the gate TF")
    # session params (H18 box-less London-open trend ride)
    ap.add_argument("--sl-atr", type=float, default=0.0, help="ATR-mult stop for session strat (0 = ride to close, no stop)")
    ap.add_argument("--stop-slip", type=float, default=0.0, help="pips the SL fills WORSE than stop price (models a headline candle gapping through the stop)")
    ap.add_argument("--cost", type=float, default=1.4)
    ap.add_argument("--resample", default="", help="resample M5 up to this TF (e.g. 15min/30min/1h) before running")
    p = ap.parse_args()

    if p.unseal:
        s, e = SPLITS["test"]
        print("!!! UNSEALING TEST SLICE -- this is a one-shot, log it. !!!")
    else:
        s, e = SPLITS[p.split]
    d = load_mt5_csv(p.csv).loc[s:e]
    if p.resample:
        d = d.resample(p.resample, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    sig = STRATS[p.strat]
    print(f"\n=== {p.strat}  {p.csv}  split={'TEST' if p.unseal else p.split}  "
          f"asia[{p.asia_start_h}-{p.asia_end_h}) bo[{p.bo_start_h}-{p.bo_end_h}) "
          f"exit{p.force_exit_h} rr{p.rr} dir={p.dir} cost{p.cost}p ===")
    print(f"  {len(d):,} M5 bars  {d.index[0]} -> {d.index[-1]}")
    dir_, sl_px, tp_px = sig(d, p)
    dir_, sl_px, tp_px = vol_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = daily_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = htf_trend_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = kama_gate(d, dir_, sl_px, tp_px, p)
    show(backtest(d, dir_, sl_px, tp_px, p), "base", byyear=p.byyear)

    if p.sweep:
        print("  -- robustness sweep (plateau=good, spike=overfit) --")
        if p.strat == "squeeze":
            keys = [("sq_ratio", [round(p.sq_ratio - 0.1, 2), p.sq_ratio, round(p.sq_ratio + 0.1, 2)]),
                    ("sq_win", [p.sq_win - 4, p.sq_win, p.sq_win + 4]),
                    ("sq_look", [p.sq_look - 24, p.sq_look, p.sq_look + 24])]
        elif p.strat == "bounce":
            keys = [("tol_atr", [round(p.tol_atr - 0.2, 2), p.tol_atr, round(p.tol_atr + 0.2, 2)]),
                    ("level_lb", [p.level_lb - 96, p.level_lb, p.level_lb + 96]),
                    ("tp_lb", [p.tp_lb - 12, p.tp_lb, p.tp_lb + 12])]
        elif p.strat == "streak":
            keys = [("streak_n", [p.streak_n - 1, p.streak_n, p.streak_n + 1]),
                    ("streak_stretch", [round(p.streak_stretch - 0.5, 2), p.streak_stretch,
                                        round(p.streak_stretch + 0.5, 2)]),
                    ("rr", [p.rr - 0.5, p.rr, p.rr + 0.5])]
        elif p.strat == "vwap":
            keys = [("vwap_k", [round(p.vwap_k - 0.5, 2), p.vwap_k, round(p.vwap_k + 0.5, 2)]),
                    ("vwap_lb", [p.vwap_lb - 24, p.vwap_lb, p.vwap_lb + 24]),
                    ("vwap_anchor_h", [(p.vwap_anchor_h - 6) % 24, p.vwap_anchor_h, (p.vwap_anchor_h + 6) % 24])]
        elif p.strat == "nr7":
            keys = [("nr_lb", [p.nr_lb - 2, p.nr_lb, p.nr_lb + 2]),
                    ("ema_len", [p.ema_len - 20, p.ema_len, p.ema_len + 20]),
                    ("rr", [p.rr - 0.5, p.rr, p.rr + 0.5])]
        elif p.strat == "session":
            keys = [("bo_start_h", [p.bo_start_h - 1, p.bo_start_h, p.bo_start_h + 1]),
                    ("htf_ema", [50, p.htf_ema, 120]),
                    ("force_exit_h", [p.force_exit_h - 2, p.force_exit_h, p.force_exit_h + 2])]
        else:
            keys = [("rr", [p.rr - 0.5, p.rr, p.rr + 0.5]),
                    ("bo_end_h", [p.bo_end_h - 1, p.bo_end_h, p.bo_end_h + 1]),
                    ("asia_end_h", [p.asia_end_h - 1, p.asia_end_h, p.asia_end_h + 1])]
        for key, vals in keys:
            row = []
            for v in vals:
                setattr(p, key, v)
                dd, sp, tpp = sig(d, p)
                dd, sp, tpp = vol_gate(d, dd, sp, tpp, p)
                m = metrics(backtest(d, dd, sp, tpp, p))
                row.append(f"{v}:{m['pf']:.2f}/{m['net']:+.0f}({m['n']})" if m else f"{v}:--")
            print(f"     {key:<11} " + "   ".join(row))
            # restore
            setattr(p, key, getattr(ap.parse_args([p.strat, '--csv', p.csv]), key) if False else vals[1])


if __name__ == "__main__":
    main()
