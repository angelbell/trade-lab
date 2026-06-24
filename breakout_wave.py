"""breakout_wave.py -- Elliott "Pattern B" (3rd-wave) breakout, mechanised (4H).

The discretionary method from the video, turned into a falsifiable rule set:

  Structure : detect alternating swing points (an online swing detector, so no
              lookahead -- a swing is only KNOWN once the bar that confirms it
              has closed). We look for the bullish reversal skeleton:
                  L0 (low)  ->  H1 (high = "wave-1 high")  ->  L2 (low)
              with L2 > L0  (a HIGHER LOW = the wave-2 pullback that holds above
              the wave-1 origin).  That is the "1st wave done, 2nd wave is a
              higher low" context the video calls 環境認識.
  Entry     : the first bar that CLOSES above the wave-1 high (H1).  Confirmed
              close only -- an intrabar poke that closes back below H1 is the
              "ダマシ" and is ignored.  (This is the exact close-confirmation
              lever proved in the EMA pullback work: wicks through a level fail.)
  Stop      : L2, the wave-2 low / 3rd-wave origin.  NOT just below H1 -- the
              video is explicit that a stop just under the broken line gets
              wicked out on the post-break retest.
  Target    : measured move -- project the wave-1 length off the wave-2 low:
                  TP = L2 + (H1 - L0).
              (--tp-mode rr instead uses a fixed reward:risk, to A/B the video's
              measured-move target against a plain RR and see which profile wins.)

This is HIGH-WINRATE / LOW-RR by construction (breakouts often realise <1:1) --
the opposite profile to the RR1:3 EMA pullback. R is in each trade's OWN risk
units: a stop = -1R, the target = +(TP-entry)/risk R.

Run:
  .venv/bin/python breakout_wave.py --csv data/vantage_btcusd_h1.csv --tf 4h --swing zigzag
  .venv/bin/python breakout_wave.py --csv data/vantage_xauusd_h1.csv --tf 4h --swing pivot
"""

import argparse
import math

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule.lower() in ("1h", "h1", ""):
        return df
    o = {"4h": "4h", "h4": "4h", "1d": "1D", "d1": "1D"}.get(rule.lower(), rule)
    return pd.DataFrame({
        "open":  df["open"].resample(o).first(),
        "high":  df["high"].resample(o).max(),
        "low":   df["low"].resample(o).min(),
        "close": df["close"].resample(o).last(),
    }).dropna()


def kama_adaptive(close, n, fast=2, slow=30):
    """Kaufman Adaptive MA (inlined to avoid a circular import with research/)."""
    ch = close.diff(n).abs()
    vol = close.diff().abs().rolling(n).sum()
    erc = (ch / vol).fillna(0).values
    sc = (erc * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    c = close.values
    out = np.full(len(c), np.nan)
    if len(c) > n:
        out[n] = c[n]
        for i in range(n + 1, len(c)):
            out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def swings_zigzag(h, l, atr, k):
    """Online ATR-threshold ZigZag. Returns list of (confirm_idx, pivot_idx,
    price, kind) where kind is +1 (swing high) / -1 (swing low). confirm_idx is
    the bar at which the swing became KNOWN (reversal of k*ATR from the extreme);
    pivot_idx is where the extreme actually sat. Acting on confirm_idx => no
    lookahead."""
    out = []
    direction = 0          # +1 = currently in an up leg (tracking a high), -1 = down leg
    ext_p = h[0]; ext_i = 0
    for i in range(1, len(h)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        thr = k * atr[i]
        if direction >= 0:                       # tracking a swing HIGH
            if h[i] > ext_p:
                ext_p, ext_i = h[i], i
            elif ext_p - l[i] >= thr:            # reversed down enough -> high confirmed
                out.append((i, ext_i, ext_p, +1))
                direction = -1; ext_p, ext_i = l[i], i
        if direction <= 0:                       # tracking a swing LOW
            if l[i] < ext_p:
                ext_p, ext_i = l[i], i
            elif h[i] - ext_p >= thr:            # reversed up enough -> low confirmed
                out.append((i, ext_i, ext_p, -1))
                direction = +1; ext_p, ext_i = h[i], i
    return out


def swings_pivot(h, l, n):
    """Online N-bar fractal pivots. A bar c is a pivot high if its high is the
    strict max of [c-n, c+n]; confirmed only at bar c+n. Returns the same
    (confirm_idx, pivot_idx, price, kind) tuples."""
    out = []
    for c in range(n, len(h) - n):
        win_h = h[c - n:c + n + 1]
        win_l = l[c - n:c + n + 1]
        if h[c] == win_h.max() and (win_h == h[c]).sum() == 1:
            out.append((c + n, c, h[c], +1))
        if l[c] == win_l.min() and (win_l == l[c]).sum() == 1:
            out.append((c + n, c, l[c], -1))
    out.sort(key=lambda t: (t[0], t[1]))         # by confirmation order
    return out


def swings_renko(h, l, c, atr, k):
    """ATR-brick Renko. A new brick forms only after a full k*ATR move; a swing
    is the run-extreme at the bar where direction reverses by one brick. Time is
    discarded and small wiggles are quantised away (noise-robust). Same axis as
    ZigZag (price extremes) but accumulated in bricks rather than off the peak."""
    out = []
    trend, base = 0, c[0]
    ext_p, ext_i = h[0], 0
    for i in range(1, len(c)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        b = k * atr[i]
        if trend > 0:
            if h[i] > ext_p:
                ext_p, ext_i = h[i], i
            if c[i] >= base + b:
                base += b * math.floor((c[i] - base) / b)
            elif c[i] <= base - b:                       # reverse down -> swing HIGH
                out.append((i, ext_i, ext_p, +1))
                trend, base, ext_p, ext_i = -1, c[i], l[i], i
        elif trend < 0:
            if l[i] < ext_p:
                ext_p, ext_i = l[i], i
            if c[i] <= base - b:
                base -= b * math.floor((base - c[i]) / b)
            elif c[i] >= base + b:                       # reverse up -> swing LOW
                out.append((i, ext_i, ext_p, -1))
                trend, base, ext_p, ext_i = +1, c[i], h[i], i
        else:                                            # establish initial direction
            if c[i] >= base + b:
                trend, base, ext_p, ext_i = +1, base + b, h[i], i
            elif c[i] <= base - b:
                trend, base, ext_p, ext_i = -1, base - b, l[i], i
    return out


def swings_momentum(h, l, c, fast, slow):
    """Momentum zero-cross swings: the MACD line (EMA fast - EMA slow) sign flip
    ends a run; the swing is that run's price extreme. A DIFFERENT information
    axis from price-extreme detectors (momentum, not the high/low itself) -- it
    lags the turn but ignores price spikes that don't shift momentum."""
    macd = (pd.Series(c).ewm(span=fast, adjust=False).mean()
            - pd.Series(c).ewm(span=slow, adjust=False).mean()).values
    out = []
    trend, ext_p, ext_i = 0, h[0], 0
    for i in range(slow, len(c)):
        if np.isnan(macd[i]):
            continue
        s = 1 if macd[i] > 0 else -1
        if trend == 0:
            trend = s; ext_p, ext_i = (h[i], i) if s > 0 else (l[i], i); continue
        if s == trend:
            if trend > 0 and h[i] > ext_p: ext_p, ext_i = h[i], i
            if trend < 0 and l[i] < ext_p: ext_p, ext_i = l[i], i
        else:                                            # momentum flipped -> emit
            out.append((i, ext_i, ext_p, +1 if trend > 0 else -1))
            trend = s; ext_p, ext_i = (h[i], i) if s > 0 else (l[i], i)
    return out


def run(d: pd.DataFrame, args) -> None:
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    es = d["close"].ewm(span=args.trend_ema, adjust=False).mean().values if args.trend_ema > 0 else None

    # daily regime gate: only take longs when the PRIOR completed day's close was
    # above its daily SMA. gap-days dropped so the SMA counts real trading days only
    # (matches MT5 iMA(D1) / the structure_sma lesson); shift(1)+ffill => no lookahead.
    reg = None
    ext_arr = None
    if args.daily_sma > 0:
        dc = d["close"].resample(getattr(args, "gate_tf", "1D")).last().dropna()
        sma = dc.rolling(args.daily_sma).mean()
        up = dc > sma
        if args.daily_slope_k > 0:                       # also require the daily SMA rising
            up = up & (sma > sma.shift(args.daily_slope_k))
        up = up.shift(1)
        reg = up.reindex(d.index, method="ffill").fillna(False).values
        if getattr(args, "ext_cap", 0) > 0:              # extension cap: skip entries when the
            ext = (dc - sma) / sma * 100.0               # prior day is >ext_cap% above daily SMA
            ext_arr = ext.shift(1).reindex(d.index, method="ffill").values  # (chasing a stretched break)

    # daily KAMA-rising ENTRY gate (optional): only enter when the prior completed daily KAMA
    # is rising. shift(1)+ffill = no lookahead. (Tests whether KAMA filters chop years.)
    kreg = None
    if getattr(args, "gate_kama", 0) > 0:
        dck = d["close"].resample("1D").last().dropna()
        kmg = kama_adaptive(dck, args.gate_kama)
        krise = (kmg > kmg.shift(1)).shift(1)
        kreg = krise.reindex(d.index, method="ffill").fillna(False).values

    # regime-flip EXIT (adaptive): bail a long at the bar CLOSE if the prior completed
    # daily KAMA has turned DOWN (regime against the long). shift(1)+ffill = no lookahead.
    against = None
    if args.exit_kama > 0:
        dc2 = d["close"].resample("1D").last().dropna()
        km = kama_adaptive(dc2, args.exit_kama)
        falling = (km < km.shift(1)).shift(1)
        against = falling.reindex(d.index, method="ffill").fillna(False).values

    if args.swing == "zigzag":
        sw = swings_zigzag(h, l, a, args.zz_k)
    elif args.swing == "pivot":
        sw = swings_pivot(h, l, args.pivot_n)
    elif args.swing == "renko":
        sw = swings_renko(h, l, c, a, args.renko_k)
    else:  # momentum
        sw = swings_momentum(h, l, c, args.mom_fast, args.mom_slow)

    def first_breakout(level, after):
        """first bar > level on a CONFIRMED close, within --bo-window of `after`."""
        for j in range(after, min(after + args.bo_window, len(c))):
            if c[j] > level:
                return j
        return None

    def retest_entry(level, break_i, window, invalid_low):
        """bolt-on retest: after a CONFIRMED break at break_i, require price to pull back
        and TOUCH the broken level (now support, within tol*ATR) then CLOSE back above it =
        a retest reclaim. Returns that reclaim bar, or None if it never retests within
        `window` bars, or if it first breaks the higher-low structure (low < invalid_low)."""
        tol = getattr(args, "retest_tol", 0.10)
        touched = False
        for j in range(break_i + 1, min(break_i + 1 + window, len(c))):
            if l[j] < invalid_low:                   # higher-low broke before retest = void
                return None
            band = tol * (a[j] if not np.isnan(a[j]) else 0.0)
            if not touched and l[j] <= level + band:
                touched = True
            if touched and c[j] > level:             # closed back above the broken level
                return j
        return None

    def next_high_target(e, e_i):
        """video's Pattern-A target: a "notable high" ABOVE entry -- the 2nd
        confirmed swing high overhead ("もう一つ上の高値", skip the nearest)."""
        ups = sorted(p for (ci, ii, p, k) in sw if k == +1 and ci <= e_i and p > e)
        return ups[1] if len(ups) >= 2 else None

    entries = []        # (entry_i, entry_px, stop_px, tgt_px)

    if args.pattern == "B":
        # L0 -> H1 (wave-1 high) -> L2 (higher low) -> close breaks H1
        for t in range(2, len(sw)):
            cL2, iL2, pL2, kL2 = sw[t]
            cH1, iH1, pH1, kH1 = sw[t - 1]
            cL0, iL0, pL0, kL0 = sw[t - 2]
            if not (kL2 == -1 and kH1 == +1 and kL0 == -1):
                continue
            if pL2 <= pL0 or pH1 - pL0 <= 0:        # need a HIGHER LOW that holds
                continue
            # Elliott leg label: is H1 (this setup's wave-1 high) ALREADY a higher-high
            # vs the prior swing high? No => first impulse off a base/downtrend (wave-3).
            # Yes => continuation leg in an established uptrend (wave-5+).
            if args.wave != "all":
                prevH = sw[t - 3][2] if (t - 3 >= 0 and sw[t - 3][3] == +1) else None
                is_cont = prevH is not None and pH1 > prevH
                if (args.wave == "3" and is_cont) or (args.wave == "5" and not is_cont):
                    continue
            if es is not None and not np.isnan(es[cL2]) and pH1 < es[cL2]:
                continue
            e_i = first_breakout(pH1, cL2 + 1)
            if e_i is None:
                continue
            rt_win = getattr(args, "retest", 0)
            if rt_win > 0:                            # require a pullback-retest of the broken
                e_i = retest_entry(pH1, e_i, rt_win, pL2)  # H1 then a confirmed reclaim
                if e_i is None:
                    continue
            if reg is not None and not reg[e_i]:     # daily regime gate (longs only when up)
                continue
            if ext_arr is not None and not np.isnan(ext_arr[e_i]) and ext_arr[e_i] > args.ext_cap:
                continue                             # extension cap: skip the stretched chase
            if kreg is not None and not kreg[e_i]:   # daily KAMA-rising entry gate
                continue
            e = c[e_i]                               # SL placement (default swinglow = wave-2 low)
            slb = getattr(args, "sl_b", "swinglow"); slk = getattr(args, "sl_b_k", 1.5)
            if slb == "origin":
                stop = pL0                            # widest structural: the wave origin (lower low)
            elif slb == "atr":
                stop = e - slk * a[e_i]               # pure ATR-width noise band
            elif slb == "band":
                stop = pL2 - slk * a[e_i]             # higher-low minus an ATR noise buffer
            else:
                stop = pL2
            risk = e - stop
            if risk <= 0:
                continue
            tgt = pL2 + (pH1 - pL0) if args.tp_mode == "measured" else e + args.rr * risk
            if tgt <= e:
                continue
            entries.append((e_i, e, stop, tgt))

    else:  # Pattern A: break the "last lower high" of a downtrend = wave-1 confirm
        for t in range(2, len(sw)):
            cHt, iHt, pHt, kHt = sw[t]               # trigger = most recent swing HIGH
            if kHt != +1:
                continue
            prev_highs = [sw[u] for u in range(t) if sw[u][3] == +1]
            if not prev_highs:
                continue
            pHprev = prev_highs[-1][2]
            if pHprev <= pHt:                         # require a LOWER HIGH (downtrend)
                continue
            if es is not None and not np.isnan(es[cHt]) and pHt > es[cHt]:
                continue                              # gate: trigger below trend EMA (still down)
            e_i = first_breakout(pHt, cHt + 1)
            if e_i is None:
                continue
            e = c[e_i]
            if args.sl_mode == "line":                # video A: just below the broken line
                stop = pHt - args.sl_buf * a[e_i]
            else:                                     # swinglow: the pullback low before the break
                lo = l[iHt:e_i + 1]
                stop = lo.min() - args.sl_buf * a[e_i] if len(lo) else pHt - args.sl_buf * a[e_i]
            risk = e - stop
            if risk <= 0:
                continue
            if args.tp_mode == "nexthigh":
                tgt = next_high_target(e, e_i)
                if tgt is None:
                    continue
            else:                                     # measured undefined for A -> use RR
                tgt = e + args.rr * risk
            if tgt <= e:
                continue
            entries.append((e_i, e, stop, tgt))

    # sort by ENTRY-BAR index first: different skeletons append out of entry order, and the
    # busy_until overlap-exclusion + the eq/maxDD cumprod below are order-dependent. run() was the
    # only metrics path not ordered by time -> path-dependent trade SET and maxDD. Sort, then de-dup.
    entries.sort(key=lambda en: en[0])
    # de-dup identical entry bars (different skeletons can arm the same break)
    seen, uniq = set(), []
    for en in entries:
        if en[0] in seen:
            continue
        seen.add(en[0]); uniq.append(en)
    entries = uniq

    # evaluate: one position at a time, forward to stop / target / timeout
    trades, rr_real = [], []
    busy_until = -1
    for (i, e, stop, tgt) in entries:
        if i <= busy_until:
            continue
        risk = e - stop
        reward = tgt - e
        exit_j = min(i + args.fwd, len(c) - 1)
        R = None
        s_frac = getattr(args, "tp1_frac", 0.0)
        if s_frac > 0:
            # SCALE-OUT: bank s_frac at tp1 (RR=tp1_rr), then run the rest to the final tgt,
            # moving the stop to break-even after tp1 if tp1_be. stop checked first on same-bar
            # conflict (conservative). Total R = sum of fraction*R-multiple of each leg.
            tp1 = e + getattr(args, "tp1_rr", 1.0) * risk
            be_move = getattr(args, "tp1_be", 1)
            realized, rem, cur_stop, tp1_hit = 0.0, 1.0, stop, False
            for j in range(i + 1, min(i + 1 + args.fwd, len(c))):
                if l[j] <= cur_stop:
                    realized += rem * ((cur_stop - e) / risk); R = realized; exit_j = j; break
                if not tp1_hit and h[j] >= tp1:
                    realized += s_frac * ((tp1 - e) / risk); rem -= s_frac; tp1_hit = True
                    if be_move: cur_stop = e
                if h[j] >= tgt:
                    realized += rem * (reward / risk); R = realized; exit_j = j; break
                if against is not None and against[j]:
                    realized += rem * ((c[j] - e) / risk); R = realized; exit_j = j; break
            if R is None:
                realized += rem * ((c[exit_j] - e) / risk); R = realized
        else:
            for j in range(i + 1, min(i + 1 + args.fwd, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt:  R = reward / risk; exit_j = j; break
                if against is not None and against[j]:        # regime turned down -> bail at close
                    R = (c[j] - e) / risk; exit_j = j; break
            if R is None:
                R = (c[exit_j] - e) / risk
        R -= args.cost / risk * e
        hold = (d.index[exit_j] - d.index[i]).total_seconds() / 86400.0
        if args.swap_pct > 0:
            R -= (args.swap_pct / 100.0) * (e / risk) * hold
        trades.append((d.index[i], R, hold))
        rr_real.append(reward / risk)
        busy_until = exit_j

    if not trades:
        print("  no entries"); return None
    t = pd.DataFrame(trades, columns=["time", "R", "hold"]); t["y"] = t["time"].dt.year
    if getattr(args, "dump_trades", False):   # clean CSV only (for per-trade slice analysis)
        print("entry_time,R,hold")
        for _, r in t.iterrows():
            print(f"{r['time'].isoformat()},{r['R']:.6f},{r['hold']:.6f}")
        return t
    yrs = sorted(t["y"].unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t[t["y"] < half]["R"] if half else t["R"]
    oosr = t[t["y"] >= half]["R"] if half else t["R"]
    print(f"  n={len(t):>4}  win={(t['R']>0).mean()*100:>3.0f}%  meanR={t['R'].mean():+.2f}  "
          f"totR={t['R'].sum():+6.0f}  | IS={isr.mean():+.2f} OOS={oosr.mean():+.2f}  "
          f"| medRR={np.median(rr_real):.2f}  hold(d) med={t['hold'].median():.1f} max={t['hold'].max():.0f}"
          + (f"  [swap {args.swap_pct}%/d]" if args.swap_pct > 0 else ""))
    # real-money equity curve at constant risk%: the true risk across ALL years (incl chop)
    eq = (1 + args.risk * t["R"]).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    yrs_span = max((t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / yrs_span) - 1) * 100
    print(f"  @risk {args.risk*100:.0f}%/trade: return={ (eq.iloc[-1]-1)*100:+.0f}%  "
          f"CAGR={cagr:+.1f}%  maxDD={dd:.1f}%  ret/DD={ (eq.iloc[-1]-1)*100/max(dd,1e-9):.2f}")
    if args.peryear:
        pos = sum(1 for _, g in t.groupby("y") if g["R"].sum() > 0)
        print("       per-year totR: " + " ".join(
            f"{y}:{g['R'].sum():+.0f}(n{len(g)})" for y, g in t.groupby("y"))
            + f"   [{pos}/{t['y'].nunique()} yrs +]")
    return t


def main() -> None:
    p = argparse.ArgumentParser(description="Elliott Pattern-B (3rd-wave) breakout screener")
    p.add_argument("--csv", required=True)
    p.add_argument("--tf", default="4h")
    p.add_argument("--pattern", default="B", choices=["A", "B"],
                   help="A=break last lower-high of a downtrend (wave-1 confirm); "
                        "B=break wave-1 high after a higher-low (wave-3 confirm)")
    p.add_argument("--wave", default="all", choices=["all", "3", "5"],
                   help="Pattern-B Elliott leg: 3=first impulse (H1 not yet a higher-high); "
                        "5=continuation leg (H1 already a higher-high); all=both (current)")
    p.add_argument("--sl-mode", default="line", choices=["line", "swinglow"],
                   help="Pattern A stop: line=just below the broken high (video), "
                        "swinglow=below the pre-breakout pullback low")
    p.add_argument("--sl-buf", type=float, default=0.25, help="Pattern A stop buffer in ATRs")
    p.add_argument("--sl-b", default="swinglow", choices=["swinglow", "origin", "atr", "band"],
                   help="Pattern B stop: swinglow=wave-2 higher-low (default/tightest); "
                        "origin=wave origin pL0 (widest structural); atr=e-k*ATR (pure noise band); "
                        "band=pL2-k*ATR (higher-low + ATR noise buffer). origin/atr/band test the "
                        "OB/channel 'wider noise-band stop' idea (hold through the shakeout).")
    p.add_argument("--sl-b-k", type=float, default=1.5, help="ATR multiple for --sl-b atr/band")
    p.add_argument("--ext-cap", type=float, default=0.0,
                   help="skip entries when prior-day close is >this%% above the daily SMA "
                        "(0=off; needs --daily-sma). Filters the 'chasing a stretched breakout' setups.")
    p.add_argument("--gate-tf", default="1D",
                   help="resample rule for the SMA regime gate + ext-cap (default 1D; try 4h/8h). "
                        "--daily-sma length & --daily-slope-k are then counted in THESE bars.")
    p.add_argument("--gate-kama", type=int, default=0,
                   help="daily KAMA-rising ENTRY gate length (0=off): only enter when prior-day "
                        "daily KAMA is rising. Tests whether KAMA filters chop years.")
    p.add_argument("--swing", default="zigzag", choices=["zigzag", "pivot", "renko", "momentum"],
                   help="swing detector: zigzag=ATR-threshold reversal, pivot=N-bar fractal, "
                        "renko=ATR-brick, momentum=MACD-line sign flip")
    p.add_argument("--zz-k", type=float, default=2.0, help="ZigZag reversal threshold in ATRs")
    p.add_argument("--pivot-n", type=int, default=5, help="N bars each side for fractal pivots")
    p.add_argument("--renko-k", type=float, default=2.0, help="Renko brick size in ATRs")
    p.add_argument("--mom-fast", type=int, default=12, help="momentum: fast EMA")
    p.add_argument("--mom-slow", type=int, default=26, help="momentum: slow EMA")
    p.add_argument("--trend-ema", type=int, default=0,
                   help="optional gate: require wave-1 high above this EMA (0=off)")
    p.add_argument("--bo-window", type=int, default=20,
                   help="max bars after wave-2 low to wait for the H1 breakout")
    p.add_argument("--retest", type=int, default=0,
                   help="bolt-on: require a pullback-retest+reclaim of the broken level within "
                        "this many bars before entry (0=off, enter on the break itself)")
    p.add_argument("--retest-tol", type=float, default=0.10,
                   help="retest touch band in ATRs (how close the pullback must come to the line)")
    p.add_argument("--daily-sma", type=int, default=0,
                   help="daily-regime gate: longs only when prior daily close > daily SMA(this) (0=off)")
    p.add_argument("--daily-slope-k", type=int, default=0,
                   help="also require the daily SMA to be rising over k days (0=off)")
    p.add_argument("--exit-kama", type=int, default=0,
                   help="adaptive regime-flip EXIT: bail a long at close when daily KAMA(this) turns down (0=off)")
    p.add_argument("--tp-mode", default="measured", choices=["measured", "rr", "nexthigh"],
                   help="measured=project wave-1 length off wave-2 low (B); "
                        "rr=fixed reward:risk; nexthigh=2nd swing-high overhead (A, video)")
    p.add_argument("--rr", type=float, default=1.0, help="reward:risk when --tp-mode rr")
    p.add_argument("--tp1-frac", type=float, default=0.0,
                   help="scale-out: fraction to bank at the tp1 target (0=off, all-or-nothing)")
    p.add_argument("--tp1-rr", type=float, default=1.0, help="scale-out: tp1 target in R (must be < --rr)")
    p.add_argument("--tp1-be", type=int, default=1, help="scale-out: move stop to break-even after tp1 (1/0)")
    p.add_argument("--atr", type=int, default=14)
    p.add_argument("--fwd", type=int, default=60, help="forward bars to resolve a trade")
    p.add_argument("--cost", type=float, default=0.001, help="round-trip cost fraction")
    p.add_argument("--swap-pct", type=float, default=0.0, help="daily swap %% of notional")
    p.add_argument("--risk", type=float, default=0.01, help="risk fraction per trade for the equity/DD curve")
    p.add_argument("--peryear", action="store_true")
    p.add_argument("--dump-trades", action="store_true",
                   help="emit per-trade CSV (entry_time,R,hold) to stdout instead of the summary")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    d = load_mt5_csv(args.csv)
    if args.start or args.end:
        d = d.loc[args.start:args.end]
    d = resample(d, args.tf)
    det = {"zigzag": f"zigzag(k={args.zz_k})", "pivot": f"pivot(n={args.pivot_n})",
           "renko": f"renko(k={args.renko_k})",
           "momentum": f"mom({args.mom_fast}/{args.mom_slow})"}[args.swing]
    sl = f" SL={args.sl_mode}" if args.pattern == "A" else ""
    print(f"\n=== Pattern-{args.pattern} breakout  {args.csv}  TF={args.tf}  swing={det}{sl}  "
          f"TP={args.tp_mode}{('('+str(args.rr)+')') if args.tp_mode=='rr' else ''} ===")
    print(f"  {len(d):,} {args.tf} bars  {d.index[0].date()} -> {d.index[-1].date()}")
    run(d, args)


if __name__ == "__main__":
    main()
