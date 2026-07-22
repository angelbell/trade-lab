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
    seed_hi_p, seed_hi_i = h[0], 0     # while direction == 0 BOTH extremes must be tracked
    seed_lo_p, seed_lo_i = l[0], 0     # separately, or ext_p oscillates and never accumulates
    for i in range(1, len(h)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        thr = k * atr[i]
        if direction == 0:                       # not yet seeded: which way does the first leg run?
            if h[i] > seed_hi_p:
                seed_hi_p, seed_hi_i = h[i], i
            if l[i] < seed_lo_p:
                seed_lo_p, seed_lo_i = l[i], i
            if seed_hi_p - l[i] >= thr:          # fell k*ATR off the running high -> that was a high
                out.append((i, seed_hi_i, seed_hi_p, +1))
                direction = -1; ext_p, ext_i = l[i], i
            elif h[i] - seed_lo_p >= thr:        # rose k*ATR off the running low -> that was a low
                out.append((i, seed_lo_i, seed_lo_p, -1))
                direction = +1; ext_p, ext_i = h[i], i
            continue
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


def run(d: pd.DataFrame, args) -> pd.DataFrame:
    """Composable engine entry point. The strategy logic lives in src/engine/
    (gates / detect / plan / walk / stats — one execution walker for the whole
    breakout family); this wrapper keeps the historical call signature that
    ~200 research scripts import. Bit-identity with the pre-split monolith is
    guarded by invariants/engine_tieback.py and invariants/engine_golden.py —
    re-run both before trusting ANY engine edit."""
    from src.engine.compat import run_compat   # lazy: avoids import cycle
    return run_compat(d, args)


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
    p.add_argument("--tgt-ref", default="stop", choices=["stop", "l2"],
                   help="what the RR target is measured from. stop=the actual risk (default: a wider "
                        "stop also pushes the target out, R-multiple unchanged). l2=the wave-2-low "
                        "risk, so a wider stop keeps the SAME target price = higher win rate, lower "
                        "R per win. Use it to test 'the target is right, the stop is too tight'.")
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
    p.add_argument("--pullback-frac", type=float, default=0.0,
                   help="pullback-limit execution: keep the structural stop+fixed target, but "
                        "enter on a limit at e-frac*(e-stop) (0=market; ~0.25-0.3 = validated lever). "
                        "runaway breaks that never pull back are skipped (adverse selection).")
    p.add_argument("--fill-win", type=int, default=0,
                   help="bars the pullback limit stays live before it is cancelled (0 = use --fwd, "
                        "the historical default). The Pine strategies cancel after their own "
                        "fillWin, so set this to the same number to compare like with like.")
    p.add_argument("--max-pos", type=int, default=1,
                   help="max concurrent positions (default 1 = historical single-slot behaviour); "
                        "each position risks its own 1R, so account risk stacks on overlap")
    p.add_argument("--exec-split", type=int, default=0,
                   help="with --pullback-frac: 1 = half position at market + half at the "
                        "pullback limit (no missed runners); stop/target stay at market levels")
    p.add_argument("--gate-kama-tf", default="1D",
                   help="resample TF for the --gate-kama KAMA-rising gate (default 1D)")
    p.add_argument("--gate-kama-tf2", default="",
                   help="optional second KAMA-rising gate TF, ANDed with the first "
                        "(e.g. fast 4h entry-timing gate AND slow 1D bear-market veto)")
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
    p.add_argument("--exit-kama-tf", default="1D",
                   help="resample rule for the exit-KAMA series (e.g. 240min; default 1D)")
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
