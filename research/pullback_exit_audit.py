"""pullback_exit_audit.py -- classify HOW the EMA-pullback trades end.

The user's question: after a pullback entry, how many die by price falling back
THROUGH the fast EMA again (the 'fake pullback' / trend-resumption-down loser)?
In this strategy the stop sits at the pullback's dip extreme, which is BELOW the
fast EMA, so a -1R stop necessarily means price re-broke the EMA and kept going.

This replicates ema_pullback.run()'s entry+evaluation but tags each trade's exit
reason: STOP(-1R) / TARGET(+rr) / TIMEOUT(held fwd bars). It also flags whether
price CLOSED back below the fast EMA at any point after entry (the EMA re-break),
and -- among non-stopped trades -- whether it re-broke but recovered.

  .venv/bin/python research/pullback_exit_audit.py --csv data/vantage_btcusd_h1.csv --tf 4h
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from ema_pullback import resample


def audit(d, args):
    ef = (d["close"].rolling(args.ema_fast).mean() if args.fast_ma_type == "sma"
          else d["close"].ewm(span=args.ema_fast, adjust=False).mean()).values
    es = (d["close"].rolling(args.ema_slow).mean() if args.trend_ma_type == "sma"
          else d["close"].ewm(span=args.ema_slow, adjust=False).mean()).values
    a = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    K, N = args.slope_k, args.fwd
    slope = np.full(len(c), np.nan)
    slope[K:] = (es[K:] - es[:-K]) / (K * np.where(a[K:] > 0, a[K:], np.nan))

    # entry detection (LONG, confirmed-close, fill-at-close) -- matches the validated config.
    # also capture 4 ENTRY-TIME predictors of 'won't re-break the EMA':
    #   depth   = how far below fast EMA the dip pierced (ATR)   -> SHALLOW favourable
    #   reclaim = how far above fast EMA the entry bar closed    -> STRONG  favourable
    #   slopev  = 80EMA slope magnitude (steepness)              -> STEEP   favourable
    #   pbbars  = bars spent below the fast EMA (dip duration)   -> SHORT   favourable
    entries = []
    state, ext, pbcnt = 0, None, 0
    for i in range(K + 1, len(c) - 1):
        if np.isnan(slope[i]) or np.isnan(a[i]) or a[i] <= 0:
            continue
        if not (slope[i] >= args.thr):
            state, ext, pbcnt = 0, None, 0
            continue
        if c[i] < ef[i]:
            if state == 0:
                pbcnt = 0
            state = 1; ext = l[i] if ext is None else min(ext, l[i]); pbcnt += 1
        elif state == 1 and h[i] >= ef[i]:
            e = c[i]; stop = min(ext, l[i])
            if e - stop < args.min_stop_atr * a[i]:
                stop = e - args.min_stop_atr * a[i]
            depth = (ef[i] - ext) / a[i]; reclaim = (e - ef[i]) / a[i]
            entries.append((i, e, stop, depth, reclaim, slope[i], pbcnt))
            state, ext, pbcnt = 0, None, 0

    rows = []
    busy_until = -1
    for (i, e, stop, depth, reclaim, slopev, pbbars) in entries:
        if i <= busy_until:
            continue
        risk = e - stop
        if risk <= 0 or i + 1 >= len(c):
            continue
        tgt = e + args.rr * risk
        reason, exit_j, R = "timeout", min(i + N, len(c) - 1), None
        rebroke = False                                   # closed back below fast EMA post-entry?
        for j in range(i + 1, min(i + 1 + N, len(c))):
            if c[j] < ef[j]:
                rebroke = True
            if l[j] <= stop:
                reason, R, exit_j = "stop", -1.0, j; break
            if h[j] >= tgt:
                reason, R, exit_j = "target", float(args.rr), j; break
        if R is None:
            R = (c[exit_j] - e) / risk
        R -= args.cost / risk * e
        rows.append((d.index[i], reason, R, rebroke, depth, reclaim, slopev, pbbars))
        busy_until = exit_j
    return pd.DataFrame(rows, columns=["time", "reason", "R", "rebroke",
                                       "depth", "reclaim", "slopev", "pbbars"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--ema-fast", type=int, default=20)
    ap.add_argument("--ema-slow", type=int, default=80)
    ap.add_argument("--slope-k", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--trend-ma-type", default="sma")
    ap.add_argument("--fast-ma-type", default="ema")
    ap.add_argument("--min-stop-atr", type=float, default=0.5)
    ap.add_argument("--atr", type=int, default=14)
    ap.add_argument("--fwd", type=int, default=90)
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--split", default="2022-01-01", help="IS/VAL boundary date for the predictor test")
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    t = audit(d, a)
    n = len(t)
    print(f"\n=== EMA pullback exit audit  {os.path.basename(a.csv)} {a.tf}  RR={a.rr}  "
          f"long  {d.index[0].date()}->{d.index[-1].date()} ===")
    print(f"  total trades: {n}")
    for r in ("stop", "target", "timeout"):
        g = t[t.reason == r]
        share = len(g) / n * 100 if n else 0
        print(f"    {r:<8} {len(g):>4}  ({share:4.1f}%)   sumR={g.R.sum():+6.1f}")
    print(f"\n  EMA re-break (closed back below fast EMA AFTER entry, before exit):")
    print(f"    re-broke EMA : {t.rebroke.sum():>4} / {n}  ({t.rebroke.mean()*100:.1f}%)")
    rb = t[t.rebroke]
    print(f"      of those -> stopped: {(rb.reason=='stop').sum()}   "
          f"target: {(rb.reason=='target').sum()}   timeout: {(rb.reason=='timeout').sum()}   "
          f"(sumR {rb.R.sum():+.1f})")
    nb = t[~t.rebroke]
    print(f"    never re-broke: {len(nb):>4} / {n}  ({(~t.rebroke).mean()*100:.1f}%)  "
          f"win {(nb.R>0).mean()*100:.0f}%  sumR {nb.R.sum():+.1f}")
    # the specific answer: fake-pullback = entered then re-broke EMA and got stopped
    fake = t[(t.rebroke) & (t.reason == "stop")]
    print(f"\n  >>> FAKE pullbacks (re-broke EMA -> stopped at dip low): {len(fake)} "
          f"({len(fake)/n*100:.1f}% of all trades, sumR {fake.R.sum():+.1f})")
    t["y"] = t.time.dt.year
    print("      by year: " + " ".join(
        f"{y}:{len(g[(g.rebroke)&(g.reason=='stop')])}/{len(g)}" for y, g in t.groupby("y")))

    # ---- can any ENTRY-TIME predictor separate clean pullbacks from fakes? ----
    # threshold = IS median, applied to BOTH IS and VAL. A real predictor must show
    # the favourable half beating the weak half in BOTH splits (else it's IS-luck).
    split_ts = pd.Timestamp(a.split)
    if t.time.dt.tz is not None:
        split_ts = split_ts.tz_localize(t.time.dt.tz)
    is_t = t[t.time < split_ts].copy()
    val_t = t[t.time >= split_ts].copy()
    print(f"\n  ==== entry-time predictor test  (IS<{a.split} n={len(is_t)} | VAL>= n={len(val_t)}) ====")
    print(f"  threshold = IS median, applied to both. fav = the 'should stay clean' side.")
    preds = [("depth", "<=", "shallow dip"), ("reclaim", ">=", "strong reclaim"),
             ("slopev", ">=", "steep slope"), ("pbbars", "<=", "quick reclaim")]

    def stats(sub):
        if len(sub) == 0:
            return f"{'n=0':>30}"
        return (f"n={len(sub):>3} win={ (sub.R>0).mean()*100:>3.0f}% "
                f"meanR={sub.R.mean():+5.2f} rebrk={sub.rebroke.mean()*100:>3.0f}%")

    for col, op, label in preds:
        med = is_t[col].median()
        if op == "<=":
            is_fav, is_weak = is_t[is_t[col] <= med], is_t[is_t[col] > med]
            v_fav, v_weak = val_t[val_t[col] <= med], val_t[val_t[col] > med]
        else:
            is_fav, is_weak = is_t[is_t[col] >= med], is_t[is_t[col] < med]
            v_fav, v_weak = val_t[val_t[col] >= med], val_t[val_t[col] < med]
        print(f"\n  [{label:<14}] ({col} {op} {med:.2f})")
        print(f"      IS  fav : {stats(is_fav)}")
        print(f"      IS  weak: {stats(is_weak)}")
        print(f"      VAL fav : {stats(v_fav)}")
        print(f"      VAL weak: {stats(v_weak)}")


if __name__ == "__main__":
    main()
