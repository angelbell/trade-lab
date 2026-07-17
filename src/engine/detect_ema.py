"""EMA pullback-continuation detection (the btc_pull engine): 80MA-slope trend
state machine — pullback across the fast MA, then a reclaim = entry.
Detection emits (i, entry_px, stop_px); execution is walk.walk_ema.
Lifted verbatim from ema_pullback.run()."""
import numpy as np


def ema_slope(es, a, K):
    """slow-MA slope per bar, ATR-normalised."""
    slope = np.full(len(es), np.nan)
    slope[K:] = (es[K:] - es[:-K]) / (K * np.where(a[K:] > 0, a[K:], np.nan))
    return slope


def ema_entries(c, h, l, ef, a, slope, er, daily_up, daily_dn, side, args, thr):
    """Walk the bars: detect pullback then reclaim-entry. NaN slope/ATR bars keep the
    pullback state; a not-trending bar resets it (matches the original exactly)."""
    K = args.slope_k
    entries = []   # (i, entry_px, stop_px)
    state, ext = 0, None
    for i in range(K + 1, len(c) - 1):
        if np.isnan(slope[i]) or np.isnan(a[i]) or a[i] <= 0:
            continue
        # direction from slow-MA slope sign; gate on slope-magnitude OR efficiency
        dir_ok = (slope[i] > 0) if side == "long" else (slope[i] < 0)
        if args.filter == "er":
            gate = (not np.isnan(er[i])) and (er[i] >= thr)
            trending = dir_ok and gate
        else:  # slope-magnitude filter (original)
            trending = (slope[i] >= thr) if side == "long" else (slope[i] <= -thr)
        if daily_up is not None:
            trending = trending and (daily_up[i] if side == "long" else daily_dn[i])
        if not trending:
            state, ext = 0, None
            continue
        # entry trigger:
        #   "close" (default) = wait for a bar to CLOSE back across the fast MA -> confirmed.
        #   "touch"           = a resting stop at the fast MA fills intrabar (Pine/live).
        if side == "long":
            if args.entry_trigger == "touch" and state == 1 and h[i] >= ef[i]:
                stop = min(ext if ext is not None else l[i], l[i])
                e = ef[i]
                if e - stop < args.min_stop_atr * a[i]:
                    stop = e - args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None
            elif c[i] < ef[i]:                      # pullback: below fast MA
                state = 1; ext = l[i] if ext is None else min(ext, l[i])
            elif state == 1 and h[i] >= ef[i]:      # confirmed reclaim -> enter
                e = c[i] if args.fill_at_close else ef[i]   # close = realistic, ma = idealized
                stop = min(ext, l[i])
                if e - stop < args.min_stop_atr * a[i]:
                    stop = e - args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None
        else:
            if args.entry_trigger == "touch" and state == 1 and l[i] <= ef[i]:
                stop = max(ext if ext is not None else h[i], h[i])
                e = ef[i]
                if stop - e < args.min_stop_atr * a[i]:
                    stop = e + args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None
            elif c[i] > ef[i]:
                state = 1; ext = h[i] if ext is None else max(ext, h[i])
            elif state == 1 and l[i] <= ef[i]:
                e = c[i] if args.fill_at_close else ef[i]
                stop = max(ext, h[i])
                if stop - e < args.min_stop_atr * a[i]:
                    stop = e + args.min_stop_atr * a[i]
                entries.append((i, e, stop)); state, ext = 0, None
    return entries
