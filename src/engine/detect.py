"""Signal detection: swing skeleton → breakout setups. Detection only — no regime
gating, no stop/target math (that is plan.py), no execution (walk.py).
A Pattern-B setup is (e_i, pH1, pL0, pL2, iL0); Pattern-A is (e_i, pHt, iHt),
where e_i is the confirmed break bar (after the optional retest reclaim).
Lifted verbatim from breakout_wave.run()."""
import numpy as np

from breakout_wave import swings_zigzag, swings_pivot, swings_renko, swings_momentum


def make_swings(h, l, c, a, args):
    if args.swing == "zigzag":
        return swings_zigzag(h, l, a, args.zz_k)
    elif args.swing == "pivot":
        return swings_pivot(h, l, args.pivot_n)
    elif args.swing == "renko":
        return swings_renko(h, l, c, a, args.renko_k)
    else:  # momentum
        return swings_momentum(h, l, c, args.mom_fast, args.mom_slow)


def first_breakout(c, level, after, bo_window):
    """first bar > level on a CONFIRMED close, within bo_window of `after`."""
    for j in range(after, min(after + bo_window, len(c))):
        if c[j] > level:
            return j
    return None


def retest_entry(c, l, a, args, level, break_i, window, invalid_low):
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


def next_high_target(sw, e, e_i):
    """Pattern-A "notable high" target ABOVE entry — the 2nd confirmed swing high
    overhead ("もう一つ上の高値", skip the nearest)."""
    ups = sorted(p for (ci, ii, p, k) in sw if k == +1 and ci <= e_i and p > e)
    return ups[1] if len(ups) >= 2 else None


def pattern_b(c, l, a, es, sw, args):
    """L0 → H1 (wave-1 high) → L2 (higher low) → close breaks H1."""
    setups = []
    for t in range(2, len(sw)):
        cL2, iL2, pL2, kL2 = sw[t]
        cH1, iH1, pH1, kH1 = sw[t - 1]
        cL0, iL0, pL0, kL0 = sw[t - 2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1):
            continue
        if pL2 <= pL0 or pH1 - pL0 <= 0:        # need a HIGHER LOW that holds
            continue
        # Elliott leg label: is H1 ALREADY a higher-high vs the prior swing high?
        # No => first impulse off a base/downtrend (wave-3). Yes => continuation (wave-5+).
        wave = getattr(args, "wave", "all")
        if wave != "all":
            prevH = sw[t - 3][2] if (t - 3 >= 0 and sw[t - 3][3] == +1) else None
            is_cont = prevH is not None and pH1 > prevH
            if (wave == "3" and is_cont) or (wave == "5" and not is_cont):
                continue
        if es is not None and not np.isnan(es[cL2]) and pH1 < es[cL2]:
            continue
        e_i = first_breakout(c, pH1, cL2 + 1, args.bo_window)
        if e_i is None:
            continue
        rt_win = getattr(args, "retest", 0)
        if rt_win > 0:                            # require a pullback-retest of the broken
            e_i = retest_entry(c, l, a, args, pH1, e_i, rt_win, pL2)  # H1 then a reclaim
            if e_i is None:
                continue
        setups.append((e_i, pH1, pL0, pL2, iL0))
    return setups


def pattern_a(c, l, es, sw, args):
    """break the "last lower high" of a downtrend = wave-1 confirm."""
    setups = []
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
        e_i = first_breakout(c, pHt, cHt + 1, args.bo_window)
        if e_i is None:
            continue
        setups.append((e_i, pHt, iHt))
    return setups
