"""Setups + gate arrays → orders (e_i, entry_px, stop_px, tgt_px, i_origin).
The continue-check ORDER matches breakout_wave.run() exactly
(reg → ext → kreg → stop math → risk<=0 → target math → tgt<=e), then the
entry-bar sort + same-bar de-dup. Lifted verbatim."""
import numpy as np

from .detect import next_high_target


def plan(c, l, a, sw, setups, reg, ext_arr, kreg, args):
    entries = []        # (entry_i, entry_px, stop_px, tgt_px, i_origin)

    if args.pattern == "B":
        for (e_i, pH1, pL0, pL2, iL0) in setups:
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
            # Where the TARGET is measured from. Default: the ACTUAL risk. --tgt-ref l2 keeps
            # the target at the price the TIGHT (wave-2 low) stop would have given.
            risk_t = risk if getattr(args, "tgt_ref", "stop") == "stop" else max(e - pL2, 1e-12)
            tgt = pL2 + (pH1 - pL0) if args.tp_mode == "measured" else e + args.rr * risk_t
            if tgt <= e:
                continue
            entries.append((e_i, e, stop, tgt, iL0))

    else:  # Pattern A
        for (e_i, pHt, iHt) in setups:
            if reg is not None and not reg[e_i]:     # same regime gates as Pattern B
                continue
            if ext_arr is not None and not np.isnan(ext_arr[e_i]) and ext_arr[e_i] > args.ext_cap:
                continue
            if kreg is not None and not kreg[e_i]:
                continue
            e = c[e_i]
            if args.sl_mode == "line":                # just below the broken line
                stop = pHt - args.sl_buf * a[e_i]
            else:                                     # swinglow: the pullback low before the break
                lo = l[iHt:e_i + 1]
                stop = lo.min() - args.sl_buf * a[e_i] if len(lo) else pHt - args.sl_buf * a[e_i]
            risk = e - stop
            if risk <= 0:
                continue
            if args.tp_mode == "nexthigh":
                tgt = next_high_target(sw, e, e_i)
                if tgt is None:
                    continue
            else:                                     # measured undefined for A -> use RR
                tgt = e + args.rr * risk
            if tgt <= e:
                continue
            entries.append((e_i, e, stop, tgt, iHt))

    # sort by ENTRY-BAR index first: different skeletons append out of entry order, and the
    # busy_until overlap-exclusion + the equity cumprod downstream are order-dependent.
    entries.sort(key=lambda en: en[0])
    # de-dup identical entry bars (different skeletons can arm the same break)
    seen, uniq = set(), []
    for en in entries:
        if en[0] in seen:
            continue
        seen.add(en[0]); uniq.append(en)
    return uniq
