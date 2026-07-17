"""THE execution walker — the single forward-scan loop of the engine. Every
execution rule lives here and only here:
  - max_pos slotting (open_x semantics: a signal ON an exit bar stays excluded)
  - market entry / pullback-limit (fill_win, target-first = missed, and the
    fill bar itself checked against the stop: same-bar tie → stop wins)
  - split execution (half market + half pullback-limit)
  - tp1 scale-out (bank a fraction at tp1, optional break-even move)
  - regime-flip bail (against), timeout at forward-cap close
  - cost and swap accounting
Returns (trades DataFrame | None, rr_real). Lifted verbatim from breakout_wave.run()."""
import numpy as np
import pandas as pd


def walk(d, entries, against, args):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    trades, rr_real = [], []
    open_x = []                                        # exit bars of currently-open positions
    maxpos = max(1, int(getattr(args, "max_pos", 1)))
    pf = getattr(args, "pullback_frac", 0.0)          # 0 = market entry; >0 = pullback-limit
    for (i, e, stop, tgt, i_origin) in entries:
        open_x = [x for x in open_x if x >= i]        # a signal ON an exit bar stays excluded
                                                      # (matches the historical busy_until <=)
        if len(open_x) >= maxpos:
            continue
        # SPLIT execution (--exec-split with --pullback-frac): half at market, half at the
        # pullback limit. Both halves keep the market stop/target levels; each half risks
        # 0.5R on its own stop distance. Recorded risk = harmonic combination so post-hoc
        # absolute cost stays spread/risk. Not combinable with tp1_frac scale-out.
        if pf > 0.0 and getattr(args, "exec_split", 0):
            risk_m = e - stop
            if risk_m <= 0:
                continue
            lim = e - pf * risk_m

            def _walk(px, sb):
                rk = px - stop
                for j in range(sb + 1, min(sb + 1 + args.fwd, len(c))):
                    if l[j] <= stop: return -1.0, j
                    if h[j] >= tgt:  return (tgt - px) / rk, j
                    if against is not None and against[j]:
                        return (c[j] - px) / rk, j
                rj = min(sb + args.fwd, len(c) - 1)
                return (c[rj] - px) / rk, rj

            Rm, xm = _walk(e, i)
            Rm -= args.cost / risk_m * e
            fj = None
            for j in range(i + 1, min(i + 1 + args.fwd, len(c))):
                if h[j] >= tgt: break                 # ran to target first = limit missed
                if l[j] <= lim: fj = j; break
            R, exit_j, w_risk = 0.5 * Rm, xm, 2.0 * risk_m
            if fj is not None:
                risk_l = lim - stop
                Rl, xl = _walk(lim, fj)
                Rl -= args.cost / risk_l * lim
                R += 0.5 * Rl
                exit_j = max(xm, xl)
                w_risk = 1.0 / (0.5 * (1.0 / risk_m + 1.0 / risk_l))
            hold = (d.index[exit_j] - d.index[i]).total_seconds() / 86400.0
            if args.swap_pct > 0:
                R -= (args.swap_pct / 100.0) * (e / risk_m) * hold
            trades.append((d.index[i], R, hold, w_risk, e, Rm, 1.0 if fj is not None else 0.0,
                           i - i_origin))
            rr_real.append((tgt - e) / risk_m)
            open_x.append(exit_j)
            continue
        # PULLBACK-LIMIT execution: keep the structural stop AND the fixed target at their
        # MARKET levels, but lower the entry to e-pf*(e-stop). Fill on the pullback touch
        # BEFORE the target is reached (else MISSED = the runaway winner is skipped =
        # adverse selection modelled). Realized risk shrinks -> effective RR rises.
        e_px, e_bar = e, i
        fill_bar_stopped = False
        if pf > 0.0:
            lim = e - pf * (e - stop)
            # how long the limit stays live. Defaults to fwd, but the Pine cancels after its
            # own fillWin -- pass --fill-win to make the two agree.
            fw = getattr(args, "fill_win", 0) or args.fwd
            fj = None
            for j in range(i + 1, min(i + 1 + fw, len(c))):
                if h[j] >= tgt: break                 # ran to target first = missed
                if l[j] <= lim: fj = j; break
            if fj is None:
                continue
            e_px, e_bar = lim, fj
            # The bar that reaches the limit can reach THROUGH the stop as well: filled and
            # stopped inside the same bar. The forward walk below starts at e_bar+1, so without
            # this the fill bar gets a free pass -- and the free pass grows with pf, because
            # the limit and the stop converge ((1-pf)*risk apart). Measured on gold15m: 3% of
            # fills at pf=0.25, 23% at pf=0.70 = what made "deeper is better" look monotone.
            fill_bar_stopped = l[fj] <= stop
        risk = e_px - stop
        reward = tgt - e_px
        if risk <= 0:
            continue
        exit_j = min(e_bar + args.fwd, len(c) - 1)
        R = None
        s_frac = getattr(args, "tp1_frac", 0.0)
        if fill_bar_stopped:                          # stop wins the same-bar tie, as everywhere else
            R, exit_j = -1.0, e_bar
        elif s_frac > 0:
            # SCALE-OUT: bank s_frac at tp1 (RR=tp1_rr), then run the rest to the final tgt,
            # moving the stop to break-even after tp1 if tp1_be. stop checked first on same-bar
            # conflict (conservative). Total R = sum of fraction*R-multiple of each leg.
            tp1 = e_px + getattr(args, "tp1_rr", 1.0) * risk
            be_move = getattr(args, "tp1_be", 1)
            realized, rem, cur_stop, tp1_hit = 0.0, 1.0, stop, False
            for j in range(e_bar + 1, min(e_bar + 1 + args.fwd, len(c))):
                if l[j] <= cur_stop:
                    realized += rem * ((cur_stop - e_px) / risk); R = realized; exit_j = j; break
                if not tp1_hit and h[j] >= tp1:
                    realized += s_frac * ((tp1 - e_px) / risk); rem -= s_frac; tp1_hit = True
                    if be_move: cur_stop = e_px
                if h[j] >= tgt:
                    realized += rem * (reward / risk); R = realized; exit_j = j; break
                if against is not None and against[j]:
                    realized += rem * ((c[j] - e_px) / risk); R = realized; exit_j = j; break
            if R is None:
                realized += rem * ((c[exit_j] - e_px) / risk); R = realized
        else:
            for j in range(e_bar + 1, min(e_bar + 1 + args.fwd, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt:  R = reward / risk; exit_j = j; break
                if against is not None and against[j]:        # regime turned down -> bail at close
                    R = (c[j] - e_px) / risk; exit_j = j; break
            if R is None:
                R = (c[exit_j] - e_px) / risk
        R -= args.cost / risk * e_px
        hold = (d.index[exit_j] - d.index[e_bar]).total_seconds() / 86400.0
        if args.swap_pct > 0:
            R -= (args.swap_pct / 100.0) * (e_px / risk) * hold
        trades.append((d.index[e_bar], R, hold, risk, e_px, R, 1.0, i - i_origin))
        rr_real.append(reward / risk)
        open_x.append(exit_j)

    if not trades:
        return None, rr_real
    t = pd.DataFrame(trades, columns=["time", "R", "hold", "risk", "e_px", "r_mkt", "filled",
                                      "base_bars"])
    t["y"] = t["time"].dt.year
    return t, rr_real


def walk_ema(d, entries, side, exit_ma, a, args):
    """Walker for the EMA-pullback engine (btc_pull). Kept as its OWN loop — its
    execution semantics genuinely differ from walk() and must not be blended:
      - fills ON the signal bar (at the fast-MA level or its close), no limit stage
      - native short side (the breakout engine shorts via price inversion instead)
      - target pays exactly args.rr (not reward/risk)
      - trend-failure bail = close across exit_ma (walk() bails on a regime array)
      - one-at-a-time via busy_until when --no-overlap (walk() slots via max_pos)
    Returns (trades DataFrame | None, mfe, mae) — the excursion arrays (ATR units)
    feed the screen line. Lifted verbatim from ema_pullback.run()."""
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    N = args.fwd
    mfe, mae, trades = [], [], []
    busy_until = -1
    for (i, e, stop) in entries:
        if args.no_overlap and i <= busy_until:
            continue
        risk = abs(e - stop)
        fh, fl = h[i + 1:i + 1 + N], l[i + 1:i + 1 + N]
        if risk <= 0 or len(fh) == 0:
            continue
        exit_j = min(i + N, len(c) - 1)           # default: timeout bar
        if side == "long":
            mfe.append((fh.max() - e) / a[i]); mae.append((e - fl.min()) / a[i])
            tgt, R = e + args.rr * risk, None
            for j in range(i + 1, min(i + 1 + N, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt:  R = args.rr; exit_j = j; break
                if exit_ma is not None and not np.isnan(exit_ma[j]) and c[j] < exit_ma[j]:
                    R = (c[j] - e) / risk; exit_j = j; break        # closed below the MA -> bail
            if R is None: R = (c[exit_j] - e) / risk
        else:
            mfe.append((e - fl.min()) / a[i]); mae.append((fh.max() - e) / a[i])
            tgt, R = e - args.rr * risk, None
            for j in range(i + 1, min(i + 1 + N, len(c))):
                if h[j] >= stop: R = -1.0; exit_j = j; break
                if l[j] <= tgt:  R = args.rr; exit_j = j; break
                if exit_ma is not None and not np.isnan(exit_ma[j]) and c[j] > exit_ma[j]:
                    R = (e - c[j]) / risk; exit_j = j; break        # closed above the MA -> bail
            if R is None: R = (e - c[exit_j]) / risk
        R -= args.cost / risk * e          # round-trip cost in R units
        # calendar days held (incl. weekends) -> daily swap cost on NOTIONAL.
        hold_days = (d.index[exit_j] - d.index[i]).total_seconds() / 86400.0
        if args.swap_pct > 0:
            R -= (args.swap_pct / 100.0) * (e / risk) * hold_days  # notional/risk * %/day * days
        # `risk` (the stop distance in PRICE units) and `e_px` are what a real order needs.
        trades.append((d.index[i], R, hold_days, risk, e))
        busy_until = exit_j

    if not trades:
        return None, mfe, mae
    t = pd.DataFrame(trades, columns=["time", "R", "hold", "risk", "e_px"])
    t["y"] = t["time"].dt.year
    return t, mfe, mae

