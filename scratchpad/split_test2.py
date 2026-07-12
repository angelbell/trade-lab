"""案1 SPLIT, done right: EXACT canonical market/limit resolve logic (copied from
btc15m_pullback_gauntlet.evaluate / pullback_cost_abs), so market & limit rows reproduce canon
before split is trusted. Builds reused from exec_ceiling_diag (verified: gold N354, BTC N657).

  market : own loop, busy = market exit every signal.            (canon)
  limit  : own loop, busy = limit exit only when filled; miss -> busy unchanged, trade skipped. (canon)
  split  : same busy evolution as market (market half always fills at e). Per signal:
           R = 0.5*(Rm - sp/u_m) + [filled]*0.5*(Rl - sp/u_l).   Rm in u_m=e-stop, Rl in u_l=lim-stop.
Cost = absolute spread$ / risk (gold $0.6, BTC $15), matching canon.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scratchpad.exec_ceiling_diag import build_gold, build_btc, FWD


def leg_market(E, idx, h, l, c, sp):
    busy = -1; tr = []
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        u = e - stop; rew = tgt - e; xj = min(e_i + FWD, len(c) - 1); R = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= tgt: R = rew / u; xj = j; break
        if R is None: R = (c[xj] - e) / u
        tr.append((idx[e_i], R - sp / u)); busy = xj
    return tr


def leg_limit(E, idx, h, l, c, frac, sp):
    busy = -1; tr = []; miss = 0
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        lim = e - frac * (e - stop)
        if lim <= stop or lim >= e: miss += 1; continue
        fill_j = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: miss += 1; continue
        u = lim - stop; rew = tgt - lim
        if l[fill_j] <= stop: R = -1.0; xj = fill_j
        else:
            xj = min(fill_j + FWD, len(c) - 1); R = None
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                if l[j] <= stop: R = -1.0; xj = j; break
                if h[j] >= tgt: R = rew / u; xj = j; break
            if R is None: R = (c[xj] - lim) / u
        tr.append((idx[fill_j], R - sp / u)); busy = xj
    return tr, miss


def leg_split(E, idx, h, l, c, frac, sp):
    """market half (always) + limit half (if it fills), risk 0.5 each. busy = market half exit."""
    busy = -1; tr = []; recovered = 0
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        u_m = e - stop; rew_m = tgt - e
        xj = min(e_i + FWD, len(c) - 1); Rm = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if l[j] <= stop: Rm = -1.0; xj = j; break
            if h[j] >= tgt: Rm = rew_m / u_m; xj = j; break
        if Rm is None: Rm = (c[xj] - e) / u_m
        Rm_net = Rm - sp / u_m
        # limit half
        lim = e - frac * u_m; filled = False; Rl_net = 0.0; ran_to_tgt = False
        if lim > stop and lim < e:
            fill_j = None
            for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
                if h[j] >= tgt: ran_to_tgt = True; break
                if l[j] <= lim: fill_j = j; break
            if fill_j is not None:
                u_l = lim - stop; rew_l = tgt - lim
                if l[fill_j] <= stop: Rl = -1.0
                else:
                    xl = min(fill_j + FWD, len(c) - 1); Rl = None
                    for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                        if l[j] <= stop: Rl = -1.0; break
                        if h[j] >= tgt: Rl = rew_l / u_l; break
                    if Rl is None: Rl = (c[xl] - lim) / u_l
                filled = True; Rl_net = Rl - sp / u_l
        if ran_to_tgt and not filled: recovered += 1   # a runner the market half caught
        Rs = 0.5 * Rm_net + (0.5 * Rl_net if filled else 0.0)
        tr.append((idx[e_i], Rs)); busy = xj
    return tr, recovered


def leg_chase(E, idx, h, l, c, frac, sp):
    """案7: pure-limit, but on a MISS (ran to tgt without pulling back to lim) enter FULL at
    market at the first higher close (c[j]>c[e_i]) that occurred before tgt = chase the runner.
    stop/tgt at original levels -> chased entry is above e -> risk bigger, RR degraded (the tax)."""
    busy = -1; tr = []; chased = 0
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        lim = e - frac * (e - stop)
        fill_j = None; tgt_j = None; hc_j = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if l[j] <= lim: fill_j = j; break         # limit fills first -> normal pullback trade
            if h[j] >= tgt: tgt_j = j; break          # ran to tgt without pulling back = MISS
            if hc_j is None and c[j] > c[e_i]: hc_j = j  # first higher close = chase trigger
        if fill_j is not None:
            ep, eb, u = lim, fill_j, lim - stop       # pullback trade (same as pure-limit)
        elif tgt_j is not None and hc_j is not None and hc_j < tgt_j:
            ep, eb, u = c[hc_j], hc_j, c[hc_j] - stop; chased += 1   # chase the runner
        else:
            continue                                  # true miss, no chase possible
        if u <= 0: continue
        rew = tgt - ep; xj = min(eb + FWD, len(c) - 1); R = None
        for j in range(eb + 1, min(eb + 1 + FWD, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= tgt: R = rew / u; xj = j; break
        if R is None: R = (c[xj] - ep) / u
        tr.append((idx[eb], R - sp / u)); busy = xj
    return tr, chased


def stats(tr, span):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
    return dict(N=len(R), npy=len(R)/span, win=(R > 0).mean()*100, pf=pf, meanR=R.mean(),
                totR=R.sum(), IS=R[yr < half].mean(), OOS=R[yr >= half].mean(),
                retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))


def report(name, E, idx, h, l, c, frac, sp, canon):
    span = (idx[-1] - idx[0]).days / 365.25
    print(f"\n===== {name}  frac={frac}  cost=${sp} =====   (canon: {canon})")
    print(f"  {'mode':>10}{'N':>5}{'N/yr':>6}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'IS/OOS':>13}{'ret/DD':>8}{'grn':>6}")
    sm = stats(leg_market(E, idx, h, l, c, sp), span)
    trl, miss = leg_limit(E, idx, h, l, c, frac, sp); sl = stats(trl, span)
    trs, rec = leg_split(E, idx, h, l, c, frac, sp); ss = stats(trs, span)
    trc, chn = leg_chase(E, idx, h, l, c, frac, sp); sc = stats(trc, span)
    for mode, s, extra in (("market", sm, ""), ("limit", sl, f" miss={miss}"),
                           ("split", ss, f" recov_runners={rec}"), ("chase", sc, f" chased={chn}")):
        io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
        print(f"  {mode:>10}{s['N']:>5}{s['npy']:>6.0f}{s['win']:>4.0f}%{s['pf']:>6.2f}"
              f"{s['meanR']:>+8.3f}{s['totR']:>+7.0f}{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}{extra}")


if __name__ == "__main__":
    Eg, ig, hg, lg, cg = build_gold()
    report("GOLD 15m", Eg, ig, hg, lg, cg, 0.25, 0.6,
           "market meanR+0.342 / frac0.25 +0.471")
    Eb, ib, hb, lb, cb = build_btc()
    report("BTC 15m", Eb, ib, hb, lb, cb, 0.30, 15.0,
           "market +0.175/N657 / frac0.3 +0.322/N614")
