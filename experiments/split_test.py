"""案1 SPLIT execution: half at market (e), half at frac-limit (lim=e-frac*(e-stop)).
Recovers the miss tax (D1: pure-limit throws away the ~10% runaway +RR winners) while keeping
some of the limit's effective-RR gain on the trades that DO pull back.

Compare 3 execution styles as separate equity legs, NET, both 15m breakout legs:
  market     : full size at e (pays full cost/risk on every trade, catches all runners)
  pure-limit : full size at lim, SKIP misses (the adopted leg; loses the runners)
  split      : 0.5 at e + 0.5 at lim (lim half only if it fills)

R-unit = market risk (e - stop), so 1R = what the trader risks per signal at full market size.
Cost applied per half-notional. Faithful reuse of the canonical builds.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from experiments.exec_ceiling_diag import build_gold, build_btc, FWD


def walk(E, idx, h, l, c, mode, frac, cost_frac=0.0, cost_abs=0.0):
    busy = -1; tr = []
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        u = e - stop                                   # R-unit = market risk
        if u <= 0: continue
        lim = e - frac * u
        c_unit = cost_frac * e / u + cost_abs / u      # full-position round-trip cost in R
        # resolve forward: track limit fill + first stop/tgt
        filled = False; fill_j = None; exit_j = min(e_i + FWD, len(c) - 1); outcome = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if l[j] <= stop: outcome = "stop"; exit_j = j; break
            if (not filled) and l[j] <= lim: filled = True; fill_j = j
            if h[j] >= tgt: outcome = "tgt"; exit_j = j; break
        if outcome is None:                            # timeout: mark-to-close
            close_R = (c[exit_j] - e) / u
        # --- per-mode P&L in R-units of u ---
        if mode == "market":
            if outcome == "stop": R = -1.0
            elif outcome == "tgt": R = (tgt - e) / u
            else: R = close_R
            R -= c_unit
        elif mode == "limit":
            if not filled:                             # missed runaway -> skipped (no trade)
                busy = exit_j; continue
            rk = lim - stop
            if outcome == "stop": R = (stop - lim) / rk
            elif outcome == "tgt": R = (tgt - lim) / rk
            else: R = (c[exit_j] - lim) / rk
            # keep R in the limit's OWN risk unit (this is how the adopted leg reports)
            R -= cost_frac * lim / rk + cost_abs / rk
        elif mode == "split":
            # market half (always) in u; limit half (if filled) in u
            if outcome == "stop":
                Rm = -1.0
                Rl = (stop - lim) / u if filled else 0.0
            elif outcome == "tgt":
                Rm = (tgt - e) / u
                Rl = (tgt - lim) / u if filled else 0.0
            else:
                Rm = close_R
                Rl = (c[exit_j] - lim) / u if filled else 0.0
            cost = 0.5 * c_unit + (0.5 * c_unit if filled else 0.0)
            R = 0.5 * Rm + 0.5 * Rl - cost
        tr.append((idx[e_i], R)); busy = exit_j
    return tr


def stats(tr, span):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
    return dict(N=len(R), npy=len(R)/span, win=(R > 0).mean()*100, pf=pf, meanR=R.mean(),
                totR=R.sum(), IS=R[yr < half].mean(), OOS=R[yr >= half].mean(),
                retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))


def report(name, E, idx, h, l, c, frac, cost_frac, cost_abs):
    span = (idx[-1] - idx[0]).days / 365.25
    print(f"\n===== {name}  frac={frac} =====")
    print(f"  {'mode':>10}{'N':>5}{'N/yr':>6}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'IS/OOS':>13}{'ret/DD':>8}{'grn':>6}")
    for tag, cf, ca in (("GROSS", 0.0, 0.0), ("NET", cost_frac, cost_abs)):
        print(f"  --- {tag} ---")
        for mode in ("market", "limit", "split"):
            s = stats(walk(E, idx, h, l, c, mode, frac, cf, ca), span)
            io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
            print(f"  {mode:>10}{s['N']:>5}{s['npy']:>6.0f}{s['win']:>4.0f}%{s['pf']:>6.2f}"
                  f"{s['meanR']:>+8.3f}{s['totR']:>+7.0f}{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")


if __name__ == "__main__":
    Eg, ig, hg, lg, cg = build_gold()
    report("GOLD 15m (cost 0.001 frac)", Eg, ig, hg, lg, cg, 0.25, 0.001, 0.0)
    Eb, ib, hb, lb, cb = build_btc()
    report("BTC 15m (cost $15 abs)", Eb, ib, hb, lb, cb, 0.30, 0.0, 15.0)
