"""案3 TIME-STOP test on both 15m breakout legs.
Rule: if a taken trade has NOT reached +1R (high >= e+risk) within K bars of entry, exit at
that bar's CLOSE (market) and RELEASE the busy_until slot -> the walk catches the next signal
SEQUENTIALLY (no added concurrency). Trades that reached +1R keep running to tgt/stop normally.
K sweep {none, 20, 50, 100}. GROSS then NET at the leg's real cost. Reuses the canonical builds.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scratchpad.exec_ceiling_diag import build_gold, build_btc, FWD


def walk(E, idx, h, l, c, K, cost_frac=0.0, cost_abs=0.0):
    """K=None -> baseline (busy until stop/tgt/FWD). Else time-stop laggards at bar e_i+K."""
    busy = -1; tr = []
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        risk = e - stop; reward = tgt - e; one = e + risk
        exit_j = min(e_i + FWD, len(c) - 1); R = None; hit1R = False
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if l[j] <= stop: R = -1.0; exit_j = j; break
            if h[j] >= tgt: R = reward / risk; exit_j = j; break
            if h[j] >= one: hit1R = True
            if K is not None and (not hit1R) and (j - e_i) >= K:
                R = (c[j] - e) / risk; exit_j = j; break     # laggard time-stop at market
        if R is None: R = (c[exit_j] - e) / risk
        cst = cost_frac / risk * e + cost_abs / risk
        R -= cst
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


def report(name, E, idx, h, l, c, cost_frac, cost_abs):
    span = (idx[-1] - idx[0]).days / 365.25
    print(f"\n===== {name} =====")
    print(f"  {'K':>6}{'N':>5}{'N/yr':>6}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'IS/OOS':>13}{'ret/DD':>8}{'grn':>6}")
    for tag, cf, ca in (("GROSS", 0.0, 0.0), ("NET", cost_frac, cost_abs)):
        print(f"  --- {tag} ---")
        for K in (None, 20, 50, 100):
            s = stats(walk(E, idx, h, l, c, K, cf, ca), span)
            io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
            print(f"  {str(K):>6}{s['N']:>5}{s['npy']:>6.0f}{s['win']:>4.0f}%{s['pf']:>6.2f}"
                  f"{s['meanR']:>+8.3f}{s['totR']:>+7.0f}{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")


if __name__ == "__main__":
    Eg, ig, hg, lg, cg = build_gold()
    report("GOLD 15m  (cost 0.001 frac)", Eg, ig, hg, lg, cg, 0.001, 0.0)
    Eb, ib, hb, lb, cb = build_btc()
    report("BTC 15m  (cost $15 abs)", Eb, ib, hb, lb, cb, 0.0, 15.0)
