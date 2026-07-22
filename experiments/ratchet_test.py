"""User's idea: NO early partial TP (full size runs to the final RR4 target, so the effRR runner
is untouched) — but once price reaches TP2, ratchet the stop up to TP1 (or breakeven). This only
rescues trades that reach TP2 then reverse to a loss; it does NOT bank runners early.
Tax = trades that reach TP2, dip back to TP1/BE (stopped), then WOULD have run to the final tgt.
Levels in the limit's OWN risk u=lim-pL2 (final tgt sits ~5.7R for frac0.25 / ~ (RR+frac)/(1-frac)).
Conservative same-bar: OLD stop checked first, ratchet applied AFTER (live from next bar) = no lookahead."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from stop_placement import build_gold, build_btc, FWD


def run_leg(name, E, idx, o, h, l, c, frac, sp):
    span = (idx[-1] - idx[0]).days / 365.25

    def walk(tp2_rr, to_rr):
        """tp2_rr=None -> baseline. Else at high>=lim+tp2_rr*u, move stop to lim+to_rr*u (0=BE)."""
        busy = -1; tr = []
        for (e_i, e, pL2, pL0, tgt, atre) in E:
            if e_i <= busy: continue
            lim = e - frac * (e - pL2)
            if lim <= pL2: continue
            fill_j = None
            for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
                if h[j] >= tgt: fill_j = None; break
                if l[j] <= lim: fill_j = j; break
            if fill_j is None: continue
            u = lim - pL2; rew = tgt - lim
            cur_stop = pL2; ratched = False
            tp2p = lim + tp2_rr * u if tp2_rr is not None else None
            newstop = lim + to_rr * u if tp2_rr is not None else None
            xj = min(fill_j + FWD, len(c) - 1); R = None
            if l[fill_j] <= pL2:
                R = -1.0; xj = fill_j
            else:
                for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                    if l[j] <= cur_stop:                     # OLD stop first (conservative)
                        R = (cur_stop - lim) / u; xj = j; break
                    if h[j] >= tgt:
                        R = rew / u; xj = j; break
                    if tp2_rr is not None and not ratched and h[j] >= tp2p:
                        ratched = True; cur_stop = newstop   # live from NEXT bar
                if R is None: R = (c[xj] - lim) / u
            tr.append((idx[fill_j], R - sp / u)); busy = xj
        return tr

    def st(tr):
        R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
        pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
        cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
        eq = np.cumprod(1 + 0.01 * R); ddpct = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
        yrs = np.unique(yr); half = yrs[len(yrs) // 2]
        grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
        return dict(N=len(R), win=(R > 0).mean()*100, pf=pf, meanR=R.mean(), totR=R.sum(),
                    ddR=dd, ddpct=ddpct, retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))

    print(f"\n===== {name}  frac={frac} cost=${sp}  (equity DD @ risk 1%/trade) =====")
    print(f"  {'variant':>18}{'N':>5}{'win':>5}{'meanR':>8}{'totR':>7}{'maxDD_R':>9}{'maxDD%':>8}{'ret/DD':>8}{'grn':>6}")
    combos = [("baseline", None, None), ("TP2=2->BE", 2.0, 0.0), ("TP2=2->TP1(1)", 2.0, 1.0),
              ("TP2=3->BE", 3.0, 0.0), ("TP2=3->TP1(1)", 3.0, 1.0), ("TP2=3->TP1(2)", 3.0, 2.0),
              ("TP2=4->BE", 4.0, 0.0), ("TP2=4->TP1(2)", 4.0, 2.0)]
    for tag, t2, to in combos:
        s = st(walk(t2, to))
        print(f"  {tag:>18}{s['N']:>5}{s['win']:>4.0f}%{s['meanR']:>+8.3f}{s['totR']:>+7.0f}"
              f"{s['ddR']:>9.1f}{s['ddpct']:>7.1f}%{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")


if __name__ == "__main__":
    run_leg("GOLD 15m", *build_gold(), 0.25, 0.6)
    run_leg("BTC 15m", *build_btc(), 0.30, 15.0)
