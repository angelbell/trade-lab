"""User's ACTUAL trail: stop = PREVIOUS BAR's low, ratcheting up each bar (exit on the first
candle that takes out the prior bar's low = 'one strong red candle and you're out'). NOT an ATR
trail. Key difference from what was tested: with NO fixed target it UNCAPS the runner (can capture
>RR4 on runaways) — trades 'exit pullback-trades early' for 'ride the full extent'. Untested.
Entry = the validated pullback-limit (fill at lim); R-unit = lim-L2 (same as baseline, comparable).
Variants: immediate trail vs trail-only-after-+1R, each with/without the RR4 target cap.
Conservative: prior-bar low is known before the bar; stop fills at the trail level (gap = slippage, noted)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from stop_placement import build_gold, build_btc, FWD


def run_leg(name, E, idx, o, h, l, c, frac, sp):
    span = (idx[-1] - idx[0]).days / 365.25

    def walk(mode, use_tgt):
        """mode: 'baseline'|'immediate'|'after1R'.  use_tgt: keep the RR4 target cap."""
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
            u = lim - pL2; cur_stop = pL2; trailing = (mode == "immediate")
            R = None; xj = min(fill_j + FWD, len(c) - 1)
            if l[fill_j] <= pL2:
                R = -1.0; xj = fill_j
            else:
                for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                    if mode == "after1R" and not trailing and h[j] >= lim + u:
                        trailing = True
                    if trailing and j - 1 >= fill_j:
                        cur_stop = max(cur_stop, l[j - 1])     # prior-bar low, ratchet up
                    if l[j] <= cur_stop:
                        R = (cur_stop - lim) / u; xj = j; break
                    if use_tgt and h[j] >= tgt:
                        R = (tgt - lim) / u; xj = j; break
                if R is None:
                    R = (c[xj] - lim) / u
            tr.append((idx[fill_j], R - sp / u)); busy = xj
        return tr

    def st(tr):
        R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
        pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
        cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
        eq = np.cumprod(1 + 0.01 * R); ddp = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
        yrs = np.unique(yr); half = yrs[len(yrs) // 2]
        grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
        return dict(N=len(R), win=(R > 0).mean()*100, pf=pf, meanR=R.mean(), totR=R.sum(),
                    IS=R[yr < half].mean(), OOS=R[yr >= half].mean(), ddp=ddp,
                    retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs), mx=R.max())

    print(f"\n===== {name}  frac={frac} cost=${sp} =====")
    print(f"  {'variant':>22}{'N':>5}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'maxR':>6}{'IS/OOS':>13}{'maxDD%':>7}{'ret/DD':>8}{'grn':>6}")
    combos = [("baseline L2+RR4", "baseline", True),
              ("prevlow trail, no tgt", "immediate", False),
              ("prevlow trail + RR4 cap", "immediate", True),
              ("trail after +1R, no tgt", "after1R", False),
              ("trail after +1R + RR4", "after1R", True)]
    for tag, mode, ut in combos:
        s = st(walk(mode, ut)); io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
        print(f"  {tag:>22}{s['N']:>5}{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}{s['totR']:>+7.0f}"
              f"{s['mx']:>6.1f}{io:>13}{s['ddp']:>6.1f}%{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")


if __name__ == "__main__":
    run_leg("GOLD 15m", *build_gold(), 0.25, 0.6)
    run_leg("BTC 15m", *build_btc(), 0.30, 15.0)
