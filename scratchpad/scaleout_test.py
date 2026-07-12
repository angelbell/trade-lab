"""#2 target/partial-TP side. If the deep-limit effRR is load-bearing, scaling out early (banking
part at RR1, running the rest to the RR4 level, stop->BE after tp1) should HURT: it banks the very
runners that supply the leverage. Test on the gold 15m pullback-limit leg. f=0 = baseline (canon)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from stop_placement import build_gold, FWD

FRAC, SP = 0.25, 0.6
Eg, idx, o, h, l, c = build_gold()
span = (idx[-1] - idx[0]).days / 365.25


def walk(f, tp1_rr=1.0):
    """f = fraction banked at tp1_rr (in the limit's own risk); rest runs to tgt, stop->BE after tp1."""
    busy = -1; tr = []
    for (e_i, e, pL2, pL0, tgt, atre) in Eg:
        if e_i <= busy: continue
        lim = e - FRAC * (e - pL2)
        if lim <= pL2: continue
        fill_j = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if h[j] >= tgt: fill_j = None; break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        u = lim - pL2; rew = tgt - lim
        realized = 0.0; rem = 1.0; cur_stop = pL2; tp1 = lim + tp1_rr * u; tp1_hit = False
        xj = min(fill_j + FWD, len(c) - 1); done = False
        if l[fill_j] <= pL2:
            realized = -1.0; done = True; xj = fill_j
        for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
            if done: break
            if l[j] <= cur_stop:
                realized += rem * ((cur_stop - lim) / u); done = True; xj = j; break
            if f > 0 and not tp1_hit and h[j] >= tp1:
                realized += f * tp1_rr; rem -= f; tp1_hit = True; cur_stop = lim   # BE
            if h[j] >= tgt:
                realized += rem * (rew / u); done = True; xj = j; break
        if not done: realized += rem * ((c[xj] - lim) / u)
        tr.append((idx[fill_j], realized - SP / u)); busy = xj
    return tr


def st(tr):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
    return dict(N=len(R), win=(R > 0).mean()*100, pf=pf, meanR=R.mean(), totR=R.sum(),
                IS=R[yr < half].mean(), OOS=R[yr >= half].mean(),
                retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))


print("GOLD 15m pullback-limit: scale-out (bank f at RR1, stop->BE, run rest to RR4)")
print(f"  {'variant':>16}{'N':>5}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'IS/OOS':>13}{'ret/DD':>8}{'grn':>6}")
for f in (0.0, 0.3, 0.5, 0.7):
    s = st(walk(f))
    io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
    tag = "baseline (f=0)" if f == 0 else f"scale f={f}@RR1"
    print(f"  {tag:>16}{s['N']:>5}{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}{s['totR']:>+7.0f}"
          f"{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")
# also RR2 tp1
print("  -- tp1 at RR2 instead --")
for f in (0.3, 0.5):
    s = st(walk(f, tp1_rr=2.0))
    io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
    print(f"  {'scale f='+str(f)+'@RR2':>16}{s['N']:>5}{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}"
          f"{s['totR']:>+7.0f}{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")
