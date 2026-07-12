"""BTC 15m TP2-ratchet (TP2=3->stop TP1=1) DD-reduction gauntlet.
 (1) PLATEAU: sweep TP2 x TP1 -> is the DD cut a broad region (robust) or a lone cell (overfit)?
 (2) IS/OOS split: does the DD cut appear in BOTH halves, or only OOS/recent (= regime-luck, the
     signature that killed vol-targeting)?
 (3) per-year totR + within-year maxDD: which years give the DD cut.
 (4) block-bootstrap maxDD%: is the ratchet's DD distribution clearly below baseline's?
Return-for-safety, not a new edge -> the bar is 'robust DD cut across regimes at small totR cost'."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from stop_placement import build_btc, FWD

FRAC, SP = 0.30, 15.0
E, idx, o, h, l, c = build_btc()
span = (idx[-1] - idx[0]).days / 365.25


def walk(tp2, to):
    busy = -1; tr = []
    for (e_i, e, pL2, pL0, tgt, atre) in E:
        if e_i <= busy: continue
        lim = e - FRAC * (e - pL2)
        if lim <= pL2: continue
        fill_j = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if h[j] >= tgt: fill_j = None; break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        u = lim - pL2; rew = tgt - lim; cur_stop = pL2; ratched = False
        tp2p = lim + tp2 * u if tp2 is not None else None
        newstop = lim + to * u if tp2 is not None else None
        xj = min(fill_j + FWD, len(c) - 1); R = None
        if l[fill_j] <= pL2:
            R = -1.0; xj = fill_j
        else:
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                if l[j] <= cur_stop: R = (cur_stop - lim) / u; xj = j; break
                if h[j] >= tgt: R = rew / u; xj = j; break
                if tp2 is not None and not ratched and h[j] >= tp2p: ratched = True; cur_stop = newstop
            if R is None: R = (c[xj] - lim) / u
        tr.append((idx[fill_j], R - SP / u)); busy = xj
    return tr


def ddpct(R):
    eq = np.cumprod(1 + 0.01 * np.asarray(R))
    return ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100


def summ(tr):
    R = np.array([r for _, r in tr]); return dict(N=len(R), totR=R.sum(), meanR=R.mean(),
        dd=ddpct(R), retdd=R.sum() / (np.maximum.accumulate(np.cumsum(R)) - np.cumsum(R)).max())


print(f"BTC 15m TP2-ratchet gauntlet  (span {span:.1f}yr, cost ${SP})\n")
print("==== (1) PLATEAU: maxDD% (ret/DD) across TP2 x TP1 ====")
print(f"  {'TP2\\TP1':>8}" + "".join(f"{'BE':>14}" if i == 0 else f"{'TP1='+str(v):>14}" for i, v in enumerate([0, 1, 2])))
base = summ(walk(None, None))
print(f"  baseline: totR{base['totR']:+.0f} maxDD {base['dd']:.1f}% ret/DD {base['retdd']:.2f}")
for tp2 in (2.0, 2.5, 3.0, 3.5, 4.0):
    cells = []
    for to in (0.0, 1.0, 2.0):
        if to >= tp2: cells.append("  -"); continue
        s = summ(walk(tp2, to)); cells.append(f"{s['dd']:.1f}%({s['retdd']:.2f})")
    print(f"  {tp2:>8}" + "".join(f"{x:>14}" for x in cells))

print("\n==== (2) IS/OOS split (baseline vs TP2=3->TP1=1) ====")
for tag, tr in (("baseline", walk(None, None)), ("ratchet(3,1)", walk(3.0, 1.0))):
    yrs = np.array([t.year for t, _ in tr]); half = np.unique(yrs)[len(np.unique(yrs)) // 2]
    R = np.array([r for _, r in tr]); ism = R[yrs < half]; oos = R[yrs >= half]
    print(f"  {tag:<14} IS: meanR{ism.mean():+.3f} totR{ism.sum():+.0f} maxDD{ddpct(ism):.1f}%   "
          f"OOS: meanR{oos.mean():+.3f} totR{oos.sum():+.0f} maxDD{ddpct(oos):.1f}%")

print("\n==== (3) per-year totR / within-year maxDD% ====")
for tag, tr in (("baseline", walk(None, None)), ("ratchet(3,1)", walk(3.0, 1.0))):
    yr = np.array([t.year for t, _ in tr]); R = np.array([r for _, r in tr])
    print(f"  {tag:<14} " + " ".join(f"{y}:{R[yr==y].sum():+.0f}/DD{ddpct(R[yr==y]):.0f}%" for y in sorted(set(yr))))

print("\n==== (4) block-bootstrap maxDD% (block=20, 2000x) ====")
rng = np.random.default_rng(0)
def boot_dd(tr):
    R = np.array([r for _, r in tr]); n = len(R); out = []
    for _ in range(2000):
        idxs = []
        while len(idxs) < n:
            s = rng.integers(0, n); idxs.extend(range(s, min(s + 20, n)))
        out.append(ddpct(R[np.array(idxs[:n])]))
    return np.array(out)
for tag, tr in (("baseline", walk(None, None)), ("ratchet(3,1)", walk(3.0, 1.0))):
    b = boot_dd(tr)
    print(f"  {tag:<14} maxDD% median={np.median(b):.1f} [5/95={np.percentile(b,5):.1f}/{np.percentile(b,95):.1f}]")
