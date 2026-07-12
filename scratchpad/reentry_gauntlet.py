"""gold 15m shakeout RE-ENTRY (案2) overfit gauntlet.
 (1) audit: DSR / PBO(over M choice) / null-p / bootCI on {pure-limit, +reentry M20/50/100}.
 (2) TIMING null: for the SAME stopout windows that had a re-break, re-enter at the re-break bar
     [real] vs a RANDOM bar in (stopout, stopout+M] [null, 2000 seeds]. If the re-break timing
     is just 'buy any dip in gold's uptrend', real sits mid-distribution. Edge => high percentile.
 (3) per-year of the M50 leg + subset.
Reuses reentry_test build/play_limit (canon-faithful)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from reentry_test import build_gold, play_limit, walk, st, peryr, FWD
from research.edge_harness import audit

FRAC, SP = 0.25, 0.6
Eg, idx, o, h, l, c = build_gold()
span = (idx[-1] - idx[0]).days / 365.25


def primary_stopouts(E):
    busy = -1; ops = []
    for (e_i, e, stop, tgt, pH1) in E:
        if e_i <= busy: continue
        r = play_limit(e_i, e, stop, tgt, FRAC, SP, h, l, c)
        if r is None: continue
        R, xj, was_stop, ej = r; busy = xj
        if was_stop: ops.append((xj, stop, tgt, pH1))
    return ops


def rebreak_bar(xj, pH1, M):
    for j in range(xj + 1, min(xj + 1 + M, len(c))):
        if c[j] > pH1: return j
    return None


# ---------- (1) AUDIT ----------
cfgs = {"pure-limit": walk(Eg, idx, o, h, l, c, FRAC, SP, None)[0]}
for M in (20, 50, 100):
    cfgs[f"reentry_M{M}"] = walk(Eg, idx, o, h, l, c, FRAC, SP, M)[0]
print("==== (1) OVERFIT AUDIT (flagship = reentry_M50) ====")
for k, v in cfgs.items():
    R = np.array([r for _, r in v]); print(f"  {k:<13} n={len(R)} meanR={R.mean():+.3f} totR={R.sum():+.0f}")
audit(cfgs, flagship="reentry_M50")

# ---------- (2) TIMING NULL (M=50) ----------
M = 50
ops = primary_stopouts(Eg)
ops_rb = [(xj, stop, tgt, pH1, rebreak_bar(xj, pH1, M)) for (xj, stop, tgt, pH1) in ops]
ops_rb = [op for op in ops_rb if op[4] is not None]     # had a re-break within M
real = []
for (xj, stop, tgt, pH1, y) in ops_rb:
    r = play_limit(y, c[y], stop, tgt, FRAC, SP, h, l, c)
    if r is not None: real.append(r[0])
real = np.array(real)
rng = np.random.default_rng(0); null_means = []; null_tots = []
for _ in range(2000):
    Rs = []
    for (xj, stop, tgt, pH1, y) in ops_rb:
        hi = min(xj + M, len(c) - 1)
        if hi <= xj: continue
        rb = int(rng.integers(xj + 1, hi + 1))
        res = play_limit(rb, c[rb], stop, tgt, FRAC, SP, h, l, c)
        if res is not None: Rs.append(res[0])
    if Rs: null_means.append(np.mean(Rs)); null_tots.append(np.sum(Rs))
null_means = np.array(null_means); null_tots = np.array(null_tots)
print(f"\n==== (2) TIMING NULL (M=50, {len(ops_rb)} stopout windows w/ a re-break) ====")
print(f"  REAL re-break re-entry:  n={len(real)} meanR={real.mean():+.3f} totR={real.sum():+.0f}")
print(f"  RANDOM-timing re-entry:  meanR med={np.median(null_means):+.3f} "
      f"[5/95={np.percentile(null_means,5):+.3f}/{np.percentile(null_means,95):+.3f}]  "
      f"totR med={np.median(null_tots):+.0f}")
print(f"  real meanR percentile vs random-timing null = {(null_means < real.mean()).mean()*100:.0f}%  "
      f"(>90 = the SHAKEOUT TIMING adds edge, not generic dip-buy)")

# ---------- (3) PER-YEAR ----------
allt, ret = walk(Eg, idx, o, h, l, c, FRAC, SP, M)
print(f"\n==== (3) PER-YEAR (M=50) ====")
print(f"  leg      : {peryr(allt)}")
print(f"  subset   : {peryr(ret)}")
s = st(allt, span)
print(f"  leg totals: n={s['N']} meanR={s['meanR']:+.3f} totR={s['totR']:+.0f} ret/DD={s['retdd']:.2f} "
      f"IS/OOS={s['IS']:+.2f}/{s['OOS']:+.2f} green={s['grn']}/{s['ny']}")
