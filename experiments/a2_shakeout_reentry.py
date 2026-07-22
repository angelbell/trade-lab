"""A2: shakeout re-entry. After a -1R stop-out, if price CLOSES back above the original
break level (H1) within M bars, re-enter at that close (market), same stop / same target,
once per signal. Isolate the re-entry cohort first (meanR>0 net?), then combined leg vs
base (totR/yr, ret/DD non-degrading). M in {8,16,32} plateau. Canonical walks via
exec-split (gold: pullback_fixedtgt frac0.25 $0.6-anchor -> $0.3 real; BTC: gauntlet
kama1d frac0.3 $15)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")

# ---- canonical entry sets (same trick as duo15m_book.py) ----
gns = {}
exec(open("experiments/pullback_fixedtgt.py").read().split("m=eval_market()")[0], gns)
bns = {}
exec(open("experiments/btc15m_pullback_gauntlet.py").read().split("TFS = [")[0], bns)
bcell = bns["build"](bns["base"], 4.0, True)     # (df, E, h, l, c) kama1d-gated 15m RR4

def eval_reentry(dfi, E, h, l, c, frac, spread, FWD, M):
    """canonical pullback walk + one shakeout re-entry per signal.
    returns list of (time, netR, is_reentry)."""
    busy = -1; out = []
    n = len(c)
    for tup in E:
        i, e, stop, tgt = tup[0], tup[1], tup[2], tup[3]
        H1 = tup[4] if len(tup) > 4 else e
        if i <= busy: continue
        lim = e - frac * (e - stop)
        if lim <= stop or lim >= e: continue
        fill_j = None
        for j in range(i + 1, min(i + 1 + FWD, n)):
            if h[j] >= tgt: break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        risk = lim - stop
        if risk <= 0: continue
        # first attempt
        stopped_at = None
        if l[fill_j] <= stop: R = -1.0; exit_j = fill_j; stopped_at = fill_j
        else:
            exit_j = min(fill_j + FWD, n - 1); R = None
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, n)):
                if l[j] <= stop: R = -1.0; exit_j = j; stopped_at = j; break
                if h[j] >= tgt: R = (tgt - lim) / risk; exit_j = j; break
            if R is None: R = (c[exit_j] - lim) / risk
        out.append((dfi.index[fill_j], R - spread / risk, False))
        busy = exit_j
        # shakeout re-entry: reclaim close above H1 within M bars of the stop-out
        if stopped_at is not None:
            for j2 in range(stopped_at + 1, min(stopped_at + 1 + M, n - 1)):
                if c[j2] > H1:
                    e2 = c[j2]; risk2 = e2 - stop
                    if risk2 <= 0 or e2 >= tgt: break
                    ex2 = min(j2 + FWD, n - 1); R2 = None
                    for j3 in range(j2 + 1, min(j2 + 1 + FWD, n)):
                        if l[j3] <= stop: R2 = -1.0; ex2 = j3; break
                        if h[j3] >= tgt: R2 = (tgt - e2) / risk2; ex2 = j3; break
                    if R2 is None: R2 = (c[ex2] - e2) / risk2
                    out.append((dfi.index[j2], R2 - spread / risk2, True))
                    busy = max(busy, ex2)
                    break
                if l[j2] <= stop - (e - stop):    # fell a full risk below: give up
                    break
    return out

def report(name, dfi, E, h, l, c, frac, spread, FWD):
    span = (dfi.index[-1] - dfi.index[0]).days / 365.25
    print(f"\n===== {name} =====")
    base = eval_reentry(dfi, E, h, l, c, frac, spread, FWD, M=0)
    Rb = np.array([r for _, r, _ in base])
    eqb = np.cumsum(Rb); ddb = (np.maximum.accumulate(eqb) - eqb).max()
    print(f"  base       : N/yr={len(Rb)/span:5.1f}  meanR={Rb.mean():+.3f}  "
          f"totR/yr={Rb.sum()/span:+5.1f}  ret/DD={Rb.sum()/ddb:5.2f}")
    for M in (8, 16, 32):
        allt = eval_reentry(dfi, E, h, l, c, frac, spread, FWD, M=M)
        Ra = np.array([r for _, r, _ in allt])
        re = np.array([r for _, r, isre in allt if isre])
        eq = np.cumsum(Ra); dd = (np.maximum.accumulate(eq) - eq).max()
        pf = re[re>0].sum()/abs(re[re<=0].sum()) if (re<=0).any() and len(re) else 0
        print(f"  M={M:2d} 再入場単離: n={len(re):3d}({len(re)/span:4.1f}/yr) win={(re>0).mean()*100 if len(re) else 0:4.1f}% "
              f"PF={pf:4.2f} meanR={re.mean() if len(re) else 0:+.3f} | 合算 totR/yr={Ra.sum()/span:+5.1f} ret/DD={Ra.sum()/dd:5.2f}")

report("GOLD 15m (canon, $0.3)", gns["d"], gns["entries"], gns["h"], gns["l"], gns["c"],
       0.25, 0.3, gns["FWD"])
dfb, Eb, hb, lb, cb = bcell
report("BTC 15m (kama1d, $15)", dfb, Eb, hb, lb, cb, 0.3, 15.0, bns["FWD"])

# ---- gold cohort robustness: per-year + IS/OOS at M=16 (plateau middle) ----
print("\n--- GOLD 再入場コホート頑健性 (M=16) ---")
allt = eval_reentry(gns["d"], gns["entries"], gns["h"], gns["l"], gns["c"], 0.25, 0.3, gns["FWD"], 16)
re = [(t, r) for t, r, isre in allt if isre]
R = np.array([r for _, r in re]); ts = pd.DatetimeIndex([t for t, _ in re])
yr = ts.year.values; half = np.median(yr)
print(f"  IS/OOS: {R[yr<half].mean():+.3f}(n={int((yr<half).sum())}) / {R[yr>=half].mean():+.3f}(n={int((yr>=half).sum())})")
print("  per-year totR: " + "  ".join(f"{y}:{R[yr==y].sum():+.1f}({(yr==y).sum()})" for y in np.unique(yr)))
