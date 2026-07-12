"""Refine the BTC SHORT at its own time-home = 1h. Confirm the zz-k plateau (fine grid),
the exit (RR), and whether a bear-timing gate CONCENTRATES the all-signals +base.
"""
import sys
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag, kama_adaptive
from research.overfit_audit import cdd_R

D = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "1h")
H = D["high"].values.astype(float); L = D["low"].values.astype(float); C = D["close"].values.astype(float)
A = ta.atr(D["high"], D["low"], D["close"], 14).values
EMA = ta.ema(D["close"], 80).values
DC = D["close"].resample("1D").last().dropna()
KFALL = (kama_adaptive(DC, 14).diff() < 0).shift(1).reindex(D.index, method="ffill").fillna(False).values


def run(rr, zzk, fwd=500, cost=0.0005, gate=False):
    sw = swings_zigzag(H, L, A, zzk); n = len(C); R, times = [], []
    for t in range(2, len(sw)):
        cH2, _, pH2, kH2 = sw[t]; cL1, _, pL1, kL1 = sw[t - 1]; _, _, pH0, kH0 = sw[t - 2]
        if not (kH2 == +1 and kL1 == -1 and kH0 == +1): continue
        if pH2 >= pH0 or pH0 - pL1 <= 0: continue
        if not (not np.isnan(EMA[cL1]) and pL1 < EMA[cL1]): continue
        e_i = None
        for j in range(cH2 + 1, n):
            if C[j] < pL1: e_i = j; break
        if e_i is None: continue
        if gate and not KFALL[e_i]: continue
        e = C[e_i]; stop = pH2; risk = stop - e
        if risk <= 0: continue
        tgt = e - rr * risk; r = None
        for j in range(e_i + 1, min(e_i + 1 + fwd, n)):
            if H[j] >= stop: r = -1.0; break
            if L[j] <= tgt: r = rr; break
        if r is None: r = (e - C[min(e_i + fwd, n - 1)]) / risk
        r -= cost / risk * e
        R.append(r); times.append(D.index[e_i])
    return np.array(R), pd.to_datetime(times)


def show(tag, R, tm):
    if len(R) < 8: print(f"  {tag:<22} n={len(R)} (too few)"); return
    yr = tm.year.values; med = np.median(np.unique(yr))
    IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yrs = max((tm.max() - tm.min()).days / 365.25, .5); cdd = cdd_R(R, yrs)[2]
    uy = np.unique(yr); g = sum(1 for y in uy if R[yr == y].sum() > 0)
    fl = "  <== +base" if (R.mean() > 0 and IS > 0 and OOS > 0) else ""
    print(f"  {tag:<22} n={len(R):>4} win={(R>0).mean():.0%} meanR={R.mean():+.3f} "
          f"CAGR/DD={cdd:+.2f} IS/OOS={IS:+.2f}/{OOS:+.2f} grn={g}/{len(uy)}{fl}")


print("BTC SHORT @1h — refine (cost5bp)")
print("[1] zz-k plateau (all-signals, rr2.0):")
for z in (1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2):
    show(f"zz-k={z}", *run(2.0, z))
print("[2] exit RR sweep (all-signals, zz-k1.5):")
for rr in (1.0, 1.5, 2.0, 2.5, 3.0):
    show(f"rr={rr}", *run(rr, 1.5))
print("[3] bear-timing gate = does KAMA-fall CONCENTRATE the +base? (zz-k1.5):")
for rr in (1.5, 2.0, 2.5):
    show(f"rr={rr} +KAMAfall", *run(rr, 1.5, gate=True))
print("[4] per-year, flagship zz-k1.5 rr2.0 all-signals:")
R, tm = run(2.0, 1.5); yr = tm.year.values
print("     " + " ".join(f"{y}:{R[yr==y].sum():+.0f}(n{ (yr==y).sum() })" for y in np.unique(yr)))
