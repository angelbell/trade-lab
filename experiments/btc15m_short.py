"""BTC 15M Pattern-B SHORT (breakdown) — mirror of the long engine, to test whether a
short side exists at all. Skeleton (mirror of L0->H1->L2):
  H0 (swing high) -> L1 (swing low = wave-1 low = breakdown line) -> H2 (LOWER high = stop),
  with H2 < H0 and L1 < H0.  Entry = first CLOSE below L1.  Stop = H2.  tgt = e - rr*risk.
Falsification order: ALL-SIGNALS base first (no gate); then daily-KAMA-FALLING gate.
Reports n/win/meanR/IS-OOS/CAGR-DD/per-year at realistic BTC cost 5bp.  No lookahead:
entry at the breakdown bar close; intrabar stop/target forward; KAMA gate uses shift(1).
"""
import sys
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag, kama_adaptive
from research.overfit_audit import cdd_R


def run_short(csv, tf, rr, zzk=2.0, atr_len=14, fwd=500, cost=0.0005, gate=None):
    d = resample(load_mt5_csv(csv), tf)
    h = d["high"].values.astype(float); l = d["low"].values.astype(float); c = d["close"].values.astype(float)
    a = ta.atr(d["high"], d["low"], d["close"], atr_len).values
    ema = ta.ema(d["close"], 80).values
    sw = swings_zigzag(h, l, a, zzk); n = len(c)

    # daily KAMA-falling gate (causal: prior completed daily bar), mapped to entry bars
    gmask = None
    if gate == "kama_fall":
        dc = d["close"].resample("1D").last().dropna()
        km = kama_adaptive(dc, 14)
        fall = (km.diff() < 0).shift(1).reindex(d.index, method="ffill").fillna(False)
        gmask = fall.values

    R, times = [], []
    for t in range(2, len(sw)):
        cH2, iH2, pH2, kH2 = sw[t]         # trigger swing = lower high (stop)
        cL1, iL1, pL1, kL1 = sw[t - 1]     # wave-1 low = breakdown line
        cH0, iH0, pH0, kH0 = sw[t - 2]     # prior high
        if not (kH2 == +1 and kL1 == -1 and kH0 == +1):
            continue
        if pH2 >= pH0 or pH0 - pL1 <= 0:   # need a LOWER HIGH holding below H0
            continue
        # gate: wave-1 low below trend EMA (mirror of long's H1>ema)
        if not (not np.isnan(ema[cL1]) and pL1 < ema[cL1]):
            continue
        # first bar after H2-confirm that CLOSES below L1
        e_i = None
        for j in range(cH2 + 1, n):
            if c[j] < pL1:
                e_i = j; break
        if e_i is None:
            continue
        if gmask is not None and not gmask[e_i]:
            continue
        e = c[e_i]; stop = pH2; risk = stop - e
        if risk <= 0:
            continue
        tgt = e - rr * risk
        if tgt >= e:
            continue
        r = None
        for j in range(e_i + 1, min(e_i + 1 + fwd, n)):
            if h[j] >= stop:            # stop first (conservative on same-bar)
                r = -1.0; break
            if l[j] <= tgt:
                r = rr; break
        if r is None:
            r = (e - c[min(e_i + fwd, n - 1)]) / risk   # mark-to-horizon
        r -= cost / risk * e
        R.append(r); times.append(d.index[e_i])
    return np.array(R), pd.to_datetime(times)


def report(name, R, times):
    if len(R) < 5:
        print(f"  {name:<22} n={len(R)} (too few)"); return
    yr = times.year.values; med = np.median(np.unique(yr))
    IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yrs = max((times.max() - times.min()).days / 365.25, 0.5)
    cdd = cdd_R(R, yrs)[2]
    uy = np.unique(yr); g = sum(1 for y in uy if R[yr == y].sum() > 0)
    win = (R > 0).mean()
    yy = " ".join(f"{y}:{R[yr==y].sum():+.0f}" for y in uy)
    print(f"  {name:<22} n={len(R):>4} win={win:.0%} meanR={R.mean():+.3f} "
          f"CAGR/DD={cdd:+.2f} IS/OOS={IS:+.2f}/{OOS:+.2f} green={g}/{len(uy)}")
    print(f"      per-year: {yy}")


print("BTC 15M Pattern-B SHORT (breakdown) — structured + bear-gate + FAST exit (RR-inversion)")
print("[cost 5bp]  the untested cell: does inverting the exit to fast/tight rescue the short?")
print("  --- ALL-SIGNALS base (fast->slow RR) ---")
for rr in (1.0, 1.5, 2.0, 3.0):
    R, tm = run_short("data/vantage_btcusd_m15.csv", "15min", rr)
    report(f"ALL rr{rr}", R, tm)
print("  --- with daily-KAMA-FALLING gate (fast->slow RR) ---")
for rr in (1.0, 1.5, 2.0, 3.0):
    Rg, tmg = run_short("data/vantage_btcusd_m15.csv", "15min", rr, gate="kama_fall")
    report(f"KAMA-fall rr{rr}", Rg, tmg)
