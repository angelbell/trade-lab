"""Search the SHORT's OWN correct settings on the TIME/SCALE axis — do NOT inherit the
long's (zz-k=2, TF=15m). The long/short asymmetry lives on the time axis (down=time-
compressed / up=time-extended), so sweep TIMEFRAME x reversal-scale(zz-k) independently.
All-signals base first (checklist #1): if no TF/scale gives a positive base, nothing
downstream can. Then drill the positive cells with a bear-timing gate + exit.
"""
import sys
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag, kama_adaptive
from research.overfit_audit import cdd_R

_CACHE = {}
def _load(csv, tf):
    key = (csv, tf)
    if key not in _CACHE:
        _CACHE[key] = resample(load_mt5_csv(csv), tf)
    return _CACHE[key]


def run_short(csv, tf, rr, zzk=2.0, atr_len=14, fwd=500, cost=0.0005, gate=None):
    d = _load(csv, tf)
    h = d["high"].values.astype(float); l = d["low"].values.astype(float); c = d["close"].values.astype(float)
    a = ta.atr(d["high"], d["low"], d["close"], atr_len).values
    ema = ta.ema(d["close"], 80).values
    sw = swings_zigzag(h, l, a, zzk); n = len(c)
    gmask = None
    if gate == "kama_fall":
        dc = d["close"].resample("1D").last().dropna()
        km = kama_adaptive(dc, 14)
        fall = (km.diff() < 0).shift(1).reindex(d.index, method="ffill").fillna(False)
        gmask = fall.values

    R, times = [], []
    for t in range(2, len(sw)):
        cH2, iH2, pH2, kH2 = sw[t]; cL1, iL1, pL1, kL1 = sw[t - 1]; cH0, iH0, pH0, kH0 = sw[t - 2]
        if not (kH2 == +1 and kL1 == -1 and kH0 == +1): continue
        if pH2 >= pH0 or pH0 - pL1 <= 0: continue
        if not (not np.isnan(ema[cL1]) and pL1 < ema[cL1]): continue
        e_i = None
        for j in range(cH2 + 1, n):
            if c[j] < pL1: e_i = j; break
        if e_i is None: continue
        if gmask is not None and not gmask[e_i]: continue
        e = c[e_i]; stop = pH2; risk = stop - e
        if risk <= 0: continue
        tgt = e - rr * risk
        r = None
        for j in range(e_i + 1, min(e_i + 1 + fwd, n)):
            if h[j] >= stop: r = -1.0; break
            if l[j] <= tgt: r = rr; break
        if r is None: r = (e - c[min(e_i + fwd, n - 1)]) / risk
        r -= cost / risk * e
        R.append(r); times.append(d.index[e_i])
    return np.array(R), pd.to_datetime(times)


def line(R, times):
    if len(R) < 8: return f"n={len(R):>4} (too few)"
    yr = times.year.values; med = np.median(np.unique(yr))
    IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yrs = max((times.max() - times.min()).days / 365.25, 0.5)
    cdd = cdd_R(R, yrs)[2]; uy = np.unique(yr)
    g = sum(1 for y in uy if R[yr == y].sum() > 0)
    flag = "  <== +base" if (R.mean() > 0 and IS > 0 and OOS > 0) else ""
    return f"n={len(R):>4} win={ (R>0).mean():.0%} meanR={R.mean():+.3f} CAGR/DD={cdd:+.2f} IS/OOS={IS:+.2f}/{OOS:+.2f} grn={g}/{len(uy)}{flag}"


TFS = [("data/vantage_btcusd_m5.csv", "5min"),
       ("data/vantage_btcusd_m15.csv", "15min"),
       ("data/vantage_btcusd_h1.csv", "1h"),
       ("data/vantage_btcusd_h1.csv", "4h")]
ZZK = [1.5, 2.0, 2.5, 3.0, 4.0]

print("BTC SHORT — TIME/SCALE search (all-signals base, rr2.0, cost5bp). Find any +base cell.")
for csv, tf in TFS:
    print(f"\n--- TF={tf} ---")
    for zzk in ZZK:
        R, tm = run_short(csv, tf, 2.0, zzk=zzk)
        print(f"  zz-k={zzk:<3}  {line(R, tm)}")
