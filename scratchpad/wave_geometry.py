"""Measure Elliott wave-1 vs wave-3 geometry for the ZigZag Pattern-B breakout legs.

For every ACTUAL entry (all-signals base; gates only SELECT a subset, they do not
change the wave geometry) reconstruct the skeleton  L0 -> H1 -> L2 -> break(H1):
  wave1 = H1 - L0                       (the first impulse)
  retr  = (H1 - L2)/wave1               (wave-2 retrace fraction of wave1)
  risk  = e - L2                        (entry above H1, stop at wave-2 low)
  DESIGNED wave3 (what the FIXED-RR target aims for, measured from L2):
        tgt = e + rr*risk ,  designed3 = tgt - L2 = (rr+1)*risk
  MEASURED-MOVE target (Elliott "wave3 = wave1"):  Lm = L2 + wave1
  REALIZED wave3 = how far price actually ran before the wave-2 low broke
        (max high from e_i.. until low<=L2 or horizon) -> mfe_fromL2 = maxH - L2

Reports median / mean / std / p25 / p75 per config (distributions are skewed, so
quantiles too), plus the share of trades whose fixed-RR TP sits ABOVE the
measured-move (wave3=wave1) level, and the share that realized >=1.0x / 1.618x wave1.
"""
import sys, os
import numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag
import pandas_ta as ta


def measure(csv, tf, rr, zz_k=2.0, atr_len=14, fwd=500):
    d = resample(load_mt5_csv(csv), tf)
    h = d["high"].values.astype(float)
    l = d["low"].values.astype(float)
    c = d["close"].values.astype(float)
    a = ta.atr(d["high"], d["low"], d["close"], atr_len).values
    sw = swings_zigzag(h, l, a, zz_k)
    n = len(c)

    rows = []
    for t in range(2, len(sw)):
        cL2, iL2, pL2, kL2 = sw[t]
        cH1, iH1, pH1, kH1 = sw[t - 1]
        cL0, iL0, pL0, kL0 = sw[t - 2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1):
            continue
        if pL2 <= pL0 or pH1 - pL0 <= 0:          # higher-low that holds
            continue
        # first bar after L2-confirm that CLOSES above H1
        e_i = None
        for j in range(cL2 + 1, n):
            if c[j] > pH1:
                e_i = j
                break
        if e_i is None:
            continue
        e = c[e_i]
        stop = pL2                                # canonical swinglow stop
        risk = e - stop
        if risk <= 0:
            continue
        w1 = pH1 - pL0
        tgt = e + rr * risk
        Lm = pL2 + w1                             # measured-move (wave3 = wave1) level

        # realized favorable run: until the wave-2 low breaks, or horizon
        hi = e
        for j in range(e_i + 1, min(e_i + 1 + fwd, n)):
            if l[j] <= stop:
                hi = max(hi, h[j])
                break
            hi = max(hi, h[j])

        rows.append(dict(
            w1=w1, w1_pct=100 * w1 / e,
            retr=(pH1 - pL2) / w1,
            risk_over_w1=risk / w1,
            designed3_over_w1=(tgt - pL2) / w1,          # crux
            tgt_dist=tgt - e,
            measured_dist=Lm - e,
            tgt_above_measured=1.0 if tgt > Lm else 0.0,
            designed_vs_measured=(tgt - e) / (Lm - e) if (Lm - e) > 0 else np.nan,
            mfe_over_w1=(hi - pL2) / w1,                  # realized wave3 / wave1
            reach_1_0=1.0 if hi >= Lm else 0.0,
            reach_1_618=1.0 if (hi - pL2) >= 1.618 * w1 else 0.0,
        ))
    return rows


def stat(rows, key):
    v = np.array([r[key] for r in rows], float)
    v = v[~np.isnan(v)]
    q = np.percentile(v, [25, 50, 75])
    return f"med={q[1]:+.2f} mean={v.mean():+.2f} sd={v.std():.2f} [p25 {q[0]:+.2f} / p75 {q[2]:+.2f}]"


def share(rows, key):
    v = np.array([r[key] for r in rows], float)
    return f"{100 * v.mean():.0f}%"


CONFIGS = [
    ("GOLD 1H  rr3", "data/vantage_xauusd_h1.csv", "1h", 3.0, 500),
    ("BTC  4H  rr2", "data/vantage_btcusd_h1.csv", "4h", 2.0, 300),
    ("GOLD 15M rr4", "data/vantage_xauusd_m5.csv", "15min", 4.0, 500),
]

for name, csv, tf, rr, fwd in CONFIGS:
    rows = measure(csv, tf, rr, fwd=fwd)
    print(f"\n===== {name}  (all-signals Pattern-B; n={len(rows)}) =====")
    print(f"  wave1 size (% of price)     : {stat(rows,'w1_pct')}")
    print(f"  wave2 retrace / wave1       : {stat(rows,'retr')}")
    print(f"  risk / wave1                : {stat(rows,'risk_over_w1')}")
    print(f"  DESIGNED wave3 / wave1  <-- : {stat(rows,'designed3_over_w1')}")
    print(f"  designed dist / measured    : {stat(rows,'designed_vs_measured')}")
    print(f"  fixed-RR TP ABOVE measured  : {share(rows,'tgt_above_measured')} of trades")
    print(f"  REALIZED wave3 / wave1      : {stat(rows,'mfe_over_w1')}")
    print(f"  reached 1.0x wave1 (measured): {share(rows,'reach_1_0')}   reached 1.618x: {share(rows,'reach_1_618')}")
