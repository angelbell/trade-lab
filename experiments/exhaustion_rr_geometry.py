"""Does HIGH RR rescue a direction-neutral entry? The user's exact strategy geometry:
entry at exhaustion-bounce fill, stop = cluster low - 0.25*ATR (structural), target =
big-down-candle high (the strategy's own far target). Measure win% vs the MARTINGALE
breakeven risk/(risk+reward) per event, meanR gross, and vs SAME-GEOMETRY random entries
(same hour, same stop/target distances in ATR units). Net = $0.3 rt + stop slip 0.27ATR.
If win% == geometry breakeven == random-same-geometry, RR does NOT create EV; cost makes
it negative. That is the claim under test."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exhaustion_bounce as EB
from src.data_loader import load_mt5_csv

d = load_mt5_csv("data/vantage_xauusd_m5.csv")
cnt = d.groupby(d.index.date).size()
ok = cnt[cnt.rolling(30).median() >= 150]
d = d[d.index.date >= ok.index[0]]
sig = EB.build_signals(d)
h, l, c = d["high"].values, d["low"].values, d["close"].values
atr_arr = EB.atr(h, l, c)
n = len(c); FWD = 288   # 24h of 5m bars
hours = d.index.hour.values

def walk(i, e, stop, tgt, slip=0.27, ac=None):
    risk = e - stop; reward = tgt - e
    if risk <= 0 or reward <= 0 or i + 1 >= n: return None
    end = min(i + FWD, n - 1); R = None
    for j in range(i + 1, min(i + 1 + FWD, n)):
        if l[j] <= stop: R = -1.0 - slip * (ac if ac else atr_arr[i]) / risk; end = j; break
        if h[j] >= tgt: R = reward / risk; end = j; break
    if R is None: R = (c[end] - e) / risk
    return R - 0.3 / risk, reward / risk, risk

rows = []
for (fi, arm, aH, aL, tgt, ac) in sig["fills"]:
    if fi + 1 >= n or np.isnan(ac) or ac <= 0: continue
    e = c[fi]; stop = aL - 0.25 * ac
    r = walk(fi, e, stop, tgt, ac=ac)
    if r: rows.append((fi, *r))
R = np.array([x[1] for x in rows]); RR = np.array([x[2] for x in rows]); RISK = np.array([x[3] for x in rows])
idx = np.array([x[0] for x in rows], int)
win = R > 0
be = 1.0 / (1.0 + RR)          # martingale breakeven win% per event
print(f"REAL exhaustion events n={len(R)}  medRR={np.median(RR):.2f}  med(risk$)={np.median(RISK):.2f}"
      f"  cost/risk med={np.median(0.3/RISK)*100:.1f}%")
print(f"  win={win.mean()*100:.1f}%  martingale-breakeven(медRR)={np.mean(be)*100:.1f}%"
      f"  meanR net={R.mean():+.3f}  totR={R.sum():+.0f}")

rng = np.random.default_rng(7)
mr, wr = [], []
for t in range(200):
    Rr = []
    for (fi, rr, risk) in zip(idx, RR, RISK):
        for _ in range(3):
            j = rng.integers(200, n - FWD - 2)
            if np.isnan(atr_arr[j]) or atr_arr[j] <= 0: continue
            scale = atr_arr[j] / atr_arr[fi] if atr_arr[fi] > 0 else 1.0
            e2 = c[j]; stop2 = e2 - RISK[0] * 0 - (risk * scale); tgt2 = e2 + rr * (risk * scale)
            r = walk(j, e2, stop2, tgt2, ac=atr_arr[j])
            if r: Rr.append(r[0])
    if Rr:
        Rr = np.array(Rr); mr.append(Rr.mean()); wr.append((Rr > 0).mean())
    if t >= 30: break
mr = np.array(mr); wr = np.array(wr)
print(f"RANDOM same-geometry (hour-agnostic, {len(mr)} trials x ~{len(R)*3}):"
      f"  win med={np.median(wr)*100:.1f}%  meanR med={np.median(mr):+.3f} sd={mr.std():.3f}")
print(f"  -> real meanR percentile vs random: {(mr < R.mean()).mean()*100:.0f}%ile")
