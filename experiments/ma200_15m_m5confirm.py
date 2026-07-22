"""A. M5-CONFIRMED reversal entry (vs falling-knife limit).
15m signal layer unchanged (strict-causal, conditions at [s-1], MA=sma[s-1]).
Execution: walk M5 from the entry 15m bar; (1) wait for touch (low5<=MA),
(2) then wait for the FIRST M5 bar that CLOSES back above MA = bounce confirm,
(3) enter at the NEXT M5 open. Stop/target resolved on M5 in true time order.
Sizing unit = k*ATR(15m). Sweeps target/stop + confirm-window."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
idx15 = d.index.values; SK, W = 20, 30

cand, cnt = [], 0
for s in range(222, len(c) - 1):
    if np.isnan(sma[s - 1]) or np.isnan(a[s - 1]) or a[s - 1] <= 0: continue
    if c[s - 1] > sma[s - 1] + 1.5 * a[s - 1]: cnt = 0
    if not (sma[s - 1] > sma[s - 1 - SK] and l[s] <= sma[s - 1]): continue
    cnt += 1
    win = slice(s - W, s); ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
    vel = ((ext - sma[s - 1]) / a[s - 1]) / max(bd, 1)
    cand.append((s, sma[s - 1], a[s - 1], vel, cnt, e20[s - 1] > e20[s - 1 - 5]))
C = pd.DataFrame(cand, columns=["s", "limit", "atr", "vel", "atk", "u5"])
C = C[(C.atk == 1) & C.u5]
C = C[C.vel >= C.vel.quantile(0.667)].reset_index(drop=True)

m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values; o5, h5, l5, c5 = (m5[k].values for k in ("open", "high", "low", "close"))
COST = 0.40

def sim(df, TGT, KST, CWIN):
    res, busy_t = [], np.datetime64("1900-01-01")
    for r in df.itertuples():
        ts = idx15[r.s]
        if ts <= busy_t: continue
        p0 = int(np.searchsorted(t5, ts))
        if p0 + CWIN + 1 >= len(t5): continue
        lim = r.limit; touched = False; ei = None
        for j in range(p0, p0 + CWIN):           # confirm window (M5 bars)
            if l5[j] <= lim: touched = True
            if touched and c5[j] > lim:           # bounce confirmed on close
                ei = j + 1; break                 # enter next M5 open
        if ei is None or ei >= len(t5): continue
        e = o5[ei]; sd = KST * r.atr; stop = e - sd; tgt = e + TGT * r.atr
        R = None; end = min(ei + 900, len(t5))
        for j in range(ei, end):
            if l5[j] <= stop: R = -1.0; xj = j; break
            if h5[j] >= tgt:  R = TGT / KST; xj = j; break
        if R is None: R = (c5[end - 1] - e) / sd; xj = end - 1
        res.append((t5[xj], R - COST / sd)); busy_t = t5[xj]
    return pd.DataFrame(res, columns=["time", "R"])

def report(tag, t):
    if len(t) < 8: print(f"  {tag:<22} n={len(t)} too few"); return
    x = t.R.values; t = t.assign(y=pd.to_datetime(t.time).dt.year)
    yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2]
    pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else 9.0
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    print(f"  {tag:<22} n={len(x):>3} win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={x.mean():+.3f} "
          f"med={np.median(x):+.2f} IS={t[t.y<half].R.mean():+.2f} OOS={t[t.y>=half].R.mean():+.2f} grn={grn}/{t.y.nunique()}")

print("A. M5-CONFIRMED reversal entry — target x stop (confirm window = 6 M5 bars = 30min):")
for KST in (0.75, 1.0, 1.5):
    for TGT in (0.75, 1.0, 1.5, 2.0):
        report(f"tgt{TGT}/stop{KST}", sim(C, TGT, KST, 6))
print("\nconfirm-window sweep (tgt1.0/stop1.0):")
for CW in (3, 6, 9, 12):
    report(f"cwin{CW}", sim(C, 1.0, 1.0, CW))
print("\nper-year (best-looking cell tgt1.0/stop1.0, cwin6):")
t = sim(C, 1.0, 1.0, 6); t = t.assign(y=pd.to_datetime(t.time).dt.year)
for y, g in t.groupby("y"):
    gx = g.R.values
    print(f"  {y}: n={len(gx):>3} win={(gx>0).mean()*100:>3.0f}% sumR={gx.sum():+6.1f} meanR={gx.mean():+.2f}")
