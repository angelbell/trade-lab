"""M5-RESOLVED execution of the strict-causal 15m 200MA long limit-bounce.
Signal lives on 15m (conditions confirmed at [s-1], limit @ sma[s-1]); EXECUTION
walks the 3 M5 sub-bars per 15m bar so we resolve (a) the real limit-fill moment
mid-forming-bar and (b) the TRUE time-order of stop vs target — instead of the
conservative 15m-OHLC 'stop-first' assumption. Compares to the 15m approximation."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

# ---- 15m signal layer (strict causal, identical to audit_ma200_long_strict) ----
d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
idx15 = d.index.values; SK, W = 20, 30

cand, cnt = [], 0
for s in range(222, len(c) - 1):
    if np.isnan(sma[s - 1]) or np.isnan(a[s - 1]) or a[s - 1] <= 0: continue
    if c[s - 1] > sma[s - 1] + 1.5 * a[s - 1]: cnt = 0
    if not (sma[s - 1] > sma[s - 1 - SK] and l[s] <= sma[s - 1]): continue  # limit @ confirmed MA reached
    cnt += 1
    win = slice(s - W, s); ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
    vel = ((ext - sma[s - 1]) / a[s - 1]) / max(bd, 1)
    cand.append((s, sma[s - 1], a[s - 1], vel, cnt, e20[s - 1] > e20[s - 1 - 5]))
C = pd.DataFrame(cand, columns=["s", "limit", "atr", "vel", "atk", "u5"])
C = C[(C.atk == 1) & C.u5]
C = C[C.vel >= C.vel.quantile(0.667)].reset_index(drop=True)

# ---- M5 execution layer ----
m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values; o5, h5, l5, c5 = (m5[k].values for k in ("open", "high", "low", "close"))
TGT, KST, COST = 1.5, 1.0, 0.40

def sim_m5(df):
    res, busy_t = [], np.datetime64("1900-01-01")
    for r in df.itertuples():
        t_start = idx15[r.s]                       # 15m entry bar opens here; conditions known from prior close
        if t_start <= busy_t: continue
        p0 = int(np.searchsorted(t5, t_start))     # first M5 bar of the 15m entry bar
        if p0 >= len(t5): continue
        lim = r.limit; filled = False; e = None
        end = min(p0 + 900, len(t5))               # 300 15m bars ~ 900 M5 bars horizon
        for j in range(p0, end):
            if not filled:
                if l5[j] <= lim:                   # limit reached mid-forming-bar
                    e = min(o5[j], lim); stop = e - KST * r.atr; tgt = e + TGT * r.atr
                    filled = True
                    # same M5 bar can also hit stop/target after fill
                    if l5[j] <= stop: res.append((t5[j], -1.0 - COST / (KST * r.atr))); busy_t = t5[j]; break
                    if h5[j] >= tgt: res.append((t5[j], TGT / KST - COST / (KST * r.atr))); busy_t = t5[j]; break
                continue
            if l5[j] <= stop: res.append((t5[j], -1.0 - COST / (KST * r.atr))); busy_t = t5[j]; break
            if h5[j] >= tgt:  res.append((t5[j], TGT / KST - COST / (KST * r.atr))); busy_t = t5[j]; break
        else:
            if filled:
                R = (c5[end - 1] - e) / (KST * r.atr) - COST / (KST * r.atr)
                res.append((t5[end - 1], R)); busy_t = t5[end - 1]
    return pd.DataFrame(res, columns=["time", "R"])

def report(tag, t):
    x = t.R.values; t = t.assign(y=pd.to_datetime(t.time).dt.year)
    yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2]
    pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else 9.0
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    print(f"  {tag:<18} n={len(x):>3} win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={x.mean():+.3f} "
          f"med={np.median(x):+.2f} IS={t[t.y<half].R.mean():+.2f} OOS={t[t.y>=half].R.mean():+.2f} grn={grn}/{t.y.nunique()}")
    return t

print("M5-resolved LONG, flagship (tgt1.5/stop1.0):")
t = report("flagship", sim_m5(C))
print("  (15m-OHLC stop-first approx was: meanR +0.22 / PF 1.42 / n160)")
print("\nper-year (flagship):")
for y, g in t.groupby("y"):
    gx = g.R.values
    print(f"  {y}: n={len(gx):>3} win={(gx>0).mean()*100:>3.0f}% sumR={gx.sum():+6.1f} meanR={gx.mean():+.2f}")

print("\nM5-resolved target x stop sweep (reflex='small&fast' hypothesis):")
for KST in (0.75, 1.0, 1.5):
    for TGT in (0.5, 0.75, 1.0, 1.5, 2.0):
        report(f"tgt{TGT}/stop{KST}", sim_m5(C))
