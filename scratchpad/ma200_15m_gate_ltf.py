"""Lower/intermediate-TF trend filters on the M5-resolved LIMIT-entry 15m 200MA
long (tgt1.0/stop1.0). Compares 1H/2H/4H/8H trend gates (close>SMA200, SMA200 up,
close>SMA50, SMA50>200) — standalone AND stacked on the daily-SMA200-up gate.
All gates causal (prior completed HTF bar). Random keep-% null on the best."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import cdd_R

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
sma = cl.rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
idx15 = d.index; iv = idx15.values; SK, W = 20, 30
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

def trend_gates(rule_close):
    sm200 = rule_close.rolling(200).mean(); sm50 = rule_close.rolling(50).mean()
    return {
        "c>SMA200": rule_close > sm200,
        "SMA200up": sm200 > sm200.shift(10),
        "c>SMA50": rule_close > sm50,
        "SMA50>200": sm50 > sm200,
    }

# daily base gate (the previous winner)
dd = d.resample("1D").agg(AGG).dropna()
dsm200 = dd.close.rolling(200).mean()
dgate = (dsm200 > dsm200.shift(10)).shift(1).fillna(False).reindex(idx15, method="ffill").fillna(False).values

# intermediate TF gates
ltf = {}
for tf in ["1H", "2H", "4H", "8H"]:
    r = d.resample(tf).agg(AGG).dropna()
    for nm, g in trend_gates(r.close).items():
        gg = g.shift(1).fillna(False).reindex(idx15, method="ffill").fillna(False).values
        ltf[f"{tf} {nm}"] = gg

# 15m signal collection (strict causal)
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
TGT, KST, COST = 1.0, 1.0, 0.40

def sim(df):
    res, busy_t = [], np.datetime64("1900-01-01")
    for r in df.itertuples():
        ts = iv[r.s]
        if ts <= busy_t: continue
        p0 = int(np.searchsorted(t5, ts))
        if p0 >= len(t5): continue
        lim = r.limit; e = None; end = min(p0 + 900, len(t5))
        for j in range(p0, end):
            if e is None:
                if l5[j] <= lim:
                    e = min(o5[j], lim); sd = KST * r.atr; stop = e - sd; tgt = e + TGT * r.atr
                    if l5[j] <= stop: res.append((t5[j], -1.0 - COST/sd)); busy_t = t5[j]; break
                    if h5[j] >= tgt: res.append((t5[j], TGT/KST - COST/sd)); busy_t = t5[j]; break
                continue
            if l5[j] <= stop: res.append((t5[j], -1.0 - COST/sd)); busy_t = t5[j]; break
            if h5[j] >= tgt:  res.append((t5[j], TGT/KST - COST/sd)); busy_t = t5[j]; break
        else:
            if e is not None:
                res.append((t5[end-1], (c5[end-1]-e)/sd - COST/sd)); busy_t = t5[end-1]
    return pd.DataFrame(res, columns=["time", "R"])

def row(tag, keep):
    t = sim(C[keep].reset_index(drop=True))
    if len(t) < 8: print(f"  {tag:<26} n={len(t)} too few"); return
    x = t.R.values; t = t.assign(y=pd.to_datetime(t.time).dt.year)
    yrs = sorted(t.y.unique()); half = yrs[len(yrs)//2]
    pf = x[x>0].sum()/abs(x[x<=0].sum()) if (x<=0).any() else 9.0
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    span = (t.time.iloc[-1]-t.time.iloc[0]).days/365.25
    cdd = cdd_R(x, span)[2]
    print(f"  {tag:<26} {len(x):>3} {(x>0).mean()*100:>3.0f}% {pf:.2f} {x.mean():+.3f} "
          f"{t[t.y<half].R.mean():+.2f} {t[t.y>=half].R.mean():+.2f} {grn}/{t.y.nunique()}  {cdd:+.2f}")

sv = C.s.values
print("Intermediate-TF trend filters — M5-resolved LIMIT long (tgt1.0/stop1.0)")
print(f"  {'GATE':<26} n   win  PF   meanR   IS    OOS   grn   CAGR/DD")
row("none", np.ones(len(C), bool))
row("[ref] D SMA200up", dgate[sv])
print("  -- intermediate standalone --")
for nm, g in ltf.items():
    row(nm, g[sv])
print("  -- stacked on D SMA200up --")
for nm, g in ltf.items():
    row(f"D+{nm}", dgate[sv] & g[sv])
