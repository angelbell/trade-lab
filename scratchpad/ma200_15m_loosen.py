"""C1: loosen the frequency-killing filters (first-touch / V-cut / gate stack)
on the M5-resolved LIMIT 15m 200MA long, keeping tgt1.0/stop1.0. Goal: recover
to ~40-50 trades/yr while holding PF~2 (the median-2x target). Reports freq/yr."""
import sys, itertools; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
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

dd = d.resample("1D").agg(AGG).dropna(); dsm200 = dd.close.rolling(200).mean()
dgate = (dsm200 > dsm200.shift(10)).shift(1).fillna(False).reindex(idx15, method="ffill").fillna(False).values
def htf_c_gt_sma200(tf):
    r = d.resample(tf).agg(AGG).dropna(); g = r.close > r.close.rolling(200).mean()
    return g.shift(1).fillna(False).reindex(idx15, method="ffill").fillna(False).values
g8 = htf_c_gt_sma200("8H"); g4 = htf_c_gt_sma200("4H")

# collect EVERY touch (no first-touch / no V filter yet); keep attack count + vel + u5
cand, cnt = [], 0
for s in range(222, len(c) - 1):
    if np.isnan(sma[s - 1]) or np.isnan(a[s - 1]) or a[s - 1] <= 0: continue
    if c[s - 1] > sma[s - 1] + 1.5 * a[s - 1]: cnt = 0
    if not (sma[s - 1] > sma[s - 1 - SK] and l[s] <= sma[s - 1]): continue
    cnt += 1
    win = slice(s - W, s); ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
    vel = ((ext - sma[s - 1]) / a[s - 1]) / max(bd, 1)
    cand.append((s, sma[s - 1], a[s - 1], vel, cnt, e20[s - 1] > e20[s - 1 - 5]))
A = pd.DataFrame(cand, columns=["s", "limit", "atr", "vel", "atk", "u5"])

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

GATES = {"D+8H": dgate & g8, "D+4H": dgate & g4, "D": dgate, "8H": g8, "none": np.ones(len(dgate), bool)}
def subset(atk_mode, vcut, u5_on, gate):
    df = A.copy()
    if u5_on: df = df[df.u5]
    if atk_mode == "1st": df = df[df.atk == 1]
    elif atk_mode == "<=2": df = df[df.atk <= 2]
    elif atk_mode == "<=3": df = df[df.atk <= 3]
    df = df[GATES[gate][df.s.values]]
    if vcut > 0: df = df[df.vel >= df.vel.quantile(vcut)]
    return df.reset_index(drop=True)

def report(tag, df):
    if len(df) < 20: print(f"  {tag:<34} (signals {len(df)} too few)"); return
    t = sim(df)
    if len(t) < 15: print(f"  {tag:<34} n={len(t)} too few"); return
    x = t.R.values; t = t.assign(y=pd.to_datetime(t.time).dt.year)
    yrs_span = (t.time.iloc[-1] - t.time.iloc[0]).days / 365.25
    pf = x[x>0].sum()/abs(x[x<=0].sum()) if (x<=0).any() else 9.0
    half = sorted(t.y.unique())[t.y.nunique()//2]
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    cdd = cdd_R(x, yrs_span)[2]
    print(f"  {tag:<34} n={len(x):>3} {len(x)/yrs_span:>4.0f}/yr win={(x>0).mean()*100:>3.0f}% "
          f"PF={pf:.2f} meanR={x.mean():+.3f} IS={t[t.y<half].R.mean():+.2f} OOS={t[t.y>=half].R.mean():+.2f} "
          f"grn={grn}/{t.y.nunique()} CDD={cdd:+.2f}")

print("== axis 1: loosen attack x V-cut  (gate=D+8H, 20MAup on) ==")
print("  [flagship = 1st / V0.667]")
for atk in ("1st", "<=2", "<=3", "all"):
    for vc in (0.667, 0.5, 0.0):
        report(f"atk{atk} V{vc}", subset(atk, vc, True, "D+8H"))
    print()

print("== axis 2: loosen the GATE  (attack=all, V0.5, 20MAup on) ==")
for gate in ("D+8H", "D+4H", "D", "8H", "none"):
    report(f"gate={gate}", subset("all", 0.5, True, gate))

print("\n== axis 3: drop 20MAup too  (attack=all, V0.0) ==")
for gate in ("D+8H", "D", "8H"):
    report(f"gate={gate} no-u5 noV", subset("all", 0.0, False, gate))
