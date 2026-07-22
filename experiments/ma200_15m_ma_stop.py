"""(3) touch-MA type/length: EMA200 (user's original spec) vs SMA200, +-length plateau.
(2) stop type: fixed-ATR vs STRUCTURAL (causal swing-low) x RR geometry.
Gated+V long (gate = daily SMA200-up & 8H c>SMA200, unchanged), M5-resolved fills,
news window 12-15 UTC skipped, realistic stop-slippage haircut (0.5*overshoot)."""
import sys, itertools; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from research.overfit_audit import cdd_R

d = load_mt5_csv("data/vantage_xauusd_m15.csv"); cl = d["close"]
a = ta.atr(d["high"], d["low"], d["close"], 14).values
e20 = cl.ewm(span=20, adjust=False).mean().values
o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
idx = d.index; iv = idx.values; SK, W = 20, 30
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
dd = d.resample("1D").agg(AGG).dropna(); dsm = dd.close.rolling(200).mean()
dgate = (dsm > dsm.shift(10)).shift(1).fillna(False).reindex(idx, method="ffill").fillna(False).values
r8 = d.resample("8H").agg(AGG).dropna(); g8s = r8.close > r8.close.rolling(200).mean()
g8 = g8s.shift(1).fillna(False).reindex(idx, method="ffill").fillna(False).values

def make_ma(kind, n):
    return (cl.ewm(span=n, adjust=False).mean() if kind == "EMA" else cl.rolling(n).mean()).values

def collect(ma):
    cand, cnt = [], 0
    for s in range(max(260, 2 + SK), len(c) - 1):
        if np.isnan(ma[s - 1]) or np.isnan(a[s - 1]) or a[s - 1] <= 0: continue
        if c[s - 1] > ma[s - 1] + 1.5 * a[s - 1]: cnt = 0
        if not (ma[s - 1] > ma[s - 1 - SK] and l[s] <= ma[s - 1]): continue
        cnt += 1
        win = slice(s - W, s); ext = h[win].max(); bd = W - 1 - int(np.argmax(h[win]))
        vel = ((ext - ma[s - 1]) / a[s - 1]) / max(bd, 1)
        swlo = l[s - 20:s].min()                    # causal swing low over prior 20 bars
        if not (cnt == 1 and (e20[s - 1] > e20[s - 1 - 5]) and dgate[s] and g8[s]): continue
        cand.append((s, iv[s], ma[s - 1], a[s - 1], vel, swlo))
    C = pd.DataFrame(cand, columns=["s", "ts", "ma", "atr", "vel", "swlo"])
    C = C[C.vel >= C.vel.quantile(0.667)].reset_index(drop=True)
    C["ts_ns"] = pd.to_datetime(C.ts).astype("int64")
    return C

m5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
t5 = m5.index.values.astype("datetime64[ns]").astype("int64")
o5, h5, l5, c5 = (m5[k].values for k in ("open", "high", "low", "close"))
COST = 0.40

def sim(C, stop_mode, kparam, RR, skip_news=True, slip=True):
    res, busy = [], -1
    for r in C.itertuples():
        if r.ts_ns <= busy: continue
        if skip_news and 12 <= pd.Timestamp(r.ts_ns).hour < 15: continue
        p0 = int(np.searchsorted(t5, r.ts_ns))
        if p0 >= len(t5): continue
        lim = r.ma; fj = None
        for j in range(p0, min(p0 + 12, len(t5))):
            if l5[j] <= lim: fj = j; break
        if fj is None: continue
        e = min(o5[fj], lim)
        if stop_mode == "atr":
            sd = kparam * r.atr
        else:                                        # structural: below prior swing low
            sd = e - (r.swlo - 0.1 * r.atr)
            if sd <= 0.2 * r.atr: sd = 0.2 * r.atr   # floor (avoid absurd-tight)
        stop = e - sd; tgt = e + RR * sd
        R = None; over = 0.0; end = min(fj + 1500, len(t5))
        for j in range(fj, end):
            if l5[j] <= stop: R = -1.0; over = (stop - l5[j]) / sd; break
            if h5[j] >= tgt: R = RR; break
        if R is None: R = (c5[end - 1] - e) / sd
        if slip and R == -1.0: R -= 0.5 * over        # stop-slippage haircut
        res.append((t5[fj], R - COST / sd)); busy = r.ts_ns
    t = pd.DataFrame(res, columns=["time", "R"]); t["time"] = pd.to_datetime(t["time"])
    return t

def line(tag, t):
    if len(t) < 15: print(f"  {tag:<22} n={len(t)} too few"); return
    x = t.R.values; t = t.assign(y=t.time.dt.year); span = (t.time.iloc[-1]-t.time.iloc[0]).days/365.25
    pf = x[x>0].sum()/abs(x[x<=0].sum()) if (x<=0).any() else 9.0
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    print(f"  {tag:<22} n={len(x):>3} win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={x.mean():+.3f} "
          f"med={np.median(x):+.2f} totR={x.sum():+.0f} CDD={cdd_R(x,span)[2]:+.2f} grn={grn}/{t.y.nunique()}")

print("=== (3) touch-MA type/length (stop=ATR1.0, RR1, news-skip+slip) ===")
for kind in ("SMA", "EMA"):
    for n in (150, 200, 250):
        line(f"{kind}{n}", sim(collect(make_ma(kind, n)), "atr", 1.0, 1.0))
    print()

print("=== (2) stop type x RR  (touch MA from above winner) ===")
for matag in (("SMA", 200), ("EMA", 200)):
    C = collect(make_ma(*matag)); print(f" -- touch {matag[0]}{matag[1]} --")
    for k in (0.75, 1.0, 1.5):
        line(f"ATR{k} RR1", sim(C, "atr", k, 1.0))
    for RR in (1.0, 1.5, 2.0, 3.0):
        line(f"struct-swing RR{RR}", sim(C, "struct", None, RR))
    print()
