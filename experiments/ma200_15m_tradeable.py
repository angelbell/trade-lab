"""15m gold 200MA reflex-bounce TRADEABILITY (confirmed-bar entry, machine-runnable).
Signal = first-touch + steep-V approach + long lower-wick, touch bar closes bullish
above MA (rejection confirmed) -> enter NEXT bar open. Stop = touch low. Fixed
thresholds (no tercile lookahead). Sweep quick targets; full falsification."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

RNG = np.random.default_rng(5)

d = load_mt5_csv("data/vantage_xauusd_m15.csv")
sma = d["close"].rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], length=14).values
o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
idx = d.index
slopeK, zoneW, swingW = 20, 0.5, 30

# pass 1: collect feature distribution to FIX thresholds (then frozen as constants)
vels, wicks = [], []
tmp, cnt = [], 0
for s in range(221, len(c) - 1):
    if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
        continue
    z = zoneW * a[s]
    if c[s] > sma[s] + 1.5 * a[s]:
        cnt = 0
    if not (sma[s] > sma[s - slopeK] and c[s] > sma[s] and l[s] <= sma[s] + z and c[s] > o[s]):
        continue
    cnt += 1
    rng = max(h[s] - l[s], 1e-9); win = slice(s - swingW + 1, s + 1)
    sh = h[win].max(); bd = swingW - 1 - int(np.argmax(h[win])); vel = ((sh - l[s]) / a[s]) / max(bd, 1)
    lw = (min(o[s], c[s]) - l[s]) / rng
    vels.append(vel); wicks.append(lw)
    tmp.append((s, cnt, vel, lw))
V_THR = float(np.quantile(vels, 2 / 3)); W_THR = float(np.quantile(wicks, 2 / 3))
print(f"fixed thresholds: V(vel)>= {V_THR:.2f}   wick(lwick)>= {W_THR:.2f}   (frozen tercile cuts)")

# signals = first-touch + V + wick
sig = [(s, l[s]) for (s, ct, vel, lw) in tmp if ct == 1 and vel >= V_THR and lw >= W_THR]
print(f"signals (1st+V+wick, confirmed bullish touch bar): n={len(sig)}  ~{len(sig)/((idx[-1]-idx[0]).days/365):.0f}/yr\n")


def run(rr, cost, sigs=sig):
    rows, busy = [], -1
    for (s, lowtouch) in sigs:
        i = s + 1                      # confirmed entry: next bar open
        if i <= busy or i >= len(c):
            continue
        e = o[i]; stop = lowtouch
        if e - stop < 0.5 * a[s]:
            stop = e - 0.5 * a[s]
        risk = e - stop
        if risk <= 0:
            continue
        tgt = e + rr * risk; R = None; xj = min(i + 200, len(c) - 1)
        for j in range(i, min(i + 200, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= tgt: R = rr; xj = j; break
        if R is None: R = (c[xj] - e) / risk
        R -= cost / risk * e
        rows.append((idx[i], R)); busy = xj
    return pd.DataFrame(rows, columns=["time", "R"])


def beta(rr, cost, n, niter=300):
    ctx = np.where((sma[:-1] > np.roll(sma, slopeK)[:-1]) & (c[:-1] > sma[:-1]))[0]
    ctx = ctx[(ctx > 200) & (ctx < len(c) - 201)]
    real = run(rr, cost)["R"].mean(); means = []
    for _ in range(niter):
        rs = []
        for ix in RNG.choice(ctx, n):
            e = o[ix]; risk = 0.7 * a[ix]
            if risk <= 0: continue
            stop = e - risk; tgt = e + rr * risk; R = None
            for j in range(ix, min(ix + 200, len(c))):
                if l[j] <= stop: R = -1.0; break
                if h[j] >= tgt: R = rr; break
            if R is None: R = (c[min(ix + 200, len(c) - 1)] - e) / risk
            rs.append(R - cost / risk * e)
        means.append(np.mean(rs))
    return (real > np.array(means)).mean() * 100


def stats(t):
    t = t.copy(); t["y"] = t.time.dt.year; yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2]
    x = t.R.values; pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else np.inf
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    return (len(t), (x > 0).mean() * 100, pf, x.mean(),
            t[t.y < half].R.mean(), t[t.y >= half].R.mean(), grn, t.y.nunique())


for cost in (0.0002, 0.0005):
    print(f"--- cost {cost} (15m gold spread/slip) ---")
    print(f"{'RR':<6}{'n':>5}{'win%':>6}{'PF':>6}{'meanR':>8}{'IS':>7}{'OOS':>7}{'grn':>7}{'beta%':>7}")
    for rr in (0.75, 1.0, 1.5, 2.0):
        t = run(rr, cost)
        n, win, pf, mr, is_, oos, grn, ny = stats(t)
        bp = beta(rr, cost, n)
        print(f"{rr:<6}{n:>5}{win:>6.0f}{pf:>6.2f}{mr:>8.2f}{is_:>7.2f}{oos:>7.2f}{grn:>3}/{ny:<3}{bp:>6.0f}")
    print()
