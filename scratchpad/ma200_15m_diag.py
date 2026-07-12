"""15m gold 200MA bounce diagnostic: does a better exit/filter revive it at 15m?
base vs first-touch x exit{structural-target / fixed RR1.5 / RR2}, with IS/OOS,
PF, per-year greens, and a BETA-null (beat random-long-in-uptrend same geometry?).
Cost modeled (15m is cost-sensitive: tight stops -> cost is a big fraction of R)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

RNG = np.random.default_rng(3)
COST = 0.0003


def fractals(h, l, p=3):
    lows = []
    for t in range(p, len(h) - p):
        if l[t] == min(l[t - p:t + p + 1]):
            lows.append((t + p, l[t]))
    return np.array([x[0] for x in lows]), np.array([x[1] for x in lows])


def find(d, slopeK=20, zoneW=0.5, atrlen=14, swingW=30, cleanN=10):
    sma = d["close"].rolling(200).mean().values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    rows, cnt = [], 0
    for s in range(max(slopeK, swingW, 200) + 1, len(c) - 1):
        if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        z = zoneW * a[s]
        if c[s] > sma[s] + 1.5 * a[s]:
            cnt = 0
        if not (sma[s] > sma[s - slopeK] and c[s] > sma[s]):
            continue
        if not (l[s] <= sma[s] + z and c[s] > sma[s] - z and c[s] > o[s]):
            continue
        cnt += 1
        e = o[s + 1]; stop = l[s]
        if e - stop < 0.5 * a[s]:
            stop = e - 0.5 * a[s]
        if e <= stop:
            continue
        win = slice(s - swingW + 1, s + 1)
        swing_hi = h[win].max(); bars_down = swingW - 1 - int(np.argmax(h[win]))
        vel = ((swing_hi - l[s]) / a[s]) / max(bars_down, 1)
        clean = int(np.sum(l[s - cleanN:s] <= sma[s - cleanN:s] + z))
        rows.append(dict(i=s + 1, e=e, stop=stop, target=swing_hi, vel=vel,
                         bars_down=bars_down, attack=cnt, clean=clean))
    return d, rows


def sim(d, rows, mode, rr, fwd=400):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    idx = d.index
    res, busy = [], -1
    for r in rows:
        i, e, stop, tgt = r["i"], r["e"], r["stop"], r["target"]
        if i <= busy:
            continue
        risk = e - stop
        if risk <= 0:
            continue
        target = tgt if mode == "target" else e + rr * risk
        if target <= e:
            continue
        R = None; xj = min(i + fwd, len(c) - 1)
        for j in range(i, min(i + fwd, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= target: R = (target - e) / risk; xj = j; break
        if R is None: R = (c[xj] - e) / risk
        R -= COST / risk * e
        res.append((idx[i], R)); busy = xj
    return pd.DataFrame(res, columns=["time", "R"])


def beta_null(d, rows, mode, rr, niter=300):
    sma = d["close"].rolling(200).mean().values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], length=14).values
    ctx = np.where((sma[:-1] > np.roll(sma, 20)[:-1]) & (c[:-1] > sma[:-1]))[0]
    ctx = ctx[(ctx > 200) & (ctx < len(c) - 401)]
    satr = [(r["e"] - r["stop"]) / a[r["i"]] for r in rows if a[r["i"]] > 0]
    real = sim(d, rows, mode, rr)["R"].mean()
    means = []
    for _ in range(niter):
        picks = RNG.choice(ctx, len(rows)); sds = RNG.choice(satr, len(rows)); rs = []
        for ix, sd in zip(picks, sds):
            e = o[ix]; risk = sd * a[ix]
            if risk <= 0: continue
            stop = e - risk; tgt = e + (3 * risk if mode == "target" else rr * risk)
            R = None
            for j in range(ix, min(ix + 400, len(c))):
                if l[j] <= stop: R = -1.0; break
                if h[j] >= tgt: R = (tgt - e) / risk; break
            if R is None: R = (c[min(ix + 400, len(c) - 1)] - e) / risk
            rs.append(R - COST / risk * e)
        means.append(np.mean(rs))
    return (real > np.array(means)).mean() * 100


def stats(t):
    if len(t) < 8: return None
    t = t.copy(); t["y"] = t.time.dt.year
    yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2]
    isr = t[t.y < half].R; oosr = t[t.y >= half].R
    x = t.R.values; pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else np.inf
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    return dict(n=len(t), win=(x > 0).mean() * 100, pf=pf, meanR=x.mean(),
                IS=isr.mean(), OOS=oosr.mean(), grn=grn, ny=t.y.nunique())


d = load_mt5_csv("data/vantage_xauusd_h1.csv")  # placeholder; replaced below
d, rows = find(load_mt5_csv("data/vantage_xauusd_m15.csv"))
dd = load_mt5_csv("data/vantage_xauusd_m15.csv")
print(f"GOLD 15m 200MA bounce  (zone0.5, V-avoid, cost {COST}, long)  base touches={len(rows)}")
print(f"{'filter':<12}{'exit':<8}{'n':>5}{'win%':>6}{'PF':>6}{'meanR':>8}{'IS':>7}{'OOS':>7}{'grn':>7}{'beta%':>7}")
nv = [r for r in rows if r["bars_down"] >= 4 and r["vel"] <= 0.6]
ft = [r for r in nv if r["attack"] == 1]
for tag, rws in [("base+Vavoid", nv), ("first-touch", ft)]:
    for mode, rr in [("target", 3), ("rr", 1.5), ("rr", 2.0)]:
        t = sim(dd, rws, mode, rr); st = stats(t)
        if st is None:
            print(f"{tag:<12}{(mode+str(rr)):<8} too few"); continue
        bp = beta_null(dd, rws, mode, rr)
        ex = "target" if mode == "target" else f"RR{rr}"
        print(f"{tag:<12}{ex:<8}{st['n']:>5}{st['win']:>6.0f}{st['pf']:>6.2f}{st['meanR']:>8.2f}"
              f"{st['IS']:>7.2f}{st['OOS']:>7.2f}{st['grn']:>3}/{st['ny']:<3}{bp:>6.0f}")
