"""15m gold 200MA bounce TRADEABILITY, LONG & SHORT separate, FIXED $3 stop.
Confirmed-close entry (touch bar closes back across MA) -> next bar open.
Stop = $3 fixed (long e-3 / short e+3). Target = RR x $3 (sweep). Cost in $.
Per-year shown so the era effect of a FIXED $ stop is visible (old gold small ATR)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

STOP_USD = 3.0
d = load_mt5_csv("data/vantage_xauusd_m15.csv")
sma = d["close"].rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], length=14).values
o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
idx = d.index
slopeK, zoneW, swingW = 20, 0.5, 30


def collect(side):
    rows, cnt = [], 0
    for s in range(221, len(c) - 1):
        if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        z = zoneW * a[s]; rng = max(h[s] - l[s], 1e-9); win = slice(s - swingW + 1, s + 1)
        if side == "long":
            if c[s] > sma[s] + 1.5 * a[s]: cnt = 0
            if not (sma[s] > sma[s - slopeK] and l[s] <= sma[s] + z and c[s] > sma[s] and c[s] > o[s]):
                continue
            cnt += 1
            ext = h[win].max(); bd = swingW - 1 - int(np.argmax(h[win])); vel = ((ext - l[s]) / a[s]) / max(bd, 1)
            wick = (min(o[s], c[s]) - l[s]) / rng
        else:
            if c[s] < sma[s] - 1.5 * a[s]: cnt = 0
            if not (sma[s] < sma[s - slopeK] and h[s] >= sma[s] - z and c[s] < sma[s] and c[s] < o[s]):
                continue
            cnt += 1
            ext = l[win].min(); bd = swingW - 1 - int(np.argmin(l[win])); vel = ((h[s] - ext) / a[s]) / max(bd, 1)
            wick = (h[s] - max(o[s], c[s])) / rng
        rows.append(dict(s=s, vel=vel, wick=wick, attack=cnt))
    df = pd.DataFrame(rows)
    df["Vth"] = df.vel >= df.vel.quantile(2 / 3)
    df["Wth"] = df.wick >= df.wick.quantile(2 / 3)
    return df


def run(side, sigs, rr, cost):
    res, busy = [], -1
    for _, r in sigs.iterrows():
        s = int(r.s); i = s + 1
        if i <= busy or i >= len(c): continue
        e = o[i]
        if side == "long":
            stop = e - STOP_USD; tgt = e + rr * STOP_USD
            R = None; xj = min(i + 200, len(c) - 1)
            for j in range(i, min(i + 200, len(c))):
                if l[j] <= stop: R = -1.0; xj = j; break
                if h[j] >= tgt: R = rr; xj = j; break
            if R is None: R = (c[xj] - e) / STOP_USD
        else:
            stop = e + STOP_USD; tgt = e - rr * STOP_USD
            R = None; xj = min(i + 200, len(c) - 1)
            for j in range(i, min(i + 200, len(c))):
                if h[j] >= stop: R = -1.0; xj = j; break
                if l[j] <= tgt: R = rr; xj = j; break
            if R is None: R = (e - c[xj]) / STOP_USD
        R -= cost / STOP_USD
        res.append((idx[i], R)); busy = xj
    t = pd.DataFrame(res, columns=["time", "R"]); t["y"] = t.time.dt.year
    return t


def line(tag, t):
    if len(t) < 8: print(f"  {tag:<26} n={len(t)} too few"); return
    yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2]
    x = t.R.values; pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else 9
    grn = sum(1 for _, g in t.groupby("y") if g.R.sum() > 0)
    rec = t[t.y >= 2022].R                         # recent-era (where $3 stop is sensible)
    print(f"  {tag:<26} n={len(t):>4} win={ (x>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={x.mean():+.2f} "
          f"IS={t[t.y<half].R.mean():+.2f} OOS={t[t.y>=half].R.mean():+.2f} grn={grn}/{t.y.nunique()} "
          f"| 22+:{rec.mean():+.2f}(n{len(rec)})")


for side in ("long", "short"):
    df = collect(side)
    ft = df[df.attack == 1]
    combo = ft[ft.Vth & ft.Wth]
    print(f"\n===== {side.upper()}  ($3 stop, confirmed entry, cost $0.40) =====")
    for fname, fdf in [("1st-touch", ft), ("1st+V+wick", combo)]:
        for rr in (0.5, 1.0, 1.5, 2.0):
            line(f"{fname} RR{rr}", run(side, fdf, rr, 0.40))
        print()
