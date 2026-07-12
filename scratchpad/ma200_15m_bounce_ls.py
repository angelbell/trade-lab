"""15m gold 200MA bounce-RATE, LONG and SHORT (mirror). Classification only (no exit).
LONG : uptrend (200SMA rising, price above), low touches MA zone; bounce = price reaches
       low+UP*ATR before low-DOWN*ATR.
SHORT: downtrend (200SMA falling, price below), high touches MA zone; bounce(=drop) = price
       reaches high-UP*ATR before high+DOWN*ATR.
Stratify by touch-count / V-approach / wick, vs a same-direction trend baseline."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

UP, DOWN, FWD = 1.5, 0.5, 200
d = load_mt5_csv("data/vantage_xauusd_m15.csv")
sma = d["close"].rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], length=14).values
o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
yr = d.index.year.values
slopeK, zoneW, swingW = 20, 0.5, 30


def barr_long(s, ref):
    hi = ref + UP * a[s]; lo = ref - DOWN * a[s]
    for j in range(s + 1, min(s + 1 + FWD, len(h))):
        if l[j] <= lo: return 0
        if h[j] >= hi: return 1
    return 0


def barr_short(s, ref):
    lo = ref - UP * a[s]; hi = ref + DOWN * a[s]
    for j in range(s + 1, min(s + 1 + FWD, len(h))):
        if h[j] >= hi: return 0
        if l[j] <= lo: return 1
    return 0


def collect(side):
    rows, cnt = [], 0
    for s in range(221, len(c) - 1):
        if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        z = zoneW * a[s]; rng = max(h[s] - l[s], 1e-9); win = slice(s - swingW + 1, s + 1)
        if side == "long":
            if c[s] > sma[s] + 1.5 * a[s]: cnt = 0
            if not (sma[s] > sma[s - slopeK] and c[s] > sma[s] and l[s] <= sma[s] + z):
                continue
            cnt += 1
            ext = h[win].max(); bd = swingW - 1 - int(np.argmax(h[win])); vel = ((ext - l[s]) / a[s]) / max(bd, 1)
            wick = (min(o[s], c[s]) - l[s]) / rng
            b = barr_long(s, l[s])
        else:
            if c[s] < sma[s] - 1.5 * a[s]: cnt = 0
            if not (sma[s] < sma[s - slopeK] and c[s] < sma[s] and h[s] >= sma[s] - z):
                continue
            cnt += 1
            ext = l[win].min(); bd = swingW - 1 - int(np.argmin(l[win])); vel = ((h[s] - ext) / a[s]) / max(bd, 1)
            wick = (h[s] - max(o[s], c[s])) / rng
            b = barr_short(s, h[s])
        rows.append(dict(y=yr[s], attack=cnt, vel=vel, wick=wick, b=b))
    return pd.DataFrame(rows)


def baseline(side):
    if side == "long":
        ctx = np.where((sma[:-1] > np.roll(sma, slopeK)[:-1]) & (c[:-1] > sma[:-1]))[0]
    else:
        ctx = np.where((sma[:-1] < np.roll(sma, slopeK)[:-1]) & (c[:-1] < sma[:-1]))[0]
    ctx = ctx[(ctx > 200) & (ctx < len(c) - FWD - 1)]
    samp = np.random.default_rng(0).choice(ctx, min(3000, len(ctx)), replace=False)
    f = barr_long if side == "long" else barr_short
    ref = l if side == "long" else h
    return np.mean([f(s, ref[s]) for s in samp]) * 100


for side in ("long", "short"):
    t = collect(side); base = baseline(side)
    print(f"\n===== {side.upper()}  (touches n={len(t)}) =====")
    print(f"  overall bounce={t.b.mean()*100:.1f}%   baseline(trend)={base:.1f}%")
    t["atk"] = np.where(t.attack == 1, "1", np.where(t.attack == 2, "2", "3+"))
    t["vb"] = pd.qcut(t.vel, 3, labels=["grad", "mid", "V"])
    t["wb"] = pd.qcut(t.wick, 3, labels=["body", "mid", "wick"])
    print("  touch#:   " + "  ".join(f"{k}:{g.b.mean()*100:.0f}%(n{len(g)})" for k, g in t.groupby("atk")))
    print("  V-ness:   " + "  ".join(f"{k}:{g.b.mean()*100:.0f}%" for k, g in t.groupby("vb")))
    print("  wick:     " + "  ".join(f"{k}:{g.b.mean()*100:.0f}%" for k, g in t.groupby("wb")))
    combo = t[(t.atk == "1") & (t.vb == "V") & (t.wb == "wick")]
    print(f"  COMBO 1st+V+wick: n={len(combo)} bounce={combo.b.mean()*100:.1f}% (base {base:.1f}%)")
    print("    per-year: " + " ".join(f"{int(y)}:{g.b.mean()*100:.0f}%" for y, g in combo.groupby("y") if len(g) >= 5))
