"""15m gold 200MA: pure BOUNCE-RATE measurement (no RR/exit).
bounce = from the touch low, price reaches low+UP*ATR before low-DOWN*ATR.
Stratify by touch-count / V-approach / wick-touch vs a random-uptrend baseline."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

UP, DOWN, FWD = 1.5, 0.5, 200


def barrier(h, l, s, lowref, up, dn, fwd):
    hi = lowref + up; lo = lowref - dn
    for j in range(s + 1, min(s + 1 + fwd, len(h))):
        if l[j] <= lo: return 0
        if h[j] >= hi: return 1
    return 0  # timeout = no bounce (conservative)


d = load_mt5_csv("data/vantage_xauusd_m15.csv")
sma = d["close"].rolling(200).mean().values
a = ta.atr(d["high"], d["low"], d["close"], length=14).values
o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
yr = d.index.year.values
slopeK, zoneW, swingW = 20, 0.5, 30

rows, cnt = [], 0
for s in range(max(slopeK, swingW, 200) + 1, len(c) - 1):
    if np.isnan(sma[s]) or np.isnan(a[s]) or a[s] <= 0:
        continue
    z = zoneW * a[s]
    if c[s] > sma[s] + 1.5 * a[s]:
        cnt = 0
    if not (sma[s] > sma[s - slopeK] and c[s] > sma[s]):
        continue
    if not (l[s] <= sma[s] + z):           # low entered the zone = a touch
        continue
    cnt += 1
    rng = max(h[s] - l[s], 1e-9)
    win = slice(s - swingW + 1, s + 1)
    swing_hi = h[win].max(); bars_down = swingW - 1 - int(np.argmax(h[win]))
    vel = ((swing_hi - l[s]) / a[s]) / max(bars_down, 1)        # V-ness of the approach
    lwick = (min(o[s], c[s]) - l[s]) / rng                      # lower-wick ratio
    b = barrier(h, l, s, l[s], UP * a[s], DOWN * a[s], FWD)
    rows.append(dict(y=yr[s], attack=cnt, vel=vel, lwick=lwick, bounce=b))

t = pd.DataFrame(rows)

# baseline: random uptrend bars (200SMA rising & price above), same barrier from their low
ctx = np.where((sma[:-1] > np.roll(sma, slopeK)[:-1]) & (c[:-1] > sma[:-1]))[0]
ctx = ctx[(ctx > 200) & (ctx < len(c) - FWD - 1)]
rng_ = np.random.default_rng(0); samp = rng_.choice(ctx, 4000, replace=False)
base = np.mean([barrier(h, l, s, l[s], UP * a[s], DOWN * a[s], FWD) for s in samp])

print(f"GOLD 15m 200MA touch  bounce = low+{UP}ATR before low-{DOWN}ATR (fwd{FWD})")
print(f"  touches n={len(t)}   OVERALL bounce rate = {t.bounce.mean()*100:.1f}%")
print(f"  BASELINE (random uptrend bar)     = {base*100:.1f}%   <- beat this to have signal\n")

def show(name, grp):
    print(f"  by {name}:")
    for k, g in grp:
        print(f"    {str(k):<10} n={len(g):>5}  bounce={g.bounce.mean()*100:>5.1f}%")

# touch count
t["atk"] = np.where(t.attack == 1, "1", np.where(t.attack == 2, "2", "3+"))
show("touch-count", t.groupby("atk"))
# V-approach terciles (low vel = gradual, high vel = steep V)
t["vbin"] = pd.qcut(t.vel, 3, labels=["gradual", "mid", "steepV"])
show("approach (V-ness)", t.groupby("vbin"))
# wick terciles
t["wbin"] = pd.qcut(t.lwick, 3, labels=["body", "mid", "longwick"])
show("wick-touch", t.groupby("wbin"))
# combo: first-touch + gradual + longwick
best = t[(t.atk == "1") & (t.vbin == "gradual") & (t.wbin == "longwick")]
print(f"\n  COMBO 1st+gradual+longwick: n={len(best)} bounce={best.bounce.mean()*100:.1f}% (vs base {base*100:.1f}%)")
# per-year of that combo
print("  combo per-year bounce%:", " ".join(f"{int(y)}:{g.bounce.mean()*100:.0f}%(n{len(g)})"
      for y, g in best.groupby("y") if len(g) >= 5))
