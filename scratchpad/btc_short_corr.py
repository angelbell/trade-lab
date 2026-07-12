"""Does the funding-gated BTC short DIVERSIFY the book? Annual & monthly-R correlation vs
each leg (esp. the BTC legs = redundancy risk) + the portfolio CAGR/DD with the short added.
A short only earns its place if it is LOW/NEGATIVELY correlated (cuts DD); if it tracks the
BTC longs it is just redundant. Correlations on the short's active window (2019-2026).
"""
import sys
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag
from research.portfolio_kama import get_legs, cagr_dd

# ---- build the funding-gated short leg (1h / zzk1.5 / RR2 / fund>p50) ----
D = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "1h")
H = D["high"].values.astype(float); L = D["low"].values.astype(float); C = D["close"].values.astype(float)
A = ta.atr(D["high"], D["low"], D["close"], 14).values
EMA = ta.ema(D["close"], 80).values
fund = pd.read_csv("data/btc_funding.csv")
fund["time"] = pd.to_datetime(fund["time"], utc=True, format="ISO8601")
FR = pd.merge_asof(pd.DataFrame({"t": D.index}), fund.sort_values("time").rename(columns={"time": "t"}),
                   on="t", direction="backward")["fundingRate"].values


def short_leg(rr=2.0, zzk=1.5, fwd=500, cost=0.0005, fq=0.5):
    sw = swings_zigzag(H, L, A, zzk); n = len(C); rows = []
    for t in range(2, len(sw)):
        cH2, _, pH2, kH2 = sw[t]; cL1, _, pL1, kL1 = sw[t - 1]; _, _, pH0, kH0 = sw[t - 2]
        if not (kH2 == +1 and kL1 == -1 and kH0 == +1): continue
        if pH2 >= pH0 or pH0 - pL1 <= 0: continue
        if not (not np.isnan(EMA[cL1]) and pL1 < EMA[cL1]): continue
        e_i = None
        for j in range(cH2 + 1, n):
            if C[j] < pL1: e_i = j; break
        if e_i is None or np.isnan(FR[e_i]): continue
        e = C[e_i]; stop = pH2; risk = stop - e
        if risk <= 0: continue
        tgt = e - rr * risk; r = None
        for j in range(e_i + 1, min(e_i + 1 + fwd, n)):
            if H[j] >= stop: r = -1.0; break
            if L[j] <= tgt: r = rr; break
        if r is None: r = (e - C[min(e_i + fwd, n - 1)]) / risk
        r -= cost / risk * e
        rows.append((D.index[e_i], r, FR[e_i]))
    df = pd.DataFrame(rows, columns=["time", "R", "fund"])
    thr = np.nanpercentile(df.fund, fq * 100)
    return df[df.fund > thr][["time", "R"]].reset_index(drop=True)


legs = get_legs()
gold, btc_k, pb = legs["gold_bo"], legs["btc_bo_kama"], legs["btc_pull"]
sh = short_leg()

def ann(t): return t.assign(y=t.time.dt.year).groupby("y").R.sum()
def mon(t): return t.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()

# align on the short's active window
Y0, Y1 = sh.time.dt.year.min(), sh.time.dt.year.max()
def corr(series_fn, a, b):
    A_ = series_fn(a); B_ = series_fn(b)
    df = pd.concat([A_, B_], axis=1).fillna(0.0)
    if series_fn is ann:
        df = df[(df.index >= Y0) & (df.index <= Y1)]
    df.columns = ["x", "y"]
    return df.x.corr(df.y)

print(f"funding-gated SHORT leg: n={len(sh)}  window {Y0}-{Y1}  meanR={sh.R.mean():+.3f}")
print(f"  standalone CAGR/DD={cagr_dd(sh)[2]:+.2f}\n")

print("=== correlation of SHORT vs each book leg ===")
print(f"  {'leg':<16}{'annual-R':>10}{'monthly-R':>11}")
for nm, lg in [("gold_bo", gold), ("btc_bo_kama", btc_k), ("btc_pull", pb),
               ("BTC longs (k+pull)", pd.concat([btc_k, pb])),
               ("whole book (3)", pd.concat([gold, btc_k, pb]))]:
    print(f"  {nm:<16}{corr(ann, sh, lg):>+10.2f}{corr(mon, sh, lg):>+11.2f}")

print("\n=== portfolio CAGR/DD: does adding the short help? (1%/leg unless noted) ===")
def L(name, t): c, dd, cdd, ret = cagr_dd(t); print(f"  {name:<30} n={len(t):>4}  CAGR/DD={cdd:+.2f}  maxDD={dd:4.1f}%")
book = pd.concat([gold, btc_k, pb])
L("book 3-leg", book)
L("book + short@1.0%", pd.concat([book, sh]))
L("book + short@0.5%", pd.concat([book, sh.assign(R=sh.R * 0.5)]))
print("  --- BTC-family only (where redundancy would bite) ---")
L("btc_k + btc_pull", pd.concat([btc_k, pb]))
L("btc_k + btc_pull + short@1%", pd.concat([btc_k, pb, sh]))

print("\n=== risk-balanced allocation sweep (does ANY dose beat book 2.63?) ===")
for w in (0.1, 0.15, 0.25, 0.35):
    L(f"book + short@{w}%", pd.concat([book, sh.assign(R=sh.R * w)]))
# also: short as a targeted HEDGE for the btc breakout leg only (-0.60 corr)
print("  --- targeted: btc_bo_kama + short (hedge the -0.60-corr leg) ---")
L("btc_bo_kama alone", btc_k)
for w in (0.25, 0.5):
    L(f"btc_bo_kama + short@{w}%", pd.concat([btc_k, sh.assign(R=sh.R * w)]))
