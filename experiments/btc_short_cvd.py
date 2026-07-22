"""CHEAP SCREEN: does order-flow (CVD) carry any edge for the BTC short beyond the leverage
family (funding/OI)? Proxy CVD from the taker buy/sell RATIO already on hand (2021+, hourly):
imbalance i = (r-1)/(r+1) in (-1,1); recent net flow = rolling_sum(i, K). Test as a gate
(momentum: short when flow sell-heavy) and as a bearish DIVERGENCE (price up but flow down).
Judge: null vs covered pool + directional null vs the funding pool (does CVD ADD to funding?).
CAVEAT: ratio proxy is NOT volume-weighted -> if it screens dead, fetch real klines before
concluding. Compare against funding>p50 on the same window.
"""
import sys
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag
from research.overfit_audit import cdd_R

D = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "1h")
H = D["high"].values.astype(float); L = D["low"].values.astype(float); C = D["close"].values.astype(float)
A = ta.atr(D["high"], D["low"], D["close"], 14).values
EMA = ta.ema(D["close"], 80).values
oi = pd.read_csv("data/btc_oi.csv", parse_dates=["create_time"]).set_index("create_time").sort_index()
def align(s): return pd.merge_asof(pd.DataFrame({"t": D.index}), s.rename("v").reset_index().rename(columns={"create_time": "t"}), on="t", direction="backward")["v"].values
TAK = align(oi["sum_taker_long_short_vol_ratio"])
IMB = pd.Series((TAK - 1) / (TAK + 1))                       # net taker imbalance in (-1,1)
FLOW = {K: IMB.rolling(K).sum().values for K in (24, 72, 168)}   # recent net aggressive flow
fund = pd.read_csv("data/btc_funding.csv"); fund["time"] = pd.to_datetime(fund["time"], utc=True, format="ISO8601")
FR = pd.merge_asof(pd.DataFrame({"t": D.index}), fund.sort_values("time").rename(columns={"time": "t"}), on="t", direction="backward")["fundingRate"].values


def base_trades(rr=2.0, zzk=1.5, fwd=500, cost=0.0005):
    sw = swings_zigzag(H, L, A, zzk); n = len(C); rows = []
    for t in range(2, len(sw)):
        cH2, _, pH2, kH2 = sw[t]; cL1, _, pL1, kL1 = sw[t - 1]; _, _, pH0, kH0 = sw[t - 2]
        if not (kH2 == +1 and kL1 == -1 and kH0 == +1): continue
        if pH2 >= pH0 or pH0 - pL1 <= 0: continue
        if not (not np.isnan(EMA[cL1]) and pL1 < EMA[cL1]): continue
        e_i = None
        for j in range(cH2 + 1, n):
            if C[j] < pL1: e_i = j; break
        if e_i is None: continue
        e = C[e_i]; stop = pH2; risk = stop - e
        if risk <= 0: continue
        tgt = e - rr * risk; r = None
        for j in range(e_i + 1, min(e_i + 1 + fwd, n)):
            if H[j] >= stop: r = -1.0; break
            if L[j] <= tgt: r = rr; break
        if r is None: r = (e - C[min(e_i + fwd, n - 1)]) / risk
        r -= cost / risk * e
        rows.append((D.index[e_i], r, e_i))
    return pd.DataFrame(rows, columns=["time", "R", "ei"])


T = base_trades()
ei = T.ei.values
cov = ~np.isnan(FLOW[72][ei])
yrs = max((pd.DatetimeIndex(T.time).max() - pd.DatetimeIndex(T.time).min()).days / 365.25, .5)
FRc = FR[ei]; Fg = (~np.isnan(FRc)) & (FRc > np.nanpercentile(FRc[cov & ~np.isnan(FRc)], 50))


def ev(tag, mask, pool):
    df = T[mask]
    if len(df) < 25: print(f"  {tag:<28} n={len(df)} (too few)"); return
    R = df.R.values; yr = pd.DatetimeIndex(df.time).year.values
    med = np.median(np.unique(yr)); IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yy = max((pd.DatetimeIndex(df.time).max() - pd.DatetimeIndex(df.time).min()).days / 365.25, .5)
    cdd = cdd_R(R, yy)[2]; poolR = T[pool].R.values; k = len(df); rng = np.random.default_rng(7)
    nl = np.array([cdd_R(rng.choice(poolR, min(k, len(poolR)), replace=False), yy)[2] for _ in range(2000)])
    g = sum(1 for y in np.unique(yr) if R[yr == y].sum() > 0)
    print(f"  {tag:<28} n={len(R):>4} meanR={R.mean():+.3f} CAGR/DD={cdd:+.2f} IS/OOS={IS:+.2f}/{OOS:+.2f} grn={g}/{len(np.unique(yr))} null%ile={(nl<cdd).mean()*100:.0f}")


print(f"CVD-proxy covered trades: {cov.mean():.0%}  window {pd.DatetimeIndex(T[cov].time).year.min()}+")
ev("ungated (CVD-cov)", cov, cov)
ev("funding>p50 (ref)", Fg & cov, cov)

print("\n=== momentum gate: recent net flow SELL-heavy (flowK < 0) ===")
for K in (24, 72, 168):
    f = FLOW[K][ei]
    ev(f"flow{K} < 0 (selling)", cov & (f < 0), cov)
    ev(f"flow{K} > 0 (buying)", cov & (f > 0), cov)

print("\n=== bearish DIVERGENCE: price rose over M bars but flow was net SELL ===")
for M in (24, 72, 168):
    f = FLOW[M][ei]
    pr_up = np.array([C[i] > C[max(i - M, 0)] for i in ei])
    ev(f"div M{M}: price up & flow<0", cov & pr_up & (f < 0), cov)

print("\n=== does CVD add to funding? (best flow gate WITHIN funding pool, null vs fund-pool) ===")
f72 = FLOW[72][ei]
ev("flow72<0 & funding>p50", cov & Fg & (f72 < 0), cov & Fg)
