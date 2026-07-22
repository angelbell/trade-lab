"""Test the REAL Fear&Greed index (alternative.me, 0-100, uncapped) as the greed gate for the
BTC short, vs the saturated funding gate. Short into GREED = F&G high. F&G is causal: use the
prior day's value (known only at end of day) -> shift +1 day, ffill to 1h bars. Judge on
CAGR/DD + IS/OOS + per-year + random-drop null (vs the ungated pool) -- not meanR alone.
Bonus: F&G covers 2018+ (funding only 2019-09+), so the 2018 gap closes.
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

# --- F&G, causal: day D value available from D+1 00:00 ---
fng = pd.read_csv("data/btc_fng.csv")
fng["t"] = pd.to_datetime(fng["date"], utc=True) + pd.Timedelta(days=1)
FNG = pd.merge_asof(pd.DataFrame({"t": D.index}), fng[["t", "fng"]].sort_values("t"),
                    on="t", direction="backward")["fng"].values
# --- funding (baseline gate), causal ---
fund = pd.read_csv("data/btc_funding.csv"); fund["time"] = pd.to_datetime(fund["time"], utc=True, format="ISO8601")
FR = pd.merge_asof(pd.DataFrame({"t": D.index}), fund.sort_values("time").rename(columns={"time": "t"}),
                   on="t", direction="backward")["fundingRate"].values


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
FNGv = FNG[T.ei.values].astype(float); FRv = FR[T.ei.values]


def ev(tag, mask, pool_mask=None):
    df = T[mask]
    if len(df) < 30: print(f"  {tag:<24} n={len(df)} (too few)"); return
    R = df.R.values; tm = pd.DatetimeIndex(df.time); yr = tm.year.values
    med = np.median(np.unique(yr)); IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yrs = max((tm.max() - tm.min()).days / 365.25, .5); cdd = cdd_R(R, yrs)[2]
    uy = np.unique(yr); g = sum(1 for y in uy if R[yr == y].sum() > 0)
    y20 = R[yr == 2020].sum() if 2020 in uy else 0; y23 = R[yr == 2023].sum() if 2023 in uy else 0
    # random-drop null: keep same N at random from the reference pool
    pm = pool_mask if pool_mask is not None else np.ones(len(T), bool)
    pool = T[pm].R.values; k = len(df); rng = np.random.default_rng(5)
    nl = np.array([cdd_R(rng.choice(pool, min(k, len(pool)), replace=False), yrs)[2] for _ in range(2000)])
    pct = (nl < cdd).mean() * 100
    print(f"  {tag:<24} n={len(R):>4} meanR={R.mean():+.3f} CAGR/DD={cdd:+.2f} "
          f"IS/OOS={IS:+.2f}/{OOS:+.2f} grn={g}/{len(uy)} '20={y20:+.0f} '23={y23:+.0f}  null%ile={pct:.0f}")


cov = ~np.isnan(FNGv)
print(f"F&G coverage of trades: {cov.mean():.0%}   (funding cov: {np.mean(~np.isnan(FRv)):.0%})")
print("\n=== ungated & funding baseline (reference) ===")
ev("ungated (F&G-covered)", cov)
ev("funding>p50 (baseline)", (~np.isnan(FRv)) & (FRv > np.nanpercentile(FRv, 50)))

print("\n=== F&G gate: short only when F&G >= threshold (into greed) ===  [null vs F&G-covered pool]")
for th in (45, 50, 55, 60, 65, 70):
    ev(f"F&G>={th}", cov & (FNGv >= th), pool_mask=cov)

print("\n=== F&G percentile gate (relative greed) ===")
for q in (50, 60, 70):
    th = np.nanpercentile(FNGv, q); ev(f"F&G>p{q} ({th:.0f})", cov & (FNGv > th), pool_mask=cov)

print("\n=== combine: F&G>=55 AND funding>p50 ===")
fmask = (~np.isnan(FRv)) & (FRv > np.nanpercentile(FRv, 50))
ev("F&G>=55 & fund>p50", cov & (FNGv >= 55) & fmask, pool_mask=cov & fmask)
