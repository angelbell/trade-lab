"""Improve the funding-gated BTC short. Bottleneck = raw funding saturates at the +0.0001
cap (dynamic range dies past p50). Attack it with SHARPER greed transforms (rolling z-score,
momentum, trailing sum) that restore range even under the cap; also re-optimise the EXIT on
the gated population. Every candidate judged on meanR AND IS/OOS balance AND per-year AND the
random-drop null -- not in-sample meanR alone.  Flagship baseline = level>p50 (meanR+0.163).
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
fund = pd.read_csv("data/btc_funding.csv")
fund["time"] = pd.to_datetime(fund["time"], utc=True, format="ISO8601")
FRs = pd.merge_asof(pd.DataFrame({"t": D.index}), fund.sort_values("time").rename(columns={"time": "t"}),
                    on="t", direction="backward")["fundingRate"]
FR = FRs.values
# sharper greed transforms (all causal: only past/current funding)
Z500 = ((FRs - FRs.rolling(500).mean()) / FRs.rolling(500).std()).values      # relative greed
MOM24 = (FRs - FRs.shift(24)).values                                          # 1-day rising funding
MOM168 = (FRs - FRs.shift(168)).values                                        # 1-week rising funding
SUM500 = FRs.rolling(500).sum().values                                        # accumulated crowding


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


T2 = base_trades(2.0, 1.5)  # cache the RR2 stream for the greed-signal comparison


def evalg(tag, df):
    if len(df) < 30: print(f"  {tag:<30} n={len(df)} (too few)"); return None
    R = df.R.values; tm = pd.DatetimeIndex(df.time); yr = tm.year.values
    med = np.median(np.unique(yr)); IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yrs = max((tm.max() - tm.min()).days / 365.25, .5); cdd = cdd_R(R, yrs)[2]
    uy = np.unique(yr); g = sum(1 for y in uy if R[yr == y].sum() > 0)
    y20 = R[yr == 2020].sum() if 2020 in uy else 0; y23 = R[yr == 2023].sum() if 2023 in uy else 0
    print(f"  {tag:<30} n={len(R):>4} meanR={R.mean():+.3f} CAGR/DD={cdd:+.2f} "
          f"IS/OOS={IS:+.2f}/{OOS:+.2f} grn={g}/{len(uy)} '20={y20:+.0f} '23={y23:+.0f}")
    return df


def sig_at(arr): return arr[T2.ei.values]

print("=== BASELINE greed gate (level>p50) ===")
lvl = sig_at(FR); cov = ~np.isnan(lvl)
base = T2[cov & (lvl > np.nanpercentile(lvl, 50))]; evalg("level>p50", base)

print("\n=== SHARPER greed signals (restore range past the cap) ===")
for name, arr, ths in [("z500 (rel-greed)", Z500, (0.0, 0.5, 1.0)),
                       ("mom24 rising", MOM24, (0.0,)),
                       ("mom168 rising", MOM168, (0.0,)),
                       ("sum500 accumulated", SUM500, None)]:
    s = sig_at(arr); c = ~np.isnan(s)
    if ths is None: ths = (np.nanpercentile(s, 50),)
    for th in ths:
        evalg(f"{name} > {th:g}", T2[c & (s > th)])

print("\n=== combine: level>p50 AND z500>0 (sustained + relative greed) ===")
z = sig_at(Z500); evalg("lvl>p50 & z500>0", T2[cov & (lvl > np.nanpercentile(lvl, 50)) & (z > 0)])

print("\n=== EXIT re-optimisation on the greed-gated (level>p50) population ===")
lmask = (FR > np.nanpercentile(FR, 50))
for rr in (1.5, 2.0, 2.5, 3.0, 4.0):
    tt = base_trades(rr, 1.5); s = FR[tt.ei.values]; c = ~np.isnan(s)
    evalg(f"RR{rr} + level>p50", tt[c & (s > np.nanpercentile(FR, 50))])

# random-drop null helper on the best sharper candidate vs the baseline population
print("\n=== null check: does the BEST sharper gate beat random same-keep% (vs level>p50 pop)? ===")
z = sig_at(Z500); cand = T2[cov & (z > 0.5)]
pool = T2[cov & (lvl > np.nanpercentile(lvl, 50))]  # baseline greed pop as the honest reference
yrs = max((pd.DatetimeIndex(cand.time).max() - pd.DatetimeIndex(cand.time).min()).days / 365.25, .5)
obs = cdd_R(cand.R.values, yrs)[2]; k = len(cand); rng = np.random.default_rng(3)
nl = np.array([cdd_R(rng.choice(pool.R.values, min(k, len(pool)), replace=False), yrs)[2] for _ in range(3000)])
print(f"  z500>0.5  n={k}  CAGR/DD={obs:+.2f}  rand(from greed-pop) med={np.median(nl):+.2f}  %ile={(nl<obs).mean()*100:.0f}")
