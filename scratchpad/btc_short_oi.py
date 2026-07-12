"""Test Open-Interest / long-short-ratio (leverage-family, uncapped) as the greed gate for the
BTC short, vs the funding gate on the SAME 2021+ window (OI starts 2021-01). Mechanism =
crowded leverage -> liquidation flush -> short pays. Signals: OI level (z), OI value (z),
OI momentum, toptrader long/short ratio, taker buy/sell ratio. Judge on CAGR/DD + IS/OOS +
per-year + random-drop null (vs covered pool) -- not meanR alone. Compare head-to-head to
funding>p50 restricted to the same window (fair overlap).
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
def align(series):
    return pd.merge_asof(pd.DataFrame({"t": D.index}), series.rename("v").reset_index().rename(columns={"create_time": "t"}),
                         on="t", direction="backward")["v"].values
OI = align(oi["sum_open_interest"]); OIV = align(oi["sum_open_interest_value"])
TLS = align(oi["sum_toptrader_long_short_ratio"]); TAK = align(oi["sum_taker_long_short_vol_ratio"])
def zroll(a, n=500):
    s = pd.Series(a); return ((s - s.rolling(n).mean()) / s.rolling(n).std()).values
OIz = zroll(OI); OIVz = zroll(OIV); TLSz = zroll(TLS, 250)
OImom = (pd.Series(OI) - pd.Series(OI).shift(72)).values  # 3-day OI change

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
S = {k: v[T.ei.values] for k, v in dict(OIz=OIz, OIVz=OIVz, TLS=TLS, TLSz=TLSz, TAK=TAK, OImom=OImom, FR=FR).items()}
covOI = ~np.isnan(S["OIz"])   # OI-covered window (2021+)


def ev(tag, mask, pool_mask):
    df = T[mask]
    if len(df) < 25: print(f"  {tag:<26} n={len(df)} (too few)"); return
    R = df.R.values; tm = pd.DatetimeIndex(df.time); yr = tm.year.values
    med = np.median(np.unique(yr)); IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yrs = max((tm.max() - tm.min()).days / 365.25, .5); cdd = cdd_R(R, yrs)[2]
    uy = np.unique(yr); g = sum(1 for y in uy if R[yr == y].sum() > 0)
    y23 = R[yr == 2023].sum() if 2023 in uy else 0
    pool = T[pool_mask].R.values; k = len(df); rng = np.random.default_rng(5)
    nl = np.array([cdd_R(rng.choice(pool, min(k, len(pool)), replace=False), yrs)[2] for _ in range(2000)])
    print(f"  {tag:<26} n={len(R):>4} meanR={R.mean():+.3f} CAGR/DD={cdd:+.2f} "
          f"IS/OOS={IS:+.2f}/{OOS:+.2f} grn={g}/{len(uy)} '23={y23:+.0f} null%ile={(nl<cdd).mean()*100:.0f}")


print(f"OI-covered trades: {covOI.mean():.0%}  window {pd.DatetimeIndex(T[covOI].time).year.min()}+")
print("\n=== reference on the SAME 2021+ window ===")
ev("ungated (OI-cov)", covOI, covOI)
frc = covOI & ~np.isnan(S["FR"])
ev("funding>p50 (2021+)", frc & (S["FR"] > np.nanpercentile(S["FR"][frc], 50)), covOI)

print("\n=== OI-family leverage gates (short into crowded leverage) ===")
for nm, arr, ths in [("OI level z>", "OIz", (0.0, 0.5, 1.0)),
                     ("OI value z>", "OIVz", (0.0, 0.5, 1.0)),
                     ("toptrader L/S >", "TLS", (1.0, 1.1, 1.2)),
                     ("toptrader L/S z>", "TLSz", (0.0, 0.5)),
                     ("taker buy/sell >", "TAK", (1.0, 1.05)),
                     ("OI 3d-momentum >", "OImom", (0.0,))]:
    s = S[arr]; c = covOI & ~np.isnan(s)
    for th in ths:
        ev(f"{nm}{th:g}", c & (s > th), covOI)

print("\n=== combine OI-crowding with funding>p50 ===")
ev("OIz>0.5 & fund>p50", frc & (S["OIz"] > 0.5) & (S["FR"] > np.nanpercentile(S["FR"][frc], 50)), frc)
