"""Is FUNDING redundant once we have OI? Decisive directional nulls on the SAME window
(2021+, both covered): does funding add to OI (funding>p50 beats random-drop FROM the
OI-gated pool?) and does OI add to funding (OIz>0 beats random-drop FROM the funding pool)?
Plus the signal correlation (are they the same leverage read?) and the 2x2 factorial.
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
OI = align(oi["sum_open_interest"])
OIz = ((pd.Series(OI) - pd.Series(OI).rolling(500).mean()) / pd.Series(OI).rolling(500).std()).values
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
fr = FR[T.ei.values]; oiz = OIz[T.ei.values]
cov = (~np.isnan(fr)) & (~np.isnan(oiz))     # both-covered window
T, fr, oiz = T[cov].reset_index(drop=True), fr[cov], oiz[cov]
yrs = max((pd.DatetimeIndex(T.time).max() - pd.DatetimeIndex(T.time).min()).days / 365.25, .5)

Fg = fr > np.nanpercentile(fr, 50)   # funding gate
Og = oiz > 0                         # OI gate


def stats(m):
    R = T.R.values[m]; yr = pd.DatetimeIndex(T.time[m]).year.values
    med = np.median(np.unique(yr)); IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    return len(R), R.mean(), cdd_R(R, yrs)[2], IS, OOS


def show(tag, m):
    n, mr, cdd, IS, OOS = stats(m)
    print(f"  {tag:<22} n={n:>4} meanR={mr:+.3f} CAGR/DD={cdd:+.2f} IS/OOS={IS:+.2f}/{OOS:+.2f}")


def add_null(tag, sub_mask, pool_mask, reps=4000):
    """does sub (a filter applied WITHIN pool) beat random same-N drop FROM pool?"""
    pool = T.R.values[pool_mask]; k = int(sub_mask.sum())
    obs = cdd_R(T.R.values[sub_mask], yrs)[2]; rng = np.random.default_rng(11)
    nl = np.array([cdd_R(rng.choice(pool, min(k, len(pool)), replace=False), yrs)[2] for _ in range(reps)])
    print(f"  {tag:<40} obs CAGR/DD={obs:+.2f}  rand-med={np.median(nl):+.2f}  %ile={(nl<obs).mean()*100:.0f}")


print(f"both-covered window: n={len(T)}  {pd.DatetimeIndex(T.time).year.min()}-{pd.DatetimeIndex(T.time).year.max()}")
print(f"\nsignal correlation  corr(funding, OIz) = {np.corrcoef(fr, oiz)[0,1]:+.2f}")
print("\n=== 2x2 factorial ===")
show("ungated", np.ones(len(T), bool))
show("funding>p50 (F)", Fg)
show("OIz>0 (O)", Og)
show("F & O", Fg & Og)

print("\n=== DECISIVE directional nulls ===")
add_null("does FUNDING add to OI?  (F within O-pool)", Fg & Og, Og)
add_null("does OI add to FUNDING?  (O within F-pool)", Fg & Og, Fg)
print("  (>90 = the added gate is real selection, not redundant n-trimming)")
