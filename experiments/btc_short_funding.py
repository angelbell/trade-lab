"""Can a FUNDING-RATE gate suppress the BTC-short's bull-year bleed (2020, 2023) WITHOUT
overfitting to those 2 years? Mechanism: high positive funding = crowded leveraged longs =
euphoric melt-up = the exact condition the short bleeds into. Gate = 'do not short when
funding is high-positive'. Honest test: does it keep the GOOD years' shorts while cutting
2020/2023, AND beat a random same-keep% drop (not just n-trimming)?
Flagship short = 1h / zz-k1.5 / RR2 (the plateau center found on the time-axis search).
Caveat: funding data starts 2019-09 -> 2018 & early-2019 trades drop from the gated view.
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

# funding, causal: last published rate at or before the bar
fund = pd.read_csv("data/btc_funding.csv")
fund["time"] = pd.to_datetime(fund["time"], utc=True, format="ISO8601")
fund = fund.sort_values("time")
bars = pd.DataFrame({"t": D.index})
merged = pd.merge_asof(bars, fund.rename(columns={"time": "t"}), on="t", direction="backward")
FR = merged["fundingRate"].values  # aligned to D.index; NaN before 2019-09


def trades(rr=2.0, zzk=1.5, fwd=500, cost=0.0005):
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
        rows.append((D.index[e_i], r, FR[e_i]))
    return pd.DataFrame(rows, columns=["time", "R", "fund"])


def summ(tag, df):
    R = df.R.values; tm = pd.DatetimeIndex(df.time)
    yr = tm.year.values; med = np.median(np.unique(yr))
    IS = R[yr < med].mean(); OOS = R[yr >= med].mean()
    yrs = max((tm.max() - tm.min()).days / 365.25, .5); cdd = cdd_R(R, yrs)[2]
    uy = np.unique(yr); g = sum(1 for y in uy if R[yr == y].sum() > 0)
    print(f"  {tag:<26} n={len(R):>4} meanR={R.mean():+.3f} CAGR/DD={cdd:+.2f} "
          f"IS/OOS={IS:+.2f}/{OOS:+.2f} grn={g}/{len(uy)}")
    return R, tm


def peryear(df):
    R = df.R.values; yr = pd.DatetimeIndex(df.time).year.values
    return " ".join(f"{y}:{R[yr==y].sum():+.0f}" for y in np.unique(yr))


t = trades()
print("funding coverage: {:.0%} of trades have funding".format(np.mean(~np.isnan(t.fund))))
print(f"funding distribution (all): p10/50/90 = "
      f"{np.nanpercentile(t.fund,10):.5f} / {np.nanpercentile(t.fund,50):.5f} / {np.nanpercentile(t.fund,90):.5f}")

print("\n[BASE] ungated (full 2018+):")
summ("ungated ALL", t); print("     per-year:", peryear(t))

# restrict to funding-covered trades for a fair gated comparison
tc = t[~np.isnan(t.fund)].copy()
print("\n[BASE] ungated, funding-covered only (2019-09+):")
summ("ungated (covered)", tc); print("     per-year:", peryear(tc))

print("\n[GATE] short only when funding < threshold (avoid crowded-long euphoria):")
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    thr = np.nanpercentile(tc.fund, q * 100)
    g = tc[tc.fund < thr]
    Rg, _ = summ(f"fund<p{int(q*100)} ({thr:+.5f})", g)
    print("     per-year:", peryear(g))

# random-drop null on the p70 gate (does the SELECTION beat keeping the same count at random?)
print("\n[NULL] random same-keep% vs the fund<p70 gate (CAGR/DD %ile; >90 = real selection):")
thr70 = np.nanpercentile(tc.fund, 70)
gate70 = tc[tc.fund < thr70]
keep = len(gate70)
R_all = tc.R.values
yrs = max((pd.DatetimeIndex(tc.time).max() - pd.DatetimeIndex(tc.time).min()).days / 365.25, .5)
obs = cdd_R(gate70.R.values, yrs)[2]
rng = np.random.default_rng(1)
null = np.array([cdd_R(rng.choice(R_all, keep, replace=False), yrs)[2] for _ in range(2000)])
pct = (null < obs).mean() * 100
print(f"  gate CAGR/DD={obs:+.2f}  random-keep median={np.median(null):+.2f}  gate %ile={pct:.0f}")

# opposite funding direction (short only in HIGH funding = fade crowded longs)
print("\n[GATE-OPP] short only when funding > threshold (fade crowded longs):")
for q in (0.3, 0.5, 0.7):
    thr = np.nanpercentile(tc.fund, q * 100); g = tc[tc.fund > thr]
    summ(f"fund>p{int(q*100)} ({thr:+.5f})", g); print("     per-year:", peryear(g))

# weekly cycle-phase gate: do NOT short when price is stretched ABOVE the 30-week SMA
print("\n[GATE] weekly cycle-phase: short only when close < (1+x)*30wSMA (avoid strong bull):")
wma = D["close"].resample("W").last().rolling(30).mean().shift(1).reindex(D.index, method="ffill")
WR = (D["close"].values / wma.values)  # close / 30wSMA at each bar
tw = trades().copy()
tw["wr"] = WR[[D.index.get_loc(x) for x in tw.time]]
tw = tw[~np.isnan(tw.wr)]
summ("ungated (wma-covered)", tw); print("     per-year:", peryear(tw))
yrsw = max((pd.DatetimeIndex(tw.time).max() - pd.DatetimeIndex(tw.time).min()).days / 365.25, .5)
for x in (-0.1, 0.0, 0.1, 0.2, 0.4):
    g = tw[tw.wr < 1 + x]
    if len(g) < 20:
        print(f"  close<{1+x:.2f}*wma  n={len(g)} (too few)"); continue
    Rg, _ = summ(f"close<{1+x:.2f}*wma", g); print("     per-year:", peryear(g))
    obsx = cdd_R(g.R.values, yrsw)[2]; k = len(g)
    nl = np.array([cdd_R(rng.choice(tw.R.values, k, replace=False), yrsw)[2] for _ in range(1500)])
    print(f"       random-drop null: gate CAGR/DD={obsx:+.2f}  %ile={ (nl<obsx).mean()*100:.0f}")

# DECISIVE: random-drop null on the correct-direction funding gate + plateau
print("\n[NULL] fund>threshold gate vs random same-keep% (CAGR/DD %ile; >90 = real selection):")
for q in (0.4, 0.5, 0.6):
    thr = np.nanpercentile(tc.fund, q * 100); g = tc[tc.fund > thr]
    k = len(g); obsx = cdd_R(g.R.values, yrs)[2]
    nl = np.array([cdd_R(rng.choice(tc.R.values, k, replace=False), yrs)[2] for _ in range(3000)])
    print(f"  fund>p{int(q*100)}  n={k}  meanR={g.R.mean():+.3f}  gate CAGR/DD={obsx:+.2f}  "
          f"rand-med={np.median(nl):+.2f}  %ile={(nl<obsx).mean()*100:.0f}")
