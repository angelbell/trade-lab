"""Overfit audit on the BTC-short (1h / zz-k1.5 / RR2 / funding>p50 gate).
DSR (trial-count haircut) / PBO-CSCV (is the config-winner noise?) / bootstrap-CI + null.
PBO grid = the configs we actually touched for THIS leg: zz-k x RR x funding-threshold.
HONEST CAVEAT: the grid captures within-leg selection; the earlier TF search (5m/15m/1h/4h)
and the whole exploration path are NOT in V, so DSR here is an UPPER bound on true DSR
(under-counts trials). Live-forward stays the arbiter.
"""
import sys, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from breakout_wave import resample, swings_zigzag
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

D = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "1h")
H = D["high"].values.astype(float); L = D["low"].values.astype(float); C = D["close"].values.astype(float)
A = ta.atr(D["high"], D["low"], D["close"], 14).values
EMA = ta.ema(D["close"], 80).values
fund = pd.read_csv("data/btc_funding.csv")
fund["time"] = pd.to_datetime(fund["time"], utc=True, format="ISO8601")
fund = fund.sort_values("time")
FR = pd.merge_asof(pd.DataFrame({"t": D.index}), fund.rename(columns={"time": "t"}),
                   on="t", direction="backward")["fundingRate"].values


def trades(rr, zzk, fwd=500, cost=0.0005):
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


# cache raw streams per (rr,zzk); funding threshold applied after
RAW = {}
def gated(rr, zzk, fq):
    if (rr, zzk) not in RAW:
        RAW[(rr, zzk)] = trades(rr, zzk)
    t = RAW[(rr, zzk)]; t = t[~t.fund.isna()]
    thr = np.nanpercentile(t.fund, fq * 100)
    return t[t.fund > thr]


ZZK = (1.4, 1.5, 1.6, 2.0); RR = (1.5, 2.0, 2.5, 3.0); FQ = (0.3, 0.4, 0.5)


def build_grid():
    cols, srs = {}, []; cid = 0
    for zzk, rr, fq in itertools.product(ZZK, RR, FQ):
        g = gated(rr, zzk, fq)
        if len(g) < 40: continue
        m = g.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()
        cols[f"c{cid}"] = m; srs.append(g.R.mean() / g.R.std(ddof=1)); cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0)
    return M.values, float(np.var(srs)), len(srs)


def main():
    fl = gated(2.0, 1.5, 0.5); R = fl.R.values
    yrs = max((fl.time.max() - fl.time.min()).days / 365.25, .5)
    print(f"FLAGSHIP short 1h/zzk1.5/RR2/fund>p50  n={len(R)}  meanR={R.mean():+.3f}  "
          f"SR/tr={R.mean()/R.std(ddof=1):+.3f}  yrs={yrs:.1f}")
    M, V, ncfg = build_grid()

    print("\nA. DEFLATED SHARPE")
    _, sr, g1, g4 = psr(R, 0.0); t = sr * np.sqrt(len(R))
    print(f"  SR/tr={sr:+.3f}  t={t:.2f}  skew={g1:+.2f}  kurt={g4:.1f}  (grid trials={ncfg}, V={V:.4f})")
    print("  " + "  ".join(f"DSR@{N}={psr(R, sr0(N, V))[0]:.2f}" for N in (1, 10, 25, 48, 100, 200)))
    print("  (DSR>0.95 = survives the N-trial haircut)")

    print("\nB. PBO via CSCV")
    real, noise, oosm = [], [], []
    for sd in range(24):
        pbo, om, _ = cscv(M, S=10, seed=sd); real.append(pbo); oosm.append(om)
        noise.append(cscv(np.random.default_rng(sd).standard_normal(M.shape), S=10, seed=sd)[0])
    print(f"  grid={M.shape[1]}cfg x {M.shape[0]}mo   REAL PBO={np.mean(real):.2f}  "
          f"IS-best OOS-Sharpe={np.mean(oosm):+.2f}   NOISE PBO={np.mean(noise):.2f}")

    print("\nC. BLOCK-BOOTSTRAP CI + mean-removed NULL on CAGR/DD")
    obs = cdd_R(R, yrs)[2]; rng = np.random.default_rng(7); B, Lb = 4000, 5
    boot = np.array([cdd_R(block_resample(R, Lb, rng), yrs)[2] for _ in range(B)])
    nul = np.array([cdd_R(block_resample(R - R.mean(), Lb, rng), yrs)[2] for _ in range(B)])
    c5, c50, c95 = np.percentile(boot, [5, 50, 95])
    print(f"  obs CAGR/DD={obs:+.2f}  CI[5/50/95]={c5:+.2f}/{c50:+.2f}/{c95:+.2f}  null p={(nul>=obs).mean():.3f}")
    yr = fl.time.dt.year.values
    print("  per-year:", " ".join(f"{y}:{R[yr==y].sum():+.0f}" for y in np.unique(yr)))


if __name__ == "__main__":
    main()
