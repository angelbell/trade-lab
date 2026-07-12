"""(A) STANDARD overfit audit for the BTC 15m pullback-limit cell
(15m / Pattern B zz-k2 / trend-ema80 / RR4 / daily KAMA(14)-rising / frac0.3 / net $15).

Judgment machinery REUSED from research/overfit_audit.py (psr / sr0 / cscv / block_resample)
-- no re-invention. Trade streams regenerated from the canonical gauntlet
scratchpad/btc15m_pullback_gauntlet.py (build/evaluate/net/stats via exec of its setup).

Pre-registered:
  DSR: effective trial count N=200 (session-honest haircut); report DSR@N sensitivity.
  PBO (CSCV): config grid = TF{15m,30m} x RR{2,4} x frac{0.2,0.25,0.3,0.38,0.5} = 20 cfgs,
              monthly net-R matrix, S=16 blocks. PBO<0.5 PASS; >=0.5 = real-but-size-uncertain.
  Bootstrap: block bootstrap (L=5, B=4000) 95% CI on meanR + mean-removed zero-edge null p (<0.01 PASS).
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

src = open("/home/angelbell/dev/auto-trade/scratchpad/btc15m_pullback_gauntlet.py").read()
exec(src.split("TFS = [")[0])  # base df, SPAN_YRS, build/evaluate/net/stats -- canonical parity

from research.overfit_audit import psr, sr0, cscv, block_resample

SP = 15.0
FRACS = (0.2, 0.25, 0.3, 0.38, 0.5)

# ---- build only the gated cells the pre-registered grid needs ----
cells = {}
for tf, fr in (("15m", None), ("30m", "30min")):
    dfx = base if fr is None else base.resample(fr).agg(AGG).dropna()
    for RR in (2.0, 4.0):
        cells[(tf, RR)] = build(dfx, RR, True)

# ---- anchor check: flagship 15m RR4 frac0.3 net$15 ----
df, E, h, l, c = cells[("15m", 4.0)]
tr, miss = evaluate(df, E, h, l, c, 0.3)
trn = net(tr, SP)
s = stats(trn)
green = sum(1 for v in s["ann"].values() if v > 0)
print("ANCHOR CHECK  flagship = 15m RR4 KAMA frac0.3 net$15")
print(f"  N={s['N']} N/yr={s['npy']:.1f} win={s['win']:.1f}% PF={s['pf']:.2f} meanR={s['meanR']:+.3f} "
      f"IS/OOS={s['IS']:+.2f}/{s['OOS']:+.2f} maxDD_R={s['maxDD']:.1f} ret/DD={s['retdd']:.2f} "
      f"green={green}/{len(s['ann'])}  miss={miss/(miss+s['N'])*100:.0f}%")
print("  per-year netR: " + "  ".join(f"{y}:{v:+.1f}" for y, v in s["ann"].items()))

# ---- B. PBO via CSCV on the pre-registered 20-config grid (monthly net-R matrix) ----
cols, srs = {}, []
for tf in ("15m", "30m"):
    for RR in (2.0, 4.0):
        dfx, Ex, hx, lx, cx = cells[(tf, RR)]
        for f_ in FRACS:
            t2, _ = evaluate(dfx, Ex, hx, lx, cx, f_)
            t2n = net(t2, SP)
            R = np.array([r for _, r in t2n])
            ser = pd.Series(R, index=pd.DatetimeIndex([t for t, _ in t2n]))
            cols[f"{tf}_RR{RR:.0f}_f{f_}"] = ser.groupby(ser.index.to_period("M")).sum()
            srs.append(R.mean() / R.std(ddof=1))
M = pd.concat(cols, axis=1).fillna(0.0)
V = float(np.var(srs))
pbo, oos_sr, ploss = cscv(M.values, S=16)
print(f"\nPBO (CSCV)  grid={M.shape[1]}cfg x {M.shape[0]}mo  PBO={pbo:.2f}  "
      f"IS-best mean OOS-SR={oos_sr:+.2f}  P(OOS loss)={ploss:.2f}  V_SR={V:.5f}")
rng = np.random.default_rng(1)
pbo_n, _, _ = cscv(rng.standard_normal(M.shape))
print(f"  [noise sanity] PBO={pbo_n:.2f}  (must be ~0.50)")
print("  per-config SR/trade: " + "  ".join(f"{k}={v:+.3f}" for k, v in zip(cols, srs)))

# ---- A. Deflated Sharpe, trial-count sensitivity (pre-registered N=200) ----
r = np.array([x for _, x in trn])
sr = r.mean() / r.std(ddof=1)
_, _, g1, g4 = psr(r, 0.0)
print(f"\nDSR  per-trade SR={sr:.4f} (t={sr*np.sqrt(len(r)):.2f})  skew={g1:.2f} kurt={g4:.1f}  n={len(r)}")
for N in (1, 50, 100, 200, 400):
    d_, _, _, _ = psr(r, sr0(N, V))
    tag = "  <== pre-registered" if N == 200 else ""
    print(f"  DSR@{N:<4} = {d_:.3f}  (benchmark SR0={sr0(N, V):.4f}){tag}")

# ---- C. block-bootstrap 95% CI on meanR + zero-edge null p ----
B, L = 4000, 5
rng = np.random.default_rng(7)
boot = np.array([block_resample(r, L, rng).mean() for _ in range(B)])
rn = r - r.mean()
nul = np.array([block_resample(rn, L, rng).mean() for _ in range(B)])
p = (nul >= r.mean()).mean()
c2, c50, c97 = np.percentile(boot, [2.5, 50, 97.5])
print(f"\nBOOTSTRAP meanR  obs={r.mean():+.3f}  95%CI=[{c2:+.3f}, {c97:+.3f}]  med={c50:+.3f} sd={boot.std():.3f}")
print(f"NULL (mean-removed, block L={L}, B={B})  p = P(null meanR >= obs) = {p:.4f}")
