"""overfit_audit_pull.py -- overfit_audit の BTC 4H EMA pullback + stop-幅フィルター版。

null 比較で ≤2.5x ATR フィルターが REAL (96%ile) と判明 -> 正式ゲートとして
A/B/C の標準バッテリーを通す。

Parts:
  A. Deflated Sharpe (DSR) -- フィルタなし vs ≤2.5x ATR vs ≤1.5x ATR
  B. PBO via CSCV -- ATR閾値 × slope_k × RR の全組み合わせで IS-best が OOS-below-median になる確率
  C. Block-bootstrap CI + mean-removed null

  .venv/bin/python research/overfit_audit_pull.py
"""
import os, sys, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import norm, skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample

RNG = np.random.default_rng(42)
GAMMA = 0.5772156649

# ── trade engine ──────────────────────────────────────────────────────────────
def make_trades(d, slope_k=6, rr=3.0, fwd=90, cost=0.001, min_stop_atr=0.5,
                max_atr=999.0):
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    es = d["close"].rolling(80).mean().values
    ef = d["close"].ewm(span=20, adjust=False).mean().values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    K = slope_k
    slope = np.full(len(c), np.nan)
    slope[K:] = (es[K:] - es[:-K]) / (K * np.where(atr[K:] > 0, atr[K:], np.nan))

    entries = []; state = 0; ext = None
    for i in range(K+1, len(c)-1):
        if np.isnan(slope[i]) or np.isnan(atr[i]) or atr[i] <= 0: continue
        if slope[i] < 0: state = 0; ext = None; continue
        if c[i] < ef[i]:
            state = 1; ext = l[i] if ext is None else min(ext, l[i])
        elif state == 1 and h[i] >= ef[i]:
            e = c[i]; stop = min(ext, l[i])
            if e - stop < min_stop_atr * atr[i]: stop = e - min_stop_atr * atr[i]
            risk = e - stop
            if risk > 0 and (risk / atr[i]) <= max_atr:
                entries.append((i, e, stop, atr[i]))
            state = 0; ext = None

    rows = []; busy = -1
    for (i, e, stop, a) in entries:
        if i <= busy: continue
        risk = e - stop
        if risk <= 0: continue
        tgt = e + rr * risk; R = None; xj = None
        for j in range(i+1, min(i+1+fwd, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= tgt:  R = rr;  xj = j; break
        if R is None:
            xj = min(i+fwd, len(c)-1); R = (c[xj]-e)/risk
        R -= cost/risk*e
        rows.append({"time": d.index[i], "R": R, "year": d.index[i].year,
                     "month": d.index[i].to_period("M")})
        busy = xj
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["time","R","year","month"])


# ── DSR helpers ───────────────────────────────────────────────────────────────
def psr(r, sr_bench):
    r = np.asarray(r, float); n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g1 = skew(r); g4 = kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g1*sr + (g4-1)/4*sr**2, 1e-9))
    return norm.cdf((sr - sr_bench)*np.sqrt(n-1)/denom), sr, g1, g4

def sr0(N, V):
    if N <= 1: return 0.0
    return np.sqrt(V)*((1-GAMMA)*norm.ppf(1-1/N) + GAMMA*norm.ppf(1-1/(N*np.e)))

def cdd_R(r, years, risk=0.01):
    eq = np.cumprod(1 + risk*r)
    dd = ((np.maximum.accumulate(eq)-eq)/np.maximum.accumulate(eq)).max()*100
    cagr = (eq[-1]**(1/max(years,0.5))-1)*100
    return cagr, dd, cagr/max(dd, 1e-9)

def block_resample(r, L):
    out = []
    while len(out) < len(r):
        s = RNG.integers(0, len(r))
        out.extend(r[s:s+L])
    return np.array(out[:len(r)])


# ── CSCV ──────────────────────────────────────────────────────────────────────
def cscv(M, S=8, max_combos=2000):
    T, N = M.shape
    T2 = (T//S)*S
    blocks = np.array_split(np.arange(T2), S)
    combos = list(itertools.combinations(range(S), S//2))
    if len(combos) > max_combos:
        idx = RNG.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in idx]
    lam = []
    for cmb in combos:
        isb = set(cmb)
        ir  = np.concatenate([blocks[b] for b in range(S) if b in isb])
        orr = np.concatenate([blocks[b] for b in range(S) if b not in isb])
        Ris, Roos = M[ir], M[orr]
        sr_is  = Ris.mean(0)  / (Ris.std(0)  + 1e-12)
        sr_oos = Roos.mean(0) / (Roos.std(0) + 1e-12)
        ns = int(np.argmax(sr_is))
        w = (sr_oos < sr_oos[ns]).sum() / (N-1)
        w = min(max(w, 1/(N+1)), 1-1/(N+1))
        lam.append(np.log(w/(1-w)))
    lam = np.array(lam)
    return (lam < 0).mean()


def main():
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    print(f"BTC 4H EMA pullback overfit audit  "
          f"{d.index[0].date()}→{d.index[-1].date()}  ({len(d)} bars)\n")

    # 3 候補レッグ
    t_base = make_trades(d, max_atr=999)
    t_25   = make_trades(d, max_atr=2.5)
    t_15   = make_trades(d, max_atr=1.5)
    legs = {"btc_pull (base)": t_base,
            "btc_pull ≤2.5xATR": t_25,
            "btc_pull ≤1.5xATR": t_15}

    # ── Part A: DSR ───────────────────────────────────────────────────────────
    print("=" * 72)
    print("A. DEFLATED SHARPE  (DSR>0.95 = survives N-trial haircut)")
    # SR variance across the pullback param grid for sr0()
    srs = []
    for max_a, rr, sk in itertools.product([1.5,2.0,2.5,3.0,999],[2,3,4],[4,6,8]):
        tt = make_trades(d, slope_k=sk, rr=float(rr), max_atr=max_a)
        if len(tt) >= 10:
            srs.append(tt.R.mean()/tt.R.std(ddof=1))
    V = float(np.var(srs)) if srs else 0.01
    Ns = [1, 10, 25, 50, 100]
    print(f"\n  {'leg':<22} {'n':>4} {'SR':>6} {'t':>5} {'skew':>5}  "
          + "  ".join(f"DSR@{N}" for N in Ns))
    print("  " + "-"*72)
    for name, t in legs.items():
        if len(t) < 5: continue
        r = t.R.values
        _, sr, g1, _ = psr(r, 0.0)
        tstat = sr*np.sqrt(len(r))
        dsrs = [psr(r, sr0(N, V))[0] for N in Ns]
        print(f"  {name:<22} {len(r):>4} {sr:>6.3f} {tstat:>5.2f} {g1:>5.2f}  "
              + "  ".join(f"{d_:>6.2f}" for d_ in dsrs))

    # ── Part B: PBO via CSCV ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("B. PBO via CSCV  (monthly R matrix; PBO<0.50 better, <0.20 = robust)")
    print("   grid: max_atr {1.5,2.0,2.5,3.0,inf} × slope_k {4,6,8} × RR {2,3,4} = 45 configs")

    cols = {}; cid = 0
    for max_a, sk, rr in itertools.product([1.5,2.0,2.5,3.0,999.0],[4,6,8],[2.0,3.0,4.0]):
        tt = make_trades(d, slope_k=sk, rr=rr, max_atr=max_a)
        if len(tt) < 10: continue
        cols[f"c{cid}"] = tt.groupby("month").R.sum()
        cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0).values
    pbo = cscv(M)
    # noise sanity
    pbo_n = cscv(RNG.standard_normal((M.shape[0], M.shape[1])))
    print(f"\n  grid shape: {M.shape[1]} configs × {M.shape[0]} months")
    print(f"  PBO = {pbo:.2f}   (noise sanity: {pbo_n:.2f}, must be ~0.50)")
    verdict = ("ROBUST (PBO<0.20)" if pbo < 0.20 else
               "OK (PBO<0.35)"     if pbo < 0.35 else
               "MARGINAL"          if pbo < 0.50 else "HIGH OVERFIT")
    print(f"  → {verdict}")

    # ── Part C: bootstrap CI + null ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("C. BLOCK-BOOTSTRAP CI on CAGR/DD + mean-removed NULL  (p<0.05 = not luck)")
    B, L = 4000, 5
    print(f"\n  {'leg':<22} {'obs CDD':>8} {'CI5':>6} {'CI50':>6} {'CI95':>6}  {'null p':>7}")
    print("  " + "-"*60)
    for name, t in legs.items():
        if len(t) < 5: continue
        r = t.R.values
        yrs = (t.time.max()-t.time.min()).days/365.25
        obs = cdd_R(r, yrs)[2]
        boot = np.array([cdd_R(block_resample(r, L), yrs)[2] for _ in range(B)])
        nul  = np.array([cdd_R(block_resample(r-r.mean(), L), yrs)[2] for _ in range(B)])
        p = (nul >= obs).mean()
        c5, c50, c95 = np.percentile(boot, [5, 50, 95])
        print(f"  {name:<22} {obs:>8.2f} {c5:>6.2f} {c50:>6.2f} {c95:>6.2f}  {p:>7.3f}")

    print("\n" + "=" * 72)
    print("PASS 基準: A=DSR@50>0.95 / B=PBO<0.35 / C=null p<0.05 の3つ揃い")


if __name__ == "__main__":
    main()
