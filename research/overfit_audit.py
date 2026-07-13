"""overfit_audit.py -- QUANTIFY statistical-overfit risk on the book (don't just assert it).

Three established measurements applied to the book's trade streams:
  A. Deflated Sharpe Ratio (Lopez de Prado 2014) + Harvey-Liu Bonferroni haircut -- is the Sharpe
     still significant after deflating for (i) number of trials N and (ii) non-normal returns
     (skew/kurtosis)? Reported as a SENSITIVITY over N (trial count is fuzzy).
  B. PBO -- Probability of Backtest Overfitting via CSCV (Lopez de Prado 2016) -- across a realistic
     parameter grid, when we pick the IS-best config, how often is it OOS-below-median? = the p-hack
     metric. Sanity-checked against a pure-noise matrix (must give PBO~0.5).
  C. Block-bootstrap CI on CAGR/DD + a mean-removed NULL test (empirical p-value of the edge vs luck).

PBO grid universe (gold/BTC breakout) = the configs we'd actually consider, from CLAUDE.md:
  zz_k in {1.5,2.0,2.5} x trend_ema in {0,50,80,120} x rr in {2,3,4} x daily_sma in {0,150} = 72.
This UNDER-counts the true cross-session search, so real PBO >= measured. These tests quantify
STATISTICAL overfit only -- they cannot sample a future regime absent from history; live-forward
stays the arbiter for regime-change risk.

  .venv/bin/python research/overfit_audit.py
"""
import os, sys, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run as run_bo, resample
from research.regime_gate_lab import CFG
from research.portfolio_kama import get_legs
from types import SimpleNamespace

GAMMA = 0.5772156649  # Euler-Mascheroni
SPLIT = 2022


# ============================== Part A: Deflated Sharpe ==============================
def psr(r, sr_bench):
    """probabilistic Sharpe ratio: P(true per-trade SR > sr_bench), adjusts for skew/kurtosis."""
    r = np.asarray(r, float); n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g1 = skew(r); g4 = kurtosis(r, fisher=False)        # g4: normal=3
    denom = np.sqrt(max(1 - g1 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))
    return norm.cdf((sr - sr_bench) * np.sqrt(n - 1) / denom), sr, g1, g4


def sr0(N, V):
    """expected max per-trade Sharpe of N independent trials with cross-trial SR variance V."""
    if N <= 1:
        return 0.0
    return np.sqrt(V) * ((1 - GAMMA) * norm.ppf(1 - 1.0 / N) + GAMMA * norm.ppf(1 - 1.0 / (N * np.e)))


def part_a(legs, Vmap):
    print("\n" + "=" * 78)
    print("A. DEFLATED SHARPE (per-trade) -- DSR=P(true SR>deflation benchmark for N trials)")
    print("   Bonferroni p = one-sided t-stat p-value x N.  DSR>0.95 = survives the haircut.")
    Ns = [1, 10, 25, 50, 100, 200]
    print(f"\n  {'leg':<16} {'n':>4} {'SR/tr':>6} {'t':>5} {'skew':>5} {'kurt':>5}  "
          + " ".join(f"DSR@{N:<3}" for N in Ns))
    for name, t in legs.items():
        r = t.R.values
        _, sr, g1, g4 = psr(r, 0.0)
        tstat = sr * np.sqrt(len(r))
        V = Vmap.get(name, np.mean(list(Vmap.values())))
        dsrs = [psr(r, sr0(N, V))[0] for N in Ns]
        print(f"  {name:<16} {len(r):>4} {sr:>6.3f} {tstat:>5.2f} {g1:>5.2f} {g4:>5.1f}  "
              + " ".join(f"{d:>6.2f}" for d in dsrs))
    # Bonferroni haircut on the flagship gold leg's t-stat
    r = legs["gold_bo"].R.values
    t = (r.mean() / r.std(ddof=1)) * np.sqrt(len(r))
    p1 = 1 - norm.cdf(t)
    print(f"\n  Harvey-Liu Bonferroni (gold_bo, t={t:.2f}, raw 1-sided p={p1:.1e}):")
    print("   " + "  ".join(f"N={N}:p={min(1,p1*N):.1e}" for N in Ns))


# ============================== Part B: PBO via CSCV ==============================
def build_grid(csv, tf):
    """run the breakout param grid once -> (monthly matrix MxN, per-trial SR array for V)."""
    d = load_mt5_csv(csv)
    if "xauusd_h1" in csv:
        d = d.loc[GOLD_H1_START:]          # pre-2018 gold H1 is daily data wearing an H1 label
    d = resample(d, tf)
    cols, srs = {}, []
    cid = 0
    for zz, te, rr, ds in itertools.product([1.5, 2.0, 2.5], [0, 50, 80, 120],
                                            [2.0, 3.0, 4.0], [0, 150]):
        a = SimpleNamespace(**{**CFG, "csv": csv, "tf": tf, "zz_k": zz, "trend_ema": te,
                               "rr": rr, "daily_sma": ds, "fwd": 500 if tf == "1h" else 300})
        t = run_bo(d, a)
        if len(t) < 20:
            continue
        s = t.copy(); s["m"] = s.time.dt.to_period("M")
        cols[f"c{cid}"] = s.groupby("m").R.sum()
        srs.append(t.R.mean() / t.R.std(ddof=1))
        cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0)
    return M.values, np.var(srs)


def cscv(M, S=16, max_combos=2000, seed=0):
    T, N = M.shape
    T2 = (T // S) * S
    blocks = np.array_split(np.arange(T2), S)
    combos = list(itertools.combinations(range(S), S // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    lam, is_best_oos, oos_loss = [], [], 0
    for cmb in combos:
        isb = set(cmb)
        ir = np.concatenate([blocks[b] for b in range(S) if b in isb])
        orr = np.concatenate([blocks[b] for b in range(S) if b not in isb])
        Ris, Roos = M[ir], M[orr]
        sr_is = Ris.mean(0) / (Ris.std(0) + 1e-12)
        sr_oos = Roos.mean(0) / (Roos.std(0) + 1e-12)
        ns = int(np.argmax(sr_is))
        w = (sr_oos < sr_oos[ns]).sum() / (N - 1)          # OOS relative rank of IS-best
        w = min(max(w, 1 / (N + 1)), 1 - 1 / (N + 1))
        lam.append(np.log(w / (1 - w)))
        is_best_oos.append(sr_oos[ns])
        oos_loss += sr_oos[ns] <= 0
    lam = np.array(lam)
    return (lam < 0).mean(), np.mean(is_best_oos), oos_loss / len(combos)


def part_b():
    print("\n" + "=" * 78)
    print("B. PBO via CSCV -- P(IS-best config is OOS-below-median).  PBO<0.5 better; <~0.2 = robust.")
    Vmap = {}
    for name, csv, tf, leg in [("gold breakout", "data/vantage_xauusd_h1.csv", "1h", "gold_bo"),
                               ("BTC breakout",  "data/vantage_btcusd_h1.csv", "4h", "btc_bo_kama")]:
        M, V = build_grid(csv, tf)
        Vmap[leg] = V
        pbo, oos_med, ploss = cscv(M)
        print(f"  {name:<14} grid={M.shape[1]}cfg x {M.shape[0]}mo  PBO={pbo:.2f}  "
              f"IS-best mean OOS-Sharpe={oos_med:+.2f}  P(OOS loss)={ploss:.2f}  (V_SR={V:.4f})")
    # noise sanity: random matrix must give PBO ~ 0.5
    rng = np.random.default_rng(1)
    pbo_n, _, _ = cscv(rng.standard_normal((120, 72)))
    print(f"  {'[noise sanity]':<14} PBO={pbo_n:.2f}  (must be ~0.50 -> CSCV code is valid)")
    return Vmap


# ============================== Part C: bootstrap CI + null ==============================
def cdd_R(r, years, risk=0.01):
    eq = np.cumprod(1 + risk * r)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (1 / years) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def block_resample(r, L, rng):
    out = []
    while len(out) < len(r):
        s = rng.integers(0, len(r))
        out.extend(r[s:s + L])
    return np.array(out[:len(r)])


def part_c(legs, book):
    print("\n" + "=" * 78)
    print("C. BLOCK-BOOTSTRAP CI on CAGR/DD (1% risk) + mean-removed NULL (empirical p).")
    print("   p = P(null CAGR/DD >= observed) -- small p => edge unlikely to be luck.")
    streams = dict(legs); streams["BOOK 2-leg(inv-vol)"] = book
    B, L = 4000, 5
    rng = np.random.default_rng(7)
    print(f"\n  {'stream':<22} {'obs CDD':>8} {'CI5':>6} {'CI50':>6} {'CI95':>6}  {'null p':>7}")
    for name, t in streams.items():
        r = t.R.values
        yrs = max((t.time.max() - t.time.min()).days / 365.25, 0.5)
        obs = cdd_R(r, yrs)[2]
        boot = np.array([cdd_R(block_resample(r, L, rng), yrs)[2] for _ in range(B)])
        rn = r - r.mean()
        nul = np.array([cdd_R(block_resample(rn, L, rng), yrs)[2] for _ in range(B)])
        p = (nul >= obs).mean()
        c5, c50, c95 = np.percentile(boot, [5, 50, 95])
        print(f"  {name:<22} {obs:>8.2f} {c5:>6.2f} {c50:>6.2f} {c95:>6.2f}  {p:>7.3f}")


def main():
    legs = get_legs()
    book = pd.concat([legs["gold_bo"].assign(R=legs["gold_bo"].R * 0.79),
                      legs["btc_bo_kama"].assign(R=legs["btc_bo_kama"].R * 1.21)]).sort_values("time")
    Vmap = part_b()           # build grids first -> supplies V to Part A
    part_a(legs, Vmap)
    part_c(legs, book)
    print("\n" + "=" * 78)
    print("NOTE: A/B/C quantify STATISTICAL overfit (trials/noise) only. They cannot measure")
    print("regime-change risk (a regime absent from history is unsampled). Live-forward = arbiter.")


if __name__ == "__main__":
    main()
