"""overfit_audit_extcap.py -- DSR / PBO / bootstrap on the 15M gold_bo + extension-cap leg.

Same math as overfit_audit.py (psr/sr0/cscv/cdd_R/block_resample), fed by the breakout_wave
15M Pattern-B trade stream (R is already in R-multiples from --dump-trades). PBO grid = the
configs we'd actually consider for THIS leg: ext_cap x rr x zz_k. Flagship = ext_cap 8 / rr4 /
zz_k2 (plateau center, NOT the IS-best 9% peak). Sealed-TEST note as in the H17-S audit: a
statistical-overfit measurement on full 2019+ history, not a fresh peek; live-forward = arbiter.

  .venv/bin/python research/overfit_audit_extcap.py
"""
import os, sys, subprocess, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

BASE = ["--csv", "data/vantage_xauusd_m15.csv", "--tf", "15min", "--pattern", "B",
        "--swing", "zigzag", "--trend-ema", "80", "--bo-window", "20", "--tp-mode", "rr",
        "--fwd", "500", "--daily-sma", "150", "--daily-slope-k", "10", "--risk", "0.01",
        "--cost", "0.0002"]


def run_trades(cap, rr, zzk):
    args = [*BASE, "--ext-cap", str(cap), "--rr", str(rr), "--zz-k", str(zzk), "--dump-trades"]
    out = subprocess.run([".venv/bin/python", "breakout_wave.py", *args],
                         capture_output=True, text=True).stdout.splitlines()
    try:
        i = next(k for k, l in enumerate(out) if l.startswith("entry_time,"))
    except StopIteration:
        return None
    rows = [l.split(",") for l in out[i + 1:] if l]
    if len(rows) < 20:
        return None
    t = pd.DataFrame(rows, columns=["time", "R", "hold"])
    t["time"] = pd.to_datetime(t["time"], utc=True); t["R"] = t["R"].astype(float)
    return t.sort_values("time")


def build_grid():
    cols, srs = {}, []; cid = 0
    for cap, rr, zzk in itertools.product((6, 8, 9, 10, 12), (3, 4, 5), (1.5, 2.0, 2.5)):
        t = run_trades(cap, rr, zzk)
        if t is None:
            continue
        m = t.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()
        cols[f"c{cid}"] = m; srs.append(t.R.mean() / t.R.std(ddof=1)); cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0)
    return M.values, float(np.var(srs))


def part_a(R, V):
    print("\n" + "=" * 72)
    print("A. DEFLATED SHARPE (per-trade) -- DSR=P(true SR>deflation benchmark for N trials)")
    Ns = [1, 10, 25, 45, 100, 200]
    _, sr, g1, g4 = psr(R, 0.0); tstat = sr * np.sqrt(len(R))
    print(f"  15M+ext8/rr4  n={len(R)}  SR/tr={sr:+.3f}  t={tstat:.2f}  skew={g1:+.2f}  kurt={g4:.1f}")
    print("  " + "  ".join(f"DSR@{N}={psr(R, sr0(N, V))[0]:.2f}" for N in Ns))
    print("  (DSR>0.95 = survives the N-trial haircut)")


def part_b(M, S=10, seeds=24):
    print("\n" + "=" * 72)
    print("B. PBO via CSCV -- P(IS-best config OOS-below-median). <0.5 better; <~0.2 robust.")
    real, noise, oosm = [], [], []
    for sd in range(seeds):
        pbo, om, _ = cscv(M, S=S, seed=sd)
        real.append(pbo); oosm.append(om)
        noise.append(cscv(np.random.default_rng(sd).standard_normal(M.shape), S=S, seed=sd)[0])
    real, noise = np.array(real), np.array(noise)
    print(f"  grid={M.shape[1]} cfg x {M.shape[0]} months  (S={S}, {seeds} seeds)")
    print(f"  REAL  PBO={real.mean():.2f}   IS-best mean OOS-Sharpe={np.mean(oosm):+.2f}")
    print(f"  NOISE PBO={noise.mean():.2f} (must center ~0.50)   gap={real.mean()-noise.mean():+.2f}")


def part_c(t):
    print("\n" + "=" * 72)
    print("C. BLOCK-BOOTSTRAP CI on CAGR/DD (1% risk) + mean-removed NULL (empirical p).")
    R = t.R.values
    yrs = max((t.time.max() - t.time.min()).days / 365.25, 0.5)
    obs = cdd_R(R, yrs)[2]
    B, L = 4000, 5; rng = np.random.default_rng(7)
    boot = np.array([cdd_R(block_resample(R, L, rng), yrs)[2] for _ in range(B)])
    rn = R - R.mean()
    nul = np.array([cdd_R(block_resample(rn, L, rng), yrs)[2] for _ in range(B)])
    p = (nul >= obs).mean(); c5, c50, c95 = np.percentile(boot, [5, 50, 95])
    print(f"  15M+ext8/rr4  obs CAGR/DD={obs:+.2f}  CI[5/50/95]={c5:+.2f}/{c50:+.2f}/{c95:+.2f}  null p={p:.3f}")


def main():
    print("15M gold_bo + extension-cap overfit audit")
    M, V = build_grid()
    t = run_trades(8, 4, 2.0)
    part_b(M)
    part_a(t.R.values, V)
    part_c(t)
    print("\n" + "=" * 72)
    print("NOTE: statistical-overfit only. 15M regime-dependence (2023 chop = the lone red year)")
    print("is unmeasurable here -- live-forward stays the arbiter for SIZE.")


if __name__ == "__main__":
    main()
