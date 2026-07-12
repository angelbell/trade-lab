"""overfit audit on the BTC 15M Pattern-B breakout + daily-KAMA-rising gate leg.

Same math as overfit_audit_extcap.py (psr/sr0/cscv/cdd_R/block_resample), fed by the
breakout_wave 15M trade stream. No ext-cap / no daily-SMA (BTC uses the KAMA gate, not the
gold daily-SMA regime). PBO grid = rr x zz_k (the configs we'd actually consider for THIS
leg). Flagship = rr4 / zz_k2.0 (fine-grid plateau center ~1.9-2.0, NOT an IS-best peak).
Cost = 3bp round-trip (BTC spread ~$15 on ~$60k). Live-forward stays the SIZE arbiter.

  .venv/bin/python scratchpad/btc15m_audit.py
"""
import os, sys, subprocess, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

BASE = ["--csv", "data/vantage_btcusd_m15.csv", "--tf", "15min", "--pattern", "B",
        "--swing", "zigzag", "--trend-ema", "80", "--tp-mode", "rr",
        "--fwd", "500", "--risk", "0.01", "--cost", "0.0003",
        "--gate-kama", "14", "--gate-kama-tf", "1D"]


def run_trades(rr, zzk):
    args = [*BASE, "--rr", str(rr), "--zz-k", str(zzk), "--dump-trades"]
    out = subprocess.run([".venv/bin/python", "breakout_wave.py", *args],
                         capture_output=True, text=True, cwd="/home/angelbell/dev/auto-trade").stdout.splitlines()
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
    for rr, zzk in itertools.product((2, 3, 4, 5), (1.5, 1.8, 2.0, 2.2, 2.5)):
        t = run_trades(rr, zzk)
        if t is None:
            continue
        m = t.set_index("time").R.groupby(pd.Grouper(freq="M")).sum()
        cols[f"c{cid}"] = m; srs.append(t.R.mean() / t.R.std(ddof=1)); cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0)
    return M.values, float(np.var(srs))


def part_a(R, V):
    print("\n" + "=" * 72)
    print("A. DEFLATED SHARPE (per-trade) -- DSR=P(true SR>deflation benchmark for N trials)")
    Ns = [1, 10, 20, 45, 100, 200]
    _, sr, g1, g4 = psr(R, 0.0); tstat = sr * np.sqrt(len(R))
    print(f"  BTC15M/rr4/zzk2.0  n={len(R)}  SR/tr={sr:+.3f}  t={tstat:.2f}  skew={g1:+.2f}  kurt={g4:.1f}")
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
    print(f"  BTC15M/rr4/zzk2.0  obs CAGR/DD={obs:+.2f}  CI[5/50/95]={c5:+.2f}/{c50:+.2f}/{c95:+.2f}  null p={p:.3f}")
    # per-year green spread
    yr = t.time.dt.year.values
    yy = " ".join(f"{y}:{R[yr==y].sum():+.0f}" for y in np.unique(yr))
    g = sum(1 for y in np.unique(yr) if R[yr == y].sum() > 0)
    print(f"  per-year totR: {yy}   [{g}/{len(np.unique(yr))} yrs +]")


def main():
    print("BTC 15M Pattern-B breakout + daily-KAMA-rising gate -- overfit audit")
    M, V = build_grid()
    t = run_trades(4, 2.0)
    print(f"  flagship n={len(t)}  meanR={t.R.mean():+.3f}  SR/tr={t.R.mean()/t.R.std(ddof=1):+.3f}")
    part_b(M)
    part_a(t.R.values, V)
    part_c(t)
    print("\n" + "=" * 72)
    print("NOTE: statistical-overfit only; BTC 15M regime/beta-concentration is unmeasurable here.")


if __name__ == "__main__":
    main()
