"""audit_kama_stack.py -- formal overfit audit of the C1-defense candidate: kama4h AND stack4h>0
gate on the BTC 15m leg (RR4 frac0.3 net $15).

Trial universe (the HONEST multiple-comparison set = what was actually compared today):
  12 gate variants from radar_gate_race (kama1d / kama4h / kama4h&1d / stack>0 / stack==3 /
  up&{ER,ADX,slope,ATRexp,s10} matched / radar s>=5 / kama4h&stack) + the 27-cell EMA-window
  plateau grid of the candidate (emaFast {15,20,25} x emaSlow {40,50,60} x shift {7,10,13}).
All configs share the one ungated trade set (post-hoc labels) -- this audits GATE selection.

  A. +-1 PLATEAU on the EMA windows (real edge = neighbors agree, no lone spike)
  B. PBO via CSCV (S=16 month-blocks) on the 39-config monthly-R matrix + noise sanity
  C. Deflated Sharpe of the candidate cell over a trial-count ladder (V = cross-config SR var)
  D. Block-bootstrap CI on CAGR/DD (1% risk) + mean-removed null p, candidate vs incumbents
"""
import os, sys, itertools, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_gate_race import BASE, comps_tf, kama_up, matched
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.overfit_audit import cscv, psr, sr0, cdd_R, block_resample

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
RT = 15.0


def stack4h(d15, emaF, emaS, lb):
    d = d15.resample("240min").agg(AGG).dropna()
    c = d["close"]
    f = c.ewm(span=emaF, adjust=False).mean()
    s = c.ewm(span=emaS, adjust=False).mean()
    st = np.sign(c - f) + np.sign(f - s) + np.sign(s - s.shift(lb))
    return ((st > 0).shift(1).reindex(d15.index, method="ffill")
            .fillna(False).values.astype(bool))


def main():
    d = load_mt5_csv("data/vantage_btcusd_m15.csv")
    d15 = resample(d[d.index >= "2018-10-01"], "15min")
    span = (d15.index[-1] - d15.index[0]).days / 365.25
    t = run(d15, SimpleNamespace(**BASE, pullback_frac=0.3))
    Rn = t["R"].values - RT / t["risk"].values
    pos = d15.index.get_indexer(t["time"])
    mon = t["time"].dt.to_period("M")

    k4, k1 = kama_up(d15, "240min"), kama_up(d15, "1D")
    C4 = comps_tf(d15, "240min")
    up4 = C4["stack"] > 0
    target = k4.mean()

    gates = {"kama1d": k1, "kama4h": k4, "kama4h&1d": k4 & k1,
             "stack>0": up4, "stack==3": C4["stack"] >= 3,
             "radar_s5": up4 & (C4["s10"] >= 5.0)}
    for ck in ("er", "adx", "slope", "atrexp", "s10"):
        gates[f"up&{ck}"] = matched(up4, C4[ck], target)

    # ---------- A. plateau grid (candidate = kama4h & stack4h(emaF,emaS,lb)) ----------
    print(f"\nA. EMA-window +-1 PLATEAU of kama4h & stack4h  ({span:.1f}yr, net $15)")
    print(f"   {'emaF':>4} {'emaS':>4} {'lb':>3}  {'N/yr':>5} {'meanR':>7} {'PF':>5} "
          f"{'totR/yr':>8} {'maxDD':>6} {'ret/DD':>6}")
    grid_stats = []
    for emaF, emaS, lb in itertools.product([15, 20, 25], [40, 50, 60], [7, 10, 13]):
        g = k4 & stack4h(d15, emaF, emaS, lb)
        gates[f"cand_{emaF}_{emaS}_{lb}"] = g
        r = Rn[g[pos]]
        pf = r[r > 0].sum() / abs(r[r <= 0].sum())
        eq = np.cumsum(r)
        dd = (np.maximum.accumulate(eq) - eq).max()
        grid_stats.append((emaF, emaS, lb, len(r) / span, r.mean(), pf,
                           r.sum() / span, dd, r.sum() / dd))
        star = " <== default" if (emaF, emaS, lb) == (20, 50, 10) else ""
        print(f"   {emaF:>4} {emaS:>4} {lb:>3}  {len(r)/span:>5.1f} {r.mean():>+7.3f} {pf:>5.2f} "
              f"{r.sum()/span:>+8.1f} {dd:>5.1f}R {r.sum()/dd:>6.2f}{star}")
    rdds = np.array([g[-1] for g in grid_stats])
    mrs = np.array([g[4] for g in grid_stats])
    print(f"   grid ret/DD: min={rdds.min():.2f} med={np.median(rdds):.2f} max={rdds.max():.2f}"
          f"  | meanR: min={mrs.min():+.3f} med={np.median(mrs):+.3f} max={mrs.max():+.3f}")

    # ---------- B. PBO via CSCV on the full config universe ----------
    labels = list(gates.keys())
    midx = pd.period_range(mon.min(), mon.max(), freq="M")
    cols, srs = {}, []
    for k in labels:
        m = gates[k][pos]
        s = pd.Series(Rn[m], index=mon[m]).groupby(level=0).sum().reindex(midx, fill_value=0.0)
        cols[k] = s
        srs.append(Rn[m].mean() / Rn[m].std(ddof=1))
    M = pd.concat(cols, axis=1)
    V = float(np.var(srs))
    pbo, oos_sr, ploss = cscv(M.values)
    rng = np.random.default_rng(1)
    pbo_n, _, _ = cscv(rng.standard_normal(M.shape))
    print(f"\nB. PBO via CSCV: {M.shape[1]} configs x {M.shape[0]} months")
    print(f"   PBO={pbo:.2f}  IS-best mean OOS-Sharpe={oos_sr:+.2f}  P(OOS loss)={ploss:.2f}"
          f"   [noise sanity PBO={pbo_n:.2f}, must be ~0.50]   V_SR={V:.5f}")

    # ---------- C. Deflated Sharpe of the candidate ----------
    cand = gates["cand_20_50_10"][pos]
    r = Rn[cand]
    _, sr, g1, g4 = psr(r, 0.0)
    Ns = [1, 10, 25, 39, 100, 200]
    dsrs = [psr(r, sr0(N, V))[0] for N in Ns]
    print(f"\nC. DEFLATED SHARPE (candidate kama4h&stack, n={len(r)}, SR/tr={sr:.3f}, "
          f"skew={g1:.2f}, kurt={g4:.1f})")
    print("   " + "  ".join(f"DSR@{N}={v:.2f}" for N, v in zip(Ns, dsrs)) + "   (>0.95 = PASS)")

    # ---------- D. bootstrap CI + null, candidate vs incumbents ----------
    print(f"\nD. BLOCK-BOOTSTRAP CI on CAGR/DD (1% risk) + mean-removed NULL p  (B=4000, L=5)")
    print(f"   {'gate':<14} {'obs':>6} {'CI5':>6} {'CI50':>6} {'CI95':>6} {'null p':>7}")
    rng = np.random.default_rng(7)
    for k in ("kama4h", "kama4h&1d", "cand_20_50_10"):
        rr = Rn[gates[k][pos]]
        obs = cdd_R(rr, span)[2]
        boot = np.array([cdd_R(block_resample(rr, 5, rng), span)[2] for _ in range(4000)])
        rn0 = rr - rr.mean()
        nul = np.array([cdd_R(block_resample(rn0, 5, rng), span)[2] for _ in range(4000)])
        c5, c50, c95 = np.percentile(boot, [5, 50, 95])
        print(f"   {k:<14} {obs:>6.2f} {c5:>6.2f} {c50:>6.2f} {c95:>6.2f} "
              f"{(nul >= obs).mean():>7.3f}")


if __name__ == "__main__":
    main()
