"""wick_gauntlet.py -- full falsification gauntlet on the volume-selected rejection-wick FADE (USDJPY 4h).

Leg: fade a new-L-bar extreme that prints a long rejection wick, KEEPING only the top-volume-quantile
(climax) of those — the combination that survived the stress test (gross +0.43, IS~OOS, beats random-drop,
cost-robust, MC-clean). Now the standard gate: DSR + PBO/CSCV + block-null + selection-quantile plateau +
correlation with the trend book (is it a real MR diversifier?) + per-year + cost.

Falsifier: PASS = DSR survives a sane trial count AND PBO not awful AND null p small AND plateau on the
keep-quantile AND low correlation with the book. Thin leg (~5 trades/yr) => flag estimation noise.
In-sample; live-forward arbitrates.
  .venv/bin/python research/wick_gauntlet.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample
from research.wick_select import wick_trades
from research.portfolio_kama import get_legs, cagr_dd
from research.overfit_audit import psr, sr0, cscv, block_resample

SPLIT = 2018


def leg(d, sel_q=0.30, wick_k=1.0, L=10, rr=2.0, cost=0.0005):
    t = wick_trades(d, L=L, wick_k=wick_k, rr=rr, cost=cost)
    return t[t["volpr"] >= t["volpr"].quantile(1 - sel_q)][["time", "R"]]


def info(tag, t, rr=2.0):
    cg, dd, cdd, _ = cagr_dd(t)
    is_ = t[t.time.dt.year < SPLIT].R.mean(); oos = t[t.time.dt.year >= SPLIT].R.mean()
    print(f"  {tag:<22} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f}(BE{100/(1+rr):.0f}) meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.1f} CAGR/DD={cdd:5.2f} | IS={is_:+.2f} OOS={oos:+.2f}")


def main():
    d = resample(load_mt5_csv("data/vantage_usdjpy_h1.csv"), "4h")
    base = leg(d); r = base.R.values
    print("== volume-selected wick-fade (USDJPY 4h, top30% vol, wick1.0 RR2 cost0.05%) ==")
    info("base", base)

    print("\n== 1. PLATEAU (selection quantile / wick_k / RR) ==")
    print(" selection quantile (keep top X% by volume):")
    for q in (0.20, 0.25, 0.30, 0.40, 0.50):
        info(f"  top{int(q*100)}%", leg(d, sel_q=q))
    print(" wick_k:"); [info(f"  wk={wk}", leg(d, wick_k=wk)) for wk in (0.8, 1.0, 1.3)]
    print(" RR:"); [info(f"  RR{x}", leg(d, rr=x), x) for x in (1.5, 2.0, 2.5, 3.0)]

    print("\n== 2. OVERFIT AUDIT ==")
    srs, monthly = [], {}
    for q in (0.25, 0.30, 0.40):
        for wk in (0.8, 1.0, 1.3):
            for rr in (1.5, 2.0, 3.0):
                t = leg(d, sel_q=q, wick_k=wk, rr=rr)
                if len(t) < 20:
                    continue
                srs.append(t.R.mean() / t.R.std(ddof=1))
                monthly[(q, wk, rr)] = t.set_index("time").R.resample("1ME").sum()
    V = float(np.var(srs, ddof=1))
    print(f"  per-trade SR={r.mean()/r.std(ddof=1):+.3f}  (V over {len(srs)} cfgs={V:.4f})")
    for N in (1, 10, 25, 50):
        dsr, *_ = psr(r, sr0(N, V))
        print(f"     DSR@N={N:<3} = {dsr:5.3f}{'  <-survives' if dsr > 0.95 else ''}")
    months = sorted(set().union(*[set(s.index) for s in monthly.values()]))
    M = pd.DataFrame({str(k): s.reindex(months).fillna(0.0) for k, s in monthly.items()}, index=months)
    pbo, *_ = cscv(M.values, S=12, max_combos=2000, seed=0)
    print(f"  PBO (CSCV {M.shape[1]}cfg x {M.shape[0]}mo) = {pbo:.2f}")
    rng = np.random.default_rng(0); obs = r.sum()
    null = np.array([block_resample(r - r.mean(), 4, rng).sum() for _ in range(3000)])
    print(f"  block-null: obs totR={obs:+.1f} vs null {null.mean():+.1f}, p(edge<=luck)={(null >= obs).mean():.3f}")

    print("\n== 3. CORRELATION with the trend book (annual-R; MR fade should be ~0 = diversifier) ==")
    legs = get_legs()
    def ac(a, b):
        x = pd.concat([a.groupby(a.time.dt.year).R.sum(), b.groupby(b.time.dt.year).R.sum()], axis=1).dropna()
        return x.iloc[:, 0].corr(x.iloc[:, 1]) if len(x) >= 4 else np.nan
    print(f"  corr vs gold_bo={ac(base, legs['gold_bo']):+.2f}  btc_bo_kama={ac(base, legs['btc_bo_kama']):+.2f}  "
          f"btc_pull={ac(base, legs['btc_pull']):+.2f}")

    print("\n== 4. PER-YEAR + COST ==")
    by = base.groupby(base.time.dt.year).R.sum()
    print("  " + " ".join(f"{y}:{v:+.0f}" for y, v in by.items()) + f"  (green {int((by>0).sum())}/{len(by)})")
    for c in (0.0005, 0.001, 0.0015):
        info(f"cost={c*100:.2f}%", leg(d, cost=c))
    print("\n  verdict: DSR survive + PBO sane + null small + quantile plateau + low book-corr => MR diversifier candidate.")


if __name__ == "__main__":
    main()
