"""squeeze_gauntlet.py -- full falsification gauntlet on the CORRECTLY-understood strategy:
BTC 4H Donchian breakout filtered to the EXPANSION regime (TOP-quantile ATR), no-overlap (tradeable).
(The user's 'squeeze' code had an inverted rank metric -> it actually traded high-ATR expansion, which
de-overlaps to meanR+0.43, green every year, corr+0.30 with btc_bo_kama. This stress-tests it.)

Gauntlet: 1 plateau (atr-thr/don/rr/L)  2 DSR+PBO+null  3 cost  4 random-long null (is it beta?)
          5 correlation/portfolio with the book.  In-sample; live-forward arbitrates.
  .venv/bin/python research/squeeze_gauntlet.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.squeeze_breakout import run, stats, atr
from research.portfolio_kama import get_legs, cagr_dd
from research.overfit_audit import psr, sr0, cscv, block_resample

SPLIT = 2022
EXP = dict(hi_atr=True, no_overlap=True)        # the tradeable expansion-breakout


def line(tag, t):
    s = stats(t)
    if s["n"] == 0:
        print(f"  {tag:<22} n=0"); return
    print(f"  {tag:<22} n={s['n']:>4} win%={s['win']:>3.0f} meanR={s['meanR']:+5.2f} "
          f"totR={s['totR']:>+6.1f} PF={s['PF']:4.2f} | IS={s['IS']:+6.1f} OOS={s['OOS']:+6.1f}")


def rand_null(d, side, n_real, real_meanR, rr=3.0, fwd=60, cost=0.001, iters=2000, seed=0):
    """random entries (matched count), same 1ATR-stop / RR-TP / fwd exit -> is real > drift+structure?"""
    a = atr(d, 14).values
    H, Lw, C = d["high"].values, d["low"].values, d["close"].values
    valid = np.where(np.isfinite(a) & (a > 0))[0]
    valid = valid[valid < len(d) - fwd - 1]
    rng = np.random.default_rng(seed)

    def one(i):
        e = C[i]; risk = a[i]
        if side == "L":
            sl, tp = e - risk, e + rr * risk
        else:
            sl, tp = e + risk, e - rr * risk
        for j in range(i + 1, min(i + 1 + fwd, len(d))):
            if side == "L":
                if Lw[j] <= sl: return -1 - cost * e / risk
                if H[j] >= tp: return rr - cost * e / risk
            else:
                if H[j] >= sl: return -1 - cost * e / risk
                if Lw[j] <= tp: return rr - cost * e / risk
        jj = min(i + fwd, len(d) - 1)
        return ((C[jj] - e) if side == "L" else (e - C[jj])) / risk - cost * e / risk

    means = np.array([np.mean([one(i) for i in rng.choice(valid, n_real, replace=False)])
                      for _ in range(iters)])
    pct = (means < real_meanR).mean() * 100
    return means.mean(), np.percentile(means, 95), pct


def main():
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    print(f"EXPANSION breakout (high-ATR, no-overlap) -- BTC 4h {d.index.min().date()}..{d.index.max().date()}")

    print("\n== 1. PLATEAU (each varies ONE knob; want a high plateau, not a lone spike) ==")
    print(" ATR top-quantile (expansion strength):")
    for q in (0.15, 0.20, 0.25, 0.30, 0.40):
        line(f"  top {int(q*100)}%", run(d, rr=3, sqz=q, **EXP))
    print(" Donchian len:");  [line(f"  don={n}", run(d, rr=3, don=n, **EXP)) for n in (20, 25, 30, 40, 55)]
    print(" RR:");            [line(f"  RR{rr}", run(d, rr=rr, **EXP)) for rr in (2, 2.5, 3, 3.5, 4)]
    print(" L window:");      [line(f"  L={x}", run(d, rr=3, L=x, **EXP)) for x in (80, 100, 120, 150, 200)]
    print(" ATR def:");       line("  SMA", run(d, rr=3, **EXP)); line("  Wilder", run(d, rr=3, wilder=True, **EXP))

    base = run(d, rr=3, **EXP); r = base["R"].values
    print("\n== 2. OVERFIT AUDIT (DSR / PBO / null) on the no-overlap expansion-breakout ==")
    grid, srs, monthly = [], [], {}
    for q in (0.20, 0.25, 0.30):
        for don in (20, 30, 40):
            for rr in (2.0, 3.0, 4.0):
                t = run(d, rr=rr, sqz=q, don=don, **EXP)
                if len(t) < 20: continue
                grid.append((q, don, rr)); srs.append(t.R.mean() / t.R.std(ddof=1))
                monthly[(q, don, rr)] = t.set_index("time").R.resample("1ME").sum()
    V = float(np.var(srs, ddof=1))
    print(f"  per-trade SR={r.mean()/r.std(ddof=1):+.3f}  (V over {len(srs)} cfgs={V:.4f})")
    for N in (1, 10, 25, 50, 100, 200):
        dsr, *_ = psr(r, sr0(N, V))
        print(f"     DSR@N={N:<4} = {dsr:5.3f}{'  <-survives(>0.95)' if dsr > 0.95 else ''}")
    months = sorted(set().union(*[set(s.index) for s in monthly.values()]))
    M = pd.DataFrame({str(k): s.reindex(months).fillna(0.0) for k, s in monthly.items()}, index=months)
    pbo, *_ = cscv(M.values, S=16, max_combos=2000, seed=0)
    print(f"  PBO (CSCV {M.shape[1]}cfg x {M.shape[0]}mo) = {pbo:.2f}  (~0.5=param pick doesn't generalize)")
    rng = np.random.default_rng(0); obs = r.sum()
    null = np.array([block_resample(r - r.mean(), 5, rng).sum() for _ in range(3000)])
    print(f"  block-bootstrap null: obs totR={obs:+.1f}, null mean={null.mean():+.1f}, p(edge<=luck)={(null >= obs).mean():.3f}")

    print("\n== 3. COST STRESS ==")
    [line(f"  cost={c*100:.2f}%", run(d, rr=3, cost=c, **EXP)) for c in (0.001, 0.002, 0.003, 0.005)]

    print("\n== 4. RANDOM-LONG/SHORT NULL (is meanR just BTC drift+exit-structure?) ==")
    for side in ("L", "S"):
        t = run(d, rr=3, side=("long" if side == "L" else "short"), **EXP)
        nm, n95, pct = rand_null(d, side, len(t), t.R.mean())
        verdict = "BEATS beta" if pct >= 95 else ("marginal" if pct >= 80 else "= beta/noise")
        print(f"  {('LONG' if side=='L' else 'SHORT')}: real meanR={t.R.mean():+.2f} (n={len(t)}) vs "
              f"random mean={nm:+.2f} 95%ile={n95:+.2f}  -> real at {pct:.0f}%ile [{verdict}]")

    print("\n== 5. CORRELATION + PORTFOLIO (does +0.30-corr leg cut book DD?) ==")
    legs = get_legs()
    bexp = base.rename(columns={"time": "time"})[["time", "R"]]
    def ac(a, b):
        x = pd.concat([a.groupby(a.time.dt.year).R.sum(), b.groupby(b.time.dt.year).R.sum()], axis=1).dropna()
        return x.iloc[:, 0].corr(x.iloc[:, 1]) if len(x) >= 4 else np.nan
    print(f"  annual corr: gold_bo={ac(bexp, legs['gold_bo']):+.2f}  btc_bo_kama={ac(bexp, legs['btc_bo_kama']):+.2f}  "
          f"btc_pull={ac(bexp, legs['btc_pull']):+.2f}")
    for name in ("gold_bo", "btc_bo_kama", "btc_pull"):
        c, dd, cdd, _ = cagr_dd(legs[name]); print(f"    {name:<12} standalone CAGR/DD={cdd:.2f}")
    c, dd, cdd, _ = cagr_dd(bexp); print(f"    {'EXP_breakout':<12} standalone CAGR/DD={cdd:.2f}")


if __name__ == "__main__":
    main()
