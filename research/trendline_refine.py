"""trendline_refine.py -- FINAL push on Trendlines-with-Breaks [LuxAlgo]: can the REAL entry edge
(structural-stop+RR2 gold = meanR +0.35, beat random at 98th pctile) be HARVESTED into something
adoptable? 3 pre-registered probes, each with a kill condition.

Re-test of a NON-ADOPTED method => raised bar + multiple-comparisons honesty: a pass must clear
overfit_audit afterward, and this is the FINAL TLB attempt (no reopening to fish). Justified only by
the documented real mfe/mae edge. Imports the validated causal `signals` from trendline_break.

  P1 SURFACE   : len x mult x method -- is len14 a lone SPIKE or part of a 2D plateau?
  P2 EXIT      : structural-TRAIL / KAMA-trail vs fixed RR2 -- can a runner exit smooth the equity?
  P3 PORTFOLIO : MEASURE corr(TLB-gold, gold_bo/book) -- low corr could add CAGR/DD despite weak standalone.

  .venv/bin/python research/trendline_refine.py
"""
import os, sys, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.trendline_break import signals
from research.regime_gate_lab import metrics
from research.regime_adaptive import kama
from research.portfolio_kama import get_legs, cagr_dd

RNG = np.random.default_rng(7)
CSV, TF, COST = "data/vantage_xauusd_h1.csv", "4h", 0.30


def build(d, sig_bars, exit_mode, rr=2.0, swing=14, kama_n=10, km=None):
    """long trades from given signal bars; structural stop = recent `swing`-bar low; R per initial risk."""
    op = d["open"].values; H = d["high"].values; Lo = d["low"].values; C = d["close"].values
    tm = d.index; n = len(C)
    swl = pd.Series(Lo).rolling(swing).min().values
    rows = []; busy = -1
    for i in sig_bars:
        ei = i + 1
        if ei <= busy or ei >= n or np.isnan(swl[i]):
            continue
        e = op[ei]; stop0 = swl[i]
        if e <= stop0:
            continue
        risk = e - stop0; R = None; xj = None
        if exit_mode == "rr":
            tgt = e + rr * risk; stop = stop0
            for j in range(ei, min(ei + 400, n)):
                if Lo[j] <= stop: R, xj = -1.0, j; break
                if H[j] >= tgt: R, xj = rr, j; break
        elif exit_mode == "strail":
            stop = stop0
            for j in range(ei, min(ei + 400, n)):
                if Lo[j] <= stop: R, xj = (stop - e) / risk, j; break
                if not np.isnan(swl[j]): stop = max(stop, swl[j])
        elif exit_mode == "kama":
            for j in range(ei, min(ei + 400, n)):
                if Lo[j] <= stop0: R, xj = -1.0, j; break
                if not np.isnan(km[j]) and C[j] < km[j]: R, xj = (C[j] - e) / risk, j; break
        if R is None:
            xj = min(ei + 400, n - 1); R = (C[xj] - e) / risk
        R -= COST / risk
        rows.append((tm[ei], R, 1)); busy = xj
    return pd.DataFrame(rows, columns=["time", "R", "dir"])


def cdd(t):
    m = metrics(t)
    return m


def rand_pctile(d, n_trades, exit_mode, swing, km, length, draws=150):
    valid = np.arange(swing + length + 2, len(d) - 2)
    if n_trades < 5 or len(valid) < n_trades:
        return np.nan
    vals = []
    for _ in range(draws):
        bars = np.sort(RNG.choice(valid, n_trades, replace=False))
        m = cdd(build(d, bars, exit_mode, swing=swing, km=km))
        if m: vals.append(m["cdd"])
    return np.array(vals)


def main():
    d = resample(load_mt5_csv(CSV), TF)
    print(f"TLB refine -- gold {TF} {d.index[0].date()}->{d.index[-1].date()}  (FINAL attempt; bar RAISED)")

    # anchor sanity: reproduce the prior len14/mult1/Atr/RR2 result (~meanR +0.35, CAGR/DD ~0.39)
    upb, _ = signals(d, 14, 1.0, "Atr")
    anchor = build(d, np.where(upb)[0], "rr", rr=2.0)
    m = cdd(anchor)
    print(f"\n  [anchor sanity] len14 mult1 Atr RR2: n={m['n']} meanR={anchor.R.mean():+.3f} "
          f"win={(anchor.R>0).mean()*100:.0f}% CAGR/DD={m['cdd']:.2f} IS={m['isr']:+.2f} OOS={m['oos']:+.2f} "
          f"(prior: meanR~+0.35 cdd~0.39)")

    # ================= P1: parameter SURFACE =================
    print("\n" + "=" * 80)
    print("P1 SURFACE (RR2): meanR / CAGR-DD per len x mult x method -- spike or 2D plateau?")
    print(f"  {'method':<8}{'mult':>5} | " + "  ".join(f"len{L:<2}(mR/cdd)" for L in (8, 10, 14, 21)))
    good = []
    for method in ("Atr", "Stdev", "Linreg"):
        for mult in (0.5, 1.0, 2.0):
            cells = []
            for L in (8, 10, 14, 21):
                upb, _ = signals(d, L, mult, method)
                t = build(d, np.where(upb)[0], "rr", rr=2.0)
                m = cdd(t)
                if m is None:
                    cells.append("   --     "); continue
                cells.append(f"{t.R.mean():+.2f}/{m['cdd']:4.2f}")
                if m["cdd"] >= 0.30 and t.R.mean() > 0:
                    good.append((method, mult, L, m["cdd"], t, np.where(upb)[0]))
            print(f"  {method:<8}{mult:>5} | " + "  ".join(f"{c:<11}" for c in cells))
    print("\n  random-pctile on cells with CAGR/DD>=0.30 (>=80 across NEIGHBORS = plateau, lone = spike):")
    for method, mult, L, c, t, bars in good:
        rv = rand_pctile(d, len(t), "rr", 14, None, L)
        pct = (rv < c).mean() * 100 if rv is not None and not np.isscalar(rv) else np.nan
        print(f"    {method:<7} mult{mult} len{L:<3} CAGR/DD={c:4.2f}  rand-pctile={pct:3.0f}  (n={len(t)})")
    if not good:
        print("    (no cell reached CAGR/DD>=0.30 -> nothing to harvest)")

    # ================= P2: EXIT lever (anchor params) =================
    print("\n" + "=" * 80)
    print("P2 EXIT (len14 mult1 Atr): can a runner exit smooth equity above RR2's ~0.39?")
    upb, _ = signals(d, 14, 1.0, "Atr"); bars = np.where(upb)[0]
    km = kama(d["close"], 10).values
    for mode, kw in [("rr (RR2)", dict(exit_mode="rr", rr=2.0)), ("rr (RR3)", dict(exit_mode="rr", rr=3.0)),
                     ("structural-TRAIL", dict(exit_mode="strail")), ("KAMA-trail", dict(exit_mode="kama", km=km))]:
        t = build(d, bars, **kw); m = cdd(t)
        if m is None:
            print(f"  {mode:<18} (too few)"); continue
        print(f"  {mode:<18} n={m['n']:>4} meanR={t.R.mean():+.3f} CAGR={m['cagr']:+5.1f}% DD={m['dd']:4.1f}% "
              f"CAGR/DD={m['cdd']:5.2f} | IS={m['isr']:+.2f} OOS={m['oos']:+.2f}")

    # ================= P3: PORTFOLIO correlation (MEASURED, not assumed) =================
    print("\n" + "=" * 80)
    print("P3 PORTFOLIO: corr(TLB-gold, gold_bo / book)?  low corr could add CAGR/DD despite weak standalone")
    legs = get_legs()
    gold = legs["gold_bo"]; btc_k = legs["btc_bo_kama"]
    tlb = build(d, bars, "rr", rr=2.0)[["time", "R"]]
    # CRITICAL: TLB-gold spans 2007-2026 but the book legs start ~2018 -> clip ALL to the common
    # window, else concatenated R=0/early TLB trades stretch the CAGR span and distort corr (span-bug).
    cstart = max(gold.time.min(), btc_k.time.min())
    gold = gold[gold.time >= cstart]; btc_k = btc_k[btc_k.time >= cstart]
    tlb = tlb[tlb.time >= cstart]
    print(f"  (common window from {cstart.date()}; TLB n in-window={len(tlb)})")

    def ann(x): return x.assign(y=x.time.dt.year).groupby("y").R.sum()
    def mon(x): return x.assign(m=x.time.dt.to_period("M")).groupby("m").R.sum()
    A = pd.concat([ann(gold), ann(btc_k), ann(tlb)], axis=1).fillna(0); A.columns = ["gold", "btc", "tlb"]
    M = pd.concat([mon(gold), mon(btc_k), mon(tlb)], axis=1).fillna(0); M.columns = ["gold", "btc", "tlb"]
    print(f"  annual-R  corr(TLB, gold_bo)={A.tlb.corr(A.gold):+.2f}  corr(TLB, btc_bo_kama)={A.tlb.corr(A.btc):+.2f}")
    print(f"  monthly-R corr(TLB, gold_bo)={M.tlb.corr(M.gold):+.2f}  corr(TLB, btc_bo_kama)={M.tlb.corr(M.btc):+.2f}")
    print("  (corr>=~0.6 => dilutive/redundant; <=~0.3 => could diversify -> then gauntlet)")
    # book add at CONSTANT total risk (2-leg adopted 0.79/1.21 = 2.0%; add tlb, renorm to 2.0%)
    base = cagr_dd(pd.concat([gold.assign(R=gold.R * 0.79), btc_k.assign(R=btc_k.R * 1.21)]))
    for wt in (0.0, 0.5, 1.0):
        wg, wb, wtlb = np.array([0.79, 1.21, wt]) / (2.0 + wt) * 2.0     # renormalize to 2.0% total
        c = cagr_dd(pd.concat([gold.assign(R=gold.R * wg), btc_k.assign(R=btc_k.R * wb),
                               tlb.assign(R=tlb.R * wtlb)]))
        tag = "(=2-leg book)" if wt == 0 else ""
        print(f"    tlb_w={wt:.1f}% (renorm 2%): book CAGR/DD={c[2]:.2f} (CAGR{c[0]:+.0f}/DD{c[1]:.0f}) {tag}")

    print("\n" + "=" * 80)
    print("VERDICT inputs: P1 plateau? P2 exit>0.39? P3 corr low & book-add? -> else FINAL non-adopted.")


if __name__ == "__main__":
    main()
