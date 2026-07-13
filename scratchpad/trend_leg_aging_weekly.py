"""Do trend legs age AT THE WEEKLY SCALE?  (the scale the user actually asked about)

The 4h/1d answer was "no" (26 cells: statistical aging +0.07 in Weibull shape, but the REMAINING
RUN is flat with age once the detector's own bias is subtracted). That was an INFERENCE for the
weekly scale, not a measurement -- BTC has only ~20 weekly legs in 9 years, so it cannot decide.

The N lives in FX: weekly bars pulled fresh from the MT5 bridge go back to 1993-94 (2000+ used
here to avoid the synthetic pre-euro EURUSD and the fixed-rate yen). 6 pairs x 26 years -> ~1000
pooled weekly legs. gold/BTC weekly (9 yr each) are carried along as the trend-instrument
reference, with their small n stated.

Same machinery as trend_leg_aging.py:
  detector : ZigZag 2xATR on weekly bars
  statistic: Weibull shape k of leg durations (k>1 aging, k=1 memoryless)
  null     : THE SAME DETECTOR on block-shuffled weekly bars -> reproduces the detector's
             minimum-leg bias, so only real memory survives
  plus     : remaining run (|leg end - now| / ATR) by leg age, observed vs null -- the money test
Run: .venv/bin/python scratchpad/trend_leg_aging_weekly.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
from trend_leg_aging import atr, block_shuffle, shape_k, legs_from
from trend_leg_residual import residuals

ROOT = "/home/angelbell/dev/auto-trade"
NNULL = 200
FX = ["eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdjpy"]
AGES = [(1, 2), (3, 4), (5, 7), (8, 11), (12, 17), (18, 99)]


def main():
    cells = [(s.upper(), f"data/vantage_{s}_w1.csv", "2000-01-01") for s in FX]
    cells += [("gold", "data/vantage_xauusd_w1.csv", None),
              ("BTC", "data/vantage_btcusd_w1.csv", None)]

    print("WEEKLY trend legs: do they age?   ZigZag 2xATR on weekly bars")
    print(f"null = the same detector on {NNULL} block-shuffled weekly series\n")
    print(f"{'cell':<10}{'from':<9}{'wks':>6}{'dir':<6}{'legs':>6}{'med':>5}{'sd':>6}"
          f"{'k_obs':>7}{'k_null':>8}{'null95':>8}{'p':>7}  verdict")
    obs_all, null_all, rows = [], [], []
    for name, path, start in cells:
        with contextlib.redirect_stderr(io.StringIO()):
            d = load_mt5_csv(os.path.join(ROOT, path))
        if start: d = d.loc[start:]
        h, l, c = d["high"].values, d["low"].values, d["close"].values
        if len(c) < 200: continue
        up, dn = legs_from(h, l, c)
        r = residuals(h, l, c); r["cell"] = name; obs_all.append(r)
        nu, nd = [], []
        for _ in range(NNULL):
            h2, l2, c2 = block_shuffle(h, l, c)
            u2, d2 = legs_from(h2, l2, c2)
            nu.append(shape_k(u2)); nd.append(shape_k(d2))
            if len(null_all) < 40 * NNULL:
                rn = residuals(h2, l2, c2); rn["cell"] = name; null_all.append(rn)
        for dirn, L, nl in (("UP", up, np.array(nu)), ("DOWN", dn, np.array(nd))):
            ko = shape_k(L); nn = nl[np.isfinite(nl)]
            if not np.isfinite(ko) or len(nn) < 30: continue
            p = float((nn >= ko).mean())
            v = "AGING" if p < 0.05 else ("younger-dies" if p > 0.95 else "memoryless")
            print(f"{name:<10}{str(d.index[0].date()):<9}{len(c):>6}{dirn:<6}{len(L):>6}"
                  f"{np.median(L):>5.0f}{L.std():>6.1f}{ko:>7.2f}{np.median(nn):>8.2f}"
                  f"{np.percentile(nn,95):>8.2f}{p:>7.3f}  {v}")
            rows.append(dict(cell=name, dir=dirn, n=len(L), k=ko, knull=np.median(nn), p=p, v=v))

    R = pd.DataFrame(rows)
    print(f"\nSUMMARY over {len(R)} weekly cells:")
    print(f"  AGING (p<0.05): {(R.p < 0.05).sum()}   memoryless: {((R.p>=0.05)&(R.p<=0.95)).sum()}"
          f"   younger-dies: {(R.p > 0.95).sum()}   (chance -> ~{0.05*len(R):.1f} aging)")
    print(f"  k_obs mean {R.k.mean():.2f} vs k_null mean {R.knull.mean():.2f} "
          f"(median excess {np.median(R.k - R.knull):+.3f})")
    print(f"  total weekly legs measured: {R.n.sum()}")

    # ---- ERA SPLIT (the user's objection, made testable) ---------------------
    # "the 1990s were a different world, and any real edge gets mined out" -> if trend legs USED to
    # age and the market learned to exit old trends, aging should be present early and gone now.
    # That is a measurable claim, not an assumption. Same detector, same null, per era.
    print("\nERA SPLIT -- did legs age in the old world and stop aging in the new one?")
    eras = [("1993-1999", "1993-01-01", "2000-01-01"), ("2000-2008", "2000-01-01", "2009-01-01"),
            ("2009-2017", "2009-01-01", "2018-01-01"), ("2018-2026", "2018-01-01", "2027-01-01")]
    print(f"  {'era':<11}{'dir':<6}{'legs':>6}{'med':>5}{'k_obs':>8}{'k_null':>8}{'excess':>8}{'p':>7}")
    for ename, e0, e1 in eras:
        agg = {"UP": [], "DOWN": []}
        for s in FX:
            with contextlib.redirect_stderr(io.StringIO()):
                d = load_mt5_csv(os.path.join(ROOT, f"data/vantage_{s}_w1.csv"))
            d = d.loc[e0:e1]
            if len(d) < 150: continue
            h, l, c = d["high"].values, d["low"].values, d["close"].values
            up, dn = legs_from(h, l, c)
            nu, nd = [], []
            for _ in range(60):
                h2, l2, c2 = block_shuffle(h, l, c)
                u2, d2 = legs_from(h2, l2, c2)
                nu.append(shape_k(u2)); nd.append(shape_k(d2))
            for dirn, L, nl in (("UP", up, np.array(nu)), ("DOWN", dn, np.array(nd))):
                ko = shape_k(L); nn = nl[np.isfinite(nl)]
                if np.isfinite(ko) and len(nn) >= 20:
                    agg[dirn].append((len(L), np.median(L), ko, np.median(nn),
                                      float((nn >= ko).mean())))
        for dirn in ("UP", "DOWN"):
            a = agg[dirn]
            if not a: continue
            n = sum(x[0] for x in a)
            med = np.mean([x[1] for x in a])
            ko = np.mean([x[2] for x in a]); kn = np.mean([x[3] for x in a])
            # pooled p: Fisher-style count of how many pairs individually clear p<0.05
            sig = sum(1 for x in a if x[4] < 0.05)
            print(f"  {ename:<11}{dirn:<6}{n:>6}{med:>5.0f}{ko:>8.2f}{kn:>8.2f}{ko-kn:>+8.3f}"
                  f"   {sig}/{len(a)} pairs p<.05")

    # ---- the money test: remaining run by age, pooled over FX ---------------
    P = pd.concat(obs_all); N = pd.concat(null_all)
    P, N = P[P.cell.isin([s.upper() for s in FX])], N[N.cell.isin([s.upper() for s in FX])]
    print(f"\nREMAINING RUN by leg age (weekly, 6 FX pooled; {len(P)} in-leg weeks)")
    print("  rem_run = |price at the leg's end - price now| / ATR(now)")
    for dirn, nm in ((+1, "UP legs"), (-1, "DOWN legs")):
        p, n = P[P.dir == dirn], N[N.dir == dirn]
        print(f"\n  {nm}")
        print(f"    {'age (weeks)':<13}{'n':>7}{'rem_run med':>13}{'mean':>8}{'sd':>7}"
              f"{'rem_wks med':>13}{'NULL med':>10}{'obs-null':>10}")
        for lo, hi in AGES:
            g = p[(p.age >= lo) & (p.age <= hi)]
            gn = n[(n.age >= lo) & (n.age <= hi)]
            if len(g) < 30: continue
            print(f"    {f'{lo}-{hi}':<13}{len(g):>7}{g.rem_run.median():>13.2f}"
                  f"{g.rem_run.mean():>8.2f}{g.rem_run.std():>7.2f}{g.rem_bars.median():>13.0f}"
                  f"{gn.rem_run.median():>10.2f}{g.rem_run.median()-gn.rem_run.median():>+10.2f}")


if __name__ == "__main__":
    main()
