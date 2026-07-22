"""Statistical aging is real but tiny (Weibull k 1.27 obs vs 1.20 null). Translate it into the
only currency that matters: REMAINING RUN.

    Standing inside a trend leg that has already run k bars, how much move is LEFT?

  aging that matters  -> remaining run SHRINKS with age -> time stops / "it has run far enough"
                         exits / age-based size-down are levers.
  flat remaining run  -> the leg's age tells you nothing about the money left on the table, even
                         though the leg is statistically "older" -> every age-based exit is dead,
                         and the lab's "fixed far target beats cutting winners" law gets its
                         mechanism.

Measured for every bar INSIDE a leg (not just at the pivots):
    age            = bars since the leg started
    remaining_bars = bars to the leg's end
    remaining_run  = |price at the leg's end - price now| / ATR(now)     <- in risk units, so it is
                     comparable across instruments and timeframes
Bucketed by age. Reported with median AND sd (skewed distributions), plus the same measurement on
the block-shuffled null so the detector's own bias is subtracted.

NOTE this is a LOOKAHEAD measurement by construction (it uses the leg's end to score the present).
That is legitimate here: it is a description of the process, NOT a trading rule. Any rule built on
it must be re-derived causally.
Run: .venv/bin/python experiments/trend_leg_residual.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag
from trend_leg_aging import atr, block_shuffle, resample, ZZK, CELLS

ROOT = "/home/angelbell/dev/auto-trade"
NNULL = 30
AGES = [(1, 2), (3, 4), (5, 7), (8, 11), (12, 17), (18, 99)]


def residuals(h, l, c):
    """For every bar inside a leg: (age, remaining_bars, remaining_run_in_ATR, direction)."""
    a = atr(h, l, c)
    sw = swings_zigzag(h, l, a, ZZK)
    P = [(p[1], p[3]) for p in sw]
    out = []
    for (b0, k0), (b1, k1) in zip(P[:-1], P[1:]):
        if b1 <= b0: continue
        end_px = c[b1]
        for j in range(b0 + 1, b1):
            if not np.isfinite(a[j]) or a[j] <= 0: continue
            out.append((j - b0, b1 - j, abs(end_px - c[j]) / a[j], k1))
    return pd.DataFrame(out, columns=["age", "rem_bars", "rem_run", "dir"])


def table(df, tag):
    rows = []
    for lo, hi in AGES:
        g = df[(df.age >= lo) & (df.age <= hi)]
        if len(g) < 30: rows.append(None); continue
        rows.append((len(g), g.rem_run.median(), g.rem_run.mean(), g.rem_run.std(),
                     g.rem_bars.median()))
    return rows


def main():
    print("REMAINING RUN inside a trend leg, by how long the leg has already run.")
    print("rem_run = |leg's final price - price now| / ATR(now)   (risk units; comparable across cells)")
    print(f"null = the same measurement on {NNULL} block-shuffled series (detector bias subtracted)\n")

    pooled, pooled_null = [], []
    for name, path, tfs in CELLS:
        with contextlib.redirect_stderr(io.StringIO()):
            base = load_mt5_csv(os.path.join(ROOT, path))
        for tf in tfs:
            if tf == "1h": continue                     # 4h/1d only: the decision timeframes
            d = resample(base, tf)
            if len(d) < 500: continue
            h, l, c = d["high"].values, d["low"].values, d["close"].values
            r = residuals(h, l, c); r["cell"] = f"{name} {tf}"
            pooled.append(r)
            for _ in range(NNULL):
                h2, l2, c2 = block_shuffle(h, l, c)
                rn = residuals(h2, l2, c2); rn["cell"] = f"{name} {tf}"
                pooled_null.append(rn)
    P = pd.concat(pooled); N = pd.concat(pooled_null)
    print(f"pooled: {len(P)} in-leg bars from {P.cell.nunique()} cells "
          f"({len(N)} null bars)\n")

    for dirn, nm in ((+1, "UP legs"), (-1, "DOWN legs")):
        p, n = P[P.dir == dirn], N[N.dir == dirn]
        print(f"{nm}")
        print(f"  {'age (bars)':<12}{'n':>8}{'rem_run med':>13}{'mean':>8}{'sd':>7}"
              f"{'rem_bars med':>14}{'NULL rem_run med':>18}{'obs - null':>12}")
        for lo, hi in AGES:
            g = p[(p.age >= lo) & (p.age <= hi)]
            gn = n[(n.age >= lo) & (n.age <= hi)]
            if len(g) < 30: continue
            d = g.rem_run.median() - gn.rem_run.median()
            print(f"  {f'{lo}-{hi}':<12}{len(g):>8}{g.rem_run.median():>13.2f}{g.rem_run.mean():>8.2f}"
                  f"{g.rem_run.std():>7.2f}{g.rem_bars.median():>14.0f}"
                  f"{gn.rem_run.median():>18.2f}{d:>+12.2f}")
        print()

    # per-cell slope: does rem_run fall with age anywhere?
    print("per-cell: median rem_run at age 1-2 vs age 12-17  (does ANY cell show the run drying up?)")
    print(f"  {'cell':<14}{'dir':<6}{'young':>8}{'old':>8}{'old/young':>11}{'null old/young':>16}")
    for cell in sorted(P.cell.unique()):
        for dirn, nm in ((+1, "UP"), (-1, "DOWN")):
            g = P[(P.cell == cell) & (P.dir == dirn)]
            gn = N[(N.cell == cell) & (N.dir == dirn)]
            y = g[(g.age >= 1) & (g.age <= 2)].rem_run.median()
            o = g[(g.age >= 12) & (g.age <= 17)].rem_run.median()
            yn = gn[(gn.age >= 1) & (gn.age <= 2)].rem_run.median()
            on = gn[(gn.age >= 12) & (gn.age <= 17)].rem_run.median()
            if not np.isfinite(o) or not np.isfinite(y) or y <= 0: continue
            print(f"  {cell:<14}{nm:<6}{y:>8.2f}{o:>8.2f}{o/y:>11.2f}{on/yn:>16.2f}")


if __name__ == "__main__":
    main()
