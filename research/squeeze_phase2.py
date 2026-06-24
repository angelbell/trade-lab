"""squeeze_phase2.py -- Phase 2: partial SCALE-OUT exit on the BTC 4h expansion breakout.

The leg's CAGR/DD ceiling is lumpiness (RR3 = 33-38% win => clusters of losers => DD). Scale-out (sell
half at T1, move stop to breakeven, let the rest run to T2) might cut DD without killing the runners.
Trailing already died (BTC retraces shake you out); scale-out is a different mechanism.

Falsifier (up front): adopt ONLY if CAGR/DD rises AND the gain is DD-DRIVEN (maxDD down, max consec loss
down), with IS~OOS held. A scale-out that only raises return / doesn't cut DD = no improvement (keep the
clean fixed-RR3). In-sample; live-forward arbitrates.
  .venv/bin/python research/squeeze_phase2.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.squeeze_breakout import atr
from research.portfolio_kama import cagr_dd

SPLIT = 2022


def trades(d, rr1=2.0, rr2=3.0, frac=0.5, be=True, single_rr=None,
           L=120, sqz=0.25, don=30, fwd=60, cost=0.001, side="both"):
    """expansion breakout (high-ATR, no-overlap). single_rr set => plain fixed-RR (baseline).
       else scale-out: sell `frac` at rr1, move stop to BE (if be), rest to rr2."""
    a = atr(d, 14)
    rank = a.rolling(L).rank(pct=True)
    gate = (rank >= 1 - sqz).shift(1).fillna(False)
    dHi = d["high"].rolling(don).max().shift(1)
    dLo = d["low"].rolling(don).min().shift(1)
    longsig = (gate & (d["close"] > dHi)).values
    shortsig = (gate & (d["close"] < dLo)).values
    H, Lw, C = d["high"].values, d["low"].values, d["close"].values
    av = a.values
    rows = []; last = -1
    for i in range(len(d) - 1):
        if not np.isfinite(av[i]) or av[i] <= 0 or i <= last:
            continue
        isL = longsig[i] and side in ("both", "long")
        isS = shortsig[i] and side in ("both", "short")
        if not (isL or isS):
            continue
        e = C[i]; risk = av[i]; sgn = 1 if isL else -1
        stop = e - sgn * risk
        end = min(i + 1 + fwd, len(d)); R = None
        if single_rr is not None:                            # baseline fixed RR
            tp = e + sgn * single_rr * risk
            for j in range(i + 1, end):
                if (Lw[j] <= stop) if isL else (H[j] >= stop):
                    R = -1; break
                if (H[j] >= tp) if isL else (Lw[j] <= tp):
                    R = single_rr; break
            if R is None:
                R = sgn * (C[end - 1] - e) / risk; j = end - 1
        else:                                                # scale-out
            t1 = e + sgn * rr1 * risk; t2 = e + sgn * rr2 * risk
            got1 = False; realized = 0.0
            for j in range(i + 1, end):
                hit_stop = (Lw[j] <= stop) if isL else (H[j] >= stop)
                if hit_stop:                                 # stop first (conservative)
                    realized += (1 - frac if not got1 else 0.0) * (sgn * (stop - e) / risk)
                    realized += (frac * rr1) if got1 else (frac * (sgn * (stop - e) / risk))
                    R = realized; break
                if not got1 and ((H[j] >= t1) if isL else (Lw[j] <= t1)):
                    realized += frac * rr1; got1 = True
                    if be:
                        stop = e                             # move remaining stop to breakeven
                if got1 and ((H[j] >= t2) if isL else (Lw[j] <= t2)):
                    realized += (1 - frac) * rr2; R = realized; break
            if R is None:                                    # time stop: remaining half MTM
                mtm = sgn * (C[end - 1] - e) / risk
                R = realized + (1 - frac) * mtm if got1 else mtm
                j = end - 1
        rows.append((d.index[i], "L" if isL else "S", R - cost * e / risk)); last = j
    return pd.DataFrame(rows, columns=["time", "side", "R"])


def streak(R):
    m = c = 0
    for x in R:
        c = c + 1 if x < 0 else 0; m = max(m, c)
    return m


def line(tag, t):
    c, dd, cdd, ret = cagr_dd(t[["time", "R"]])
    is_ = t[t.time.dt.year < SPLIT].R.sum(); oos = t[t.time.dt.year >= SPLIT].R.sum()
    print(f"  {tag:<28} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f} meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.1f} maxDD%={dd:4.1f} CAGR/DD={cdd:5.2f} loss{streak(t.R.values):>3} "
          f"| IS={is_:+5.1f} OOS={oos:+5.1f}")


def main():
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    print("== BTC 4h expansion breakout -- Phase 2: scale-out vs fixed RR3 ==")
    print("  -- baseline --")
    line("fixed RR3 (current)", trades(d, single_rr=3.0))
    line("fixed RR2", trades(d, single_rr=2.0))
    print("  -- scale-out (half at T1 -> BE -> runner at T2) --")
    for rr1, rr2 in [(1.5, 3.0), (2.0, 3.0), (2.0, 4.0), (1.5, 4.0), (2.0, 5.0)]:
        line(f"BE  half@{rr1}->run@{rr2}", trades(d, rr1=rr1, rr2=rr2, be=True))
    print("  -- scale-out WITHOUT breakeven move (isolate the BE effect) --")
    for rr1, rr2 in [(2.0, 3.0), (2.0, 4.0)]:
        line(f"noBE half@{rr1}->run@{rr2}", trades(d, rr1=rr1, rr2=rr2, be=False))
    print("\n  -- plateau on the runner target for the best BE config --")
    for rr2 in (3.0, 3.5, 4.0, 4.5):
        line(f"BE half@2.0->run@{rr2}", trades(d, rr1=2.0, rr2=rr2, be=True))
    print("\n  verdict: adopt ONLY if CAGR/DD up AND maxDD/loss-streak down (DD-driven), IS~OOS held.")


if __name__ == "__main__":
    main()
