"""squeeze_phase3.py -- Phase 3: SHORT-side decision on the scale-out expansion breakout.

Short was the weak component (94%ile vs beta on fixed-RR3). Decide: keep long+short, or go long-only?
Decision rule (up front): keep shorts ONLY if (a) long+short CAGR/DD > long-only AND (b) short-only
beats the random-short null >=95%ile. BUT also weigh the PORTFOLIO angle -- a standalone-weak short can
still earn its place if it DECORRELATES the (all-long) book. The real arbiter = which leg version lifts
the BOOK's CAGR/DD more. (Phase 1 showed the edge is direction-agnostic, so a downtrend gate on shorts is
expected to fail; tested for completeness.) In-sample; live-forward arbitrates.
  .venv/bin/python research/squeeze_phase3.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from research.squeeze_breakout import atr
from research.squeeze_phase2 import trades, streak
from research.portfolio_kama import get_legs, cagr_dd
from research.portfolio_alloc import monthly_matrix, cagr_dd_monthly

SPLIT = 2022
SO = dict(rr1=2.0, rr2=4.0, be=False)        # the adopted scale-out exit


def scaleout_R(H, Lw, C, av, i, side, rr1=2.0, rr2=4.0, frac=0.5, fwd=60, cost=0.001):
    e = C[i]; risk = av[i]; sgn = 1 if side == "L" else -1
    stop = e - sgn * risk; t1 = e + sgn * rr1 * risk; t2 = e + sgn * rr2 * risk
    got1 = False; realized = 0.0; end = min(i + 1 + fwd, len(C))
    for j in range(i + 1, end):
        hit = (Lw[j] <= stop) if side == "L" else (H[j] >= stop)
        if hit:
            realized += (frac * rr1) if got1 else (frac * (sgn * (stop - e) / risk))
            realized += 0.0 if got1 else (1 - frac) * (sgn * (stop - e) / risk)
            return realized - cost * e / risk
        if not got1 and ((H[j] >= t1) if side == "L" else (Lw[j] <= t1)):
            realized += frac * rr1; got1 = True
        if got1 and ((H[j] >= t2) if side == "L" else (Lw[j] <= t2)):
            return realized + (1 - frac) * rr2 - cost * e / risk
    mtm = sgn * (C[end - 1] - e) / risk
    return (realized + (1 - frac) * mtm if got1 else mtm) - cost * e / risk


def rand_null(d, side, n_real, real_meanR, iters=2000, seed=0):
    a = atr(d, 14).values
    H, Lw, C = d["high"].values, d["low"].values, d["close"].values
    valid = np.where(np.isfinite(a) & (a > 0))[0]; valid = valid[valid < len(d) - 61]
    rng = np.random.default_rng(seed)
    means = np.array([np.mean([scaleout_R(H, Lw, C, a, i, side) for i in rng.choice(valid, n_real, replace=False)])
                      for _ in range(iters)])
    return means.mean(), (means < real_meanR).mean() * 100


def info(tag, t):
    c, dd, cdd, _ = cagr_dd(t[["time", "R"]])
    by = t.groupby(t.time.dt.year).R.sum()
    print(f"  {tag:<16} n={len(t):>4} meanR={t.R.mean():+5.2f} totR={t.R.sum():>+6.1f} maxDD%={dd:4.1f} "
          f"CAGR/DD={cdd:5.2f} | IS={t[t.time.dt.year<SPLIT].R.sum():+5.1f} OOS={t[t.time.dt.year>=SPLIT].R.sum():+5.1f} "
          f"green {int((by>0).sum())}/{len(by)}")
    return t


def main():
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    print("== Phase 3: short-side decision (scale-out 2R/4R exit) ==")
    both = info("long+short", trades(d, side="both", **SO))
    lon = info("long-only", trades(d, side="long", **SO))
    sh = info("short-only", trades(d, side="short", **SO))

    print("\n  -- (b) does short-only beat the random-SHORT null (>=95%ile)? --")
    nm, pct = rand_null(d, "S", len(sh), sh.R.mean())
    print(f"     short real meanR={sh.R.mean():+.2f} vs random-short mean={nm:+.2f} -> {pct:.0f}%ile "
          f"[{'BEATS beta' if pct>=95 else 'marginal/fails'}]")

    print("\n  -- downtrend-gated shorts (Phase1 says direction-agnostic -> expect no help) --")
    a = atr(d, 14)
    dc = d["close"].resample("1D").last().dropna()
    sma = dc.rolling(50).mean()
    down = ((sma < sma.shift(1)).shift(1)).reindex(d.index, method="ffill").fillna(False)
    sh2 = trades(d, side="short", **SO)
    tt = pd.DatetimeIndex(sh2.time.values)
    if tt.tz is None and down.index.tz is not None:
        tt = tt.tz_localize(down.index.tz)
    keep = down.values[np.clip(down.index.searchsorted(tt, side="right") - 1, 0, len(down) - 1)]
    info("short|SMA50-down", sh2[keep])

    print("\n  -- (portfolio) which leg version lifts the BOOK more? (inv-vol, total 2% risk) --")
    legs = get_legs()
    def book_with(expt):
        names = {"gold_bo": legs["gold_bo"], "btc_bo_kama": legs["btc_bo_kama"],
                 "btc_pull": legs["btc_pull"], "exp": expt[["time", "R"]]}
        M = monthly_matrix(names)
        inv = {k: 1 / M[k].std() for k in names}; s = sum(inv.values())
        w = {k: inv[k] / s * 0.02 for k in names}
        blend = sum(M[k] * w[k] for k in names)
        return cagr_dd_monthly(blend), w["exp"] * 100, M["exp"]
    (r_b, _, _), w_b, mb = book_with(both)
    (r_l, _, _), w_l, ml = book_with(lon)
    # correlation of each version with btc_bo_kama (decorrelation value of shorts)
    Mk = monthly_matrix({"btc_bo_kama": legs["btc_bo_kama"]})["btc_bo_kama"]
    cb = pd.concat([mb, Mk], axis=1).dropna().corr().iloc[0, 1]
    cl = pd.concat([ml, Mk], axis=1).dropna().corr().iloc[0, 1]
    print(f"     book + long+short exp : CAGR/DD={r_b:.2f}  (exp wt {w_b:.2f}%, corr w/ btc_bo_kama {cb:+.2f})")
    print(f"     book + long-only  exp : CAGR/DD={r_l:.2f}  (exp wt {w_l:.2f}%, corr w/ btc_bo_kama {cl:+.2f})")
    print("\n  verdict: keep shorts if they lift CAGR/DD (standalone or via book decorrelation); else long-only.")


if __name__ == "__main__":
    main()
