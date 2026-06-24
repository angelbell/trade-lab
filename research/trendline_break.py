"""trendline_break.py -- faithful mechanization of "Trendlines with Breaks [LuxAlgo]".

Auto DIAGONAL trendline breakout: anchor a line at the last fractal pivot (high/low), slope it by
ATR(length)/length*mult, fire a BULL break when close crosses above the descending (resistance) line,
BEAR break when close crosses below the ascending (support) line. Causal: the break test uses only
past/current bars (pivots are length-lagged, normal for any swing system; backpaint = display only).

This is a BREAKOUT-family signal (our surviving family), so it earns a real test (unlike the dead
trend-flip family). Generator only -> we bolt on next-bar-open entry + RR exit + cost and run the
falsification checklist (mfe/mae screen, beta split long/short, per-year, IS/OOS) and compare to
breakout_wave. Prior: CLAUDE.md already found trendline DETECTORS ~0 lift; the lever is exit/gate.

  .venv/bin/python research/trendline_break.py --csv data/vantage_xauusd_h1.csv --tf 4h
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample


def _slope(d, length, mult, method):
    C = d["close"]
    if method == "Atr":
        return (ta.atr(d["high"], d["low"], C, length=length) / length * mult).values
    if method == "Stdev":
        return (C.rolling(length).std(ddof=0) / length * mult).values
    if method == "Linreg":
        idx = pd.Series(np.arange(len(C)), index=C.index)
        cov = (C * idx).rolling(length).mean() - C.rolling(length).mean() * idx.rolling(length).mean()
        var = idx.rolling(length).var(ddof=0)
        return (cov.abs() / var / 2 * mult).values
    raise ValueError(method)


def signals(d, length, mult, method="Atr"):
    H = d["high"].values; Lo = d["low"].values; C = d["close"].values
    atr = _slope(d, length, mult, method)
    n = len(C)
    rmaxH = pd.Series(H).rolling(2 * length + 1, center=True).max().values
    rminL = pd.Series(Lo).rolling(2 * length + 1, center=True).min().values
    ph_val = np.full(n, np.nan); pl_val = np.full(n, np.nan)
    for j in range(length, n - length):
        if H[j] == rmaxH[j]:
            ph_val[j + length] = H[j]                 # confirmed length bars later
        if Lo[j] == rminL[j]:
            pl_val[j + length] = Lo[j]
    upper = lower = 0.0; sph = spl = 0.0; upos = dnos = 0
    upb = np.zeros(n, bool); dnb = np.zeros(n, bool)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        ph = not np.isnan(ph_val[i]); pl = not np.isnan(pl_val[i])
        if ph: sph = atr[i]
        if pl: spl = atr[i]
        upper = ph_val[i] if ph else upper - sph
        lower = pl_val[i] if pl else lower + spl
        pu, pd_ = upos, dnos
        upos = 0 if ph else (1 if C[i] > upper - sph * length else upos)
        dnos = 0 if pl else (1 if C[i] < lower + spl * length else dnos)
        upb[i] = upos > pu; dnb[i] = dnos > pd_
    return upb, dnb


def screen(d, length, mult, fwd):
    upb, dnb = signals(d, length, mult)
    C = d["close"].values; H = d["high"].values; Lo = d["low"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    n = len(C)
    def mm(sig, dr):
        mfe = []; mae = []
        for i in np.where(sig)[0]:
            if i + 1 >= n or np.isnan(atr[i + 1]) or atr[i + 1] == 0:
                continue
            e = C[i]; a = atr[i + 1]; hh = H[i + 1:i + 1 + fwd]; ll = Lo[i + 1:i + 1 + fwd]
            if len(hh) == 0:
                continue
            if dr > 0:
                mfe.append((hh.max() - e) / a); mae.append((e - ll.min()) / a)
            else:
                mfe.append((e - ll.min()) / a); mae.append((hh.max() - e) / a)
        mfe = np.array(mfe); mae = np.clip(np.array(mae), 1e-9, None)
        return len(mfe), mfe.mean(), mae.mean(), mfe.mean() / mae.mean()
    for tag, sig, dr in [("bull break(long)", upb, 1), ("bear break(short)", dnb, -1)]:
        nn, fe, ae, r = mm(sig, dr)
        flag = "DEAD" if r < 1.0 else ("marg" if r < 1.2 else "WORTH")
        print(f"    {tag:<20} n={nn:>5}  MFE={fe:.2f} MAE={ae:.2f}  MFE/MAE={r:.2f} [{flag}]")


def trades(d, length, mult, rr, sl_atr, cost_usd):
    upb, dnb = signals(d, length, mult)
    op = d["open"].values; H = d["high"].values; Lo = d["low"].values; C = d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    tm = d.index; n = len(C)
    sigs = sorted([(i, 1) for i in np.where(upb)[0]] + [(i, -1) for i in np.where(dnb)[0]])
    rows = []; busy = -1
    for i, dr in sigs:
        ei = i + 1
        if ei <= busy or ei >= n or np.isnan(atr[ei]) or atr[ei] == 0:
            continue
        e = op[ei]; risk = sl_atr * atr[ei]
        stop = e - dr * risk; tgt = e + dr * rr * risk; R = None
        for j in range(ei, min(ei + 400, n)):
            if dr > 0:
                if Lo[j] <= stop: R = -1.0; break
                if H[j] >= tgt: R = rr; break
            else:
                if H[j] >= stop: R = -1.0; break
                if Lo[j] <= tgt: R = rr; break
        if R is None:
            R = dr * (C[min(ei + 400, n - 1)] - e) / risk
            j = min(ei + 400, n - 1)
        R -= cost_usd / risk
        rows.append((tm[ei], R, dr)); busy = j
    return pd.DataFrame(rows, columns=["time", "R", "dir"])


def stat(t, tag, rr):
    if len(t) < 20:
        print(f"  {tag:<22} n={len(t)}"); return
    w = t.R[t.R > 0].sum(); l = t.R[t.R < 0].sum(); pf = w / abs(l) if l else 9.9
    be = 100 / (1 + rr)
    isr = t[t.time.dt.year < 2022].R; oos = t[t.time.dt.year >= 2022].R
    print(f"  {tag:<22} n={len(t):>5} win={(t.R>0).mean()*100:>3.0f}%(be{be:.0f}) PF={pf:4.2f} "
          f"meanR={t.R.mean():+.3f} | IS={isr.mean():+.3f} OOS={oos.mean():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_xauusd_h1.csv")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--length", type=int, default=14)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--sl-atr", type=float, default=1.5)
    ap.add_argument("--cost", type=float, default=0.30)
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    print(f"\n=== Trendline-Breaks  {os.path.basename(a.csv)} {a.tf} {d.index[0].date()}->{d.index[-1].date()}"
          f"  len{a.length} mult{a.mult} RR{a.rr} SL{a.sl_atr}ATR cost{a.cost} ===")
    print("  -- 1. mfe/mae SCREEN (>1.2 worth, <1.0 dead) --")
    screen(d, a.length, a.mult, fwd=100)
    print(f"  -- 2. RR{a.rr} strategy + BETA split --")
    t = trades(d, a.length, a.mult, a.rr, a.sl_atr, a.cost)
    stat(t, "all", a.rr); stat(t[t.dir == 1], "LONG", a.rr); stat(t[t.dir == -1], "SHORT", a.rr)
    print("  -- length sweep (plateau?) --")
    for L in (8, 14, 21, 34):
        stat(trades(d, L, a.mult, a.rr, a.sl_atr, a.cost), f"len={L}", a.rr)


if __name__ == "__main__":
    main()
