"""ml_supertrend.py -- mechanize AlgoAlpha's "ML Adaptive SuperTrend" to falsify it.

The "ML" is K-means (3 clusters) on the last `train` ATR values -> snaps the current
ATR to its cluster centroid (low/med/high vol) and feeds THAT into a standard
SuperTrend. SuperTrend is a stop-and-reverse trend-follower: long while price is above
the line, short below, flip when crossed.

We trade the flips (next-bar-open fill, always-in unless --side long), 1R = the
entry-to-SuperTrend distance (the line IS the trailing stop). The KEY test is the A/B:
does the K-means adaptive ATR beat a PLAIN fixed-ATR SuperTrend (--plain)? Per project
priors, adaptive "detectors" usually add ~0 lift.

  .venv/bin/python ml_supertrend.py --csv data/vantage_btcusd_h1.csv --tf 4h --peryear
  .venv/bin/python ml_supertrend.py --csv data/vantage_btcusd_h1.csv --tf 4h --plain --peryear
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv


def resample(df, rule):
    if rule.lower() in ("1h", "h1", ""):
        return df
    o = {"4h": "4h", "h4": "4h", "1d": "1D", "d1": "1D", "2h": "2h", "12h": "12h"}.get(rule.lower(), rule)
    return pd.DataFrame({"open": df["open"].resample(o).first(), "high": df["high"].resample(o).max(),
                         "low": df["low"].resample(o).min(), "close": df["close"].resample(o).last()}).dropna()


def adaptive_atr(atr, train, hi_g, mid_g, lo_g, max_iter=30):
    """per-bar K-means(3) on the last `train` ATRs -> current ATR's cluster centroid."""
    n = len(atr)
    out = np.copy(atr)
    for i in range(train - 1, n):
        w = atr[i - train + 1:i + 1]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        lo, hi = w.min(), w.max()
        c = np.array([lo + (hi - lo) * hi_g, lo + (hi - lo) * mid_g, lo + (hi - lo) * lo_g])
        for _ in range(max_iter):
            d = np.abs(w[:, None] - c[None, :])
            lab = d.argmin(axis=1)
            newc = np.array([w[lab == k].mean() if np.any(lab == k) else c[k] for k in range(3)])
            if np.allclose(newc, c):
                c = newc; break
            c = newc
        out[i] = c[np.abs(atr[i] - c).argmin()]   # snap current ATR to nearest centroid
    return out


def supertrend(h, l, c, atr_series, factor):
    n = len(c)
    src = (h + l) / 2.0
    upper = src + factor * atr_series
    lower = src - factor * atr_series
    st = np.full(n, np.nan); direction = np.zeros(n, np.int8)
    for i in range(n):
        if i == 0 or np.isnan(atr_series[i - 1]):
            direction[i] = 1; st[i] = upper[i]; continue
        pl, pu = lower[i - 1], upper[i - 1]
        lower[i] = lower[i] if (lower[i] > pl or c[i - 1] < pl) else pl
        upper[i] = upper[i] if (upper[i] < pu or c[i - 1] > pu) else pu
        if st[i - 1] == pu:
            direction[i] = -1 if c[i] > upper[i] else 1
        else:
            direction[i] = 1 if c[i] < lower[i] else -1
        st[i] = lower[i] if direction[i] == -1 else upper[i]
    return st, direction


def run(d, args):
    h, l, c, o = (d[x].values for x in ("high", "low", "close", "open"))
    atr = ta.atr(d["high"], d["low"], d["close"], length=args.atr_len).values
    aatr = atr if args.plain else adaptive_atr(atr, args.train, args.hi, args.mid, args.lo)
    st, direction = supertrend(h, l, c, aatr, args.factor)
    sma = d["close"].rolling(args.sma).mean().values if args.sma > 0 else None   # regime filter
    # dir == -1 bullish (price above line), +1 bearish. trade the flips, next-bar-open fill.
    n = len(c); cost = args.cost
    trades = []
    pos = 0; e_px = e_risk = 0.0; ei = 0
    for i in range(n - 1):
        flip = direction[i] != direction[i - 1] if i > 0 else False
        want = -direction[i]                                   # long(+1) when dir==-1
        if args.side == "long" and want < 0:
            want = 0
        if args.side == "short" and want > 0:
            want = 0
        if sma is not None and not np.isnan(sma[i]):            # SMA regime: skip counter-regime
            if (want > 0 and c[i] <= sma[i]) or (want < 0 and c[i] >= sma[i]):
                want = 0
        if flip and want != pos:
            if pos != 0:                                       # close at next open
                ret = (o[i + 1] - e_px) / e_px if pos > 0 else (e_px - o[i + 1]) / e_px
                R = (ret - cost) / e_risk if e_risk > 0 else np.nan
                trades.append((d.index[ei], pos, R, i - ei))
            if want != 0:                                      # open at next open
                e_px = o[i + 1]; ei = i + 1; pos = want
                e_risk = abs(e_px - st[i]) / e_px              # entry-to-line = 1R
            else:
                pos = 0
    t = pd.DataFrame(trades, columns=["t", "dir", "R", "bars"]).dropna()
    if len(t) == 0:
        print("  no trades"); return
    wins = t.R[t.R > 0]; loss = t.R[t.R < 0]
    pf = wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() else float("inf")
    eq = (1 + args.risk * t.R).cumprod(); dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    span = max((t.t.iloc[-1] - t.t.iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100
    t["y"] = t.t.dt.year; yrs = sorted(t.y.unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t.R[t.y < half] if half else t.R; oos = t.R[t.y >= half] if half else t.R
    print(f"  n={len(t):>4} win={(t.R>0).mean()*100:>3.0f}% PF={pf:4.2f} meanR={t.R.mean():+.2f} "
          f"totR={t.R.sum():+5.0f} | IS={isr.mean():+.2f} OOS={oos.mean():+.2f} | "
          f"CAGR={cagr:+5.1f}% maxDD={dd:4.1f}% retDD={ (eq.iloc[-1]-1)*100/max(dd,1e-9):.2f}")
    if args.peryear:
        print("      per-year totR: " + " ".join(f"{y}:{g.R.sum():+.0f}({len(g)})" for y, g in t.groupby("y")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--atr-len", type=int, default=10)
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--hi", type=float, default=0.75)
    ap.add_argument("--mid", type=float, default=0.50)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--plain", action="store_true", help="A/B: plain fixed-ATR SuperTrend (no K-means)")
    ap.add_argument("--side", default="both", choices=["both", "long", "short"])
    ap.add_argument("--sma", type=int, default=0, help="SMA regime filter: skip flips against close-vs-SMA (0=off)")
    ap.add_argument("--cost", type=float, default=0.001, help="round-trip cost fraction")
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--peryear", action="store_true")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    if a.start: d = d.loc[a.start:]
    if a.end: d = d.loc[:a.end]
    print(f"\n=== {os.path.basename(a.csv)} {a.tf} {'PLAIN' if a.plain else 'ADAPTIVE(kmeans)'} "
          f"atr{a.atr_len} fac{a.factor} side={a.side} cost{a.cost} {d.index[0]}->{d.index[-1]} ===")
    run(d, a)


if __name__ == "__main__":
    main()
