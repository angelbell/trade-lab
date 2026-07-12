"""ma200_bounce.py -- "touch-and-hold" 200EMA bounce screener (all-signals base).

The user's proposed rule, mechanised faithfully (NOT the pierce-and-reclaim that
ema_pullback.py does -- this is the "wick touches the 200EMA, body closes back
above = rejection candle" reading the user picked):

  Trend    : 200EMA rising (ema[i] > ema[i-slopeK]) AND price above it (uptrend).
  Touch    : the bar's LOW reaches down to within tol*ATR of the 200EMA
             (low <= ema + tol*ATR -- a near-touch or shallow pierce) ...
  Hold     : ... but the CLOSE stays above the 200EMA and the bar is bullish
             (close > ema and close > open) = the line is defended = 反発確認.
  Entry    : the NEXT bar's OPEN (confirm-then-enter; no lookahead).
  Stop     : just below the touch bar's low (the dip extreme), ATR-floored.
  Target   : entry + rr * stop-distance.  Mirror everything for shorts.

Reports per (instrument, TF, side): n, win%, meanR, totR, IS/OOS split, per-year
green count, and a BETA-NULL percentile (does it beat a random long taken in the
SAME uptrend bars with the SAME stop-distance & RR? <~90%ile => just trend beta).

Run:
  .venv/bin/python research/ma200_bounce.py
"""
import os, sys
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv

RNG = np.random.default_rng(42)


def resample(df, rule):
    if rule.lower() in ("1h", "h1", ""):
        return df
    o = {"2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D"}.get(rule.lower(), rule)
    return pd.DataFrame({
        "open": df["open"].resample(o).first(), "high": df["high"].resample(o).max(),
        "low": df["low"].resample(o).min(), "close": df["close"].resample(o).last(),
    }).dropna()


def find_signals(d, side, emalen=200, slopeK=20, tol=0.25, atrlen=14):
    """Return list of (i, entry_px, stop_px) for touch-and-hold bounces.
    entry is the NEXT bar's open; i is the entry bar index (signal = i-1)."""
    ema = d["close"].ewm(span=emalen, adjust=False).mean().values
    a = ta.atr(d["high"], d["low"], d["close"], length=atrlen).values
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    out = []
    for s in range(slopeK + 1, len(c) - 1):           # s = signal bar; enter at s+1 open
        if np.isnan(ema[s]) or np.isnan(a[s]) or a[s] <= 0:
            continue
        if side == "long":
            rising = ema[s] > ema[s - slopeK]
            touch = l[s] <= ema[s] + tol * a[s]
            hold = c[s] > ema[s] and c[s] > o[s]
            if rising and touch and hold:
                e = o[s + 1]; stop = l[s]
                if e - stop < 0.5 * a[s]:
                    stop = e - 0.5 * a[s]
                if e > stop:
                    out.append((s + 1, e, stop))
        else:
            falling = ema[s] < ema[s - slopeK]
            touch = h[s] >= ema[s] - tol * a[s]
            hold = c[s] < ema[s] and c[s] < o[s]
            if falling and touch and hold:
                e = o[s + 1]; stop = h[s]
                if stop - e < 0.5 * a[s]:
                    stop = e + 0.5 * a[s]
                if stop > e:
                    out.append((s + 1, e, stop))
    return out, ema, a


def simulate(d, sigs, side, rr, fwd, cost):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    rows, busy = [], -1
    for (i, e, stop) in sigs:
        if i <= busy:
            continue
        risk = abs(e - stop)
        if risk <= 0:
            continue
        exit_j = min(i + fwd, len(c) - 1); R = None
        if side == "long":
            tgt = e + rr * risk
            for j in range(i, min(i + fwd, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt: R = rr; exit_j = j; break
            if R is None: R = (c[exit_j] - e) / risk
        else:
            tgt = e - rr * risk
            for j in range(i, min(i + fwd, len(c))):
                if h[j] >= stop: R = -1.0; exit_j = j; break
                if l[j] <= tgt: R = rr; exit_j = j; break
            if R is None: R = (e - c[exit_j]) / risk
        R -= cost / risk * e
        rows.append((d.index[i], R)); busy = exit_j
    return pd.DataFrame(rows, columns=["time", "R"])


def beta_null(d, side, sigs, rr, fwd, cost, ema, a, n_iter=400):
    """Random long/short taken in the SAME trend-context bars (ema rising/falling
    & price the right side) with the SAME per-trade stop-distance (in ATR units)
    as the real signals. Percentile of the real meanR vs this null = is it edge
    or just trend beta?"""
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    if side == "long":
        ctx = np.where((ema[:-1] > np.roll(ema, 20)[:-1]) & (c[:-1] > ema[:-1]))[0]
    else:
        ctx = np.where((ema[:-1] < np.roll(ema, 20)[:-1]) & (c[:-1] < ema[:-1]))[0]
    ctx = ctx[(ctx > 20) & (ctx < len(c) - fwd - 1)]
    if len(ctx) < 30 or not sigs:
        return np.nan
    stop_atr = [abs(e - st) / a[i] for (i, e, st) in sigs if a[i] > 0]   # risk in ATR
    real = simulate(d, sigs, side, rr, fwd, cost)["R"].mean()
    means = []
    for _ in range(n_iter):
        picks = RNG.choice(ctx, size=len(sigs), replace=True)
        sd = RNG.choice(stop_atr, size=len(sigs), replace=True)
        rs = []
        for i, satr in zip(picks, sd):
            e = d["open"].values[i]; risk = satr * a[i]
            if risk <= 0: continue
            R = None
            if side == "long":
                stop, tgt = e - risk, e + rr * risk
                for j in range(i, min(i + fwd, len(c))):
                    if l[j] <= stop: R = -1.0; break
                    if h[j] >= tgt: R = rr; break
                if R is None: R = (c[min(i + fwd, len(c) - 1)] - e) / risk
            else:
                stop, tgt = e + risk, e - rr * risk
                for j in range(i, min(i + fwd, len(c))):
                    if h[j] >= stop: R = -1.0; break
                    if l[j] <= tgt: R = rr; break
                if R is None: R = (e - c[min(i + fwd, len(c) - 1)]) / risk
            rs.append(R - cost / risk * e)
        if rs: means.append(np.mean(rs))
    means = np.array(means)
    return (real > means).mean() * 100 if len(means) else np.nan


def stats(t):
    if len(t) == 0:
        return None
    t = t.copy(); t["y"] = t["time"].dt.year
    yrs = sorted(t["y"].unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t[t["y"] < half]["R"] if half else t["R"]
    oosr = t[t["y"] >= half]["R"] if half else t["R"]
    green = sum(1 for _, g in t.groupby("y") if g["R"].sum() > 0)
    return dict(n=len(t), win=(t.R > 0).mean() * 100, meanR=t.R.mean(), totR=t.R.sum(),
                IS=isr.mean(), OOS=oosr.mean(), green=green, nyr=t.y.nunique())


INSTRUMENTS = [
    ("GOLD", "data/vantage_xauusd_h1.csv"),
    ("BTC", "data/vantage_btcusd_h1.csv"),
    ("USDJPY", "data/vantage_usdjpy_h1.csv"),
    ("EURUSD", "data/vantage_eurusd_h1.csv"),
]
TFS = ["1h", "2h", "4h", "8h", "1d"]
RR = 2.0
FWD = 200
COST = 0.001


def main():
    print(f"\n=== 200EMA touch-and-hold bounce  (RR={RR}, tol=0.25ATR, slopeK=20, cost={COST}) ===")
    print("breakeven win% @RR2 = 33%. beta%ile<~90 => trend beta, not edge.\n")
    hdr = f"{'inst':<7}{'TF':<5}{'side':<6}{'n':>5}{'win%':>6}{'meanR':>8}{'totR':>8}{'IS':>7}{'OOS':>7}{'grn':>6}{'beta%':>7}"
    for name, csv in INSTRUMENTS:
        raw = load_mt5_csv(csv)
        print(f"\n----- {name} ({raw.index[0].date()}->{raw.index[-1].date()}) -----")
        print(hdr)
        for tf in TFS:
            d = resample(raw, tf)
            if len(d) < 400:
                continue
            for side in ("long", "short"):
                sigs, ema, a = find_signals(d, side)
                t = simulate(d, sigs, side, RR, FWD, COST)
                st = stats(t)
                if st is None or st["n"] < 8:
                    print(f"{name:<7}{tf:<5}{side:<6}{'  --- too few ---':>40}")
                    continue
                bp = beta_null(d, side, sigs, RR, FWD, COST, ema, a)
                flag = "  <<" if (st["meanR"] > 0 and (bp or 0) >= 90 and st["OOS"] > 0) else ""
                print(f"{name:<7}{tf:<5}{side:<6}{st['n']:>5}{st['win']:>6.0f}{st['meanR']:>8.2f}"
                      f"{st['totR']:>8.0f}{st['IS']:>7.2f}{st['OOS']:>7.2f}"
                      f"{st['green']:>3}/{st['nyr']:<2}{bp if bp==bp else 0:>7.0f}{flag}")


if __name__ == "__main__":
    main()
