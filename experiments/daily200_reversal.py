"""User's setup: touch the *DAILY* 200MA (the genuinely-watched level) then enter
only on a REVERSAL-CONFIRMATION candle (wick to the level + close back across,
bodied in the bounce direction). Structural stop = the confirmation candle's extreme.
Tests, per verification order: does CONFIRMATION beat touch-only, and does the
DAILY-200 (watched) beat the intraday-200 (unwatched)? Both sides, BTC & gold H1."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

def level_series(d, kind, tf, n):
    """causal MA level mapped to the H1 index: prior-completed-bar value of the
    n-MA computed on the tf-resampled closes (tf='1D' = the watched daily 200)."""
    r = d.resample(tf).agg(AGG).dropna()
    ma = (r.close.ewm(span=n, adjust=False).mean() if kind == "EMA" else r.close.rolling(n).mean())
    return ma.shift(1).reindex(d.index, method="ffill").values

def test(name, d, lvl, cost, conf, RR, side, gate=False):
    at = ta.atr(d["high"], d["low"], d["close"], 14).values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    KD = 24 * 20                                     # daily-200 slope over ~20 days (H1 bars)
    rows, busy = [], -1
    for s in range(260, len(c) - 1):
        if np.isnan(lvl[s]) or np.isnan(at[s]) or at[s] <= 0 or s <= busy: continue
        if gate and s - KD >= 0 and not np.isnan(lvl[s - KD]):
            rising = lvl[s] > lvl[s - KD]
            if side == "long" and not rising: continue      # trend-aligned: long only if daily-200 rising
            if side == "short" and rising: continue          # short only if daily-200 falling
        L = lvl[s]
        if side == "long":
            touch = l[s] <= L and c[s - 1] > L                  # dipped to level from above
            ok = touch and (not conf or (c[s] > L and c[s] > o[s]))   # closed back above, bullish
            if not ok: continue
            e = o[s + 1]; stop = (l[s] if conf else L) - 0.05 * at[s]
            sd = e - stop
            if sd <= 0: continue
            tgt = e + RR * sd; R = None
            for j in range(s + 1, min(s + 1 + 200, len(c))):
                if l[j] <= stop: R = -1.0; busy = j; break
                if h[j] >= tgt: R = RR; busy = j; break
            if R is None: R = (c[min(s + 200, len(c) - 1)] - e) / sd; busy = s + 200
        else:
            touch = h[s] >= L and c[s - 1] < L
            ok = touch and (not conf or (c[s] < L and c[s] < o[s]))
            if not ok: continue
            e = o[s + 1]; stop = (h[s] if conf else L) + 0.05 * at[s]
            sd = stop - e
            if sd <= 0: continue
            tgt = e - RR * sd; R = None
            for j in range(s + 1, min(s + 1 + 200, len(c))):
                if h[j] >= stop: R = -1.0; busy = j; break
                if l[j] <= tgt: R = RR; busy = j; break
            if R is None: R = (e - c[min(s + 200, len(c) - 1)]) / sd; busy = s + 200
        rows.append(R - cost / sd)
    x = np.array(rows)
    if len(x) < 12: return f"{name:<30} n={len(x)} too few"
    pf = x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else 9.0
    return f"{name:<30} n={len(x):>3} win={(x>0).mean()*100:>3.0f}% PF={pf:.2f} meanR={x.mean():+.3f} med={np.median(x):+.2f}"

btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
xau = load_mt5_csv("data/vantage_xauusd_h1.csv")
INST = [("BTC", btc, 15.0), ("GOLD", xau, 0.40)]

for label, d, cost in INST:
    print(f"\n===== {label} H1 (cost {cost}) =====")
    dl = level_series(d, "SMA", "1D", 200)    # the watched daily 200
    hl = level_series(d, "SMA", "1H", 200)    # intraday 200 (unwatched)
    for side in ("long", "short"):
        print(f"  -- {side} --")
        for RR in (1.0, 2.0):
            print("   " + test(f"daily200 +CONFIRM   RR{RR}", d, dl, cost, True, RR, side))
            print("   " + test(f"daily200 +CONFIRM +GATE RR{RR}", d, dl, cost, True, RR, side, gate=True))
