"""Reversal/fade edge across the FULL TF LADDER (default per-TF measurement).
Triggers: RSI80/20, BB(2sigma), extMA(2ATR), + USER confluence (RSI extreme + HTF
swing zone + climax bar). Exit = fade to the mean (MA20), stop = 1 ATR beyond extreme.
Hypothesis: reversion beats momentum only at HIGHER TF. meanR per TF, both sides."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}

def fade(o, h, l, c, ma, s, side, e, stopd, cost):
    if stopd <= 0: return None
    lvl = ma[s]
    if side == -1:
        if lvl >= e: return None
        stop = e + stopd
        for j in range(s + 1, min(s + 1 + 200, len(c))):
            if h[j] >= stop: return -1.0 - cost / stopd
            if l[j] <= lvl: return (e - lvl) / stopd - cost / stopd
        return (e - c[min(s + 200, len(c) - 1)]) / stopd - cost / stopd
    if lvl <= e: return None
    stop = e - stopd
    for j in range(s + 1, min(s + 1 + 200, len(c))):
        if l[j] <= stop: return -1.0 - cost / stopd
        if h[j] >= lvl: return (lvl - e) / stopd - cost / stopd
    return (c[min(s + 200, len(c) - 1)] - e) / stopd - cost / stopd

def run_tf(df, cost):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = ta.atr(df["high"], df["low"], df["close"], 14).values
    rsi = ta.rsi(df["close"], 14).values
    ma = df["close"].rolling(20).mean().values
    sd = df["close"].rolling(20).std().values
    ub, lb = ma + 2 * sd, ma - 2 * sd
    h4 = df.resample("4D" if (df.index[1]-df.index[0]).total_seconds()>=86400 else None) if False else None
    # HTF swing zone = swing over prior 50 bars of same TF (structural-level proxy)
    sh = pd.Series(h).rolling(50).max().shift(1).values
    sl = pd.Series(l).rolling(50).min().shift(1).values
    rng = h - l
    R = {}
    for s in range(60, len(c) - 1):
        if np.isnan(atr[s]) or atr[s] <= 0 or np.isnan(rsi[s]) or np.isnan(ma[s]): continue
        a = atr[s]; e = o[s + 1]
        def add(key, side):
            r = fade(o, h, l, c, ma, s + 1, side, e, a, cost)
            if r is not None: R.setdefault(key, []).append(r)
        if rsi[s] >= 80: add("RSI80 S", -1)
        if rsi[s] <= 20: add("RSI20 L", 1)
        if c[s] > ub[s]: add("BB S", -1)
        if c[s] < lb[s]: add("BB L", 1)
        if c[s] >= ma[s] + 2 * a: add("extMA S", -1)
        if c[s] <= ma[s] - 2 * a: add("extMA L", 1)
        big_bull = c[s] > o[s] and rng[s] >= 1.5 * a
        big_bear = c[s] < o[s] and rng[s] >= 1.5 * a
        if rsi[s] >= 75 and not np.isnan(sh[s]) and abs(h[s]-sh[s]) <= 0.5*a and big_bull: add("USER S", -1)
        if rsi[s] <= 25 and not np.isnan(sl[s]) and abs(l[s]-sl[s]) <= 0.5*a and big_bear: add("USER L", 1)
    return R

def ladder(name, base, cost, tfs):
    print(f"\n===== {name}  (fade-to-mean meanR per TF; >0 = reversion edge) =====")
    cols = []
    data = {}
    for lbl, fr in tfs:
        df = base if fr is None else base.resample(fr).agg(AGG).dropna()
        if len(df) < 300: continue
        cols.append(lbl); data[lbl] = run_tf(df, cost)
    keys = ["RSI80 S","RSI20 L","BB S","BB L","extMA S","extMA L","USER S","USER L"]
    print("  cell = PF/n  (PF>1 = edge; n = entry count over full history)")
    print("  trigger   " + "".join(f"{l:>13}" for l in cols))
    for k in keys:
        row = f"  {k:<9}"
        for l in cols:
            v = np.array(data[l].get(k, []))
            if len(v) >= 12:
                pf = v[v > 0].sum() / abs(v[v <= 0].sum()) if (v <= 0).any() else 9.9
                row += f"{f'{pf:.2f}/{len(v)}':>13}"
            else:
                row += f"{'-':>13}"
        print(row)

goldm5 = load_mt5_csv("data/vantage_xauusd_m5.csv")
btc = load_mt5_csv("data/vantage_btcusd_h1.csv")
jpy = load_mt5_csv("data/vantage_usdjpy_h1.csv")
gold_tfs = [("5m","5min"),("15m","15min"),("1h","60min"),("2h","120min"),("4h","240min"),("8h","480min"),("1d","1440min")]
h1_tfs = [("1h",None),("2h","120min"),("4h","240min"),("8h","480min"),("1d","1440min")]
ladder("GOLD", goldm5, 0.40, gold_tfs)
ladder("USDJPY", jpy, 0.015, h1_tfs)
ladder("BTC", btc, 15.0, h1_tfs)
