"""Use the squeeze break AS A TREND-FOLLOWER (user's point) and ask the right question:
does it ADD to the plain breakout (low-corr diversifier) or is it REDUNDANT (high-corr)?
combine() shows correlation + combined ret/DD vs plain alone. HTF, RR4 (where squeeze is +)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
from research.edge_harness import evaluate, combine, LADDERS, AGG


def plain_bo(df):
    c = df["close"].values; sma = df["close"].rolling(100).mean().values
    dch = pd.Series(df["high"].values).rolling(20).max().shift(1).values
    dcl = pd.Series(df["low"].values).rolling(20).min().shift(1).values
    sig = np.zeros(len(c)); sig[(c > dch) & (c > sma)] = 1; sig[(c < dcl) & (c < sma)] = -1
    sig[:120] = 0; return sig


def squeeze_trend(df):
    """squeeze box break, used as TREND continuation: require stacked uptrend (SMA50>SMA100)
    for longs / stacked down for shorts. Let it run (RR set by harness)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    s50 = df["close"].rolling(50).mean().values; s100 = df["close"].rolling(100).mean().values
    atr = (df["high"] - df["low"]).rolling(14).mean().values
    bh = pd.Series(h).rolling(10).max().shift(1).values
    bl = pd.Series(l).rolling(10).min().shift(1).values
    tight = (bh - bl) < 2.0 * atr
    up = (c > s50) & (s50 > s100); dn = (c < s50) & (s50 < s100)
    sig = np.zeros(len(c))
    sig[tight & (c > bh) & up] = 1
    sig[tight & (c < bl) & dn] = -1
    sig[:120] = 0; return sig


for inst, skip in [("GOLD", (12, 13, 14)), ("BTC", None)]:
    print(f"\n############ {inst}  (RR4, HTF) ############")
    p = evaluate(inst, plain_bo, rr=4.0, only=["8h", "1d"], stop_slip=0.5, skip_hours=skip, _return=True)
    s = evaluate(inst, squeeze_trend, rr=4.0, only=["8h", "1d"], stop_slip=0.5, skip_hours=skip, _return=True)
    for tf in ("8h", "1d"):
        if tf in p and tf in s:
            print(f"\n--- {inst} {tf}: does squeeze ADD to plain breakout? ---")
            combine({f"plain_{tf}": p[tf][0], f"squeeze_{tf}": s[tf][0]}, freq="Q")
