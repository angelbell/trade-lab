"""MY idea, run through the harness: VOLATILITY-CONTRACTION BREAKOUT (squeeze).
Mechanism: vol mean-reverts -> a breakout that fires right after a BB-width squeeze
should have more follow-through than a breakout from already-expanded vol.
Decisive test (lab prior = conditioners usually just n-trim the breakout):
  does squeeze_bo BEAT a RANDOM same-size subset of the plain breakout? (random_drop_null)
Protocol: check_causal -> evaluate(stop_slip, skip_hours) -> random_drop_null."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
from research.edge_harness import evaluate, check_causal, random_drop_null, LADDERS, AGG


def plain_bo(df):
    """trend breakout BASE: close breaks prior-20 Donchian, aligned with SMA100."""
    c = df["close"].values; sma = df["close"].rolling(100).mean().values
    dch = pd.Series(df["high"].values).rolling(20).max().shift(1).values
    dcl = pd.Series(df["low"].values).rolling(20).min().shift(1).values
    sig = np.zeros(len(c)); sig[(c > dch) & (c > sma)] = 1; sig[(c < dcl) & (c < sma)] = -1
    sig[:120] = 0; return sig


def squeeze_bo(df):
    """plain_bo AND a prior-bar BB-width squeeze (vol in bottom quartile of last 100)."""
    c = df["close"].values; sma = df["close"].rolling(100).mean().values
    ma20 = df["close"].rolling(20).mean(); sd20 = df["close"].rolling(20).std()
    bbw = (4 * sd20 / ma20)
    sq = (bbw < bbw.rolling(100).quantile(0.25)).shift(1).fillna(False).values   # squeeze on prior (confirmed) bar
    dch = pd.Series(df["high"].values).rolling(20).max().shift(1).values
    dcl = pd.Series(df["low"].values).rolling(20).min().shift(1).values
    sig = np.zeros(len(c))
    sig[sq & (c > dch) & (c > sma)] = 1
    sig[sq & (c < dcl) & (c < sma)] = -1
    sig[:120] = 0; return sig


def yrs(entries):
    return (pd.Timestamp(entries[-1][0]) - pd.Timestamp(entries[0][0])).days / 365.25


print("### causal self-check ###")
g4 = load_mt5_csv(LADDERS["GOLD"][0]).resample("240min").agg(AGG).dropna()
check_causal(squeeze_bo, g4)

for inst, skip in [("GOLD", (12, 13, 14)), ("BTC", None)]:
    print(f"\n################## {inst} ##################")
    print(">> plain breakout (BASE):")
    pb = evaluate(inst, plain_bo, rr=3.0, only=["1h", "4h", "1d"], stop_slip=0.5, skip_hours=skip, _return=True)
    print(">> squeeze breakout (MY IDEA):")
    sb = evaluate(inst, squeeze_bo, rr=3.0, only=["1h", "4h", "1d"], stop_slip=0.5, skip_hours=skip, _return=True)
    print(">> does squeeze BEAT randomly dropping the breakout to the same N?")
    for tf in ("4h", "1d"):
        if tf in pb and tf in sb:
            print(f"   [{tf}]", end=" ")
            random_drop_null(pb[tf][0], sb[tf][0], yrs(sb[tf][0]))
