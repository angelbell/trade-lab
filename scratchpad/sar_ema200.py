"""SAR x 200EMA (video method) falsification via edge_harness.

Rule (5m USDJPY, video spec):
  LONG : close>EMA200 AND EMA10 crosses ABOVE EMA25 AND PSAR dot BELOW price
  SHORT: close<EMA200 AND EMA10 crosses BELOW EMA25 AND PSAR dot ABOVE price
  SL=200EMA line, TP=RR2.  (harness approximates SL as katr*ATR; screens ENTRY edge.)
Causal: all indicators on closed bar i, enter i+1 open.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
import research.edge_harness as EH
from research.edge_harness import evaluate, check_causal, LADDERS
from src.data_loader import load_mt5_csv

# add low-TF USDJPY ladders (clean data: m15=8.5yr; m5 restricted to recent)
LADDERS["UJ_M15"] = ("data/vantage_usdjpy_m15.csv", 0.015,
                     [("15m","15min"),("30m","30min"),("1h","60min"),("2h","120min"),("4h","240min")])
LADDERS["UJ_M5"]  = ("data/vantage_usdjpy_m5.csv", 0.015,
                     [("5m","5min"),("15m","15min"),("1h","60min")])

def _cross_up(a, b):
    return (a > b) & (np.r_[True, (a[:-1] <= b[:-1])])
def _cross_dn(a, b):
    return (a < b) & (np.r_[True, (a[:-1] >= b[:-1])])

def sar_ema(df):
    c = df["close"].values
    ema10  = df["close"].ewm(span=10, adjust=False).mean().values
    ema25  = df["close"].ewm(span=25, adjust=False).mean().values
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values
    ps = ta.psar(df["high"], df["low"], df["close"], af0=0.02, af=0.02, max_af=0.2)
    # PSARl non-nan => SAR below price (uptrend); PSARs non-nan => SAR above price
    long_dot  = ps.filter(like="PSARl").iloc[:,0].notna().values
    short_dot = ps.filter(like="PSARs").iloc[:,0].notna().values
    cu = np.zeros(len(c), bool); cd = np.zeros(len(c), bool)
    cu[1:] = (ema10[1:] > ema25[1:]) & (ema10[:-1] <= ema25[:-1])   # golden cross on bar i
    cd[1:] = (ema10[1:] < ema25[1:]) & (ema10[:-1] >= ema25[:-1])   # dead cross on bar i
    sig = np.zeros(len(c))
    sig[cu & (c > ema200) & long_dot]  = 1
    sig[cd & (c < ema200) & short_dot] = -1
    sig[:210] = 0
    return sig

# all-signals BASE: drop the SAR gate + drop the cross, just EMA-trend-state alignment
def sar_ema_state(df):
    c = df["close"].values
    ema10  = df["close"].ewm(span=10, adjust=False).mean().values
    ema25  = df["close"].ewm(span=25, adjust=False).mean().values
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values
    sig = np.zeros(len(c))
    sig[(ema10 > ema25) & (c > ema200)] = 1
    sig[(ema10 < ema25) & (c < ema200)] = -1
    sig[:210] = 0
    return sig

if __name__ == "__main__":
    print("### FAITHFUL SAR x 200EMA (crossover trigger), RR2, cost=1.5pip ###")
    for name in ("UJ_M5","UJ_M15","USDJPY"):
        evaluate(name, sar_ema, rr=2.0, stop_slip=0.5)
    print("\n### causal check (M15/15m) ###")
    df = load_mt5_csv(LADDERS["UJ_M15"][0]).resample("15min").agg(EH.AGG).dropna()
    check_causal(sar_ema, df)
    print("\n### GROSS (cost=0) — is there ANY entry edge before costs? ###")
    for name in ("UJ_M15","USDJPY"):
        evaluate(name, sar_ema, rr=2.0, cost=0.0)
    print("\n### ALL-SIGNALS BASE (EMA-state only, no SAR/no-cross), GROSS ###")
    for name in ("UJ_M15","USDJPY"):
        evaluate(name, sar_ema_state, rr=2.0, cost=0.0)
