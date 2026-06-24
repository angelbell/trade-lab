"""Step 0 for pairs trading: is BTC/ETH a tradeable mean-reverting pair?

Before any backtest we MUST verify the statistical foundation:
  1. Correlation (do they move together at all?)
  2. Engle-Granger cointegration test (does the spread mean-revert?)
  3. Hedge ratio (beta) via OLS, and the spread's half-life of mean reversion.

If cointegration FAILS, pairs trading is structurally impossible — stop here.
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from src.data_loader import fetch_ohlcv


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion (in bars)."""
    s_lag = spread.shift(1).dropna()
    s_ret = (spread - spread.shift(1)).dropna()
    s_lag = s_lag.loc[s_ret.index]
    beta = sm.OLS(s_ret, sm.add_constant(s_lag)).fit().params.iloc[1]
    if beta >= 0:
        return float("inf")
    return -np.log(2) / beta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end",   default="2025-01-01")
    p.add_argument("--tf",    default="1d")
    p.add_argument("--a", default="BTC/USDT")
    p.add_argument("--b", default="ETH/USDT")
    args = p.parse_args()

    print(f"Fetching {args.a} and {args.b} {args.tf} {args.start} → {args.end} …")
    da = fetch_ohlcv(args.a, args.tf, args.start, args.end)["close"]
    db = fetch_ohlcv(args.b, args.tf, args.start, args.end)["close"]
    df = pd.concat([da, db], axis=1, keys=["A", "B"]).dropna()
    print(f"  {len(df):,} aligned bars\n")

    # log prices for a stable hedge ratio
    la, lb = np.log(df["A"]), np.log(df["B"])

    # 1) correlation of daily returns
    corr = df["A"].pct_change().corr(df["B"].pct_change())

    # 2) hedge ratio via OLS:  logA = const + beta * logB
    ols  = sm.OLS(la, sm.add_constant(lb)).fit()
    beta = ols.params.iloc[1]
    spread = la - beta * lb

    # 3) Engle-Granger cointegration test
    eg_t, eg_p, _ = coint(la, lb)

    # 4) ADF test on the spread itself (stationary => mean-reverting)
    adf_t, adf_p = adfuller(spread.dropna())[:2]

    hl = half_life(spread)

    print("=" * 56)
    print(f"  Return correlation        : {corr:8.3f}")
    print(f"  Hedge ratio (beta)        : {beta:8.3f}")
    print(f"  Engle-Granger coint p-val : {eg_p:8.4f}   (<0.05 = cointegrated)")
    print(f"  ADF on spread p-val       : {adf_p:8.4f}   (<0.05 = stationary)")
    print(f"  Spread half-life (bars)   : {hl:8.1f}")
    print("=" * 56)

    coint_ok = eg_p < 0.05 and adf_p < 0.05
    if coint_ok and 1 < hl < 100:
        print("  VERDICT: ✓ tradeable mean-reverting pair — proceed to backtest")
    elif coint_ok:
        print(f"  VERDICT: ~ cointegrated but half-life {hl:.0f} bars "
              "(too slow/fast for clean trading)")
    else:
        print("  VERDICT: ✗ NOT cointegrated — pairs trading not viable on this pair")


if __name__ == "__main__":
    main()
