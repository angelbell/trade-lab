"""Judge the proposed radar additions by REDUNDANCY, not narrative. Compute the existing 5-component
composite (ER, ADXn, align, slopeN, ATRexpN) and the two proposals — RSI-position and MACD-histogram
(acceleration) — then correlate. High corr with the existing composite/slope = redundant; low = it
adds an orthogonal descriptive axis. gold & BTC on 1h/4h (the meter's use TFs)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def comps(d):
    c = d["close"]; atr = ta.atr(d["high"], d["low"], d["close"], 14)
    er = (c.diff(20).abs() / c.diff().abs().rolling(20).sum())
    adxN = ((ta.adx(d["high"], d["low"], d["close"], 14)["ADX_14"] - 15) / 25).clip(0, 1)
    emaF = c.ewm(span=20, adjust=False).mean(); emaS = c.ewm(span=50, adjust=False).mean()
    stack = np.sign(c - emaF) + np.sign(emaF - emaS) + np.sign(emaS - emaS.shift(10))
    align = stack.abs() / 3.0
    slopeN = (( emaS - emaS.shift(10)).abs() / (atr * 1.5)).clip(0, 1)
    atrexpN = ((atr / atr.rolling(100).mean() - 0.8) / 0.7).clip(0, 1)
    COMP = pd.concat([er, adxN, align, slopeN, atrexpN], axis=1).mean(axis=1)
    # proposals
    rsi = ta.rsi(c, 14)
    rsiPos = (rsi - 50) / 50                      # -1..1 momentum position ("overheat" = high)
    macd = ta.macd(c, 12, 26, 9)
    hist = macd["MACDh_12_26_9"] / atr            # histogram (acceleration) normalized
    macdline = macd["MACD_12_26_9"] / atr         # macd line (momentum level)
    return pd.DataFrame({"COMP": COMP, "ER": er, "slopeN": slopeN, "align": align,
                         "RSIpos": rsiPos, "MACDhist": hist, "MACDline": macdline}).dropna()


def analyze(name, csv, rule, start="2018-01-01"):
    d = load_mt5_csv(csv).loc[start:]
    if "volume" not in d.columns: d["volume"] = 1.0
    d = d.resample(rule).agg(AGG).dropna()
    df = comps(d)
    print(f"\n===== {name} {rule} =====  n={len(df)}")
    print("  correlation with existing components:")
    for col in ("RSIpos", "MACDhist", "MACDline"):
        print(f"    {col:>9}:  |COMP|={df[col].abs().corr(df['COMP']):+.2f}  "
              f"vs ER={df[col].corr(df['ER']):+.2f}  vs slopeN={df[col].corr(df['slopeN']):+.2f}  "
              f"vs align={df[col].corr(df['align']):+.2f}")
    print(f"    (RSIpos vs MACDline corr={df['RSIpos'].corr(df['MACDline']):+.2f}  MACDhist vs MACDline={df['MACDhist'].corr(df['MACDline']):+.2f})")


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "240min")
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "60min")
    analyze("BTC", "data/vantage_btcusd_m5.csv", "240min", start="2018-10-01")
