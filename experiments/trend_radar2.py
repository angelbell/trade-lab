"""GENERAL trend radar (regime meter, not tied to a leg/RR/direction): does the CURRENT trend-
strength score predict that the NEXT K bars will TREND (travel far, cleanly) vs CHOP?  Direction-
agnostic. Forward targets: net travel |c[t+K]-c[t]|/ATR  and  efficiency |net|/path (0..1).
Radars (causal): ER, ATR-expansion, |VWAP-distance|, and a z-scored COMPOSITE. Bucket Q1->Q4;
if high radar -> higher forward travel & efficiency, a general radar is possible. Cross-instrument."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
K, ERN = 48, 20


def z(s): return (s - s.rolling(2000, min_periods=200).mean()) / s.rolling(2000, min_periods=200).std()


def analyze(name, csv, tf, start="2018-01-01"):
    d = load_mt5_csv(csv).loc[start:]
    if "volume" not in d.columns: d["volume"] = 1.0
    if tf != "5m":
        d = d.resample("15min" if tf == "15m" else tf).agg(AGG).dropna()
    c = d["close"].values; h = d["high"].values; l = d["low"].values; v = d["volume"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    cs = pd.Series(c)
    er = (cs.diff(ERN).abs() / cs.diff().abs().rolling(ERN).sum())
    atr_exp = pd.Series(atr) / pd.Series(atr).rolling(100).mean()
    hlc3 = (h + l + c) / 3.0
    idx = d.index.tz_convert("UTC") if d.index.tz is not None else d.index.tz_localize("UTC")
    nd = pd.Series(idx.normalize()).values; nd = nd != np.roll(nd, 1); nd[0] = True
    vwap = np.empty(len(c)); cpv = cv = 0.0
    for i in range(len(c)):
        if nd[i]: cpv = cv = 0.0
        cpv += hlc3[i] * v[i]; cv += v[i]; vwap[i] = cpv / cv if cv > 0 else hlc3[i]
    vwd = pd.Series(np.abs(c - vwap) / atr)
    comp = (z(er) + z(atr_exp) + z(vwd)) / 3.0

    df = pd.DataFrame({"ER": er.values, "ATRexp": atr_exp.values, "VWdist": vwd.values, "COMP": comp.values})
    abs = np.abs(cs.diff().values)
    path = pd.Series(abs).rolling(K).sum().shift(-K).values          # sum |dc| over t..t+K-1
    net = np.abs(np.roll(c, -K) - c)
    df["travel"] = net / atr                                          # net displacement in ATR
    df["eff"] = np.where(path > 0, net / path, np.nan)                # efficiency 0..1
    df = df.iloc[200:len(df) - K - 1].dropna()

    print(f"\n===== {name} {tf} =====  n={len(df)}  (forward K={K} bars)")
    print(f"  overall: fwd travel={df.travel.median():.2f}ATR  efficiency={df.eff.median():.2f}")
    print(f"  {'radar':>8}  Q1 -> Q4 : median travel(ATR)   |   median efficiency     [Q4-Q1]")
    for rad in ("ER", "ATRexp", "VWdist", "COMP"):
        try:
            q = pd.qcut(df[rad], 4, labels=[1, 2, 3, 4], duplicates="drop")
        except Exception:
            print(f"  {rad:>8}  (cannot bucket)"); continue
        gt = df.groupby(q, observed=True).travel.median()
        ge = df.groupby(q, observed=True).eff.median()
        tv = "  ".join(f"{x:.2f}" for x in gt)
        ef = "  ".join(f"{x:.2f}" for x in ge)
        print(f"  {rad:>8}  {tv}  |  {ef}   [tr{gt.iloc[-1]-gt.iloc[0]:+.2f} eff{ge.iloc[-1]-ge.iloc[0]:+.2f}]")


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "5m")
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "15m")
    analyze("BTC", "data/vantage_btcusd_m5.csv", "15m", start="2018-10-01")
