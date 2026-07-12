"""TREND RADAR diagnostic: does a causal trend-STRENGTH score separate 'trend days (longs pay)'
from 'chop days (many stops)'? For UP-context bars (close>EMA50), bucket the forward outcome of a
standardized long (+2ATR target vs -1ATR stop, K bars) by each radar's quartile. If high-radar =
high win% / low stop% and low-radar = many stops, the radar works as a deploy filter (the user's
'yesterday I got stopped many times' = a low-radar chop day). Compare Q4 vs Q1.
Radars (all causal at bar t): ER (efficiency ratio), ATR-expansion, VWAP-distance, ADX."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
K = 48          # forward bars
ERN = 20        # efficiency-ratio lookback
TGT, STP = 2.0, 1.0


def analyze(name, csv, tf, start="2018-01-01"):
    d = load_mt5_csv(csv).loc[start:]
    if "volume" not in d.columns: d["volume"] = 1.0
    if tf != "5m":
        d = d.resample("15min" if tf == "15m" else tf).agg(AGG).dropna()
    o, h, l, c, v = (d[x].values for x in ("open", "high", "low", "close", "volume"))
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    ema = d["close"].ewm(span=50, adjust=False).mean().values
    adx = ta.adx(d["high"], d["low"], d["close"], 14)["ADX_14"].values
    # efficiency ratio (ER)
    cs = pd.Series(c)
    er = (cs.diff(ERN).abs() / cs.diff().abs().rolling(ERN).sum()).values
    atr_exp = atr / pd.Series(atr).rolling(100).mean().values
    # session VWAP (UTC-day)
    hlc3 = (h + l + c) / 3.0
    idx = d.index.tz_convert("UTC") if d.index.tz is not None else d.index.tz_localize("UTC")
    nd = pd.Series(idx.normalize()).values; nd = nd != np.roll(nd, 1); nd[0] = True
    vwap = np.empty(len(c)); cpv = cv = 0.0
    for i in range(len(c)):
        if nd[i]: cpv = cv = 0.0
        cpv += hlc3[i] * v[i]; cv += v[i]; vwap[i] = cpv / cv if cv > 0 else hlc3[i]
    vw_dist = (c - vwap) / atr
    n = len(c)

    rows = []
    for t in range(200, n - K - 1):
        if np.isnan(atr[t]) or atr[t] <= 0 or c[t] <= ema[t]: continue     # UP-context only
        if np.isnan(er[t]) or np.isnan(adx[t]) or np.isnan(atr_exp[t]): continue
        tgt = c[t] + TGT * atr[t]; stp = c[t] - STP * atr[t]; R = None
        for j in range(t + 1, t + 1 + K):
            if l[j] <= stp: R = -STP; break
            if h[j] >= tgt: R = TGT; break
        if R is None: R = (c[t + K] - c[t]) / atr[t]
        rows.append((er[t], atr_exp[t], vw_dist[t], adx[t], R))
    df = pd.DataFrame(rows, columns=["ER", "ATRexp", "VWdist", "ADX", "R"])
    df["win"] = df.R > 0; df["stop"] = df.R == -STP
    print(f"\n===== {name} {tf} =====  UP-context longs n={len(df)}  (target +{TGT}ATR / stop -{STP}ATR, K={K})")
    print(f"  overall: win={df.win.mean()*100:.0f}%  stop={df.stop.mean()*100:.0f}%  meanR={df.R.mean():+.3f}")
    print(f"  {'radar':>8}  Q1(low) -> Q4(high) : win% / stop% / meanR   [Q4-Q1 meanR spread]")
    for rad in ("ER", "ATRexp", "VWdist", "ADX"):
        try:
            df["q"] = pd.qcut(df[rad], 4, labels=[1, 2, 3, 4], duplicates="drop")
        except Exception:
            print(f"  {rad:>8}  (cannot bucket)"); continue
        g = df.groupby("q", observed=True)
        cells = [f"{gg.win.mean()*100:.0f}/{gg.stop.mean()*100:.0f}/{gg.R.mean():+.2f}" for _, gg in g]
        sp = g.R.mean()
        spread = sp.iloc[-1] - sp.iloc[0]
        print(f"  {rad:>8}  " + "  ".join(cells) + f"   [{spread:+.3f}]")


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "5m")
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "15m")
    analyze("BTC", "data/vantage_btcusd_m5.csv", "15m", start="2018-10-01")
