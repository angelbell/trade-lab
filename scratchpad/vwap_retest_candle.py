"""VWAP break -> WICK retest (candle shape as the separator, like the anti-knife clpos).
Refine the retest: not 'low touched VWAP' (coin flip = beta) but the CANDLE that wicked down to
VWAP and CLOSED BACK ABOVE (lower-wick rejection). Split by close-position (clpos) and lower-wick
length; measure forward reaction from the rejection candle's CLOSE vs a beta null (random long,
same close-entry, same K). If clpos/wick separates hold-vs-break, that's the STEP2 edge."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
W, K = 40, 40


def analyze(name, csv, tf):
    d = load_mt5_csv(csv)
    if "volume" not in d.columns: d["volume"] = 1.0
    if tf != "5m":
        d = d.resample("15min" if tf == "15m" else tf).agg(AGG).dropna()
    o, h, l, c, v = (d[x].values for x in ("open", "high", "low", "close", "volume"))
    hlc3 = (h + l + c) / 3.0
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    day = (d.index.tz_convert("UTC").normalize() if d.index.tz is not None else d.index.normalize())
    day = pd.Series(day, index=d.index)
    newday = day.values != np.roll(day.values, 1); newday[0] = True
    vwap = np.empty(len(c)); cpv = 0.0; cv = 0.0
    for i in range(len(c)):
        if newday[i]: cpv = 0.0; cv = 0.0
        cpv += hlc3[i] * v[i]; cv += v[i]
        vwap[i] = cpv / cv if cv > 0 else hlc3[i]
    n = len(c)

    rows = []
    for b in range(1, n - 1):
        if not (c[b] > vwap[b] and c[b - 1] <= vwap[b - 1]): continue
        a0 = atr[b]
        if np.isnan(a0) or a0 <= 0: continue
        peak = c[b]
        for j in range(b + 1, min(b + 1 + W, n)):
            peak = max(peak, h[j])
            if l[j] <= vwap[j]:                       # wick/low reached VWAP = retest touch
                a = atr[j]
                if np.isnan(a) or a <= 0: break
                rng = h[j] - l[j]
                clpos = (c[j] - l[j]) / rng if rng > 0 else 0.5
                lw = (min(o[j], c[j]) - l[j]) / a       # lower-wick length in ATR
                closed_above = c[j] > vwap[j]
                real = peak >= vwap[b] + 0.5 * a0
                end = min(j + 1 + K, n)
                ext = (peak - vwap[b]) / a0            # how far above VWAP the break ran before retest
                if end - (j + 1) >= 5:
                    mfe = (h[j + 1:end].max() - c[j]) / a
                    mae = (c[j] - l[j + 1:end].min()) / a
                    rows.append(dict(clpos=clpos, lw=lw, above=closed_above, real=real, ext=ext, mfe=mfe, mae=mae))
                break
            if c[j] < vwap[j]: break
    df = pd.DataFrame(rows)
    span = (d.index[-1] - d.index[0]).days / 365.25
    print(f"\n===== {name} {tf} =====  n={len(df)} ({len(df)/span:.0f}/yr)")
    # beta null: random long entered at a bar close
    rng2 = np.random.default_rng(0); rb = rng2.integers(1, n - K - 1, size=3000)
    bmfe = []; bmae = []
    for j in rb:
        if np.isnan(atr[j]) or atr[j] <= 0: continue
        end = j + 1 + K
        bmfe.append((h[j + 1:end].max() - c[j]) / atr[j]); bmae.append((c[j] - l[j + 1:end].min()) / atr[j])
    bmfe = np.array(bmfe); bmae = np.array(bmae)

    def show(tag, m):
        g = df[m]
        if len(g) < 8: print(f"    {tag:<30} n={len(g)} too few"); return
        r = g.mfe.median() / max(g.mae.median(), 1e-9)
        print(f"    {tag:<30} n={len(g):>4}  MFEup={g.mfe.median():.2f} MAEdn={g.mae.median():.2f} "
              f"up/dn={r:.2f}  %MFE>MAE={ (g.mfe>g.mae).mean()*100:.0f}%")

    print(f"    {'BETA null (random long)':<30} n={len(bmfe):>4}  MFEup={np.median(bmfe):.2f} "
          f"MAEdn={np.median(bmae):.2f} up/dn={np.median(bmfe)/max(np.median(bmae),1e-9):.2f}  "
          f"%MFE>MAE={(bmfe>bmae).mean()*100:.0f}%")
    show("ALL touches", df.index >= 0)
    show("wick-reject (closed>VWAP)", df.above)
    show("close-BELOW VWAP (weak)", ~df.above)
    show("wick-reject clpos>=0.7", df.above & (df.clpos >= 0.7))
    show("wick-reject clpos>=0.85", df.above & (df.clpos >= 0.85))
    show("wick-reject clpos<0.5", df.above & (df.clpos < 0.5))
    lwm = df[df.above].lw.median()
    show(f"wick-reject long-wick(>={lwm:.2f}ATR)", df.above & (df.lw >= lwm))
    show("wick-reject + real-break + clpos>=0.7", df.above & df.real & (df.clpos >= 0.7))
    print("    -- break STRENGTH (ext = peak above VWAP before retest, ATR) --")
    show("ext>=1 (all touches)", df.ext >= 1)
    show("ext>=2 (all touches)", df.ext >= 2)
    show("ext>=3 (all touches)", df.ext >= 3)
    show("ext<1 (weak break)", df.ext < 1)
    show("ext>=2 & wick-reject", (df.ext >= 2) & df.above)
    show("ext>=2 & wick-reject & clpos>=0.7", (df.ext >= 2) & df.above & (df.clpos >= 0.7))


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "15m")
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "5m")
    analyze("BTC", "data/vantage_btcusd_m5.csv", "15m")
