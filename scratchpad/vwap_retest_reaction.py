"""VWAP break-and-retest continuation: close crosses ABOVE session-VWAP -> price pulls back and
TOUCHES VWAP from above (retest) -> does it hold as support and continue UP?  Bounce-verification
STEP1 (do NOT jump to RR): measure the FORWARD reaction at the retest touch, decided causally (no
'it held' selection). Compare to a beta null (all-bar forward reaction) and split by daily-uptrend
to see if 'support->up' is just the instrument's drift. gold 15m & 5m. VWAP = UTC-day anchored,
hlc3*volume cumulative. Retest requires a REAL prior break (high reached vwap+0.5ATR) else it's
hover-at-vwap noise; base also shown without that requirement."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
W, K = 40, 40   # retest search window / forward reaction window (bars)


def analyze(name, csv, tf):
    d = load_mt5_csv(csv)
    volcol = "volume" if "volume" in d.columns else ("tick_volume" if "tick_volume" in d.columns else None)
    if volcol is None:
        d["volume"] = 1.0; volcol = "volume"       # fallback: unweighted anchored avg price
    d = d.rename(columns={volcol: "volume"})
    if tf != "5m":
        d = d.resample("15min" if tf == "15m" else tf).agg(AGG).dropna()
    o, h, l, c, v = (d[x].values for x in ("open", "high", "low", "close", "volume"))
    hlc3 = (h + l + c) / 3.0
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    # session VWAP, UTC-day anchored
    day = d.index.tz_convert("UTC").normalize() if d.index.tz is not None else d.index.normalize()
    day = pd.Series(day, index=d.index)
    newday = day.values != np.roll(day.values, 1); newday[0] = True
    vwap = np.empty(len(c)); cpv = 0.0; cv = 0.0
    for i in range(len(c)):
        if newday[i]: cpv = 0.0; cv = 0.0
        cpv += hlc3[i] * v[i]; cv += v[i]
        vwap[i] = cpv / cv if cv > 0 else hlc3[i]
    # daily uptrend gate (prior-day close > rising daily SMA150), no lookahead
    dc = d["close"].resample("1D").last().dropna(); sma = dc.rolling(150).mean()
    up = ((dc > sma) & (sma > sma.shift(10))).shift(1)
    reg = up.reindex(d.index, method="ffill").fillna(False).values
    n = len(c)

    events = []  # (retest_bar, gated)
    for b in range(1, n - 1):
        if not (c[b] > vwap[b] and c[b - 1] <= vwap[b - 1]):   # crossover above VWAP
            continue
        a = atr[b]
        if np.isnan(a) or a <= 0: continue
        peak = c[b]
        for j in range(b + 1, min(b + 1 + W, n)):
            peak = max(peak, h[j])
            if c[j] < vwap[j]:            # closed back below before any retest touch -> failed break, stop
                # still allow the touch on this same bar if low<=vwap (it did close below = not a hold)
                pass
            if l[j] <= vwap[j]:            # retest TOUCH of VWAP from above
                real = (peak >= vwap[b] + 0.5 * a)   # had a real move up first (not hover)
                events.append((j, reg[j], real, vwap[j], a))
                break
            if c[j] < vwap[j]:
                break
    if not events:
        print(f"{name} {tf}: no events"); return

    def react(evs, tag):
        mfe = []; mae = []
        for (j, g, real, vw, a) in evs:
            end = min(j + 1 + K, n)
            if end - (j + 1) < 5: continue
            seg_h = h[j + 1:end]; seg_l = l[j + 1:end]
            mfe.append((seg_h.max() - vw) / a); mae.append((vw - seg_l.min()) / a)
        mfe = np.array(mfe); mae = np.array(mae)
        if len(mfe) < 8: print(f"    {tag:<28} n={len(mfe)} too few"); return
        ratio = np.median(mfe) / max(np.median(mae), 1e-9)
        print(f"    {tag:<28} n={len(mfe):>4}  MFEup={np.median(mfe):.2f}ATR MAEdn={np.median(mae):.2f}ATR "
              f"up/dn={ratio:.2f}  %MFE>MAE={ (mfe>mae).mean()*100:.0f}%")

    # beta null: forward reaction from a RANDOM bar (long), same K, using vwap-agnostic anchor=close
    rng = np.random.default_rng(0); rb = rng.integers(1, n - K - 1, size=min(3000, n - K - 2))
    bmfe = []; bmae = []
    for j in rb:
        if np.isnan(atr[j]) or atr[j] <= 0: continue
        end = j + 1 + K
        bmfe.append((h[j + 1:end].max() - c[j]) / atr[j]); bmae.append((c[j] - l[j + 1:end].min()) / atr[j])
    bmfe = np.array(bmfe); bmae = np.array(bmae)

    span = (d.index[-1] - d.index[0]).days / 365.25
    print(f"\n===== {name} {tf} =====  events={len(events)} ({len(events)/span:.0f}/yr)  span={span:.1f}yr")
    print(f"    {'BETA null (random long)':<28} n={len(bmfe):>4}  MFEup={np.median(bmfe):.2f}ATR "
          f"MAEdn={np.median(bmae):.2f}ATR up/dn={np.median(bmfe)/max(np.median(bmae),1e-9):.2f}  "
          f"%MFE>MAE={(bmfe>bmae).mean()*100:.0f}%")
    react(events, "ALL retests (base)")
    react([e for e in events if e[2]], "real-break retests only")
    react([e for e in events if e[1]], "daily-UPTREND retests")
    react([e for e in events if e[1] and e[2]], "uptrend + real-break")


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "15m")
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "5m")
    analyze("BTC", "data/vantage_btcusd_m5.csv", "15m")
