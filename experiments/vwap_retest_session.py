"""Does a TIME/SESSION gate (Asia-skip etc.) help the VWAP break-retest? Fair test: compare the
retest reaction in each UTC session to that SAME session's OWN beta null (random long in those
hours) -- else we'd just be picking the higher-drift session, not a real edge. Sessions (UTC =
broker clock): Asia 0-7, London 7-13, NY 13-21, Late 21-24. Multiple-comparison caution: one
popping session at coin-flip hit-rate = noise."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
W, K = 40, 40
SESS = [("Asia 0-7", 0, 7), ("London 7-13", 7, 13), ("NY 13-21", 13, 21), ("Late 21-24", 21, 24)]


def sess_of(hr):
    for name, a, b in SESS:
        if a <= hr < b: return name
    return "Late 21-24"


def analyze(name, csv, tf):
    d = load_mt5_csv(csv)
    if "volume" not in d.columns: d["volume"] = 1.0
    if tf != "5m":
        d = d.resample("15min" if tf == "15m" else tf).agg(AGG).dropna()
    o, h, l, c, v = (d[x].values for x in ("open", "high", "low", "close", "volume"))
    hlc3 = (h + l + c) / 3.0
    atr = ta.atr(d["high"], d["low"], d["close"], 14).values
    idx = d.index.tz_convert("UTC") if d.index.tz is not None else d.index.tz_localize("UTC")
    hrs = idx.hour.values
    day = pd.Series(idx.normalize(), index=d.index)
    newday = day.values != np.roll(day.values, 1); newday[0] = True
    vwap = np.empty(len(c)); cpv = 0.0; cv = 0.0
    for i in range(len(c)):
        if newday[i]: cpv = 0.0; cv = 0.0
        cpv += hlc3[i] * v[i]; cv += v[i]
        vwap[i] = cpv / cv if cv > 0 else hlc3[i]
    n = len(c)

    ev_mfe = {s[0]: [] for s in SESS}; ev_mae = {s[0]: [] for s in SESS}
    for b in range(1, n - 1):
        if not (c[b] > vwap[b] and c[b - 1] <= vwap[b - 1]): continue
        if np.isnan(atr[b]) or atr[b] <= 0: continue
        for j in range(b + 1, min(b + 1 + W, n)):
            if l[j] <= vwap[j]:
                a = atr[j]
                if not np.isnan(a) and a > 0:
                    end = min(j + 1 + K, n)
                    if end - (j + 1) >= 5:
                        s = sess_of(hrs[j])
                        ev_mfe[s].append((h[j + 1:end].max() - c[j]) / a)
                        ev_mae[s].append((c[j] - l[j + 1:end].min()) / a)
                break
            if c[j] < vwap[j]: break

    # session-matched beta null
    rng = np.random.default_rng(0)
    beta = {}
    for name_s, a, b in SESS:
        mask = (hrs >= a) & (hrs < b) & ~np.isnan(atr) & (atr > 0)
        pool = np.where(mask)[0]; pool = pool[pool < n - K - 1]
        pick = rng.choice(pool, size=min(2000, len(pool)), replace=False)
        bm = []; bd = []
        for j in pick:
            end = j + 1 + K
            bm.append((h[j + 1:end].max() - c[j]) / atr[j]); bd.append((c[j] - l[j + 1:end].min()) / atr[j])
        beta[name_s] = (np.median(bm) / max(np.median(bd), 1e-9), (np.array(bm) > np.array(bd)).mean() * 100)

    print(f"\n===== {name} {tf} =====")
    print(f"  {'session':>14}{'n':>6}{'retest up/dn':>14}{'ret %up':>9}{'beta up/dn':>12}{'beta %up':>10}{'edge?':>8}")
    for name_s, a, b in SESS:
        m = np.array(ev_mfe[name_s]); mm = np.array(ev_mae[name_s])
        if len(m) < 20:
            print(f"  {name_s:>14}{len(m):>6}  too few"); continue
        r = np.median(m) / max(np.median(mm), 1e-9); pu = (m > mm).mean() * 100
        br, bpu = beta[name_s]
        edge = "+" if (r > br + 0.05 and pu > bpu) else "."
        print(f"  {name_s:>14}{len(m):>6}{r:>14.2f}{pu:>8.0f}%{br:>12.2f}{bpu:>9.0f}%{edge:>8}")


if __name__ == "__main__":
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "15m")
    analyze("GOLD", "data/vantage_xauusd_m5.csv", "5m")
    analyze("BTC", "data/vantage_btcusd_m5.csv", "15m")
