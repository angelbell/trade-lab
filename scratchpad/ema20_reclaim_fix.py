"""ema20_reclaim_fix.py -- user hypothesis: "add a 20EMA filter and the 4STEP works?"

Two readings, both tested on 1h bars (gold + BTC, 7.7yr, RR2 fixed, net cost):
  [A] deep version : regime ON, price touches 80EMA(+0.25ATR) -> enter on the first 1h
      bar that CLOSES back above the 20EMA (the user's filter on top of STEP2).
      stop = min(pullback minlow, 80EMA) - 0.05ATR (method stop).
  [B] shallow ver. : drop the 80EMA-touch requirement entirely -- classic 20EMA pullback:
      close below 20EMA -> close back above = entry (btc_pull mechanism at 1h).
      stop = pullback lowest low, 0.5ATR floor (ema_pullback convention).
Regime for both: GC(20>80) AND Dow HH/HL (the method's own STEP1), prior completed bar.
Pre-registered priors: gold has NO pullback edge at any TF; BTC 1h pullback ~flat.
A filter can only CONCENTRATE an existing edge, not create one.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import resample
from dow_ema_4step import prep_1h

TOUCH_K, FWD = 0.25, 500


def card(tag, rows, span):
    if len(rows) < 15:
        print(f"  {tag:<38} n={len(rows)} few"); return
    t = pd.DataFrame(rows, columns=["time", "R"])
    t["y"] = t.time.dt.year
    Rn = t.R.values
    yr = t.y.values
    half = np.median(yr)
    pf = Rn[Rn > 0].sum() / max(1e-9, abs(Rn[Rn <= 0].sum()))
    g = sum(t.groupby("y")["R"].sum() > 0)
    print(f"  {tag:<38} N/yr={len(Rn)/span:5.1f} win={(Rn>0).mean()*100:4.1f}% PF={pf:4.2f} "
          f"meanR={Rn.mean():+.3f} IS/OOS={Rn[yr<half].mean():+.2f}/{Rn[yr>=half].mean():+.2f} "
          f"totR/yr={Rn.sum()/span:+5.1f} grn={g}/{t.y.nunique()}")


def walk(d1, H, rt, span, variant):
    h, l, c = d1["high"].values, d1["low"].values, d1["close"].values
    e20, e80, a1 = H["e20"], H["e80"], H["atr"]
    on = H["up"] & H["dow"]
    rows, last_x = [], -1
    state, minlow = 0, np.inf     # A: 0 idle, 1 touched(waiting reclaim). B: 1 = below 20EMA
    for i in range(2, len(c)):
        trend = on[i - 1] and not np.isnan(a1[i - 1]) and a1[i - 1] > 0
        if not trend:
            state, minlow = 0, np.inf
            continue
        if variant == "A":
            if state == 0:
                if l[i] <= e80[i - 1] + TOUCH_K * a1[i - 1]:
                    state, minlow = 1, l[i]
                continue
            minlow = min(minlow, l[i])
            if c[i] <= e20[i]:
                continue
            e = c[i]
            stop = min(minlow, e80[i]) - 0.05 * a1[i]
        else:
            if state == 0:
                if c[i] < e20[i]:
                    state, minlow = 1, l[i]
                continue
            minlow = min(minlow, l[i])
            if c[i] <= e20[i]:
                continue
            e = c[i]
            stop = min(minlow, l[i])
            if e - stop < 0.5 * a1[i]:
                stop = e - 0.5 * a1[i]
        state, minlow = 0, np.inf
        if i <= last_x or e - stop <= 0:
            continue
        risk = e - stop
        tgt = e + 2.0 * risk
        R, xj = None, min(i + FWD, len(c) - 1)
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if l[j] <= stop: R, xj = -1.0, j; break
            if h[j] >= tgt:  R, xj = 2.0, j; break
        if R is None:
            R = (c[xj] - e) / risk
        rows.append((d1.index[i], R - rt / risk))
        last_x = xj
    card({"A": "[A] 80EMAタッチ→20EMA確定奪回 (深押し)",
          "B": "[B] 20EMA奪回のみ (浅押し=btc_pull機構1h)"}[variant], rows, span)


def main():
    for name, csv, rt, start, minbars in [
            ("GOLD 1h", "data/vantage_xauusd_m5.csv", 0.3, "2018-09-14", None),
            ("BTC 1h", "data/vantage_btcusd_m15.csv", 15.0, None, 80)]:
        d = load_mt5_csv(csv)
        if start:
            d = d.loc[start:]
        if minbars:
            cnt = d.groupby(d.index.date).size()
            okd = cnt[cnt.rolling(30).median() >= minbars]
            d = d[d.index.date >= okd.index[0]]
        d1 = resample(d, "60min")
        H = prep_1h(d1)
        span = (d1.index[-1] - d1.index[0]).days / 365.25
        print(f"\n===== {name} ({span:.1f}yr, RR2, net ${rt}) =====")
        for v in ("A", "B"):
            walk(d1, H, rt, span, v)


if __name__ == "__main__":
    main()
