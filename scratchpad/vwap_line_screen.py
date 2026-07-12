"""vwap_line_screen.py -- STEP 1-2 of the line-efficacy ladder: does the daily-anchored VWAP
act as a LINE on gold 5m, and can we tell WHEN it works?

Events (causal, decided at bar-i close; indicators at [i-1]):
  touch  : first touch of VWAP from above (low<=vwap) after >=6 bars cleanly above
  reclaim: close crosses above VWAP after >=6 bars below; stretch = max (vwap-low)/ATR
           during the below-run (>=1 ATR = the validated "oversold reclaim" context)
Bounce test: from close[i], P(+0.5*ATR hit before -0.5*ATR) over the next 72 bars (6h);
same-bar double-hit counts as a LOSS (conservative). ATR14 shifted 1 bar.
Beta null: the same barrier race started from EVERY bar, aggregated per hour-of-day, then
weighted by the event set's hour distribution -> "same-time random-long" bounce rate.
Strata: 4H radar up&>=5 (the verified trend-day gate) / FOMC+NFP day exclusion (user rule:
humans skip news days) / 3 sessions (server-time thirds) / per-year for headline cells.
PASS bar = gate-ON separation vs beta, not the absolute rate. Predicted kill: touch-bounce
== beta (the §15 retest death again); reclaim+stretch is where a real lift may live.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from radar_gate_race import comps_tf
from research.fomc_screen import FOMC

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--bar", type=float, default=0.5)
_ap.add_argument("--line", default="vwap", choices=["vwap", "ema20"])
_A = _ap.parse_args()

K = 72          # forward bars for the barrier race (6h on 5m)
BAR = _A.bar    # barrier width in ATRs
MIN_RUN = 6     # bars cleanly above/below before an event counts


def main():
    d = load_mt5_csv("data/vantage_xauusd_m5.csv")
    # clip to genuine 5m density (early file is sparse): first date of a sustained >=150 bars/day
    cnt = d.groupby(d.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= 150]
    d = d[d.index.date >= ok.index[0]]
    print(f"line={_A.line} bar={BAR}ATR  span: {d.index[0].date()} -> {d.index[-1].date()}  ({len(d):,} 5m bars, "
          f"{(d.index[-1]-d.index[0]).days/365.25:.1f}yr)")

    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    day = pd.Series(d.index.date, index=d.index)
    if _A.line == "vwap":
        vw = ((tp * d["volume"]).groupby(day).cumsum() / d["volume"].groupby(day).cumsum()).values
    else:
        vw = d["close"].ewm(span=20, adjust=False).mean().values
    n = len(c)

    # ---------- barrier race from EVERY bar (win=up first, loss=down first or same-bar) ----------
    up_lvl, dn_lvl = c + BAR * atr, c - BAR * atr
    t_up = np.full(n, K + 1, dtype=np.int32)   # first k (1..K) hitting up barrier
    t_dn = np.full(n, K + 1, dtype=np.int32)
    for k in range(1, K + 1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n - k], ls[:n - k] = h[k:], l[k:]
        hs[n - k:], ls[n - k:] = -np.inf, np.inf
        t_up = np.where((t_up > K) & (hs >= up_lvl), k, t_up)
        t_dn = np.where((t_dn > K) & (ls <= dn_lvl), k, t_dn)
    resolved = (np.minimum(t_up, t_dn) <= K) & ~np.isnan(atr)
    win = resolved & (t_up < t_dn)             # ties (t_up==t_dn) -> loss, conservative
    valid = ~np.isnan(atr) & ~np.isnan(vw) & (np.arange(n) < n - K)

    hours = d.index.hour.values
    beta_h = {hh: win[valid & (hours == hh)].mean() for hh in range(24)}

    def beta_of(mask):
        hh = hours[mask]
        return np.mean([beta_h[x] for x in hh])

    # ---------- events ----------
    above = c > vw
    below = ~above
    run_above = np.zeros(n, dtype=np.int32)
    run_below = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        run_above[i] = run_above[i - 1] + 1 if above[i - 1] else 0
        run_below[i] = run_below[i - 1] + 1 if below[i - 1] else 0

    touch = valid & (l <= vw) & (run_above >= MIN_RUN)
    # first touch only: previous bar's low stayed above vwap
    prev_l = np.roll(l, 1); prev_vw = np.roll(vw, 1)
    touch &= (prev_l > prev_vw)

    cross_up = valid & above & np.roll(below, 1) & (run_below >= MIN_RUN)
    # stretch depth of the below-run, in ATRs (max over the run; causal: run is in the past)
    depth = np.zeros(n)
    dmax = 0.0
    for i in range(1, n):
        if below[i - 1]:
            if not np.isnan(atr[i - 1]) and atr[i - 1] > 0:
                dmax = max(dmax, (vw[i - 1] - l[i - 1]) / atr[i - 1])
        else:
            dmax = 0.0
        depth[i] = dmax

    # ---------- strata ----------
    C4 = comps_tf(d, "240min")
    radar = (C4["stack"] > 0) & (C4["s10"] >= 5.0)
    dates = np.array(d.index.date, dtype="datetime64[D]")
    fomc = np.isin(dates, FOMC)
    idx_dt = d.index
    nfp = (idx_dt.weekday == 4) & (idx_dt.day <= 7)
    news = fomc | np.asarray(nfp)
    sess = hours // 8                          # 0-7 / 8-15 / 16-23 server time

    def row(tag, m):
        m = m & valid
        if m.sum() < 30:
            print(f"  {tag:<34} n={m.sum()} (too few)")
            return
        r = win[m].mean() * 100
        b = beta_of(m) * 100
        unres = (~resolved[m]).mean() * 100
        print(f"  {tag:<34} n={m.sum():5d}  bounce={r:4.1f}%  beta={b:4.1f}%  "
              f"diff={r-b:+4.1f}pt  (unresolved {unres:.0f}%)")

    for name, ev in [("TOUCH-from-above", touch), ("RECLAIM cross-up", cross_up)]:
        print(f"\n===== {name} (win = +{BAR}ATR before -{BAR}ATR, next {K} bars) =====")
        row("all", ev)
        row("radar4h ON (up&>=5)", ev & radar)
        row("radar4h OFF", ev & ~radar)
        row("radar ON & news-day OUT", ev & radar & ~news)
        row("radar ON & news-day", ev & radar & news)
        for s, lab in [(0, "00-08h"), (1, "08-16h"), (2, "16-24h")]:
            row(f"radar ON & {lab}", ev & radar & (sess == s))
        if name.startswith("RECLAIM"):
            row("radar ON & stretch>=1 ATR", ev & radar & (depth >= 1.0))
            row("radar ON & stretch<1 ATR", ev & radar & (depth < 1.0))
        # per-year of the headline gated cell
        yr = d.index.year.values
        mm = ev & radar & valid
        ys = np.unique(yr[mm])
        print("  per-year (radar ON): " + "  ".join(
            f"{y}:{win[mm & (yr==y)].mean()*100:.0f}%(n={(mm&(yr==y)).sum()})" for y in ys))


if __name__ == "__main__":
    main()
