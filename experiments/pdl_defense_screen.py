"""pdl_defense_screen.py -- is "defended the prior-day low -> next day bullish" real or hindsight?

PART 1 (daily conditional, gold 19yr + BTC 9yr, from h1->1D):
  event day D vs prior-day low pdl = low[D-1]:
    sweep&reclaim : low[D] < pdl  and close[D] > pdl   (broke intraday, closed back above)
    held-above    : low[D] >= pdl                       (never touched -- control)
    closed-below  : close[D] <= pdl                     (breakdown -- control)
  outcome: next-day candle P(close>open), and (close-open)/ATRd distribution (mean/median/std).
  beta = unconditional next-day stats. Strata: trend context (close vs SMA150d), IS/OOS halves,
  per-year for the headline. Also same-day check: P(day D itself closes green | sweep happened
  before close is NOT knowable causally -> only next-day is a tradeable daily signal).
PART 2 (intraday scalp version, gold 5m): event = first 5m close back above pdl after trading
  below it today (sweep depth recorded). Outcomes: (a) +-0.5/1.0 ATR barrier race vs same-hour
  beta (same machinery as vwap_line_screen), (b) drift to the day close in ATRs vs same-hour
  beta drift. Strata: radar4h, sweep depth, first-of-day only.
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

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def part1(name, csv):
    d1 = load_mt5_csv(csv).resample("1D").agg(AGG).dropna()
    o, h, l, c = (d1[k].values for k in ("open", "high", "low", "close"))
    atr = ta.atr(d1["high"], d1["low"], d1["close"], 14).shift(1).values
    n = len(c)
    pdl = np.roll(l, 1)                      # prior-day low, known during day D
    sma = d1["close"].rolling(150).mean().shift(1).values

    nxt_up = np.roll((c > o), -1)            # next-day candle green?
    nxt_r = np.roll((c - o), -1) / atr       # next-day body in ATRs (uses day-D atr ~ fine)
    valid = (~np.isnan(atr)) & (np.arange(n) >= 1) & (np.arange(n) < n - 1)

    ev = {
        "sweep&reclaim": valid & (l < pdl) & (c > pdl),
        "held-above": valid & (l >= pdl),
        "closed-below": valid & (c <= pdl),
        "BETA (all days)": valid,
    }
    print(f"\n===== PART1 {name} daily ({d1.index[0].date()}->{d1.index[-1].date()}, "
          f"{valid.sum()} days) =====")
    print(f"  {'set':<28} {'n':>5} {'P(nextUP)':>9} {'meanR':>7} {'medR':>7} {'stdR':>6}")
    for tag, m in ev.items():
        r = nxt_r[m]
        print(f"  {tag:<28} {m.sum():>5} {nxt_up[m].mean()*100:>8.1f}% {r.mean():>+7.3f} "
              f"{np.median(r):>+7.3f} {r.std():>6.2f}")
    # trend-context + robustness strata on the candidate
    m0 = ev["sweep&reclaim"]
    yrs = d1.index.year.values
    half = np.median(yrs)
    strata = [("  above SMA150d (順張り側)", m0 & (c > sma)),
              ("  below SMA150d (逆張り側)", m0 & (c <= sma)),
              ("  IS (front half)", m0 & (yrs < half)),
              ("  OOS (back half)", m0 & (yrs >= half))]
    for tag, m in strata:
        if m.sum() < 20:
            print(f"  {tag:<28} n={m.sum()} too few"); continue
        r = nxt_r[m]
        print(f"  {tag:<28} {m.sum():>5} {nxt_up[m].mean()*100:>8.1f}% {r.mean():>+7.3f} "
              f"{np.median(r):>+7.3f} {r.std():>6.2f}")
    print("  per-year P(nextUP) sweep&reclaim vs beta:")
    line = []
    for y in np.unique(yrs):
        m, b = m0 & (yrs == y), ev["BETA (all days)"] & (yrs == y)
        if m.sum() >= 8:
            line.append(f"{y}:{nxt_up[m].mean()*100:.0f}/{nxt_up[b].mean()*100:.0f}(n={m.sum()})")
    print("   " + "  ".join(line))


def part2():
    d = load_mt5_csv("data/vantage_xauusd_m5.csv")
    cnt = d.groupby(d.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= 150]
    d = d[d.index.date >= ok.index[0]]
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c)
    day = pd.Series(d.index.date, index=d.index)
    dlow = d["low"].groupby(day).min()
    pdl = dlow.shift(1).reindex(day.values).values          # prior BROKER day's low
    dclose_px = d["close"].groupby(day).transform("last").values
    is_last_day = day.values == day.values[-1]

    K, BAR = 72, 0.5
    up_lvl, dn_lvl = c + BAR * atr, c - BAR * atr
    t_up = np.full(n, K + 1, np.int32); t_dn = np.full(n, K + 1, np.int32)
    for k in range(1, K + 1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n - k], ls[:n - k] = h[k:], l[k:]
        hs[n - k:], ls[n - k:] = -np.inf, np.inf
        t_up = np.where((t_up > K) & (hs >= up_lvl), k, t_up)
        t_dn = np.where((t_dn > K) & (ls <= dn_lvl), k, t_dn)
    win = (np.minimum(t_up, t_dn) <= K) & (t_up < t_dn)
    drift = (dclose_px - c) / atr                            # to-day-close drift in ATRs
    valid = ~np.isnan(atr) & ~np.isnan(pdl) & (np.arange(n) < n - K) & ~is_last_day

    hours = d.index.hour.values
    beta_w = {hh: win[valid & (hours == hh)].mean() for hh in range(24)}
    beta_d = {hh: drift[valid & (hours == hh)].mean() for hh in range(24)}

    below = c < pdl
    was_below_today = pd.Series(below).groupby(day.values).cummax().values.astype(bool)
    reclaim = valid & (c > pdl) & np.roll(below, 1) & np.roll(was_below_today, 1)
    first = reclaim & ~(pd.Series(reclaim & True).groupby(day.values).cumsum().values > 1)
    depth = (pdl - np.where(l < pdl, l, np.nan))             # not causal per-bar; use day-run min below
    run_min = pd.Series(np.where(below, l, np.nan)).groupby(day.values).cummin().shift(1).values
    sweep_atr = (pdl - run_min) / atr

    C4 = comps_tf(d, "240min")
    radar = (C4["stack"] > 0) & (C4["s10"] >= 5.0)

    def row(tag, m):
        m = m & valid
        if m.sum() < 30:
            print(f"  {tag:<30} n={m.sum()} too few"); return
        hh = hours[m]
        bw = np.mean([beta_w[x] for x in hh]) * 100
        bd = np.mean([beta_d[x] for x in hh])
        dr = drift[m]
        print(f"  {tag:<30} n={m.sum():5d}  race={win[m].mean()*100:4.1f}% (beta {bw:4.1f})  "
              f"toClose mean={dr.mean():+.3f} med={np.median(dr):+.3f} std={dr.std():.2f} "
              f"(beta {bd:+.3f})")

    print(f"\n===== PART2 gold 5m: reclaim of PRIOR-DAY LOW (race +-0.5ATR | drift to day close) =====")
    row("all reclaims", reclaim)
    row("first-of-day only", first)
    row("first & radar4h ON", first & radar)
    row("first & radar4h OFF", first & ~radar)
    row("first & sweep>=0.5 ATR", first & (sweep_atr >= 0.5))
    row("first & sweep<0.5 ATR", first & (sweep_atr < 0.5))
    yr = d.index.year.values
    line = []
    for y in np.unique(yr):
        m = first & valid & (yr == y)
        if m.sum() >= 20:
            hh = hours[m]
            bd = np.mean([beta_d[x] for x in hh])
            line.append(f"{y}:{drift[m].mean():+.2f}/b{bd:+.2f}(n={m.sum()})")
    print("  per-year toClose drift (first-of-day) vs beta:\n   " + "  ".join(line))


if __name__ == "__main__":
    part1("GOLD", "data/vantage_xauusd_h1.csv")
    part1("BTC", "data/vantage_btcusd_h1.csv")
    part2()
