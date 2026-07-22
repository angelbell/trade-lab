"""BTC weekly cycle structure -- the DESCRIPTIVE base rates, before any gate.

User hypothesis: "when BTC falls, it falls for about 10 weekly bars" -> is there a cycle/duration
structure at the weekly scale that btc15m_L could use?  (The 4-year halving cycle is NOT tested:
this sample holds ~2 cycles, so any halving gate would be fitted to 2 observations. Decline LEGS,
by contrast, number in the dozens -> measurable.)

Order of questions (base rate first, gate later -- the lab's rule):
  Q1  How long do weekly up/down legs actually last?  n, median, sd, quartiles, p90.
      Legs = amplitude ZigZag on WEEKLY closes (the lab's canonical swing detector), swept over
      the reversal threshold so the answer is not an artifact of one setting.
  Q2  THE DECIDING QUESTION -- is the HAZARD flat?
      P(the leg ends next week | it has already run k weeks).  If flat, "legs last ~10 weeks" is
      TRUE BUT USELESS: the process is memoryless and week 11 is no more likely to reverse than
      week 3.  If the hazard rises with age, there is timing information.
  Q3  Base rate for the halving cycle, printed with the n=2 caveat, for eyeballing only.
Run: .venv/bin/python experiments/btc_weekly_cycle_base.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv

ROOT = "/home/angelbell/dev/auto-trade"
HALVINGS = ["2016-07-09", "2020-05-11", "2024-04-20"]


def zigzag_legs(s, pct):
    """Amplitude ZigZag on a weekly close series. Returns legs as (start_i, end_i, direction)."""
    piv = [0]
    dirn = 0
    ext_i, ext = 0, s[0]
    for i in range(1, len(s)):
        if dirn >= 0:
            if s[i] > ext: ext_i, ext = i, s[i]
            elif (ext - s[i]) / ext >= pct:
                piv.append(ext_i); dirn = -1; ext_i, ext = i, s[i]
        if dirn <= 0:
            if s[i] < ext: ext_i, ext = i, s[i]
            elif (s[i] - ext) / ext >= pct:
                piv.append(ext_i); dirn = +1; ext_i, ext = i, s[i]
    legs = []
    for a, b in zip(piv[:-1], piv[1:]):
        legs.append((a, b, +1 if s[b] > s[a] else -1))
    return legs


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        h1 = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_h1.csv"))
    wk = h1["close"].resample("W").last().dropna()
    print(f"BTC weekly closes: {len(wk)} bars, {wk.index[0].date()} -> {wk.index[-1].date()} "
          f"({(wk.index[-1]-wk.index[0]).days/365.25:.1f} yr = "
          f"{(wk.index[-1]-wk.index[0]).days/365.25/4:.1f} halving cycles)\n")
    s = wk.values

    # ---- Q1: leg lengths ----------------------------------------------------
    print("Q1  weekly LEG LENGTHS (in weekly bars), by ZigZag reversal threshold")
    print(f"{'reversal':<10}{'dir':<6}{'n':>4}{'median':>8}{'mean':>7}{'sd':>7}"
          f"{'Q1':>5}{'Q3':>5}{'p90':>6}   {'lengths'}")
    store = {}
    for pct in (0.10, 0.15, 0.20, 0.25, 0.30):
        legs = zigzag_legs(s, pct)
        store[pct] = legs
        for d, nm in ((-1, "DOWN"), (+1, "UP")):
            L = np.array([b - a for a, b, dd in legs if dd == d])
            if len(L) == 0: continue
            tag = f"{int(pct*100)}%" if nm == "DOWN" else ""
            print(f"{tag:<10}{nm:<6}{len(L):>4}{np.median(L):>8.0f}{L.mean():>7.1f}{L.std():>7.1f}"
                  f"{np.percentile(L,25):>5.0f}{np.percentile(L,75):>5.0f}{np.percentile(L,90):>6.0f}"
                  f"   {sorted(L.tolist())}")

    # ---- Q2: the hazard ------------------------------------------------------
    print("\nQ2  HAZARD: P(the leg ends within the next week | it has already run k weeks)")
    print("    flat hazard  = memoryless = 'legs last ~10 weeks' is true but carries NO timing edge")
    print("    rising hazard = old legs really are more likely to turn -> a lever exists")
    for pct in (0.15, 0.20, 0.25):
        legs = store[pct]
        for d, nm in ((-1, "DOWN"), (+1, "UP")):
            L = np.array([b - a for a, b, dd in legs if dd == d])
            if len(L) < 6: continue
            print(f"\n    reversal {int(pct*100)}%  {nm} legs (n={len(L)}, median {np.median(L):.0f}w)")
            print(f"      {'age k (weeks)':<16}" + "".join(f"{f'{lo}-{hi}':>9}"
                  for lo, hi in ((1, 2), (3, 4), (5, 7), (8, 11), (12, 16), (17, 99))))
            surv, haz = [], []
            for lo, hi in ((1, 2), (3, 4), (5, 7), (8, 11), (12, 16), (17, 99)):
                at_risk = np.sum(L >= lo)                       # legs that reached age lo
                ended = np.sum((L >= lo) & (L <= hi))           # ... and ended inside [lo,hi]
                weeks = hi - lo + 1
                h = ended / at_risk / weeks if at_risk > 0 else np.nan   # per-week hazard
                surv.append(at_risk); haz.append(h)
            print(f"      {'legs still alive':<16}" + "".join(f"{v:>9d}" for v in surv))
            print(f"      {'hazard /week':<16}" + "".join(f"{v:>9.2f}" if np.isfinite(v) else f"{'--':>9}"
                                                          for v in haz))

    # ---- Q3: halving phase (eyeball only) -----------------------------------
    print("\nQ3  halving-cycle phase (EYEBALL ONLY -- n=2 cycles in this sample, not testable)")
    hv = pd.to_datetime(HALVINGS, utc=True)
    ret = wk.pct_change().dropna()
    phase = pd.Series([(t - hv[hv <= t][-1]).days / 365.25 if (hv <= t).any() else np.nan
                       for t in ret.index], index=ret.index)
    b = pd.cut(phase, [0, 1, 2, 3, 4], labels=["yr0-1", "yr1-2", "yr2-3", "yr3-4"])
    g = ret.groupby(b)
    print(f"      {'phase':<8}{'weeks':>7}{'mean wk ret':>13}{'median':>9}{'sd':>8}{'%up':>7}")
    for k, v in g:
        if len(v) == 0: continue
        print(f"      {str(k):<8}{len(v):>7}{100*v.mean():>12.2f}%{100*v.median():>8.2f}%"
              f"{100*v.std():>7.1f}%{100*(v>0).mean():>6.0f}%")
    print("      (2 cycles -> these are 2 observations dressed up as 4 buckets. Do not gate on this.)")


if __name__ == "__main__":
    main()
