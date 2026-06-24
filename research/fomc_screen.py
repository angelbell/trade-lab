"""fomc_screen.py -- does the gold breakout die on FOMC "fakeouts"? (the EVENT-TIME axis)

Hypothesis (user): gold breakouts near FOMC get faked out -- price pokes the level, the 14:00-ET
statement reverses it, the trade dies. If real, skipping near-FOMC entries lifts the edge. We TEST
the belief, not assume it: Part 1 splits trades by signed days-to-nearest-FOMC and checks the near-
event meanR deficit against a reshuffle null (random fake-event dates); if the deficit isn't beyond
the null the fakeout is a MYTH and we STOP. Only then (Part 2) the gate, vs a RANDOM same-count
exclusion (cutting trades flatters CAGR/DD by variance alone) + a window plateau. Part 3 contrasts
BTC (weaker FOMC sensitivity) -- a real macro effect should be STRONGER on gold than BTC.

Caveats: FOMC dates HARDCODED (scheduled meetings 2015-2026, 2nd-day statement; emergency 2020 cuts
excluded) -- small accuracy risk. DAY-level (robust to MT5-broker-time vs ET offset; announce 14:00ET
~= 20-22:00 broker = same calendar day). In-sample only; live-forward arbitrates.

  .venv/bin/python research/fomc_screen.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.portfolio_kama import get_legs
from research.regime_gate_lab import metrics, SPLIT

RNG = np.random.default_rng(7)

# FOMC statement-release dates (2nd meeting day, 14:00 ET), scheduled meetings 2015-2026.
FOMC = [
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-18","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17",
]
FOMC = np.array(pd.to_datetime(FOMC).values, dtype="datetime64[D]")


def to_days(times):
    """entry timestamps -> tz-naive datetime64[D] ndarray (robust to tz-aware or naive input)."""
    idx = pd.to_datetime(pd.Series(times).values)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.to_numpy().astype("datetime64[D]")


def signed_days(times):
    """signed days from each entry to the NEAREST FOMC date (neg=before, 0=same day, pos=after)."""
    e = to_days(times)
    out = np.empty(len(e), int)
    for i, d in enumerate(e):
        diff = (d - FOMC).astype(int)
        out[i] = diff[np.argmin(np.abs(diff))]
    return out


def bucket_stats(t, d):
    buckets = {"far(|d|>2)": np.abs(d) > 2, "pre(-2..-1)": (d < 0) & (d >= -2),
               "event(0)": d == 0, "post(+1..+2)": (d > 0) & (d <= 2)}
    print(f"    {'bucket':<13}{'n':>5}{'meanR':>8}{'win%':>7}{'totR':>8}")
    for name, m in buckets.items():
        r = t.R.values[m]
        if len(r) == 0:
            print(f"    {name:<13}{0:>5}"); continue
        print(f"    {name:<13}{len(r):>5}{r.mean():>+8.3f}{(r>0).mean()*100:>6.0f}%{r.sum():>+8.0f}")
    near = np.abs(d) <= 2
    return t.R.values[near].mean() - t.R.values[~near].mean(), near.mean()


def reshuffle_p(t, observed_deficit, n_events, span_days, start, B=2000):
    """null: random fake-event weekday dates, same count -> near-minus-far meanR deficit distribution."""
    alld = pd.bdate_range(start, periods=span_days)
    alld = alld.to_numpy().astype("datetime64[D]")
    e = to_days(t.time)
    null = np.empty(B)
    for b in range(B):
        fake = np.sort(RNG.choice(alld, min(n_events, len(alld)), replace=False))
        sd = np.array([((d - fake).astype(int))[np.argmin(np.abs((d - fake).astype(int)))] for d in e])
        near = np.abs(sd) <= 2
        null[b] = t.R.values[near].mean() - t.R.values[~near].mean() if near.any() and (~near).any() else 0.0
    return (null <= observed_deficit).mean()       # small p = real (near genuinely worse than random)


def random_excl_pctile(t, gated_cdd, keep_n, draws=300):
    vals = []
    for _ in range(draws):
        idx = np.sort(RNG.choice(len(t), keep_n, replace=False))
        m = metrics(t.iloc[idx])
        if m:
            vals.append(m["cdd"])
    vals = np.array(vals)
    return (vals < gated_cdd).mean() * 100, np.median(vals)


def part1(name, t):
    d = signed_days(t.time)
    inrange = (pd.to_datetime(t.time).dt.year >= 2015)
    t = t[inrange].reset_index(drop=True); d = d[inrange.values]
    print(f"\n  [{name}] n={len(t)}  -- bucket split by days-to-nearest-FOMC --")
    deficit, nearfrac = bucket_stats(t, d)
    start = pd.to_datetime(t.time.min()).tz_localize(None)
    span = (pd.to_datetime(t.time.max()) - pd.to_datetime(t.time.min())).days + 5
    nev = int(((FOMC >= np.datetime64(start, "D")) & (FOMC <= np.datetime64(pd.to_datetime(t.time.max()).tz_localize(None), "D"))).sum())
    p = reshuffle_p(t, deficit, nev, span, start)
    verdict = "REAL deficit" if (deficit < 0 and p < 0.10) else "no real effect (myth)"
    print(f"    near(|d|<=2)-far meanR deficit = {deficit:+.3f} (near {nearfrac*100:.0f}% of trades)  "
          f"reshuffle-p={p:.3f}  -> {verdict}")
    return t, d, (deficit < 0 and p < 0.10)


def main():
    legs = get_legs()
    print("FOMC fakeout screen -- bar: near-FOMC genuinely worse than random fake-dates, then gate>random+plateau")
    print(f"FOMC dates: {len(FOMC)} ({FOMC.min()}..{FOMC.max()}); weekend check: "
          f"{int(pd.to_datetime(FOMC).weekday.isin([5,6]).sum())} weekend (should be 0)")

    print("\n" + "=" * 78 + "\n1. FAKEOUT EXISTS? gold (primary)")
    gt, gd, gold_real = part1("gold_bo", legs["gold_bo"])

    if gold_real:
        print("\n" + "=" * 78 + "\n2. GATE: exclude entries within +-W days of FOMC (vs random same-count exclusion)")
        base = metrics(gt)
        print(f"  ungated gold_bo CAGR/DD={base['cdd']:.2f} (n={base['n']}, IS={base['isr']:+.2f} OOS={base['oos']:+.2f})")
        for W in (1, 2, 3):
            keep = np.abs(gd) > W
            g = gt[keep]
            m = metrics(g)
            if m is None:
                print(f"    W={W} (too few)"); continue
            pct, rmed = random_excl_pctile(gt, m["cdd"], int(keep.sum()))
            tag = "PASS?" if (m["cdd"] > base["cdd"] and pct >= 90) else ""
            print(f"    W={W}  CAGR/DD={m['cdd']:5.2f} (keep {keep.mean()*100:3.0f}%) "
                  f"IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | rand p{pct:3.0f} (med {rmed:.2f}) {tag}")
    else:
        print("\n  >>> gold shows no real near-FOMC deficit -> fakeout is a MYTH for the gold breakout. "
              "No gate fishing. (Part 2 skipped.)")

    print("\n" + "=" * 78 + "\n3. CROSS-INSTRUMENT CONTRAST: BTC (weaker FOMC sensitivity -- effect should be < gold)")
    part1("btc_bo_kama", legs["btc_bo_kama"])
    print("\n" + "=" * 78)
    print("Read: a true FOMC-fakeout = gold near-event deficit beats reshuffle null AND a +-W gate beats")
    print("random exclusion at >=90 pctile across W (plateau), STRONGER on gold than BTC. In-sample only.")


if __name__ == "__main__":
    main()
