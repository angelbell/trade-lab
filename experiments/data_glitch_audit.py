"""Audit: which feed-glitch bars SURVIVE the current _drop_corrupt_bars guard?

The guard compares close to a CENTERED 11-bar rolling median and flags >50% deviation.
That works for an ISOLATED bad print, but a RUN of >=6 consecutive corrupt bars poisons
the median itself, so the middle of the run looks perfectly normal and passes.
Known survivor: vantage_btcusd_m15.csv, 2020-08-10 01:00-03:45 = 12 bars priced ~296-301
while BTC was ~11,900 (a missing-leading-digit error).

This script loads every Vantage CSV RAW (bypassing the guard) and flags bars against a
WIDE centered median (201 bars) at an order-of-magnitude threshold, then reports which of
those the current guard already catches and which survive.

The wide/strict rule is safe by construction: no instrument moves +-80% versus the median
of the surrounding 200 bars. BTC's worst real bar (2020-03-12) is checked explicitly below.

Run: .venv/bin/python experiments/data_glitch_audit.py
"""
import sys, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")

WIDE = 201          # centered window for the run-detector
WIDE_DEV = 0.80     # order-of-magnitude corruption only
NARROW = 11         # what the current guard uses
NARROW_DEV = 0.50


def raw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close"]].astype(float)


def flags(df):
    med_n = df["close"].rolling(NARROW, center=True, min_periods=3).median()
    dev_n = (df["close"] / med_n - 1.0).abs()
    spike = (dev_n > NARROW_DEV) & med_n.notna()
    intrabar = (df["high"] / df["low"].where(df["low"] > 0)) > 3.0
    caught = spike | intrabar                                # what the guard drops today

    med_w = df["close"].rolling(WIDE, center=True, min_periods=50).median()
    dev_w = (df["close"] / med_w - 1.0).abs()
    run = (dev_w > WIDE_DEV) & med_w.notna()                 # the proposed run-detector
    return caught, run, med_w, dev_w


def runs_of(mask):
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return []
    out, s = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            out.append((s, a)); s = b
    out.append((s, idx[-1]))
    return out


def main():
    files = sorted(glob.glob("/home/angelbell/dev/auto-trade/data/vantage_*.csv"))
    print(f"{len(files)} Vantage CSVs\n")
    total_new = 0
    for f in files:
        df = raw(f)
        caught, run, med_w, dev_w = flags(df)
        survivors = run & ~caught
        if not survivors.any() and not caught.any():
            continue
        name = os.path.basename(f)
        print(f"--- {name}   n={len(df):,}  {df.index[0].date()} -> {df.index[-1].date()}")
        if caught.any():
            print(f"    guard already drops : {int(caught.sum())} bar(s)")
        for s, e in runs_of(survivors):
            seg = df.iloc[s:e + 1]
            total_new += len(seg)
            print(f"    SURVIVES the guard  : {len(seg):2d} bar(s)  {seg.index[0]} -> {seg.index[-1]}"
                  f"  close {seg['close'].min():.2f}-{seg['close'].max():.2f}"
                  f"  vs local median {med_w.iloc[s:e+1].median():.2f}")
        print()

    print(f"TOTAL bars the proposed run-detector would newly drop: {total_new}\n")

    # --- false-positive check: does the wide/strict rule ever flag a REAL move? -------------
    print("false-positive check -- worst |close/median(201)-1| on real crash windows:")
    for f, when in (("vantage_btcusd_m15.csv", "2020-03-11:2020-03-14"),
                    ("vantage_btcusd_m15.csv", "2021-05-18:2021-05-21"),
                    ("vantage_btcusd_h1.csv", "2020-03-11:2020-03-14"),
                    ("vantage_xauusd_m15.csv", "2020-03-08:2020-03-20"),
                    ("vantage_usdjpy_m15.csv", "2022-10-20:2022-10-23")):
        p = f"/home/angelbell/dev/auto-trade/data/{f}"
        if not os.path.exists(p):
            continue
        df = raw(p)
        _, _, _, dev_w = flags(df)
        a, b = when.split(":")
        w = dev_w.loc[a:b].dropna()
        if len(w):
            print(f"    {f:<26}{when:<24}max dev {w.max():.3f}   "
                  f"(threshold {WIDE_DEV}) {'FLAGGED!!' if w.max() > WIDE_DEV else 'clean'}")


if __name__ == "__main__":
    main()
