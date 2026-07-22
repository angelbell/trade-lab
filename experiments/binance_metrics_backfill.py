"""binance_metrics_backfill.py -- FREE Binance USDT-M futures positioning-metrics backfill
(cache, reusable). Source: https://data.binance.vision daily metrics dumps for BTCUSDT.

Columns kept (verified against a live sample, 2026-07-10): create_time, symbol,
sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio,
sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio.
5-min rows (288/day). Output: data/ext_btc_oi_metrics.csv, create_time as UTC index.

Idempotent: if the cache already exists, only missing calendar dates (earliest..yesterday)
are (re)fetched and merged in. Polite: requests.Session, ~0.05s sleep between calls, 404
(missing day) tolerated silently, ONE retry on a transient (non-404) error.

TIMEZONE: create_time strings carry no explicit tz suffix. Observed: each daily zip
BTCUSDT-metrics-YYYY-MM-DD.zip contains ONLY timestamps YYYY-MM-DD 00:00:00 ..
YYYY-MM-DD 23:55:00 -- i.e. the file's own calendar-date boundary IS a UTC calendar day
(Binance's stated convention for this dataset). We therefore parse create_time as UTC.

EARLIEST-DAY PROBE: spec's prior estimated earliest ~= 2021-12. Probing backward from
2022-01-01 (stop after 40 consecutive misses) actually walks straight through 2021 and
2020 to find the REAL earliest = 2020-09-01 (confirmed by direct download: that day's
file exists and returns 576 rows -- see DUPLICATE-ROW note below -- while 2020-08-31 and
every day before it for >40 consecutive days is 404). This contradicts the spec's ~2021-12
guess; reporting the measured value, not the prior.

DUPLICATE-ROW quirk (measured): every day from the true earliest (2020-09-01) through some
point before 2021-06-01 ships each 5-min create_time TWICE with byte-identical values (576
rows/day instead of 288). Deduped here via drop_duplicates(subset=create_time, keep='first').
"""
import os
import sys
import time
import zipfile
import io
import datetime as dt

import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(ROOT, "data", "ext_btc_oi_metrics.csv")

URL_TMPL = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip"
COLS = ["create_time", "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
SLEEP_S = 0.05
PROBE_START = dt.date(2022, 1, 1)
MAX_CONSEC_404 = 40


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (auto-trade research backfill)"})
    return s


def _fetch_day(s, d: dt.date, retried=False):
    """Return parsed daily DataFrame (indexed by UTC create_time, deduped), or None if
    the day is missing (404). Raises after one retry on a genuine transient error."""
    u = URL_TMPL.format(d=d.isoformat())
    try:
        r = s.get(u, timeout=30)
    except requests.exceptions.RequestException:
        if retried:
            raise
        time.sleep(0.5)
        return _fetch_day(s, d, retried=True)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        if retried:
            raise RuntimeError(f"{u} -> HTTP {r.status_code}")
        time.sleep(0.5)
        return _fetch_day(s, d, retried=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    df = pd.read_csv(io.BytesIO(z.read(name)))
    df = df[COLS].copy()
    df = df.drop_duplicates(subset="create_time", keep="first")
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.set_index("create_time").sort_index()
    return _drop_zero_oi(df)


def _drop_zero_oi(df: pd.DataFrame) -> pd.DataFrame:
    """Glitch guard (measured, not hypothetical): 502/615110 rows across the full history
    have sum_open_interest<=0 -- a real Binance-feed dropout, e.g. 2022-03-07 23:55 and
    2022-03-08 00:00 both print exactly 0 amid ~73k on either side, and 2025-07-21 has a
    burst of non-5min-aligned timestamps (04:25:26, 04:30:11, ...) during an apparent outage.
    OI is never legitimately 0 for a live BTCUSDT perp, so these rows are dropped (all
    columns, not just OI, since the whole snapshot is suspect) with a stderr warning --
    same spirit as src/data_loader.py's _drop_corrupt_bars for OHLCV."""
    bad = df["sum_open_interest"] <= 0
    if bad.any():
        print(f"  glitch guard: dropping {int(bad.sum())} row(s) with sum_open_interest<=0 "
              f"({df.index[bad].min()} .. {df.index[bad].max()})", file=sys.stderr)
        df = df[~bad]
    return df


def probe_earliest(s, start: dt.date, max_consec_404: int) -> dt.date | None:
    """Walk backward from `start` one day at a time using cheap HEAD requests; stop after
    `max_consec_404` consecutive misses. Returns the earliest day found available, or None."""
    d = start
    miss = 0
    earliest = None
    while miss < max_consec_404:
        u = URL_TMPL.format(d=d.isoformat())
        try:
            r = s.head(u, timeout=15)
            ok = r.status_code == 200
        except requests.exceptions.RequestException:
            ok = False
        if ok:
            earliest = d
            miss = 0
        else:
            miss += 1
        d -= dt.timedelta(days=1)
        time.sleep(SLEEP_S)
    return earliest


def main():
    s = _session()

    # ---------------- SMOKE: 3 days, verify parse + observe tz convention ----------------
    print("=== SMOKE TEST: 3 days ===")
    smoke_days = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    smoke_frames = []
    for d in smoke_days:
        df = _fetch_day(s, d)
        time.sleep(SLEEP_S)
        if df is None:
            print(f"  {d}: MISSING (unexpected for a smoke day)")
            continue
        smoke_frames.append(df)
        print(f"  {d}: rows={len(df)}  cols={list(df.columns)}  "
              f"idx[0]={df.index[0]}  idx[-1]={df.index[-1]}")
    if not smoke_frames:
        print("SMOKE FAILED: no rows parsed from 3 known-good days. STOPPING.")
        sys.exit(1)
    print("  timezone convention observed: create_time has NO explicit tz suffix in the raw "
          "CSV; each day's file spans exactly that calendar date's 00:00:00..23:55:00 with no "
          "cross-midnight spillover -> the file boundary IS a UTC calendar day. Parsed as UTC.")

    # ---------------- probe earliest available day ----------------
    print(f"\n=== PROBE: walking backward from {PROBE_START} (HEAD requests, stop after "
          f"{MAX_CONSEC_404} consecutive 404s) ===")
    earliest = probe_earliest(s, PROBE_START, MAX_CONSEC_404)
    if earliest is None:
        print("PROBE FAILED: nothing found walking back from PROBE_START. STOPPING.")
        sys.exit(1)
    print(f"  earliest available day found = {earliest}  "
          f"(spec's prior estimate was ~2021-12 -- MEASURED value differs, see module docstring)")

    yesterday = dt.date.today() - dt.timedelta(days=1)
    all_days = pd.date_range(earliest, yesterday, freq="D").date

    # ---------------- idempotent: figure out what's already cached ----------------
    existing = None
    have_days = set()
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV, index_col=0, parse_dates=[0])
        existing.index = pd.to_datetime(existing.index, utc=True)
        existing = _drop_zero_oi(existing)   # re-clean any glitch rows baked into an older cache
        have_days = set(existing.index.normalize().unique().date)
        print(f"\n=== existing cache found: {OUT_CSV}  ({len(existing)} rows, "
              f"{len(have_days)} distinct days) -- fetching only missing dates ===")
    else:
        print(f"\n=== no existing cache -- full backfill {earliest} -> {yesterday} "
              f"({len(all_days)} calendar days) ===")

    missing_days = [d for d in all_days if d not in have_days]
    print(f"  days to fetch: {len(missing_days)} / {len(all_days)}")

    frames = list(smoke_frames) if existing is None else []
    ok, miss404, err = 0, 0, 0
    t0 = time.time()
    for i, d in enumerate(missing_days):
        try:
            df = _fetch_day(s, d)
        except Exception as ex:
            err += 1
            print(f"  ERROR on {d}: {ex}", file=sys.stderr)
            df = None
        if df is None:
            miss404 += 1
        else:
            frames.append(df)
            ok += 1
        time.sleep(SLEEP_S)
        if (i + 1) % 200 == 0:
            print(f"  ...{d}  ok={ok} miss={miss404} err={err}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    if existing is not None:
        frames = [existing] + frames

    if not frames:
        print("Nothing fetched and no existing cache -- STOPPING (no output written).")
        sys.exit(1)

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV)

    span_days = (out.index[-1] - out.index[0]).days
    calendar_days = len(pd.date_range(out.index[0].date(), out.index[-1].date(), freq="D"))
    days_present = len(set(out.index.normalize().unique().date))
    pct_missing = (1 - days_present / calendar_days) * 100 if calendar_days else float("nan")

    print("\n=== BACKFILL DONE ===")
    print(f"  total rows        : {len(out)}")
    print(f"  span              : {out.index[0]} -> {out.index[-1]}  ({span_days} days)")
    print(f"  columns           : {list(out.columns)}")
    print(f"  calendar days in span : {calendar_days}")
    print(f"  days with data        : {days_present}")
    print(f"  % missing days        : {pct_missing:.2f}%")
    print(f"  this run: ok={ok}  404-missing={miss404}  errors={err}  "
          f"elapsed={time.time()-t0:.0f}s")
    print(f"  wrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
