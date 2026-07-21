"""案2: 経済カレンダー近接 STEP1 -- FOMC/CPI/NFPの日時を取得し、gold15m/BTC15mの先行への
影響を層別(±4h / ±24h / それ以外)で測る。

データ取得:
  - FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm (現行=2021-2027) +
    web.archive.org の2020-01-02スナップショット(2014-2020を保持) -> 声明発表日時は各会合の
    最終日 14:00 America/New_York(公式に明記された時刻; 声明"内容"は使わない、日時のみ)。
  - CPI/NFP: bls.govは直接fetchすると403(bot遮断)なので web.archive.org 経由でスナップショットを
    取得。各スナップショットの"Schedule of Releases"表は前後合わせて約14か月分を載せているため、
    約9か月おきに取得して2018-2026を全期間カバーする。表の値は "Release Date"+"Release Time"列
    (08:30 AM ET)をそのまま使う=近似ではなく公式表の転記。

Run: .venv/bin/python scratchpad/lat2_econ_calendar.py [--smoke]
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

from src.data_loader import load_mt5_csv
from lat_common import (forward_direction, forward_magnitude, layer_table,
                         month_block_bootstrap_diff, print_tz_check, utc_to_broker_index)

ROOT = "/home/angelbell/dev/auto-trade"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/120.0 Safari/537.36"}
SESSION = requests.Session()
SESSION.headers.update(UA)


def http_get(url, tries=4, timeout=40, sleep=3):
    last_err = None
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep)
    return None, last_err


def get(url, tries=4, timeout=40, sleep=3):
    last_err = None
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text, None
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep)
    return None, last_err


LOG = []  # (source, url, ok, note)


def log(source, url, ok, note=""):
    LOG.append((source, url, ok, note))
    print(f"[fetch] {source}: {'OK' if ok else 'FAIL'} {url} {note}")


# ---------------------------------------------------------------- FOMC -----
def fetch_fomc():
    """Statement date = last day of the 2-day meeting, 14:00 America/New_York."""
    rows = []

    def parse_page(html):
        # sections: <a id="...">YYYY FOMC Meetings</a>  then repeating month/date divs
        out = []
        year_spans = [(m.start(), m.group(1)) for m in
                      re.finditer(r'<a id="\d+">(\d{4}) FOMC Meetings</a>', html)]
        year_spans.append((len(html), None))
        for (start, year), (end, _) in zip(year_spans, year_spans[1:]):
            block = html[start:end]
            for mm in re.finditer(
                r'fomc-meeting__month[^"]*"[^>]*><strong>([\w/]+)</strong></div>\s*'
                r'<div class="fomc-meeting__date[^"]*"[^>]*>([^<]+)</div>', block):
                monthspec, dayspec = mm.group(1), mm.group(2).strip()
                # cross-month meetings render as e.g. "Jul/Aug" -- the LAST day of the
                # meeting (statement date) falls in the LAST month listed.
                month = monthspec.split("/")[-1]
                dayspec = dayspec.replace("*", "").strip()
                # dayspec like "27-28" or "31" or "30-31"
                last_day = dayspec.split("-")[-1].strip()
                try:
                    dt = pd.Timestamp(f"{month} {last_day} {year} 14:00")
                except Exception:
                    continue
                out.append(dt)
        return out

    html, err = get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    if html:
        log("FOMC-live(2021-2027)", "federalreserve.gov/monetarypolicy/fomccalendars.htm", True)
        rows += parse_page(html)
    else:
        log("FOMC-live", "federalreserve.gov/monetarypolicy/fomccalendars.htm", False, err)

    wb_url = ("https://web.archive.org/web/20200102204925/"
              "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    html, err = get(wb_url)
    if html:
        log("FOMC-wayback(2014-2020)", wb_url, True)
        rows += parse_page(html)
    else:
        log("FOMC-wayback", wb_url, False, err)

    dts = sorted(set(rows))
    dts = [d for d in dts if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2026-12-31")]
    return pd.DataFrame({"kind": "FOMC", "dt_local": dts})


# --------------------------------------------------------- CPI / NFP -------
CDX_API = "http://web.archive.org/cdx/search/cdx"


def find_snapshot(path, target_ym, window_months=6):
    """Find a wayback snapshot of bls.gov{path} near target_ym (YYYYMM string),
    searching [target, target+window_months]. Returns (timestamp, url) or (None, err)."""
    start = pd.Period(target_ym, freq="M")
    end = start + window_months
    frm, to = start.strftime("%Y%m01"), end.strftime("%Y%m28")
    url = (f"{CDX_API}?url=bls.gov{path}&output=json&limit=5&from={frm}&to={to}"
           f"&filter=statuscode:200")
    txt, err = get(url, tries=5, timeout=60, sleep=4)
    if not txt:
        return None, err
    try:
        import json
        rows = json.loads(txt)
    except Exception as e:
        return None, str(e)
    if len(rows) < 2:
        return None, "no snapshot in window"
    ts = rows[1][1]
    return ts, None


def parse_release_table(html, kind):
    i = html.find("Reference")
    if i < 0:
        return []
    block = html[i:i + 20000]
    out = []
    for mm in re.finditer(
        r'<td>([A-Za-z]+ \d{4})</td>\s*<td>([A-Za-z]{3,4}\.? \d{1,2}, \d{4})</td>\s*'
        r'<td>(\d{1,2}:\d{2} [AP]M)</td>', block):
        ref_month, rel_date, rel_time = mm.groups()
        rel_date_clean = rel_date.replace(".", "")
        try:
            dt = pd.Timestamp(f"{rel_date_clean} {rel_time}")
        except Exception:
            continue
        out.append(dt)
    return out


def fetch_bls_series(path, kind, targets):
    dts = []
    for ym in targets:
        ts, err = find_snapshot(path, ym)
        if ts is None:
            log(f"{kind}-wayback({ym})", f"bls.gov{path}", False, err)
            continue
        wb_url = f"https://web.archive.org/web/{ts}/https://www.bls.gov{path}"
        html, err2 = get(wb_url)
        if not html:
            log(f"{kind}-wayback({ym})", wb_url, False, err2)
            continue
        found = parse_release_table(html, kind)
        log(f"{kind}-wayback({ym})", wb_url, True, f"{len(found)} rows parsed")
        dts += found
    dts = sorted(set(dts))
    dts = [d for d in dts if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2026-12-31")]
    return pd.DataFrame({"kind": kind, "dt_local": dts})


TARGET_YMS = ["2017-11", "2018-08", "2019-05", "2020-02", "2020-11",
              "2021-08", "2022-05", "2023-02", "2023-11", "2024-08", "2025-05", "2026-02"]


def build_calendar(smoke=False):
    fomc = fetch_fomc()
    targets = TARGET_YMS[:3] if smoke else TARGET_YMS
    cpi = fetch_bls_series("/schedule/news_release/cpi.htm", "CPI", targets)
    nfp = fetch_bls_series("/schedule/news_release/empsit.htm", "NFP", targets)

    print(f"\n[counts] FOMC={len(fomc)} (目安 ~8/年) CPI={len(cpi)} (目安 ~12/年) "
          f"NFP={len(nfp)} (目安 ~12/年)")
    for name, df in [("FOMC", fomc), ("CPI", cpi), ("NFP", nfp)]:
        if len(df):
            yrs = df["dt_local"].dt.year.value_counts().sort_index()
            print(f"  {name} per-year: {dict(yrs)}")

    allcal = pd.concat([fomc, cpi, nfp], ignore_index=True).sort_values("dt_local")
    # localize: FOMC/CPI/NFP local times are America/New_York (ET) -> UTC -> broker (Europe/Riga)
    allcal["dt_ny"] = allcal["dt_local"].dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    allcal = allcal.dropna(subset=["dt_ny"])
    allcal["dt_utc"] = allcal["dt_ny"].dt.tz_convert("UTC")
    allcal["dt_broker"] = utc_to_broker_index(pd.DatetimeIndex(allcal["dt_utc"]))
    out = allcal[["kind", "dt_utc", "dt_broker"]].drop_duplicates().sort_values("dt_utc")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cal = build_calendar(smoke=args.smoke)
    csv_path = f"{ROOT}/data/ext_econ_calendar.csv"
    cal_out = cal.copy()
    cal_out["dt_utc"] = cal_out["dt_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cal_out["dt_broker"] = cal_out["dt_broker"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cal_out.to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path} n={len(cal_out)}")

    # tz check samples: one Jan event, one Jul event (if present)
    jan_rows = cal[cal["dt_utc"].dt.month == 1]
    jul_rows = cal[cal["dt_utc"].dt.month == 7]
    if len(jan_rows) and len(jul_rows):
        print_tz_check(jan_rows["dt_utc"].iloc[0].tz_localize(None),
                        jul_rows["dt_utc"].iloc[0].tz_localize(None), label="econ_calendar")
    else:
        print("[tz check] insufficient Jan/Jul rows for sample check")

    if args.smoke:
        print("\n[smoke] calendar fetch OK, skipping measurement pass")
        return

    run_measurement(cal)


def run_measurement(cal: pd.DataFrame):
    print("\n" + "=" * 70)
    print("測定: gold15m / BTC15m の発表近接レイヤー別 先行(方向/量)")
    print("=" * 70)

    with pd.option_context("mode.chained_assignment", None):
        gold = load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m15.csv").loc["2018-01-01":]
        btc = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-01-01":]

    events_broker = pd.DatetimeIndex(sorted(cal["dt_broker"]))

    for name, df in [("gold15m", gold), ("BTC15m", btc)]:
        print(f"\n--- {name} ---")
        idx = df.index
        # time-to-nearest-event and time-since-nearest-event, in hours (both directions)
        pos = np.searchsorted(events_broker.values, idx.values)
        pos = np.clip(pos, 1, len(events_broker) - 1)
        before = events_broker.values[pos - 1]
        after = events_broker.values[pos]
        dt_before_h = (idx.values - before) / np.timedelta64(1, "h")
        dt_after_h = (after - idx.values) / np.timedelta64(1, "h")
        nearest_h = np.minimum(dt_before_h, dt_after_h)

        layer = np.where(nearest_h <= 4, "A:±4h",
                 np.where(nearest_h <= 24, "B:±24h", "C:other"))

        for H, label in [(16, "H=16本(4h)"), (96, "H=96本(24h)")]:
            direction = forward_direction(df["close"], H)
            magnitude = forward_magnitude(df["close"], H)
            work = pd.DataFrame({"layer": layer, "direction": direction.values,
                                  "magnitude": magnitude.values}, index=idx)
            print(f"\n  [{label}] 方向 log-return:")
            print(layer_table(work, "layer", "direction").to_string(index=False))
            print(f"  [{label}] 量 sum|abs log-return|:")
            print(layer_table(work, "layer", "magnitude").to_string(index=False))

            med, p2, p97, nb = month_block_bootstrap_diff(
                work, "layer", "magnitude", "A:±4h", "C:other", n_boot=1000)
            print(f"  量差(A:±4h - C:other) 月次ブロックブートストラップ: "
                  f"median={med:.5f} 95%CI=[{p2:.5f}, {p97:.5f}] (n_boot={nb})")

            era_before = work[work.index < "2022-01-01"]
            era_after = work[work.index >= "2022-01-01"]
            print(f"  [{label}] 2022-01-01前 (n={len(era_before.dropna())}):")
            print(layer_table(era_before, "layer", "magnitude").to_string(index=False))
            print(f"  [{label}] 2022-01-01後 (n={len(era_after.dropna())}):")
            print(layer_table(era_after, "layer", "magnitude").to_string(index=False))

        # per-event-kind table (H=16 only, to keep output manageable)
        print(f"\n  --- イベント種別ごと (H=16, ±4h窓のみ) ---")
        for kind in ["FOMC", "CPI", "NFP"]:
            kevents = pd.DatetimeIndex(sorted(cal.loc[cal["kind"] == kind, "dt_broker"]))
            if len(kevents) == 0:
                continue
            pos_k = np.searchsorted(kevents.values, idx.values)
            pos_k = np.clip(pos_k, 1, len(kevents) - 1)
            bef_k = kevents.values[pos_k - 1]
            aft_k = kevents.values[pos_k]
            near_k = np.minimum((idx.values - bef_k) / np.timedelta64(1, "h"),
                                 (aft_k - idx.values) / np.timedelta64(1, "h"))
            lyr_k = np.where(near_k <= 4, "A:±4h", "C:other")
            direction16 = forward_direction(df["close"], 16)
            magnitude16 = forward_magnitude(df["close"], 16)
            work_k = pd.DataFrame({"layer": lyr_k, "direction": direction16.values,
                                    "magnitude": magnitude16.values}, index=idx)
            t1 = layer_table(work_k, "layer", "direction")
            t2 = layer_table(work_k, "layer", "magnitude")
            print(f"   [{kind}] 方向:\n{t1.to_string(index=False)}")
            print(f"   [{kind}] 量:\n{t2.to_string(index=False)}")


if __name__ == "__main__":
    main()
