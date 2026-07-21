"""案8: オンチェーンMVRVの入手性と冗長性 STEP1。

データ取得: coinmetrics community API (CapMVRVCur, BTC, 日次)。
  無印(paging_from省略)だと直近ページのみが返る仕様に注意 -- `paging_from=start` を付けて
  始点からページングする(2026-07-18に手動curlで確認済み: 2010-07-18始まり、single page_size=10000
  で全期間が1ページに収まる)。

冗長性 = MVRV>全期間中央値 と btc_pull の週足サイクルゲート(既存実装
  research.portfolio_kama.cycle_gate_pull と同じ定義: 週足終値 <= 30週SMA×1.10、
  BTC 4h(h1リサンプル)ベース) の週次一致率。
単体 = MVRV五分位 -> BTC日足の先行方向/量(H=7日・28日)。

Run: .venv/bin/python scratchpad/lat8_mvrv.py [--smoke]
"""
import argparse
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
from breakout_wave import resample
from lat_common import (forward_direction, forward_magnitude, layer_table,
                         month_block_bootstrap_diff, print_tz_check, utc_to_broker_index)

ROOT = "/home/angelbell/dev/auto-trade"
CM_API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

TRIED = []


def try_source(name, url, ok, note=""):
    TRIED.append((name, url, ok, note))
    print(f"[fetch] {name}: {'OK' if ok else 'FAIL'} {url} {note}")


def fetch_mvrv_coinmetrics():
    """paging_from=start is the key param (confirmed manually 2026-07-18): without it the
    community API returns only the most-recent page regardless of start_time/sort."""
    url = (f"{CM_API}?assets=btc&metrics=CapMVRVCur&frequency=1d&page_size=10000"
           f"&start_time=2009-01-01&paging_from=start")
    all_rows = []
    next_url = url
    pages = 0
    while next_url:
        try:
            r = requests.get(next_url, timeout=60)
        except Exception as e:
            try_source("coinmetrics MVRV", next_url, False, str(e))
            return None
        if r.status_code != 200:
            try_source("coinmetrics MVRV", next_url, False, f"HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        rows = data.get("data", [])
        all_rows += rows
        pages += 1
        next_url = data.get("next_page_url")
        time.sleep(0.2)
    try_source("coinmetrics MVRV", url, True, f"{pages} page(s), {len(all_rows)} rows")
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    df["dt_utc"] = pd.to_datetime(df["time"], utc=True)
    df["mvrv"] = df["CapMVRVCur"].astype(float)
    return df[["dt_utc", "mvrv"]].sort_values("dt_utc").drop_duplicates("dt_utc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    mvrv = fetch_mvrv_coinmetrics()
    if mvrv is None or mvrv["dt_utc"].min() > pd.Timestamp("2017-01-01", tz="UTC"):
        print("\n[結果] MVRV(または近い代替)の2017年以前からの日次履歴を、無料ソースから取得できなかった。")
        print("試したソース/URL/エラー:")
        for name, url, ok, note in TRIED:
            print(f"  - {name}: {url} ok={ok} {note}")
        print("測定は打ち切り。")
        return

    start = mvrv["dt_utc"].min()
    full_range = pd.date_range(start, mvrv["dt_utc"].max(), freq="D", tz="UTC")
    missing = full_range.difference(mvrv["dt_utc"])
    print(f"\n[counts] MVRV n={len(mvrv)} start={start} end={mvrv['dt_utc'].max()} "
          f"欠損日数={len(missing)}")

    out = mvrv.copy()
    out["dt_utc"] = out["dt_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(f"{ROOT}/data/ext_btc_mvrv.csv", index=False)
    print(f"[saved] {ROOT}/data/ext_btc_mvrv.csv n={len(out)}")

    jan_rows = mvrv[mvrv["dt_utc"].dt.month == 1]
    jul_rows = mvrv[mvrv["dt_utc"].dt.month == 7]
    print_tz_check(jan_rows["dt_utc"].iloc[0].tz_localize(None),
                    jul_rows["dt_utc"].iloc[0].tz_localize(None), label="mvrv")

    if args.smoke:
        print("\n[smoke] MVRV fetch OK, skipping measurement pass")
        return

    run_measurement(mvrv)


def run_measurement(mvrv: pd.DataFrame):
    print("\n" + "=" * 70)
    print("測定: MVRV>中央値 vs btc_pull週足サイクルゲート の一致率、五分位->先行")
    print("=" * 70)

    mv = mvrv.set_index("dt_utc")["mvrv"]
    mv.index = utc_to_broker_index(pd.DatetimeIndex(mv.index))

    with pd.option_context("mode.chained_assignment", None):
        h1 = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv")

    # --- redundancy: MVRV > full-period median, weekly, vs cycle_gate_pull's own definition
    # (research.portfolio_kama.cycle_gate_pull, replicated verbatim): weekly close from
    # 4h-resampled h1, <= 30-week SMA * 1.10, shifted 1wk to avoid lookahead.
    d4h = resample(h1, "4h")
    w30 = d4h["close"].resample("1W").last().rolling(30).mean().shift(1)
    ceil = (1 + 0.10) * w30
    weekly_close = d4h["close"].resample("1W").last()
    gate_on = weekly_close <= ceil  # True = btc_pull gate ON (early-recovery phase)

    mv_weekly = mv.resample("1W").last()
    full_median = mv.median()
    mv_high = (mv_weekly > full_median)

    both = pd.DataFrame({"mv_high": mv_high, "gate_on": gate_on}).dropna()
    agree = (both["mv_high"] == (~both["gate_on"])).mean()  # MVRV高=過熱 <-> gate OFF(成熟相場)が仮説
    agree_same_dir = (both["mv_high"] == both["gate_on"]).mean()
    print(f"\n[冗長性] 週次 MVRV>中央値 vs btc_pullゲートON: n={len(both)}")
    print(f"  一致率(MVRV>中央値 == gate_on 同方向)={agree_same_dir:.4f}")
    print(f"  一致率(MVRV>中央値 == gate_off 逆方向)={agree:.4f}")
    corr = both["mv_high"].astype(float).corr(both["gate_on"].astype(float))
    print(f"  相関(MVRV>中央値, gate_on)={corr:.4f}  full_median={full_median:.4f}")

    for era, sub in [("~2021", both[both.index < "2022-01-01"]),
                      ("2022~", both[both.index >= "2022-01-01"])]:
        if len(sub) == 0:
            continue
        a = (sub["mv_high"] == sub["gate_on"]).mean()
        print(f"  era={era} n={len(sub)} 一致率(同方向)={a:.4f}")

    # --- single-instrument: MVRV quintile -> BTC daily forward direction/magnitude
    daily_close = d4h["close"].resample("1D").last().dropna()
    mv_daily = mv.resample("1D").last().reindex(daily_close.index, method="ffill")
    work = pd.DataFrame(index=daily_close.index)
    work["mvrv"] = mv_daily
    work = work.dropna()
    try:
        work["Q"] = pd.qcut(work["mvrv"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    except ValueError:
        work["Q"] = pd.qcut(work["mvrv"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])

    for H, label in [(7, "H=7日"), (28, "H=28日")]:
        direction = forward_direction(daily_close, H)
        magnitude = forward_magnitude(daily_close, H)
        w2 = work.copy()
        w2["direction"] = direction.reindex(w2.index)
        w2["magnitude"] = magnitude.reindex(w2.index)
        print(f"\n  [{label}] 方向 log-return (五分位別):")
        print(layer_table(w2, "Q", "direction").to_string(index=False))
        print(f"  [{label}] 量 sum|abs log-return| (五分位別):")
        print(layer_table(w2, "Q", "magnitude").to_string(index=False))

        med, p2, p97, nb = month_block_bootstrap_diff(w2, "Q", "direction", "Q5", "Q1", n_boot=1000)
        print(f"  方向差(Q5-Q1) 月次ブロックブートストラップ: median={med:.5f} "
              f"95%CI=[{p2:.5f}, {p97:.5f}] (n_boot={nb})")

        era_before = w2[w2.index < "2022-01-01"]
        era_after = w2[w2.index >= "2022-01-01"]
        print(f"  [{label}] 2022-01-01前 (n={len(era_before.dropna())}):")
        print(layer_table(era_before, "Q", "direction").to_string(index=False))
        print(f"  [{label}] 2022-01-01後 (n={len(era_after.dropna())}):")
        print(layer_table(era_after, "Q", "direction").to_string(index=False))


if __name__ == "__main__":
    main()
