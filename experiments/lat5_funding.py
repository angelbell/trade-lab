"""案5: Binance資金調達率 -> BTC STEP1 -- 無料APIで全履歴取得、KAMA(14)日足向きとの冗長性、
BTC h1での先行(方向/量)を五分位で測る。

データ取得: https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT (8時間毎, 2019-09開始)。
  startTimeでページング(limit=1000)。UTCで data/ext_binance_funding_btcusdt.csv に保存。

X = バー確定時点で既知の直近3回分の資金調達率の平均(as-of, 先読み無し)。
冗長性 = X符号/水準 と 既存実装 research.portfolio_kama.kama_gate_btc と同じ定義の
  BTC日足KAMA(14)向き との日次相関・一致率。
単体 = BTC h1でXの五分位 -> 先行方向/量(H=24h・72h)。Q5(資金調達率高=ロング混雑)とQ1を明示。

Run: .venv/bin/python experiments/lat5_funding.py [--smoke]
"""
import argparse
import sys
import time

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

from src.data_loader import load_mt5_csv
from research.regime_adaptive import kama
from lat_common import (forward_direction, forward_magnitude, layer_table,
                         month_block_bootstrap_diff, print_tz_check, utc_to_broker_index)

ROOT = "/home/angelbell/dev/auto-trade"
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding(start_ms, end_ms, limit=1000, smoke=False):
    rows = []
    t = start_ms
    pages = 0
    while t < end_ms:
        for attempt in range(4):
            try:
                r = requests.get(FAPI, params={"symbol": "BTCUSDT", "startTime": t,
                                                "limit": limit}, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    break
                time.sleep(2)
            except Exception:
                time.sleep(2)
        else:
            print(f"[fetch] FAILED page at startTime={t}")
            break
        if not data:
            break
        rows += data
        t = data[-1]["fundingTime"] + 1
        pages += 1
        print(f"[fetch] page {pages}: {len(data)} rows, up to "
              f"{pd.Timestamp(data[-1]['fundingTime'], unit='ms', tz='UTC')}")
        if smoke and pages >= 2:
            break
        time.sleep(0.3)
    return rows


def build_funding_csv(smoke=False):
    start_ms = int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = fetch_funding(start_ms, end_ms, smoke=smoke)
    df = pd.DataFrame(rows)
    df["dt_utc"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df[["dt_utc", "fundingRate"]].drop_duplicates("dt_utc").sort_values("dt_utc")
    print(f"\n[counts] funding rows={len(df)} span={df['dt_utc'].min()} .. {df['dt_utc'].max()}")
    out = df.copy()
    out["dt_utc"] = out["dt_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(f"{ROOT}/data/ext_binance_funding_btcusdt.csv", index=False)
    print(f"[saved] {ROOT}/data/ext_binance_funding_btcusdt.csv n={len(out)}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    funding = build_funding_csv(smoke=args.smoke)

    jan_rows = funding[funding["dt_utc"].dt.month == 1]
    jul_rows = funding[funding["dt_utc"].dt.month == 7]
    if len(jan_rows) and len(jul_rows):
        print_tz_check(jan_rows["dt_utc"].iloc[0].tz_localize(None),
                        jul_rows["dt_utc"].iloc[0].tz_localize(None), label="funding")
    else:
        print("[tz check] insufficient Jan/Jul rows for sample check (smoke mode)")

    if args.smoke:
        print("\n[smoke] funding fetch OK, skipping measurement pass")
        return

    run_measurement(funding)


def run_measurement(funding: pd.DataFrame):
    print("\n" + "=" * 70)
    print("測定: 資金調達率X(直近3回平均) の冗長性(KAMA)・単体(五分位->先行)")
    print("=" * 70)

    fdf = funding.set_index("dt_utc").sort_index()
    fdf["X"] = fdf["fundingRate"].rolling(3).mean()
    fdf = fdf.dropna(subset=["X"])
    fdf.index = utc_to_broker_index(pd.DatetimeIndex(fdf.index))

    with pd.option_context("mode.chained_assignment", None):
        btc_h1 = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv")

    # as-of join: X known at fdf.index (funding confirm time); use previous confirmed value
    # for every h1 bar (merge_asof, no lookahead: bar sees only funding events <= its own time).
    X_asof = pd.merge_asof(
        pd.DataFrame(index=btc_h1.index).rename_axis("t").reset_index(),
        fdf[["X"]].rename_axis("t").reset_index(),
        on="t", direction="backward")
    X_asof = X_asof.set_index("t")["X"]

    # --- redundancy vs KAMA(14) daily direction (same def as research.portfolio_kama.kama_gate_btc)
    from breakout_wave import resample
    d4h = resample(btc_h1, "4h")
    dc = d4h["close"].resample("1D").last().dropna()
    km = kama(dc, 14)
    kama_up = (km > km.shift(1))

    X_daily = fdf["X"].resample("1D").last().reindex(kama_up.index, method="ffill")
    both = pd.DataFrame({"X": X_daily, "kama_up": kama_up}).dropna()
    both["X_pos"] = both["X"] > 0
    corr = both["X"].corr(both["kama_up"].astype(float))
    agree = (both["X_pos"] == both["kama_up"]).mean()
    print(f"\n[冗長性] 日次 X vs KAMA(14)向き: n={len(both)} corr={corr:.4f} "
          f"一致率(X>0 vs KAMA上向き)={agree:.4f}")

    # split by era
    for era, sub in [("~2021", both[both.index < "2022-01-01"]),
                      ("2022~", both[both.index >= "2022-01-01"])]:
        if len(sub) == 0:
            continue
        c = sub["X"].corr(sub["kama_up"].astype(float))
        a = (sub["X_pos"] == sub["kama_up"]).mean()
        print(f"  era={era} n={len(sub)} corr={c:.4f} 一致率={a:.4f}")

    # --- single-instrument: X quintile -> forward direction/magnitude on BTC h1
    work = pd.DataFrame(index=btc_h1.index)
    work["X"] = X_asof
    work = work.dropna()
    try:
        work["Q"] = pd.qcut(work["X"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    except ValueError:
        work["Q"] = pd.qcut(work["X"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])

    for H, label in [(24, "H=24h"), (72, "H=72h")]:
        direction = forward_direction(btc_h1["close"], H)
        magnitude = forward_magnitude(btc_h1["close"], H)
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
