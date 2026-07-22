"""Binance のバルク配信（data.binance.vision）から h1 の全履歴を取り、Vantage CSV と同じ形で保存する。

🚨 これは**売買しているフィードではない**（CLAUDE.md: 検証は Vantage で行う）。
用途は機構の検証に限る — Binance は 2017-08 以降ずっと本物の24時間なので、
「Vantage の 2018-2021（平日限定商品だった時代）で強く見えたのは構造のせいか実物か」を切り分けられる。
ファイル名は `binance_*` にして vantage_* と混ざらないようにする。時刻は UTC。
"""
import io
import sys
import zipfile
from datetime import date

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def months(start_y, start_m):
    y, m = start_y, start_m
    today = date.today()
    while (y, m) <= (today.year, today.month):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch(symbol, tf="1h", start=(2017, 1)):
    frames, missing = [], 0
    for y, m in months(*start):
        url = f"{BASE}/{symbol}/{tf}/{symbol}-{tf}-{y}-{m:02d}.zip"
        try:
            r = requests.get(url, timeout=60)
        except Exception as e:
            print(f"  {y}-{m:02d} 通信失敗 {e}", file=sys.stderr)
            continue
        if r.status_code != 200:
            missing += 1
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0]).decode()
        first = raw.split("\n", 1)[0]
        hdr = 0 if first.split(",")[0].isdigit() else 1
        d = pd.read_csv(io.StringIO(raw), header=None if hdr == 0 else 0,
                        names=COLS if hdr == 0 else None)
        if hdr:
            d.columns = [c.strip().lower() for c in d.columns]
        frames.append(d[["open_time", "open", "high", "low", "close", "volume"]])
    if not frames:
        raise SystemExit(f"{symbol}: 1件も取れなかった")
    df = pd.concat(frames, ignore_index=True)
    # 🚨 open_time の単位はファイルによって違う（2025年以降の一部がマイクロ秒、それ以前はミリ秒）。
    # 全体一括で判定すると古い分が1970年になる。行ごとにミリ秒へ揃える。
    ot = df["open_time"].astype("int64")
    ot = ot.where(ot < 10**14, ot // 1000)
    df["time"] = pd.to_datetime(ot, unit="ms", utc=True)
    df = df.drop_duplicates("time").sort_values("time")
    out = pd.DataFrame({"time": df["time"].dt.strftime("%Y.%m.%d %H:%M"),
                        "open": df["open"].values, "high": df["high"].values,
                        "low": df["low"].values, "close": df["close"].values,
                        "tick_volume": df["volume"].round().astype("int64").values})
    TFNAME = {"1h": "h1", "15m": "m15", "5m": "m5", "4h": "h4", "1d": "d1"}
    path = f"data/binance_{symbol.lower()}_{TFNAME.get(tf, tf)}.csv"
    out.to_csv(path, index=False)
    n_by_year = df.set_index("time").groupby(lambda t: t.year).size()
    print(f"{symbol}: {len(out)}本 {df['time'].iloc[0]:%Y-%m-%d} -> {df['time'].iloc[-1]:%Y-%m-%d} "
          f"(欠月{missing}) -> {path}")
    print("   年別本数: " + " ".join(f"{y}:{n}" for y, n in n_by_year.items()))
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    tf = "1h"
    if args and args[0].startswith("--tf="):
        tf = args.pop(0).split("=", 1)[1]
    syms = args or ["BTCUSDT", "ETHUSDT"]
    for s in syms:
        fetch(s, tf=tf)
