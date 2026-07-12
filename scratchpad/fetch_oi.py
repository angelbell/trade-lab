"""Fetch BTCUSDT futures OPEN-INTEREST + long/short-ratio metrics from Binance Vision
(free, daily files, 5-min res from ~2021-01). Downsample to HOURLY (last row per hour) and
save data/btc_oi.csv. Leverage-family signals (uncapped, unlike funding): sum_open_interest,
sum_open_interest_value, toptrader long/short ratio, taker buy/sell vol ratio.
"""
import urllib.request, zipfile, io, datetime as dt
import pandas as pd

START = dt.date(2021, 1, 1)
END = dt.date.today()
COLS = ["create_time", "sum_open_interest", "sum_open_interest_value",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]

frames, ok, miss = [], 0, 0
d = START
while d <= END:
    u = f"https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip"
    try:
        raw = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
        z = zipfile.ZipFile(io.BytesIO(raw)); n = z.namelist()[0]
        df = pd.read_csv(io.BytesIO(z.read(n)))
        df = df[COLS].copy()
        df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
        df = df.set_index("create_time").resample("1h").last().dropna(how="all")
        frames.append(df); ok += 1
    except Exception:
        miss += 1
    if (ok + miss) % 200 == 0:
        print(f"...{d}  ok={ok} miss={miss}", flush=True)
    d += dt.timedelta(days=1)

out = pd.concat(frames).sort_index()
out = out[~out.index.duplicated(keep="last")]
out.to_csv("/home/angelbell/dev/auto-trade/data/btc_oi.csv")
print(f"DONE: {len(out)} hourly rows  {out.index.min()} -> {out.index.max()}  (ok={ok} miss={miss})")
