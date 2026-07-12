"""Build a CONTINUOUS 5m up/down-volume series for BTCUSDT perp (2020-09 -> now) from the
Binance public monthly kline dumps (same source pattern as binance_metrics_backfill.py).
Output: data/ext_btc_5m_flow.csv  [time, up_vol, dn_vol, taker_buy, taker_sell]
  up_vol/dn_vol : volume of 5m bars closing up / down  (= the PINE-computable proxy inputs)
  taker_buy/sell: the true aggressor split from the kline field (for cross-checks)
Idempotent: skips months already present. Run: .venv/bin/python scratchpad/binance_5m_flow_backfill.py
"""
import os, io, sys, zipfile, urllib.request
import numpy as np, pandas as pd

ROOT = "/home/angelbell/dev/auto-trade"
OUT = os.path.join(ROOT, "data/ext_btc_5m_flow.csv")
URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{ym}.zip"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def month_frame(ym):
    try:
        with urllib.request.urlopen(URL.format(ym=ym), timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
    except Exception as e:
        print(f"  {ym}: skip ({e})"); return None
    with z.open(z.namelist()[0]) as f:
        d = pd.read_csv(f, header=None, names=COLS)
    if str(d.iloc[0, 0]).startswith("open"):        # some dumps carry a header row
        d = d.iloc[1:]
    d = d.astype({"open_time": np.int64, "open": float, "close": float,
                  "volume": float, "taker_buy_volume": float})
    t = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    up = np.where(d["close"] > d["open"], d["volume"], 0.0)
    dn = np.where(d["close"] < d["open"], d["volume"], 0.0)
    return pd.DataFrame({"up_vol": up, "dn_vol": dn,
                         "taker_buy": d["taker_buy_volume"].values,
                         "taker_sell": (d["volume"] - d["taker_buy_volume"]).values}, index=t)


def main():
    have = pd.read_csv(OUT, index_col=0, parse_dates=[0]) if os.path.exists(OUT) else pd.DataFrame()
    months = pd.date_range("2020-09-01", pd.Timestamp.utcnow().tz_localize(None), freq="MS").strftime("%Y-%m")
    done = set(have.index.strftime("%Y-%m")) if len(have) else set()
    frames = [have] if len(have) else []
    for ym in months:
        if ym in done and ym != months[-1]:
            continue
        f = month_frame(ym)
        if f is None or not len(f):
            continue
        print(f"  {ym}: {len(f)} bars")
        frames.append(f)
    d = pd.concat(frames).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d.to_csv(OUT)
    print(f"\nwrote {OUT}: {len(d)} rows  {d.index[0]} -> {d.index[-1]}")


if __name__ == "__main__":
    main()
