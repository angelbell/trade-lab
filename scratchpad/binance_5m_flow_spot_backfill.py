"""Extend the BTC 5m flow cache back to 2017-08 using Binance SPOT monthly kline dumps.

Why: data/ext_btc_5m_flow.csv (USD-M PERP klines) starts 2020-09 only because the backfill was
pinned to the OI-metrics endpoint's start date -- the klines themselves go back further (perp
2019-09, spot 2017-08). btc15m_L's sample starts 2018-10, so the perp cache misses the 2018 bear
and the 2019-20 cycle = exactly the regimes where the leg bleeds and where an in-hold exit rule
would have to earn its keep.

Output: data/ext_btc_5m_flow_spot.csv  [open_time(UTC), up_vol, dn_vol, taker_buy, taker_sell]
  taker_buy/sell = the TRUE aggressor split from the kline field (taker_buy_volume).
  up_vol/dn_vol  = volume of 5m bars closing up / down (the TradingView-computable proxy).
Spot is also an INDEPENDENT venue from the perp cache -> the two series give a built-in
replication check on any separation we find.

Idempotent (skips months already present). Run: .venv/bin/python scratchpad/binance_5m_flow_spot_backfill.py
"""
import os, io, zipfile, urllib.request
import numpy as np, pandas as pd

ROOT = "/home/angelbell/dev/auto-trade"
OUT = os.path.join(ROOT, "data/ext_btc_5m_flow_spot.csv")
URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{ym}.zip"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def month_frame(ym):
    try:
        with urllib.request.urlopen(URL.format(ym=ym), timeout=90) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
    except Exception as e:
        print(f"  {ym}: skip ({e})"); return None
    with z.open(z.namelist()[0]) as f:
        d = pd.read_csv(f, header=None, names=COLS)
    if str(d.iloc[0, 0]).startswith("open"):
        d = d.iloc[1:]
    d = d.astype({"open_time": np.int64, "open": float, "close": float,
                  "volume": float, "taker_buy_volume": float})
    # Binance regenerated the dumps with MICROSECOND timestamps (the older ones are ms).
    unit = "us" if d["open_time"].iloc[0] > 1e14 else "ms"
    t = pd.to_datetime(d["open_time"], unit=unit, utc=True)
    up = np.where(d["close"] > d["open"], d["volume"], 0.0)
    dn = np.where(d["close"] < d["open"], d["volume"], 0.0)
    return pd.DataFrame({"up_vol": up, "dn_vol": dn,
                         "taker_buy": d["taker_buy_volume"].values,
                         "taker_sell": (d["volume"] - d["taker_buy_volume"]).values}, index=t)


def main():
    have = pd.read_csv(OUT, index_col=0, parse_dates=[0]) if os.path.exists(OUT) else pd.DataFrame()
    months = pd.date_range("2017-08-01", pd.Timestamp.utcnow().tz_localize(None),
                           freq="MS").strftime("%Y-%m")
    done = set() if have.empty else set(have.index.strftime("%Y-%m"))
    parts = [have] if not have.empty else []
    for ym in months:
        if ym in done:
            continue
        d = month_frame(ym)
        if d is not None:
            parts.append(d)
            print(f"  {ym}: {len(d)} bars")
    if not parts:
        print("nothing to do"); return
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]          # the perp dumps carry dupes; guard here too
    out.to_csv(OUT)
    print(f"\nwrote {OUT}: {len(out)} rows, {out.index.min()} -> {out.index.max()}")
    gap = out.index.to_series().diff().dt.total_seconds().div(300).round()
    print(f"gaps > 1 bar: {(gap > 1).sum()}  (largest {gap.max():.0f} bars)")


if __name__ == "__main__":
    main()
