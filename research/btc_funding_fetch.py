"""btc_funding_fetch.py -- pull BTC perp FUNDING-RATE history from Binance (ccxt) -> data/btc_funding.csv.

Funding rate = the closest FREE, CAUSAL proxy to Hiropi's 'positioning' signal (crowded longs pay shorts
=> liquidation fuel sits below, and vice versa). The predictive liquidation HEATMAP itself is not
backtestable (CoinGlass API has no as-of timestamp => repaints); funding history is clean and long
(~2019-09 -> now, Binance BTCUSDT perp, every 8h).

Stored as UTC (Binance native). Alignment to Vantage broker-time bars + no-lookahead is handled in the
consumer (liq_sweep_fade.py): use only the funding settled BEFORE the bar opens.

  .venv/bin/python research/btc_funding_fetch.py            # fetch/refresh -> data/btc_funding.csv
"""
import os, sys, time
import pandas as pd
import ccxt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "btc_funding.csv")
SYMBOL = "BTC/USDT:USDT"   # Binance USDT-margined perpetual


def fetch_funding(symbol=SYMBOL, start="2019-01-01"):
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    now = int(pd.Timestamp.utcnow().timestamp() * 1000)
    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["timestamp"]
        if len(batch) < 1000 or last >= now:
            break
        since = last + 1
        time.sleep(0.25)
    df = pd.DataFrame([{"time": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                        "fundingRate": float(r["fundingRate"])} for r in rows])
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df


def main():
    print(f"fetching {SYMBOL} funding from Binance ...")
    df = fetch_funding()
    df.to_csv(OUT)
    fr = df["fundingRate"]
    print(f"saved {len(df)} funding points -> {OUT}")
    print(f"  span: {df.index[0]} -> {df.index[-1]}")
    print(f"  rate: mean={fr.mean()*100:+.4f}%/8h  std={fr.std()*100:.4f}  "
          f"min={fr.min()*100:+.3f}  max={fr.max()*100:+.3f}  %pos={(fr>0).mean()*100:.0f}%")
    print(f"  annualized mean (x3x365): {fr.mean()*3*365*100:+.1f}%")


if __name__ == "__main__":
    main()
