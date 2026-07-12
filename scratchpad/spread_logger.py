"""spread_logger.py -- poll the mt5-mcp bridge /tick endpoint and append bid/ask/spread rows.

Purpose (proposals B5): measure the REAL Vantage spread distribution by hour, the one datum
that decides BTC 5m's fate ($10 alive / $15 marginal / $25 dead) and calibrates the gold/
silver cost models. Runs forever (nohup); tolerant of bridge downtime (retries). Skips a row
when the server tick time hasn't advanced (stale market). ~5s cadence x 4 symbols.

Output: data/spread_log.csv  (ts_server ISO, symbol, bid, ask, spread)
Run:    nohup ../mt5-mcp/.venv/bin/python scratchpad/spread_logger.py >> scratchpad/spread_logger.out 2>&1 &
"""
import csv
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "..", "mt5-mcp", "client"))
from mt5api import Bridge

SYMBOLS = ["BTCUSD", "XAUUSD+", "XAGUSD", "USDJPY+", "NAS100.r", "GER40.r", "USOUSD", "EURUSD+", "ETHUSD", "XPTUSD.r"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "spread_log.csv")
INTERVAL = 5.0


def main():
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["ts_server", "symbol", "bid", "ask", "spread"])
        f.flush()
    b = Bridge(timeout=8)
    last_t = {}
    n = 0
    while True:
        for s in SYMBOLS:
            try:
                t = b.tick(s)
            except Exception as e:
                print(f"{datetime.now(timezone.utc).isoformat()} {s} ERR {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                time.sleep(10)
                continue
            if last_t.get(s) == t["time"]:
                continue
            last_t[s] = t["time"]
            ts = datetime.fromtimestamp(t["time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([ts, s, t["bid"], t["ask"], t["spread"]])
            n += 1
        f.flush()
        if n and n % 5000 < len(SYMBOLS):
            print(f"{datetime.now(timezone.utc).isoformat()} rows={n}", file=sys.stderr, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
