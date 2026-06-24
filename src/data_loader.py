"""OHLCV data fetcher with local parquet cache."""

import re
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

# --- data-corruption guard thresholds (see _drop_corrupt_bars) ----------------
# Feed glitches (e.g. Vantage BTC H1 around 2020-08-10 printed ~$300 bars amid
# ~$11,900 price = a missing-leading-digit error). These corrupt stop/ATR calcs.
_MEDIAN_WINDOW = 11      # centered rolling-median window for the spike test
_SPIKE_DEV = 0.5         # flag close that deviates >50% from the local median
_MAX_BAR_RATIO = 3.0     # flag a single bar whose high/low ratio exceeds this

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RATE_LIMIT_MS = 500  # conservative: 2 req/s


def _cache_path(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    safe_sym = re.sub(r"[^A-Za-z0-9]", "_", symbol)
    return CACHE_DIR / f"{safe_sym}_{timeframe}_{start.date()}_{end.date()}.parquet"


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return OHLCV DataFrame indexed by UTC timestamp.

    Loads from parquet cache when available; otherwise fetches from Binance.
    Args:
        symbol: e.g. "BTC/USDT"
        timeframe: e.g. "15m"
        start: inclusive start (ISO string or Timestamp)
        end: exclusive end
    Returns:
        DataFrame with columns [open, high, low, close, volume], DatetimeIndex UTC.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    cache = _cache_path(symbol, timeframe, start_ts, end_ts)
    if cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    df = _fetch_from_exchange(symbol, timeframe, start_ts, end_ts)
    df.to_parquet(cache)
    return df


def load_mt5_csv(path: str | Path) -> pd.DataFrame:
    """Load an MT5-exported OHLCV CSV (from export_history.mq5).

    The CSV has columns: time, open, high, low, close, tick_volume, where
    `time` is "YYYY.MM.DD HH:MM" in BROKER SERVER time. We keep those labels
    as the canonical clock (tagged UTC) on purpose: the strategy resamples
    4h / 1D on the index, so binning on the broker's own boundaries makes the
    backtest match what the live MT5 EA actually sees (server-time bars), not
    Binance/UTC. This is the whole point of validating on the traded feed.

    Returns a DataFrame matching fetch_ohlcv: DatetimeIndex, columns
    [open, high, low, close, volume].
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    cols = ["open", "high", "low", "close", "volume"]
    df = df[cols].astype(float)
    return _drop_corrupt_bars(df, source=str(path))


def _drop_corrupt_bars(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Detect and remove feed-glitch bars, printing a clear warning to stderr.

    Two conservative rules (thresholds chosen so legitimate gold/BTC/USDJPY
    volatility never trips them — only true data corruption):
      1. SPIKE: close deviates >`_SPIKE_DEV` (50%) from the centered local
         median (`_MEDIAN_WINDOW` bars). An isolated wrong-magnitude print
         (e.g. $300 amid $11,900) reverts immediately, so the local median
         stays clean and the spike stands out; a genuine sustained move drags
         the median with it and is NOT flagged.
      2. INTRABAR: a single bar whose high/low ratio exceeds `_MAX_BAR_RATIO`
         (catches a bar mixing a corrupt low with a valid high).

    Corrupt bars are dropped (resample/rolling tolerate the gaps). Warnings go
    to stderr on purpose, so stdout stays parseable for the grep-based sweeps.
    """
    if len(df) < _MEDIAN_WINDOW:
        return df

    med = df["close"].rolling(_MEDIAN_WINDOW, center=True, min_periods=3).median()
    dev = (df["close"] / med - 1.0).abs()
    spike = dev > _SPIKE_DEV
    intrabar = (df["high"] / df["low"].where(df["low"] > 0)) > _MAX_BAR_RATIO
    bad = (spike | intrabar) & med.notna()

    if bad.any():
        tag = Path(source).name if source else "data"
        n = int(bad.sum())
        print(f"⚠️  DATA CORRUPTION: {tag} — dropped {n} glitch bar(s) "
              f"(>±{int(_SPIKE_DEV*100)}% local-median spike or high/low>{_MAX_BAR_RATIO:g}):",
              file=sys.stderr)
        for ts, row in df[bad].iterrows():
            reason = []
            if spike.get(ts, False):
                reason.append(f"close {row['close']:.2f} vs median {med[ts]:.2f}")
            if intrabar.get(ts, False):
                reason.append(f"H/L={row['high']/row['low']:.0f}x")
            print(f"      {ts}  O={row['open']:.2f} H={row['high']:.2f} "
                  f"L={row['low']:.2f} C={row['close']:.2f}  [{'; '.join(reason)}]",
                  file=sys.stderr)
        df = df[~bad]

    return df


def _fetch_from_exchange(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_rows: list[list] = []
    while True:
        rows = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not rows:
            break
        rows = [r for r in rows if r[0] < end_ms]
        all_rows.extend(rows)
        if len(rows) < 1000 or rows[-1][0] >= end_ms:
            break
        since_ms = rows[-1][0] + 1
        time.sleep(_RATE_LIMIT_MS / 1000)

    if not all_rows:
        raise ValueError(f"No data returned for {symbol} {timeframe} {start}–{end}")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[start:end]
    return df.astype(float)
