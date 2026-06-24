"""Two breakout strategies for comparison.

A) Williams Volatility Breakout (daily range adaptation)
   Entry: close > prev_DAY_close + k * prev_DAY_range
   Original Williams formula using the previous FULL DAY's range as threshold.
   Naturally limits to a handful of entries per day.

B) Donchian Channel Breakout (Turtle Trading)
   Entry: close > highest HIGH of last N bars  (long, in uptrend)
          close < lowest  LOW  of last N bars  (short, in downtrend)
   Classic Richard Dennis / Turtle approach: new N-bar extreme = breakout.

Both: HTF EMA regime gate + ATR-based SL backstop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt

FEES = 0.001      # Vantage CFD spread ~0.1%/side (conservative)
SLIPPAGE = 0.001  # slippage on CFD market orders


# ---------------------------------------------------------------------------
# A: Williams Volatility Breakout (previous-day range)
# ---------------------------------------------------------------------------

def compute_signals_williams(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    k: float = 0.5,
    htf_timeframe: str = "1h",
    htf_ema_period: int = 50,
    chop_threshold: float = 61.8,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Williams VB using the PREVIOUS DAY's range — faithful to the original formula.

    Using the full day's range (not the 15m bar range) keeps entry frequency
    reasonable: threshold is ~0.5–2% of price, not 0.02%.

    chop_threshold: Choppiness Index gate. CHOP(14) < threshold = trending → allow entry.
    Classic levels: 38.2 (strong trend) / 61.8 (transitioning) / 100 (choppy).
    """
    # Previous calendar day's OHLC (no lookahead: shift(1) on daily)
    prev_daily_close = close.resample("1D").last().shift(1)
    prev_daily_range = (high.resample("1D").max() - low.resample("1D").min()).shift(1)

    upper_trigger = (prev_daily_close + k * prev_daily_range).reindex(close.index, method="ffill")
    lower_trigger = (prev_daily_close - k * prev_daily_range).reindex(close.index, method="ffill")

    # HTF regime
    htf_close   = close.resample(htf_timeframe).last()
    htf_ema     = htf_close.ewm(span=htf_ema_period, adjust=False).mean().shift(1)
    htf_ema_ltf = htf_ema.reindex(close.index, method="ffill")
    bullish     = (close > htf_ema_ltf).fillna(False)

    # Choppiness filter: block entries in ranging/choppy markets (shift(1) = no lookahead)
    chop = ta.chop(high, low, close, length=14).shift(1)
    trending = (chop < chop_threshold).fillna(False)

    long_entries  = (bullish  & (close > upper_trigger) & trending).fillna(False).astype(bool)
    short_entries = (~bullish & (close < lower_trigger) & trending).fillna(False).astype(bool)
    long_exits    = (close < lower_trigger).fillna(False).astype(bool)
    short_exits   = (close > upper_trigger).fillna(False).astype(bool)

    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# B: Donchian Channel Breakout (Turtle Trading)
# ---------------------------------------------------------------------------

def compute_signals_donchian(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    donchian_period: int = 40,
    htf_timeframe: str = "1h",
    htf_ema_period: int = 50,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Donchian channel breakout — Richard Dennis Turtle system adapted for 15m.

    shift(1) on rolling max/min ensures we use only completed prior bars.
    """
    high_n = high.rolling(donchian_period).max().shift(1)
    low_n  = low.rolling(donchian_period).min().shift(1)

    # HTF regime
    htf_close   = close.resample(htf_timeframe).last()
    htf_ema     = htf_close.ewm(span=htf_ema_period, adjust=False).mean().shift(1)
    htf_ema_ltf = htf_ema.reindex(close.index, method="ffill")
    bullish     = (close > htf_ema_ltf).fillna(False)

    long_entries  = (bullish  & (close > high_n)).fillna(False).astype(bool)
    short_entries = (~bullish & (close < low_n)).fillna(False).astype(bool)
    long_exits    = (close < low_n).fillna(False).astype(bool)
    short_exits   = (close > high_n).fillna(False).astype(bool)

    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# C: ATR Trailing Stop Crossover (ceyhun / TradingView)
# ---------------------------------------------------------------------------

def _atr_trail(close_arr: np.ndarray, sl_arr: np.ndarray) -> np.ndarray:
    """Recursive ATR trailing stop. Requires a Python loop (not vectorisable)."""
    n = len(close_arr)
    trail = np.empty(n)
    trail[0] = close_arr[0] - sl_arr[0]
    for i in range(1, n):
        c  = close_arr[i]
        cp = close_arr[i - 1]
        tp = trail[i - 1]
        sl = sl_arr[i]
        if c > tp and cp > tp:
            trail[i] = max(tp, c - sl)
        elif c < tp and cp < tp:
            trail[i] = min(tp, c + sl)
        elif c > tp:
            trail[i] = c - sl
        else:
            trail[i] = c + sl
    return trail


def compute_signals_atr_trail(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ap1: int = 5,
    af1: float = 0.5,
    ap2: int = 10,
    af2: float = 3.0,
    ema_period: int = 200,
    ma_type: str = "ema",
    adx_threshold: float = 0.0,
    adx_period: int = 14,
    htf_timeframe: str = "1D",   # accepted for API compatibility, not used
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Dual ATR Trailing Stop crossover — ceyhun's TradingView strategy.

    Trail1 (fast) = af1 * ATR(ap1)  trailing stop
    Trail2 (slow) = af2 * ATR(ap2)  trailing stop
    Buy  = Trail1 crosses above Trail2
    Sell = Trail1 crosses below Trail2

    EMA filter (ema_period): trade WITH the trend only.
      close > EMA → longs allowed, close < EMA → shorts allowed.
      Set ema_period=0 to disable the filter.

    ADX filter (adx_threshold): only enter when ADX > threshold (trending).
      Blocks entries in ranging/choppy markets — the source of the drawdown.
      Set adx_threshold=0 to disable.
    """
    atr1 = ta.atr(high, low, close, length=ap1).bfill().values.astype(np.float64)
    atr2 = ta.atr(high, low, close, length=ap2).bfill().values.astype(np.float64)
    c_arr = close.values.astype(np.float64)

    trail1 = pd.Series(_atr_trail(c_arr, af1 * atr1), index=close.index)
    trail2 = pd.Series(_atr_trail(c_arr, af2 * atr2), index=close.index)

    t1_above = trail1 > trail2
    t1_above_prev = t1_above.shift(1, fill_value=False)

    buy  = (~t1_above_prev) & t1_above    # crossover  (Trail1 crosses above Trail2)
    sell = t1_above_prev  & (~t1_above)   # crossunder (Trail1 crosses below Trail2)

    # MA trend filter (shift(1) = use only completed bars, no lookahead)
    if ema_period and ema_period > 0:
        if ma_type == "sma":
            ma = close.rolling(ema_period).mean().shift(1)
        else:
            ma = close.ewm(span=ema_period, adjust=False).mean().shift(1)
        uptrend   = (close > ma).fillna(False)
        downtrend = (close < ma).fillna(False)
    else:
        uptrend = downtrend = pd.Series(True, index=close.index)

    # ADX trend-strength filter (shift(1) = no lookahead)
    if adx_threshold and adx_threshold > 0:
        adx_df = ta.adx(high, low, close, length=adx_period)
        adx    = adx_df[f"ADX_{adx_period}"].shift(1)
        trending = (adx > adx_threshold).fillna(False)
    else:
        trending = pd.Series(True, index=close.index)

    long_entries  = (buy  & uptrend   & trending).astype(bool)  # long: uptrend + strong
    short_entries = (sell & downtrend & trending).astype(bool)  # short: downtrend + strong
    long_exits    = sell.astype(bool)                # exit long on opposite cross
    short_exits   = buy.astype(bool)                 # exit short on opposite cross

    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# D: Daily Pivot-Point mean reversion (fade to PP)
# ---------------------------------------------------------------------------

def compute_signals_pivot_fade(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    entry: str = "r1",
    htf_timeframe: str = "1D",   # accepted for API compatibility, not used
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Classic daily pivot points, traded as MEAN REVERSION (fade back to PP).

    Pivots from the PREVIOUS day (shift(1) on daily = no lookahead):
      PP = (H+L+C)/3,  R1 = 2PP-L,  S1 = 2PP-H,  R2 = PP+(H-L),  S2 = PP-(H-L)

    Fade logic (entry = which band to fade):
      price reaches support  → long  (expect bounce to PP)
      price reaches resistance → short (expect drop to PP)
      exit when price returns to PP. A hard ATR SL (in run_portfolio) caps
      failed fades — appropriate here, unlike trend following.
    """
    d_high  = high.resample("1D").max().shift(1)
    d_low   = low.resample("1D").min().shift(1)
    d_close = close.resample("1D").last().shift(1)

    pp = (d_high + d_low + d_close) / 3.0
    r1 = 2.0 * pp - d_low
    s1 = 2.0 * pp - d_high
    r2 = pp + (d_high - d_low)
    s2 = pp - (d_high - d_low)

    PP = pp.reindex(close.index, method="ffill")
    if entry == "r2":
        upper = r2.reindex(close.index, method="ffill")
        lower = s2.reindex(close.index, method="ffill")
    else:
        upper = r1.reindex(close.index, method="ffill")
        lower = s1.reindex(close.index, method="ffill")

    long_entries  = (close <= lower).fillna(False).astype(bool)   # at support → buy
    short_entries = (close >= upper).fillna(False).astype(bool)   # at resistance → sell
    long_exits    = (close >= PP).fillna(False).astype(bool)      # back to mean → exit
    short_exits   = (close <= PP).fillna(False).astype(bool)

    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# E: Market-structure (fractal swings on HTF) + LTF entry — the user's method
# ---------------------------------------------------------------------------

def _confirmed_swings(htf_high: pd.Series, htf_low: pd.Series, n: int):
    """Fractal swing highs/lows on the HTF, confirmed n bars later (no lookahead).

    Returns (last_sh, prev_sh, last_sl, prev_sl) on the HTF index — each the
    level known *as of* that bar (the swing is only confirmed n bars after it
    prints, so values are shifted forward by n).
    """
    w = 2 * n + 1
    is_sh = htf_high >= htf_high.rolling(w, center=True).max()
    is_sl = htf_low  <= htf_low.rolling(w, center=True).min()

    sh_conf = htf_high.where(is_sh).shift(n)   # confirmed n bars later
    sl_conf = htf_low.where(is_sl).shift(n)

    last_sh = sh_conf.ffill()
    last_sl = sl_conf.ffill()
    prev_sh = sh_conf.dropna().shift(1).reindex(sh_conf.index).ffill()
    prev_sl = sl_conf.dropna().shift(1).reindex(sl_conf.index).ffill()
    return last_sh, prev_sh, last_sl, prev_sl


def compute_signals_structure(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    fractal_n: int = 3,
    entry_mode: str = "support",
    tol: float = 0.003,
    min_struct: float = 0.0,
    adx_threshold: float = 0.0,
    adx_period: int = 14,
    sma_regime: int = 0,
    htf_timeframe: str = "4h",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Multi-timeframe market-structure trading (price action).

    HTF (htf_timeframe): detect fractal swing highs/lows; structure trend =
      higher-lows → uptrend, lower-highs → downtrend.
    LTF (the passed series): time the entry in the structure's direction.

    entry_mode:
      "support"  — uptrend + price dips to the HTF swing low then closes back
                   above it with an up-bar (buy the support touch / bounce)
      "pullback" — uptrend + up-bar right after a down-bar (buy the pullback)
      "breakout" — uptrend + close breaks above the last HTF swing high

    Range filters (skip choppy regimes — the source of the drawdown):
      min_struct    — A: require the higher-low rise (or lower-high drop) to
                      exceed this fraction; tight/flat swings = chop → skip.
      adx_threshold — B: require ADX(adx_period) > threshold (trend strength).

    Shorts are the mirror image in a downtrend. shift(1) on all HTF series
    before reindex = only completed HTF bars are referenced (no lookahead).
    """
    htf_high  = high.resample(htf_timeframe).max()
    htf_low   = low.resample(htf_timeframe).min()
    last_sh_h, prev_sh_h, last_sl_h, prev_sl_h = _confirmed_swings(htf_high, htf_low, fractal_n)

    uptrend_h   = (last_sl_h > prev_sl_h)   # higher lows
    downtrend_h = (last_sh_h < prev_sh_h)   # lower highs

    # A: structure-strength — relative size of the higher-low rise / lower-high drop
    hl_rise = ((last_sl_h - prev_sl_h) / prev_sl_h)        # >0 in a rising structure
    lh_drop = ((prev_sh_h - last_sh_h) / prev_sh_h)        # >0 in a falling structure

    def to_ltf(s: pd.Series) -> pd.Series:
        return s.shift(1).reindex(close.index, method="ffill")

    up      = to_ltf(uptrend_h).fillna(False)
    dn      = to_ltf(downtrend_h).fillna(False)
    sh_lvl  = to_ltf(last_sh_h)
    sl_lvl  = to_ltf(last_sl_h)
    long_strong  = (to_ltf(hl_rise) >= min_struct).fillna(False)
    short_strong = (to_ltf(lh_drop) >= min_struct).fillna(False)

    # B: ADX trend-strength gate on the trading timeframe (shift(1) = no lookahead)
    if adx_threshold and adx_threshold > 0:
        adx_df = ta.adx(high, low, close, length=adx_period)
        adx    = adx_df[f"ADX_{adx_period}"].shift(1)
        trending = (adx > adx_threshold).fillna(False)
    else:
        trending = pd.Series(True, index=close.index)

    bull = close > close.shift(1)                                   # up-bar proxy
    bear = close < close.shift(1)
    prev_down = close.shift(1) < close.shift(2)                     # prior bar was a pullback
    prev_up   = close.shift(1) > close.shift(2)

    if entry_mode == "breakout":
        long_raw  = up & (close > sh_lvl)
        short_raw = dn & (close < sl_lvl)
    elif entry_mode == "pullback":
        long_raw  = up & bull & prev_down
        short_raw = dn & bear & prev_up
    else:  # "support"
        long_raw  = up & (low <= sl_lvl * (1 + tol)) & (close > sl_lvl) & bull
        short_raw = dn & (high >= sh_lvl * (1 - tol)) & (close < sh_lvl) & bear

    # Daily SMA timing as a macro-regime gate (combine with SMA-timing edge).
    # Longs only in a bullish daily regime, shorts only in a bearish one.
    if sma_regime and sma_regime > 0:
        # dropna() BEFORE rolling so feed gaps (weekend/maintenance days that
        # resample turns into NaN rows) don't poison the SMA via NaN propagation.
        # This makes the SMA = mean of the last `sma_regime` *actual* daily bars,
        # exactly like MT5's iMA(D1,150). On a gap-free feed (Binance) dropna is
        # a no-op, so historical WFO/holdout results are unchanged; on a gappy
        # broker feed (Vantage) it fixes the regime gate firing as "none".
        d_close = close.resample("1D").last().dropna()
        d_sma   = d_close.rolling(sma_regime).mean()
        bull_reg = (d_close > d_sma).shift(1).reindex(close.index, method="ffill").fillna(False)
        bear_reg = (d_close < d_sma).shift(1).reindex(close.index, method="ffill").fillna(False)
    else:
        bull_reg = pd.Series(True, index=close.index)
        bear_reg = pd.Series(True, index=close.index)

    long_raw  = long_raw  & long_strong  & trending & bull_reg
    short_raw = short_raw & short_strong & trending & bear_reg

    long_entries  = long_raw.fillna(False).astype(bool)
    short_entries = short_raw.fillna(False).astype(bool)
    # exit when structure flips against us or the protective swing level breaks
    long_exits    = (dn | (close < sl_lvl)).fillna(False).astype(bool)
    short_exits   = (up | (close > sh_lvl)).fillna(False).astype(bool)

    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# Shared: portfolio execution and metrics
# ---------------------------------------------------------------------------

def run_portfolio(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    atr_period: int = 14,
    atr_mult_sl: float = 1.5,
    fees: float = FEES,
    slippage: float = SLIPPAGE,
    init_cash: float = 10_000.0,
    freq: str = "15min",
) -> vbt.Portfolio:
    sl_kwarg: dict = {}
    if atr_mult_sl and atr_mult_sl > 0:
        atr     = ta.atr(high, low, close, length=atr_period)
        sl_frac = (atr * atr_mult_sl / close).clip(upper=0.5).fillna(0.02)
        sl_kwarg["sl_stop"] = sl_frac

    return vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        freq=freq,
        **sl_kwarg,
    )


def profit_factor(pf: vbt.Portfolio) -> float:
    trades = pf.trades.records_readable
    if trades.empty:
        return float("nan")
    gross_profit = trades.loc[trades["PnL"] > 0, "PnL"].sum()
    gross_loss   = trades.loc[trades["PnL"] < 0, "PnL"].abs().sum()
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")
