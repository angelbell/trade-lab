"""Walk-forward optimization engine.

IS=180 days, OOS=30 days, step=30 days (rolling).
No data leakage: OOS timestamps never appear in IS optimization.
Supports multiple strategies via --strategy flag.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .strategy import (
    compute_signals_atr_trail,
    compute_signals_donchian,
    compute_signals_pivot_fade,
    compute_signals_structure,
    compute_signals_williams,
    profit_factor,
    run_portfolio,
)

PARAM_GRIDS: dict[str, dict] = {
    "williams": {
        "k":              [0.3, 0.5, 0.7, 1.0, 1.3],
        "htf_ema_period": [20, 50, 100],
        "atr_mult_sl":    [0.5, 1.0, 1.5, 2.0],
    },
    "donchian": {
        "donchian_period": [20, 30, 40, 60, 80],
        "htf_ema_period":  [20, 50, 100],
        "atr_mult_sl":     [0.5, 1.0, 1.5, 2.0],
    },
    # ATR Trail (EMA200 trend filter + ADX>20, no hard SL). Optimize af1/af2 only.
    "atr_trail": {
        "af1":           [0.3, 0.5, 1.0],   # fast ATR multiplier
        "af2":           [2.0, 3.0, 4.0],   # slow ATR multiplier
        "ema_period":    [200],
        "ma_type":       ["ema"],
        "adx_threshold": [20.0],
        "atr_mult_sl":   [0.0],             # 0 = no hard SL (proven to hurt)
    },
    # Same as atr_trail but SMA200 trend filter, for comparison.
    "atr_trail_sma": {
        "af1":           [0.3, 0.5, 1.0],
        "af2":           [2.0, 3.0, 4.0],
        "ema_period":    [200],
        "ma_type":       ["sma"],
        "adx_threshold": [20.0],
        "atr_mult_sl":   [0.0],
    },
    # Daily pivot-point mean reversion (fade to PP). Hard ATR SL for failed fades.
    "pivot_fade": {
        "entry":       ["r1", "r2"],
        "atr_mult_sl": [1.0, 1.5, 2.0, 3.0],
    },
    # Market-structure (HTF fractal swings) + LTF entry — encodes the user's method.
    # breakout won the mode search; now add A (min_struct) + B (adx) chop filters.
    "structure": {
        "fractal_n":     [3, 5],
        "entry_mode":    ["breakout"],
        "min_struct":    [0.0, 0.01, 0.02],   # A: structure-strength
        "adx_threshold": [0.0, 20.0, 25.0],   # B: ADX trend filter
        "atr_mult_sl":   [2.0, 3.0],
    },
    # COMPOSITE: structure breakout gated by daily-SMA timing (macro regime).
    "structure_sma": {
        "fractal_n":   [3, 5],
        "entry_mode":  ["breakout"],
        "sma_regime":  [100, 150, 200],
        "atr_mult_sl": [2.0, 3.0],
    },
}

SIGNAL_FNS: dict[str, Callable] = {
    "williams":      compute_signals_williams,
    "donchian":      compute_signals_donchian,
    "atr_trail":     compute_signals_atr_trail,
    "atr_trail_sma": compute_signals_atr_trail,
    "pivot_fade":    compute_signals_pivot_fade,
    "structure":     compute_signals_structure,
    "structure_sma": compute_signals_structure,
}


@dataclass
class WindowResult:
    window_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: dict[str, Any]
    is_pf: float
    is_sharpe: float
    oos_pf: float
    oos_sharpe: float
    oos_max_dd: float
    oos_trades: int


def generate_windows(
    index: pd.DatetimeIndex,
    is_days: int = 180,
    oos_days: int = 30,
    step_days: int = 30,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Return list of (is_index, oos_index) pairs with zero overlap."""
    windows = []
    start  = index[0]
    end    = index[-1]
    cursor = start

    while True:
        is_start  = cursor
        is_end    = is_start + pd.Timedelta(days=is_days)
        oos_start = is_end
        oos_end   = oos_start + pd.Timedelta(days=oos_days)

        if oos_end > end:
            break

        is_idx  = index[(index >= is_start) & (index < is_end)]
        oos_idx = index[(index >= oos_start) & (index < oos_end)]

        if len(is_idx) < 100 or len(oos_idx) < 10:
            break

        assert len(is_idx.intersection(oos_idx)) == 0, (
            f"BUG: IS/OOS overlap at window starting {is_start}"
        )

        windows.append((is_idx, oos_idx))
        cursor += pd.Timedelta(days=step_days)

    return windows


def _param_candidates(param_grid: dict) -> Any:
    keys = list(param_grid.keys())
    for combo in itertools.product(*param_grid.values()):
        yield dict(zip(keys, combo))


def _optimize_on_is(
    data_history: pd.DataFrame,
    is_idx: pd.DatetimeIndex,
    warmup_bars: int,
    htf_kwargs: dict,
    signal_fn: Callable,
    param_grid: dict,
    freq: str = "15min",
) -> tuple[dict[str, Any], float, float]:
    """Grid search over IS data. Full history passed for HTF EMA convergence."""
    best_sharpe = -np.inf
    best_params: dict[str, Any] = {}
    best_pf = float("nan")

    eval_idx   = is_idx[warmup_bars:]
    eval_close = data_history["close"].loc[eval_idx]
    eval_high  = data_history["high"].loc[eval_idx]
    eval_low   = data_history["low"].loc[eval_idx]

    for params in _param_candidates(param_grid):
        signal_params = {k: v for k, v in params.items() if k != "atr_mult_sl"}
        le, lx, se, sx = signal_fn(
            data_history["close"],
            data_history["high"],
            data_history["low"],
            **signal_params,
            **htf_kwargs,
        )

        pf = run_portfolio(
            eval_close, eval_high, eval_low,
            le.loc[eval_idx], lx.loc[eval_idx],
            se.loc[eval_idx], sx.loc[eval_idx],
            atr_mult_sl=params["atr_mult_sl"],
            freq=freq,
        )
        try:
            sharpe = pf.sharpe_ratio()
        except Exception:
            continue

        if np.isnan(sharpe):
            continue

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params.copy()
            best_pf = profit_factor(pf)

    return best_params, best_sharpe, best_pf


def run_wfo(
    data: pd.DataFrame,
    is_days: int = 180,
    oos_days: int = 30,
    step_days: int = 30,
    htf_timeframe: str = "1h",
    strategy: str = "williams",
    freq: str = "15min",
) -> list[WindowResult]:
    """Run walk-forward optimization and return per-window results."""
    if strategy not in PARAM_GRIDS:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(PARAM_GRIDS)}")

    param_grid = PARAM_GRIDS[strategy]
    signal_fn  = SIGNAL_FNS[strategy]

    close   = data["close"]
    windows = generate_windows(close.index, is_days, oos_days, step_days)

    if not windows:
        raise ValueError("No complete IS+OOS windows found — extend the date range.")

    warmup_bars = 80   # covers Donchian(80) and daily-range warm-up
    htf_kwargs  = {"htf_timeframe": htf_timeframe}
    n_combos    = sum(1 for _ in _param_candidates(param_grid))
    results: list[WindowResult] = []

    print(f"Strategy: {strategy} | HTF: {htf_timeframe} | "
          f"Grid: {n_combos} param combos per window")

    for i, (is_idx, oos_idx) in enumerate(windows):
        print(f"Window {i}: IS {is_idx[0].date()} → {is_idx[-1].date()} | "
              f"OOS {oos_idx[0].date()} → {oos_idx[-1].date()}")

        data_up_to_is = data.loc[data.index <= is_idx[-1]]
        best_params, is_sharpe, is_pf_val = _optimize_on_is(
            data_up_to_is, is_idx, warmup_bars=warmup_bars,
            htf_kwargs=htf_kwargs, signal_fn=signal_fn, param_grid=param_grid,
            freq=freq,
        )

        if not best_params:
            print(f"  No valid params found for window {i}, skipping.")
            continue

        data_up_to_oos = data.loc[data.index <= oos_idx[-1]]
        signal_params  = {k: v for k, v in best_params.items() if k != "atr_mult_sl"}
        le_w, lx_w, se_w, sx_w = signal_fn(
            data_up_to_oos["close"],
            data_up_to_oos["high"],
            data_up_to_oos["low"],
            **signal_params,
            **htf_kwargs,
        )

        close_oos    = close.loc[oos_idx]
        oos_portfolio = run_portfolio(
            close_oos,
            data["high"].loc[oos_idx],
            data["low"].loc[oos_idx],
            le_w.loc[oos_idx], lx_w.loc[oos_idx],
            se_w.loc[oos_idx], sx_w.loc[oos_idx],
            atr_mult_sl=best_params["atr_mult_sl"],
            freq=freq,
        )

        try:
            oos_sharpe = oos_portfolio.sharpe_ratio()
            oos_max_dd = oos_portfolio.max_drawdown()
        except Exception:
            oos_sharpe = float("nan")
            oos_max_dd = float("nan")

        oos_pf_val = profit_factor(oos_portfolio)
        oos_trades = oos_portfolio.trades.count()

        results.append(WindowResult(
            window_id=i,
            is_start=is_idx[0],
            is_end=is_idx[-1],
            oos_start=oos_idx[0],
            oos_end=oos_idx[-1],
            best_params=best_params,
            is_pf=is_pf_val,
            is_sharpe=is_sharpe,
            oos_pf=oos_pf_val,
            oos_sharpe=oos_sharpe,
            oos_max_dd=oos_max_dd,
            oos_trades=oos_trades,
        ))

    return results
