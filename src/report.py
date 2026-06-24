"""Report generation for walk-forward results."""

from __future__ import annotations

import math

import pandas as pd
from tabulate import tabulate

from .wfo import WindowResult


def _fmt(val: float, fmt: str = ".4f") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return format(val, fmt)


def _wfe(oos_pf: float, is_pf: float) -> str:
    if math.isnan(oos_pf) or math.isnan(is_pf) or is_pf <= 0:
        return "N/A"
    return f"{oos_pf / is_pf:.4f}"


def print_oos_table(results: list[WindowResult]) -> None:
    rows = []
    for r in results:
        rows.append([
            r.window_id,
            r.is_start.date(),
            r.oos_start.date(),
            _fmt(r.oos_pf),
            f"{r.oos_max_dd * 100:.2f}%",
            _fmt(r.oos_sharpe),
            r.oos_trades,
            _fmt(r.is_pf),
            _wfe(r.oos_pf, r.is_pf),
            str({k: v for k, v in r.best_params.items()}),
        ])

    print("\n" + "=" * 80)
    print("OOS Results by Window")
    print("=" * 80)
    print(tabulate(
        rows,
        headers=["Win", "IS Start", "OOS Start", "OOS PF", "MaxDD", "Sharpe", "Trades",
                 "IS PF", "WFE", "Best Params"],
        tablefmt="rounded_outline",
    ))


def print_wfe_summary(results: list[WindowResult]) -> None:
    valid = [r for r in results if not (math.isnan(r.oos_pf) or math.isnan(r.is_pf))]

    if not valid:
        print("\nNo valid windows to summarize.")
        return

    is_pfs  = [r.is_pf  for r in valid]
    oos_pfs = [r.oos_pf for r in valid]
    wfes    = [o / i for o, i in zip(oos_pfs, is_pfs) if i > 0]

    # Composite OOS metrics (concatenated OOS equity curve approximation via average)
    avg_oos_pf     = sum(oos_pfs) / len(oos_pfs)
    avg_is_pf      = sum(is_pfs) / len(is_pfs)
    avg_wfe        = sum(wfes) / len(wfes) if wfes else float("nan")
    avg_oos_sharpe = sum(r.oos_sharpe for r in valid) / len(valid)
    avg_oos_dd     = sum(r.oos_max_dd for r in valid) / len(valid)
    total_trades   = sum(r.oos_trades for r in results)

    print("\n" + "=" * 80)
    print("Walk-Forward Summary")
    print("=" * 80)
    summary = [
        ["Windows evaluated", len(results)],
        ["Windows with valid trades", len(valid)],
        ["Avg IS  Profit Factor", f"{avg_is_pf:.4f}"],
        ["Avg OOS Profit Factor", f"{avg_oos_pf:.4f}"],
        ["Avg WFE (OOS PF / IS PF)", f"{avg_wfe:.4f}"],
        ["Avg OOS Sharpe Ratio", f"{avg_oos_sharpe:.4f}"],
        ["Avg OOS Max Drawdown", f"{avg_oos_dd * 100:.2f}%"],
        ["Total OOS Trades", total_trades],
    ]
    print(tabulate(summary, tablefmt="rounded_outline"))

    wfe_threshold = 0.70
    print(f"\nWFE target (>= {wfe_threshold}): "
          f"{'PASS ✓' if avg_wfe >= wfe_threshold else 'FAIL ✗'} ({avg_wfe:.4f})")
