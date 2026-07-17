"""Composable backtest engine — breakout_wave.run() split into pure stages:

    bars → gates (regime arrays) → setups (detect) → orders (plan) → trades (walk) → summarize

Contract: `run_compat(d, args)` is BIT-IDENTICAL to `breakout_wave.run(d, args)`
(same returned trade table, same printed lines). Guarded by scratchpad/engine_tieback.py —
any edit to this package must re-pass that gauntlet before its numbers are trusted.

Swing/KAMA primitives stay in breakout_wave.py (201 scripts import them from there);
this package imports them, never redefines them.
"""
from .gates import gate_sma, gate_kama, exit_flip, ema_htf_gate, ema_exit_ma
from .detect import make_swings, pattern_b, pattern_a
from .detect_ema import ema_slope, ema_entries
from .plan import plan
from .walk import walk, walk_ema
from .stats import summarize, summarize_ema
from .mirror import invert
from .size import (bar_idx, pdh_series, pdl_series, pdh_soft, pdl_break_mask,
                   hh4h_series, pdh_hh4h_ladder, daily_regime_mult,
                   compute_labels, ict_label_mult)
from .arbiter import BUDGET, cd, Boot, months_union
from . import walk_ict   # ICT execution model (ask-limit/killzone) — use walk_ict.walk etc.
from .compat import run_compat, run_ema_compat
