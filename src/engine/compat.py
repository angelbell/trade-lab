"""Drop-in, bit-identical replacements built by composing the engine stages:
  run_compat(d, args)             == breakout_wave.run(d, args)
  run_ema_compat(d, side, args, thr) == ema_pullback.run(d, side, args, thr)
The stage boundaries are the composition points — swap any one stage (a new
detector, a different execution policy, a size overlay on the returned table)
without touching the others."""
import pandas_ta as ta

from .gates import gate_sma, gate_kama, exit_flip, ema_htf_gate, ema_exit_ma
from .detect import make_swings, pattern_b, pattern_a
from .detect_ema import ema_slope, ema_entries
from .plan import plan
from .walk import walk, walk_ema
from .stats import summarize, summarize_ema


def run_compat(d, args):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    es = d["close"].ewm(span=args.trend_ema, adjust=False).mean().values if args.trend_ema > 0 else None

    reg, ext_arr = gate_sma(d, args)
    kreg = gate_kama(d, args)
    against = exit_flip(d, args)

    sw = make_swings(h, l, c, a, args)
    if args.pattern == "B":
        setups = pattern_b(c, l, a, es, sw, args)
    else:
        setups = pattern_a(c, l, es, sw, args)
    orders = plan(c, l, a, sw, setups, reg, ext_arr, kreg, args)
    t, rr_real = walk(d, orders, against, args)
    if t is None:
        print("  no entries")
        return None
    return summarize(t, rr_real, args)


def run_ema_compat(d, side, args, thr):
    from ema_pullback import efficiency_ratio

    ef = (d["close"].rolling(args.ema_fast).mean() if args.fast_ma_type == "sma"
          else d["close"].ewm(span=args.ema_fast, adjust=False).mean()).values
    es = (d["close"].rolling(args.ema_slow).mean() if args.trend_ma_type == "sma"
          else d["close"].ewm(span=args.ema_slow, adjust=False).mean()).values
    a = ta.atr(d["high"], d["low"], d["close"], length=args.atr).values
    c, h, l = d["close"].values, d["high"].values, d["low"].values

    slope = ema_slope(es, a, args.slope_k)
    er = efficiency_ratio(c, args.er_period) if args.filter == "er" else None
    exit_ma = ema_exit_ma(d, args)
    daily_up, daily_dn = ema_htf_gate(d, args)

    entries = ema_entries(c, h, l, ef, a, slope, er, daily_up, daily_dn, side, args, thr)
    t, mfe, mae = walk_ema(d, entries, side, exit_ma, a, args)
    if t is None:
        print(f"  thr={thr:>4.2f}: no entries")
        return None
    return summarize_ema(t, mfe, mae, thr, args)
