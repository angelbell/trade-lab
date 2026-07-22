"""ENGINE TIE-BACK GAUNTLET — src/engine/run_compat vs breakout_wave.run
must be BIT-IDENTICAL (returned trade table exact-equal AND printed lines equal)
on the canonical book legs + a branch-coverage matrix of variant flags.
ANY diff = FAIL: report it, do not "fix" the expectation.

Run: .venv/bin/python invariants/engine_tieback.py 2>&1 | tee experiments/out_engine_tieback.txt
"""
import io, os, sys, warnings
from contextlib import redirect_stdout
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))   # 番人は invariants/ にあるが、
# 比較対象の実装は experiments/ にある（素の import を解決するため）

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from ema_pullback import run as run_ema
from src.engine import run_compat, run_ema_compat
from radar_gate_race import BASE
from short_mirror_15m import invert

PB = dict(side="long", ema_fast=20, ema_slow=80, slope_k=6, filter="slope", er_period=14,
          swap_pct=0.0, daily_ema=0, exit_sma=0, exit_ma_type="sma", peryear=False,
          no_overlap=True, entry_trigger="close", fill_at_close=True, rr=3.0,
          min_stop_atr=0.5, atr=14, fwd=90, cost=0.001, trend_ma_type="sma", fast_ma_type="ema")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def capture(fn, d, args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        t = fn(d, args)
    return t, buf.getvalue()


def compare(name, d, overrides, base=None):
    args = SimpleNamespace(**{**(base or BASE), **overrides})
    t_old, s_old = capture(run, d, args)
    t_new, s_new = capture(run_compat, d, args)
    ok_print = (s_old == s_new)
    if t_old is None and t_new is None:
        ok_df = True
    elif t_old is None or t_new is None:
        ok_df = False
    else:
        try:
            pd.testing.assert_frame_equal(t_old, t_new, check_exact=True)
            ok_df = True
        except AssertionError as ex:
            ok_df = False
            print(f"  [{name}] FRAME DIFF:\n{ex}")
    n = "None" if t_old is None else str(len(t_old))
    mark = "PASS" if (ok_df and ok_print) else "FAIL"
    print(f"  {mark}  {name:<34} n={n:<6} frame={'OK' if ok_df else 'DIFF'} print={'OK' if ok_print else 'DIFF'}")
    if not ok_print:
        print(f"    --- old stdout ---\n{s_old}    --- new stdout ---\n{s_new}")
    return ok_df and ok_print


def compare_ema(name, d, side, thr, overrides):
    args = SimpleNamespace(**{**PB, **overrides})
    buf_o, buf_n = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_o):
        t_old = run_ema(d, side, args, thr)
    with redirect_stdout(buf_n):
        t_new = run_ema_compat(d, side, args, thr)
    s_old, s_new = buf_o.getvalue(), buf_n.getvalue()
    ok_print = (s_old == s_new)
    if t_old is None and t_new is None:
        ok_df = True
    elif t_old is None or t_new is None:
        ok_df = False
    else:
        try:
            pd.testing.assert_frame_equal(t_old, t_new, check_exact=True)
            ok_df = True
        except AssertionError as ex:
            ok_df = False
            print(f"  [{name}] FRAME DIFF:\n{ex}")
    n = "None" if t_old is None else str(len(t_old))
    mark = "PASS" if (ok_df and ok_print) else "FAIL"
    print(f"  {mark}  {name:<34} n={n:<6} frame={'OK' if ok_df else 'DIFF'} print={'OK' if ok_print else 'DIFF'}")
    if not ok_print:
        print(f"    --- old stdout ---\n{s_old}    --- new stdout ---\n{s_new}")
    return ok_df and ok_print


def main():
    results = []

    print("== canonical legs ==")
    g1h = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc["2018-01-01":], "1h")
    results.append(compare("gold_bo (anchor n=208 meanR+0.49)", g1h,
                           dict(rr=3.0, daily_sma=150, daily_slope_k=10)))
    b4h = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
    results.append(compare("btc_bo_kama", b4h, dict(rr=2.0, gate_kama=14)))
    g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    results.append(compare("gold15m", g15,
                           dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0, pullback_frac=0.25)))
    d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    results.append(compare("btc15m_L (anchor n=759 meanR+0.59)", d15,
                           dict(rr=4.5, gate_kama=14, gate_kama_tf="240min",
                                pullback_frac=0.3, fill_win=200)))
    inv = invert(d15)
    results.append(compare("btc15m_S (inverted)", inv,
                           dict(rr=4.5, gate_kama=14, pullback_frac=0.3)))

    print("== branch coverage (gold h1) ==")
    G = dict(rr=3.0)
    for name, ov in [
        ("A/nexthigh/line",   dict(pattern="A", tp_mode="nexthigh")),
        ("A/rr/swinglow",     dict(pattern="A", tp_mode="rr", sl_mode="swinglow")),
        ("B retest=10",       dict(retest=10)),
        ("B tp1 scale-out",   dict(tp1_frac=0.5, tp1_rr=1.0, tp1_be=1)),
        ("B max_pos=3",       dict(max_pos=3)),
        ("B exit_kama=14",    dict(exit_kama=14)),
        ("B sl_b=band tgt=l2", dict(sl_b="band", tgt_ref="l2")),
        ("B sl_b=atr",        dict(sl_b="atr")),
        ("B swing=pivot",     dict(swing="pivot")),
        ("B swing=momentum",  dict(swing="momentum")),
        ("B swing=renko",     dict(swing="renko")),
        ("B tp=measured",     dict(tp_mode="measured")),
        ("B wave=3",          dict(wave="3")),
        ("B sma+ext_cap",     dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0)),
        ("B tp1+exit_kama",   dict(tp1_frac=0.5, exit_kama=14)),
    ]:
        results.append(compare(name, g1h, {**G, **ov}))

    print("== branch coverage (btc 15m) ==")
    for name, ov in [
        ("exec_split",        dict(rr=4.5, pullback_frac=0.3, exec_split=1)),
        ("swap_pct",          dict(rr=4.5, pullback_frac=0.3, swap_pct=0.01)),
        ("kama tf2 AND",      dict(rr=4.5, gate_kama=14, gate_kama_tf="240min",
                                   gate_kama_tf2="1D", pullback_frac=0.3)),
        ("pullback max_pos3", dict(rr=4.5, pullback_frac=0.3, max_pos=3)),
        ("split+swap",        dict(rr=4.5, pullback_frac=0.3, exec_split=1, swap_pct=0.01)),
    ]:
        results.append(compare(name, d15, ov))

    print("== ema_pullback engine (btc 4h + branch coverage) ==")
    b4h_full = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
    results.append(compare_ema("btc_pull canonical (PB thr=0)", b4h_full, "long", 0.0, {}))
    g4h = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc["2018-01-01":], "4h")
    for name, side, thr, ov in [
        ("ema short",              "short", 0.0,  {}),
        ("ema thr=0.10",           "long",  0.10, {}),
        ("ema filter=er",          "long",  0.30, dict(filter="er")),
        ("ema touch trigger",      "long",  0.0,  dict(entry_trigger="touch")),
        ("ema fill_at_ma",         "long",  0.0,  dict(fill_at_close=False)),
        ("ema exit_sma=200",       "long",  0.0,  dict(exit_sma=200)),
        ("ema exit ema200",        "long",  0.0,  dict(exit_sma=200, exit_ma_type="ema")),
        ("ema gate 1D ema-slope",  "long",  0.0,  dict(gate_tf="1D", gate_type="ema-slope", gate_n=14)),
        ("ema gate kama-rising",   "long",  0.0,  dict(gate_tf="1D", gate_type="kama-rising", gate_n=14)),
        ("ema gate sma-slope W",   "long",  0.0,  dict(gate_tf="1W", gate_type="sma-slope", gate_n=30)),
        ("ema overlap allowed",    "long",  0.0,  dict(no_overlap=False)),
        ("ema swap_pct",           "long",  0.0,  dict(swap_pct=0.01)),
        ("ema min_stop_atr=2",     "long",  0.0,  dict(min_stop_atr=2.0)),
        ("ema ma types ema/sma",   "long",  0.0,  dict(trend_ma_type="ema", fast_ma_type="sma")),
        ("ema short gate 1D",      "short", 0.0,  dict(gate_tf="1D", gate_type="ema-slope", gate_n=14)),
    ]:
        results.append(compare_ema(name, g4h, side, thr, ov))

    npass = sum(results)
    print(f"\n===== {npass}/{len(results)} PASS =====")
    if npass < len(results):
        print("!! NOT bit-identical — engine must not be used until every row passes.")
        sys.exit(1)
    print("engine == breakout_wave.run: bit-identical on all configs.")


if __name__ == "__main__":
    main()
