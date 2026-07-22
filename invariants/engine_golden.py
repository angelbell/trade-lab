"""Golden snapshot for the engine delegation (Phase 3).

  dump : run the CURRENT breakout_wave.run / ema_pullback.run on the canonical legs
         and pickle the trade tables + stdout -> invariants/engine_golden.pkl
  check: run src.engine run_compat / run_ema_compat (or the delegated run()) and
         require exact equality with the pickled goldens.

Usage:
  .venv/bin/python invariants/engine_golden.py dump
  .venv/bin/python invariants/engine_golden.py check        # engine vs golden
  .venv/bin/python invariants/engine_golden.py check-run    # breakout_wave.run vs golden (post-delegation)
"""
import io, os, pickle, sys, warnings
from contextlib import redirect_stdout
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))   # 番人は invariants/ にあるが、
# 比較対象の実装は experiments/ にある（素の import を解決するため）

from src.data_loader import load_mt5_csv
from breakout_wave import resample
from radar_gate_race import BASE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = f"{ROOT}/invariants/engine_golden.pkl"

PB = dict(side="long", ema_fast=20, ema_slow=80, slope_k=6, filter="slope", er_period=14,
          swap_pct=0.0, daily_ema=0, exit_sma=0, exit_ma_type="sma", peryear=False,
          no_overlap=True, entry_trigger="close", fill_at_close=True, rr=3.0,
          min_stop_atr=0.5, atr=14, fwd=90, cost=0.001, trend_ma_type="sma", fast_ma_type="ema")


def configs():
    from short_mirror_15m import invert
    g1h = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc["2018-01-01":], "1h")
    b4h = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
    g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    bo = [
        ("gold_bo", g1h, dict(rr=3.0, daily_sma=150, daily_slope_k=10)),
        ("btc_bo_kama", b4h, dict(rr=2.0, gate_kama=14)),
        ("gold15m", g15, dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0, pullback_frac=0.25)),
        ("btc15m_L", d15, dict(rr=4.5, gate_kama=14, gate_kama_tf="240min",
                               pullback_frac=0.3, fill_win=200)),
        ("btc15m_S", invert(d15), dict(rr=4.5, gate_kama=14, pullback_frac=0.3)),
        ("v_split", d15, dict(rr=4.5, pullback_frac=0.3, exec_split=1)),
        ("v_tp1", g1h, dict(rr=3.0, tp1_frac=0.5, tp1_rr=1.0, tp1_be=1)),
        ("v_A", g1h, dict(rr=3.0, pattern="A", tp_mode="nexthigh")),
    ]
    ema = [
        ("btc_pull", b4h, "long", 0.0, {}),
        ("v_ema_short", b4h, "short", 0.0, {}),
        ("v_ema_gate", b4h, "long", 0.0, dict(gate_tf="1D", gate_type="kama-rising", gate_n=14)),
    ]
    return bo, ema


def snap(fn_bo, fn_ema):
    bo, ema = configs()
    out = {}
    for name, d, ov in bo:
        buf = io.StringIO()
        with redirect_stdout(buf):
            t = fn_bo(d, SimpleNamespace(**{**BASE, **ov}))
        out[name] = (t, buf.getvalue())
    for name, d, side, thr, ov in ema:
        buf = io.StringIO()
        with redirect_stdout(buf):
            t = fn_ema(d, side, SimpleNamespace(**{**PB, **ov}), thr)
        out[name] = (t, buf.getvalue())
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "dump":
        from breakout_wave import run
        from ema_pullback import run as run_ema
        out = snap(run, run_ema)
        with open(PKL, "wb") as f:
            pickle.dump(out, f)
        for k, (t, s) in out.items():
            print(f"  dumped {k:<12} n={'None' if t is None else len(t)}")
        print(f"golden -> {PKL}")
        return
    if mode == "check-run":
        from breakout_wave import run as fb
        from ema_pullback import run as fe
    else:
        from src.engine import run_compat as fb, run_ema_compat as fe
    with open(PKL, "rb") as f:
        gold = pickle.load(f)
    out = snap(fb, fe)
    npass = 0
    for k, (tg, sg) in gold.items():
        tn, sn = out[k]
        ok = (sg == sn)
        if tg is None or tn is None:
            ok = ok and (tg is None and tn is None)
        else:
            try:
                pd.testing.assert_frame_equal(tg, tn, check_exact=True)
            except AssertionError as ex:
                ok = False
                print(f"  [{k}] FRAME DIFF:\n{ex}")
        npass += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print(f"===== {npass}/{len(gold)} PASS vs golden =====")
    sys.exit(0 if npass == len(gold) else 1)


if __name__ == "__main__":
    main()
