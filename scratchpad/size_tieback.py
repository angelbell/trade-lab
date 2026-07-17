"""SIZE-LAYER TIE-BACK — src/engine/size.py must be ARRAY-IDENTICAL to the frozen
evidence scripts it was lifted from (stack_size_btc15mL / book_integration inline /
ict_size_transplant.compute_labels), on the canonical btc15m_L trade set.

Run: .venv/bin/python scratchpad/size_tieback.py 2>&1 | tee scratchpad/out_size_tieback.txt
"""
import io, os, sys, warnings
from contextlib import redirect_stdout
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from src.engine.mirror import invert
from src.engine import size as SZ
from stack_size_btc15mL import comp1_ladder, comp2_daily, comp3_ict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
oks = []


def chk(name, ok):
    oks.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def main():
    d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    buf = io.StringIO()
    with redirect_stdout(buf):
        t = run(d15, SimpleNamespace(**{**BASE, "rr": 4.5, "gate_kama": 14,
                                        "gate_kama_tf": "240min", "pullback_frac": 0.3,
                                        "fill_win": 200}))
    ii = d15.index.get_indexer(pd.DatetimeIndex(t["time"]))
    print(f"canonical btc15m_L: n={len(t)} (expect 763)")

    # ladder vs stack_size_btc15mL.comp1_ladder
    W_new, ap_new, ah_new = SZ.pdh_hh4h_ladder(d15, t, ii)
    W_old, ap_old, ah_old = comp1_ladder(d15, t, ii)
    chk("pdh_hh4h_ladder == comp1_ladder",
        np.array_equal(W_new, W_old) and np.array_equal(ap_new, ap_old)
        and np.array_equal(ah_new, ah_old))

    # daily regime vs comp2_daily
    W2n, dn = SZ.daily_regime_mult(d15, t, ii)
    W2o, do = comp2_daily(d15, t, ii)
    chk("daily_regime_mult == comp2_daily",
        np.array_equal(W2n, W2o) and np.array_equal(dn, do))

    # ICT A∧B labels vs comp3_ict (transitively checks compute_labels verbatim)
    W3n, abn = SZ.ict_label_mult(d15, t, ii, x=48, weak=0.5, label="AB")
    W3o, abo = comp3_ict(d15, t, ii, x=48)
    chk("ict_label_mult(AB) == comp3_ict",
        np.array_equal(W3n, W3o) and np.array_equal(abn, abo))

    # PDH soft vs book_integration's inline formula (that module runs on import ->
    # the 2 lines are replicated here verbatim instead of imported)
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ab_bi = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    W_bi = np.where(ab_bi, 1.0, 0.5)
    W_sz, ab_sz = SZ.pdh_soft(d15, t)
    chk("pdh_soft == book_integration inline",
        np.array_equal(W_sz, W_bi) and np.array_equal(ab_sz, ab_bi))

    # PDL hard mask (short mirror) vs book_integration's inline formula
    inv = invert(d15); C = 2 * d15["high"].max()
    with redirect_stdout(buf):
        ts_ = run(inv, SimpleNamespace(**{**BASE, "rr": 4.5, "gate_kama": 14,
                                          "pullback_frac": 0.3}))
    pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
    mS_bi = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
    mS_sz = SZ.pdl_break_mask(d15, ts_, C)
    chk("pdl_break_mask == book_integration inline", np.array_equal(mS_sz, mS_bi))

    print(f"\n===== {sum(oks)}/{len(oks)} PASS =====")
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
