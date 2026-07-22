"""Follow-up: is there ANY fade edge? all-signals BASE (no SAR) vs SAR-flip, TF ladder."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
import research.edge_harness as EH
from research.edge_harness import evaluate, LADDERS, AGG
from src.data_loader import load_mt5_csv
from experiments.sar_fade import _range_ref, make_fade

_m5 = load_mt5_csv("data/vantage_usdjpy_m5.csv"); _m5 = _m5[_m5.index >= "2018-01-01"]
_real = load_mt5_csv
EH.load_mt5_csv = lambda p, *a, **k: _m5.copy() if p == "MEM_M5c" else _real(p, *a, **k)
LADDERS["UJ_L"] = ("data/vantage_usdjpy_m15.csv", 0.015,
                   [("15m", "15min"), ("30m", "30min"), ("1h", "60min"), ("2h", "120min"), ("4h", "240min")])
LADDERS["UJ_H1"] = ("data/vantage_usdjpy_h1.csv", 0.015,
                    [("1h", None), ("2h", "120min"), ("4h", "240min"), ("8h", "480min")])

def make_pure_fade(k, ref="day"):
    """all-signals base: stretched from EMA200 -> fade to mean, NO SAR filter."""
    def fade(df):
        c = df["close"].values
        e200 = df["close"].ewm(span=200, adjust=False).mean().values
        rr = _range_ref(df, ref); dev = c - e200
        sig = np.zeros(len(c))
        sig[dev < -k * rr] = 1
        sig[dev > k * rr] = -1
        sig[:210] = 0
        return sig
    return fade

if __name__ == "__main__":
    print("=== ALL-SIGNALS BASE: pure stretch->fade (NO SAR), k=1.0 daily-range, GROSS, TF ladder ===")
    evaluate("UJ_L", make_pure_fade(1.0), exit_mode="mean", cost=0.0)
    evaluate("UJ_H1", make_pure_fade(1.0), exit_mode="mean", cost=0.0)
    print("=== SAR-flip fade, k=1.0 daily-range, GROSS, TF ladder ===")
    evaluate("UJ_L", make_fade(1.0, "day"), exit_mode="mean", cost=0.0)
    evaluate("UJ_H1", make_fade(1.0, "day"), exit_mode="mean", cost=0.0)
    print("=== SAR-flip fade, NET (cost 1.5pip), TF ladder ===")
    evaluate("UJ_L", make_fade(1.0, "day"), exit_mode="mean", stop_slip=0.5)
    evaluate("UJ_H1", make_fade(1.0, "day"), exit_mode="mean", stop_slip=0.5)
