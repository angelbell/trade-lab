"""Price inversion for SHORT mirrors: run the LONG machinery on p' = C − p
(C = 2×max high). A downward Pattern-B break, rally-limit entry, stop above the
lower-high and a down-target all fall out of the long code automatically.
Note ratio-based features (ext-cap %, swap on notional) do NOT mirror cleanly.

Lifted from experiments/short_mirror_15m.py — import invert from HERE: that module
executes its whole experiment at import time (old experiments style, kept as history)."""
import pandas as pd


def invert(d):
    C = 2 * d["high"].max()
    return pd.DataFrame({"open": C - d["open"], "high": C - d["low"],
                         "low": C - d["high"], "close": C - d["close"]}, index=d.index)
