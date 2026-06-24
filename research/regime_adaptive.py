"""regime_adaptive.py -- do ADAPTIVE/multi-TF regime gates beat the fixed SMA gate, on BOTH
gold AND BTC? (generalization built in from the start, not gold-only.)

regime_headroom.py found the mech->oracle headroom is NOT predictable from static price features
(slope/vol/extension/ER all failed to separate clean). This tries genuinely ADAPTIVE estimates of
"is the trend regime on" -- still price-based, so the prior says they'll likely hit the same ceiling,
but the mechanism differs so we test:
  kama_slope  : Kaufman Adaptive MA (ER-modulated smoothing) rising / steep
  stack3      : SMA50>150>200 all rising (multi-horizon agreement)
  persistK    : daily uptrend sustained >= K days (deploy only after the regime is established)
  dwk         : daily uptrend AND weekly uptrend (multi-timeframe agreement)
Refs: always-on, mech SMA150+slope, oracle(lookahead). Judged on IS/OOS (robust=IS~=OOS both up)
and -- crucially -- it must hold on BOTH instruments (steep-slope looked great on gold, died on BTC).

  .venv/bin/python research/regime_adaptive.py --csv data/vantage_xauusd_h1.csv --tf 1h --rr 3 --fwd 500
  .venv/bin/python research/regime_adaptive.py --csv data/vantage_btcusd_h1.csv --tf 4h --rr 2 --fwd 300
"""
import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG, SPLIT, er, metrics, at


def kama(close, n=10, fast=2, slow=30):
    ch = close.diff(n).abs()
    vol = close.diff().abs().rolling(n).sum()
    erc = (ch / vol).fillna(0).values
    fsc, ssc = 2 / (fast + 1), 2 / (slow + 1)
    sc = (erc * (fsc - ssc) + ssc) ** 2
    c = close.values; out = np.full(len(c), np.nan)
    seed = n
    out[seed] = c[seed]
    for i in range(seed + 1, len(c)):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def persist(up, k):
    """True where boolean `up` has been continuously True for >= k bars."""
    grp = (~up).cumsum()
    streak = up.groupby(grp).cumsum()
    return streak >= k


def show(name, t):
    m = metrics(t)
    if m is None:
        print(f"  {name:<22} (too few)"); return
    print(f"  {name:<22} n={m['n']:>4} CAGR={m['cagr']:+5.1f}% DD={m['dd']:4.1f}% "
          f"CAGR/DD={m['cdd']:5.2f} | IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | yr+ {m['py']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/vantage_xauusd_h1.csv")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--fwd", type=int, default=500)
    ap.add_argument("--kama-n", type=int, default=10)
    a = ap.parse_args()
    d = resample(load_mt5_csv(a.csv), a.tf)
    t = run(d, SimpleNamespace(**{**CFG, "csv": a.csv, "tf": a.tf, "rr": a.rr, "fwd": a.fwd}))
    t = t.sort_values("time").reset_index(drop=True)

    dc = d["close"].resample("1D").last().dropna()
    a14 = dc.diff().abs().rolling(14).mean()
    s50, s150, s200 = dc.rolling(50).mean(), dc.rolling(150).mean(), dc.rolling(200).mean()
    km = kama(dc, a.kama_n)
    km_slope = (km - km.shift(10)) / (10 * a14)
    wk = d["close"].resample("1W").last().dropna()
    wk_up = (wk > wk.rolling(30).mean())
    wk_up_d = wk_up.reindex(dc.index, method="ffill")
    up150 = dc > s150

    gates = {
        "(ref) always-on":   pd.Series(True, index=dc.index),
        "(ref) mech 150+slp": up150 & (s150 > s150.shift(10)),
        "kama rising":        km > km.shift(1),
        "kama slope>=0.05":   km_slope >= 0.05,
        "stack3 50>150>200":  (s50 > s150) & (s150 > s200) & (s50 > s50.shift(10)),
        "persist up>=20d":    persist(up150, 20),
        "persist up>=40d":    persist(up150, 40),
        "daily&weekly up":    up150 & wk_up_d,
        "(ref) ORACLE":       dc.shift(-20) > dc,
    }
    print(f"\n=== regime_adaptive :: {os.path.basename(a.csv)} {a.tf} (entries FIXED) | IS<{SPLIT} ===")
    for name, g in gates.items():
        gs = g if "ORACLE" in name else g.shift(1)         # no lookahead except the oracle label
        show(name, t[at(gs, t.time)])


if __name__ == "__main__":
    main()
