"""The Pine and the backtest disagree about how long a pullback limit stays live.

breakout_wave.run() searches for the fill over `args.fwd` bars -- 500 for both 15m legs, i.e. the
limit sits live for 5.2 days. The shipped Pine cancels after fillWin = 200 bars (2.1 days). So the
strategy I would actually TRADE takes fewer trades than the strategy I measured, and every book
number reported today (8.27, etc.) came from the 500-bar version.

Which one is right is an empirical question, not a preference:
  - if the leg/book is flat across the fill window, the mismatch is harmless -> set the Pine to
    whatever is operationally saner (a limit that waits 5 days on a stale breakout is odd)
  - if it is NOT flat, then one of the two is wrong and the numbers must be restated.

Sweep the fill window on btc15m_L and gold15m; read the leg AND the 6-leg book.
Judged on the corrected arbiter (trade-resolution DD, trade-sigma inv-vol at 3%).
Run: .venv/bin/python scratchpad/fillwin_fidelity.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from book_spec_fix import build, book

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]


def leg_line(s, tag, yrs):
    pf = s[s > 0].sum() / abs(s[s <= 0].sum())
    eq = np.cumprod(1 + 0.01 * s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    return (f"  {tag:<20}n={len(s):>4}{len(s)/yrs:>7.0f}本/年  win={100*(s>0).mean():>4.1f}%  "
            f"PF={pf:>4.2f}  meanR={s.mean():+.3f}  legC/DD={cagr/dd:>5.2f}")


def main():
    legs0 = build("2018-01-01", False)
    print("注: `fwd` は指値の待機本数と、建てた後の最大保有本数を兼ねている。")
    print("    保有側まで短くすると別物になるので、ここでは **指値の待機だけ** を変えて測る。\n")

    with contextlib.redirect_stderr(io.StringIO()):
        b15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")

    for name, d, cfg, cost in (
        ("btc15m_L", b15, {**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                           "pullback_frac": 0.3, "rr": 4.5}, 15.0),
        ("gold15m", g15, {**BASE, "daily_sma": 150, "daily_slope_k": 10,
                          "ext_cap": 8.0, "pullback_frac": 0.25}, 0.3),
    ):
        print(f"--- {name} : 指値の待機本数を変える（保有側の fwd=500 は固定） ---")
        for fw in (50, 100, 200, 300, 500):
            with contextlib.redirect_stderr(io.StringIO()):
                t = run(d, SimpleNamespace(**{**cfg, "fill_win": fw}))
            R = t["R"].values - cost / t["risk"].values
            idx = pd.DatetimeIndex(t["time"])
            if name == "btc15m_L":                     # PDH soft size
                ei = d.index.get_indexer(t["time"])
                pdh = d["high"].resample("1D").max().dropna().shift(1).reindex(
                    d.index, method="ffill").values
                R = R * np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
            s = pd.Series(R, index=idx)
            yrs = (idx[-1] - idx[0]).days / 365.25
            L = dict(legs0); L[name] = s
            c, dd, x, n = book(L, SIX)
            tag = f"fillWin={fw}" + ("  ← Pine の既定" if fw == 200 else
                                     "  ← backtest(=fwd)" if fw == 500 else "")
            print(leg_line(s, tag, yrs) + f"   6レッグ・ブック={x:>5.2f}")
        print()


if __name__ == "__main__":
    main()
