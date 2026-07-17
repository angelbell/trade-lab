"""stop/PRICE wobbled (book 8.47 / 8.31 / 8.38 across thresholds) -- that is what a mis-specified
variable looks like. The suspect: stop/price mixes two different things. A 2%-of-price stop in a
calm month means "this consolidation was sloppy"; the same 2% in a volatile month means "volatility
is high" and says nothing about the breakout's quality. 2021's median stop was 2.09% of price vs
2023's 1.02% -- that is the volatility cycle, not a quality signal.

Divide by ATR instead. stop/ATR asks the scale-free, vol-free question the filter is actually for:
"relative to how much this thing is moving right now, is the base I am breaking out of tight or loose?"
If the effect is real, stop/ATR should give a MONOTONE plateau where stop/price gave a wobble.
Judged on the BOOK (trade-resolution CAGR/DD, inv-vol on trade-R sigma, 3% budget), not the leg.
Run: .venv/bin/python scratchpad/stop_over_atr.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import book
import pandas_ta as ta

ROOT = "/home/angelbell/dev/auto-trade"


def pf(x):
    return x[x > 0].sum() / abs(x[x <= 0].sum())


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    # ATR on the SAME 15m bars, CONFIRMED (shift 1) -- the value a trader could see at the signal bar
    atr = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1)
    ei = d15.index.get_indexer(t["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
    risk = t["risk"].values / w
    R = pd.Series(t["R"].values * w - 15.0 / risk, index=pd.DatetimeIndex(t["time"]))
    sa = risk / atr.values[ei]                        # stop / ATR  -- scale-free AND vol-free
    sp = 100.0 * risk / t["e_px"].values              # stop / price -- the wobbly one, for comparison
    ok = np.isfinite(sa)
    R, sa, sp = R[ok], sa[ok], sp[ok]
    yrs = (R.index[-1] - R.index[0]).days / 365.25

    print(f"損切り / ATR(14, 確定足):  中央値 {np.median(sa):.2f}  10/90%点 "
          f"{np.percentile(sa,10):.2f} / {np.percentile(sa,90):.2f}")
    y = pd.Series(sa, index=R.index)
    print("  年別の中央値: " + "  ".join(f"{a}:{b:.2f}" for a, b in y.groupby(y.index.year).median().items()))
    y2 = pd.Series(sp, index=R.index)
    print("  （比較・損切り/価格: " + "  ".join(f"{a}:{b:.2f}%" for a, b in y2.groupby(y2.index.year).median().items()) + "）")
    print("  → ATR で割ると年ごとのブレが消えるはず。消えなければ、そもそも変数が違う。\n")

    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    c0, d0, r0, n0 = book(legs0, SIX)
    print(f"  {'条件':<26}{'n':>6}{'本/年':>7}{'PF':>7}{'meanR':>9}{'totR/年':>10}"
          f"{'ブックCAGR/DD':>14}")
    print(f"  {'全部（現行）':<26}{len(R):>6}{len(R)/yrs:>7.0f}{pf(R.values):>7.2f}"
          f"{R.mean():>+9.3f}{R.sum()/yrs:>+10.1f}{r0:>14.2f}")
    for thr in (5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5):
        m = sa <= thr
        if m.sum() < 100:
            continue
        s = R[m]
        legs = dict(legs0); legs["btc15m_L"] = s
        c, d, rb, n = book(legs, SIX)
        print(f"  {'損切り <= ATR × '+str(thr):<26}{len(s):>6}{len(s)/yrs:>7.0f}{pf(s.values):>7.2f}"
              f"{s.mean():>+9.3f}{s.sum()/yrs:>+10.1f}{rb:>14.2f}{'  ★' if rb > r0 else ''}")


if __name__ == "__main__":
    main()
