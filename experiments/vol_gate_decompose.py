"""stop/price = (stop/ATR) x (ATR/price).  Filtering on the LEFT factor alone (base looseness) makes
the book WORSE (8.28 -> 7.21). Filtering on the PRODUCT makes it slightly better (8.28 -> 8.47). By
elimination the gain must live in the RIGHT factor -- ATR/price, i.e. the volatility LEVEL. If so,
"skip breakouts whose stop exceeds 3% of price" was never a quality filter at all; it was a
volatility gate wearing a costume, and the honest way to state it is:

    btc15m_L should not trade when 15m realised volatility is high.

Test the three factors head to head, on the BOOK, so the claim is falsifiable rather than a story:
  A. stop/ATR      <= x   (base looseness alone)          -- expect: book down (already measured)
  B. ATR/price     <= x   (volatility level alone)        -- if the gain is here, this is the truth
  C. stop/price    <= x   (the product)                   -- the original, for reference
Then, for whichever wins, the block bootstrap on the BOOK decides.
Run: .venv/bin/python experiments/vol_gate_decompose.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
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
    atr = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1)     # confirmed bar
    ei = d15.index.get_indexer(t["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
    risk = t["risk"].values / w
    R = pd.Series(t["R"].values * w - 15.0 / risk, index=pd.DatetimeIndex(t["time"]))
    px = t["e_px"].values
    A = atr.values[ei]
    F = {"A. 損切り/ATR（保ち合いの緩さ）": risk / A,
         "B. ATR/価格 %（ボラ水準）": 100.0 * A / px,
         "C. 損切り/価格 %（積＝元の案）": 100.0 * risk / px}
    ok = np.isfinite(A)
    R = R[ok]; F = {k: v[ok] for k, v in F.items()}
    yrs = (R.index[-1] - R.index[0]).days / 365.25

    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    r0 = book(legs0, SIX)[2]
    print(f"現行のブック CAGR/DD = {r0:.2f}   btc15m_L: n={len(R)} {len(R)/yrs:.0f}本/年 "
          f"PF {pf(R.values):.2f} totR/年 {R.sum()/yrs:+.1f}\n")
    print("3つの因子を、それぞれ単独で『上位を切る』フィルタにして、ブックで裁定する")
    print("（B が勝てば、これはボラ・ゲートであって品質フィルタではない）\n")
    for name, v in F.items():
        print(f"  {name}   中央値 {np.median(v):.2f}")
        print(f"    {'切る位置（分位）':<20}{'しきい値':>9}{'n':>6}{'本/年':>7}{'PF':>7}"
              f"{'totR/年':>10}{'ブックCAGR/DD':>14}")
        for q in (100, 90, 80, 70, 60, 50):
            thr = np.percentile(v, q)
            m = v <= thr
            if m.sum() < 100:
                continue
            legs = dict(legs0); legs["btc15m_L"] = R[m]
            rb = book(legs, SIX)[2]
            lab = "全部" if q == 100 else f"上位 {100-q}% を捨てる"
            print(f"    {lab:<20}{thr:>9.2f}{m.sum():>6}{m.sum()/yrs:>7.0f}{pf(R.values[m]):>7.2f}"
                  f"{R.values[m].sum()/yrs:>+10.1f}{rb:>14.2f}{'  ★' if rb > r0 else ''}")
        print()


if __name__ == "__main__":
    main()
