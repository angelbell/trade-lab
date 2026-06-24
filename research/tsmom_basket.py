"""tsmom_basket.py -- mimic the PROVEN TSMOM setup (AQR/Winton/AHL-style managed-futures): a
CROSS-ASSET basket with per-instrument vol-targeting + blended lookbacks + symmetric long/short.

Our gold+BTC TSMOM was weak (bleeding shorts, no breadth). The literature's edge comes from breadth
across UNCORRELATED asset classes (the short side works because at any time SOME market trends down).
This tests whether asset-class breadth turns the weak 2-asset TSMOM into a smooth, LOW-correlated
stream worth adding to the book. PRE-SCREEN: feed = yfinance daily (NOT Vantage), monthly TSMOM ->
direction/decorrelation check only; a Vantage-tradeable subset + the gauntlet is the real test.

  .venv/bin/python research/tsmom_basket.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.instrument_screen import load                 # cached yfinance daily close
from research.portfolio_kama import get_legs
from research.portfolio_alloc import monthly_matrix, cagr_dd_monthly, SPLIT

LOOKBACKS = (3, 6, 12)        # blended momentum horizons (months)
TGT_VOL_M = 0.04             # per-instrument target monthly vol (~14%/yr)
COST_M = 0.0005             # monthly rebalance cost

# cross-asset universe (label -> yfinance ticker), spanning equity/bond/FX/commodity/crypto
UNIV = {
    "US500": "^GSPC", "NAS100": "^NDX", "JP225": "^N225", "DAX": "^GDAXI",
    "UST20y": "TLT", "EURUSD": "EURUSD=X", "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X",
    "gold": "GC=F", "silver": "SI=F", "WTI": "CL=F", "copper": "HG=F",
    "BTC": "BTC-USD", "ETH": "ETH-USD",
}
CLASS = {"US500": "EQ", "NAS100": "EQ", "JP225": "EQ", "DAX": "EQ", "UST20y": "BOND",
         "EURUSD": "FX", "USDJPY": "FX", "AUDUSD": "FX", "gold": "COMM", "silver": "COMM",
         "WTI": "COMM", "copper": "COMM", "BTC": "CRYPTO", "ETH": "CRYPTO"}


def monthly(ticker):
    s = load(ticker)
    if s is None or len(s) < 400:
        return None
    m = s.resample("1ME").last().dropna()
    r = m.pct_change().dropna()
    r.index = r.index.to_period("M")
    return r


def tsmom_leg(rm, side="longshort"):
    """vol-targeted, lookback-blended TSMOM monthly return stream (causal)."""
    sig = sum(np.sign(rm.rolling(L).sum().shift(1)) for L in LOOKBACKS) / len(LOOKBACKS)  # in [-1,1]
    if side == "long":
        sig = sig.clip(lower=0)
    tv = rm.rolling(12).std().shift(1)
    lev = (TGT_VOL_M / tv).clip(upper=3.0).fillna(0.0)
    ret = sig * lev * rm - COST_M * (sig * lev).diff().abs().fillna(0.0)
    return ret.dropna()


def basket(legs_dict):
    df = pd.concat(legs_dict, axis=1)
    return df.mean(axis=1).dropna()                          # equal-risk avg (each vol-targeted)


def stat(tag, ret):
    if len(ret) < 12:
        print(f"  {tag:<26} n={len(ret)}"); return None
    c, dd, r = cagr_dd_monthly(ret)
    isr = ret[ret.index.year < SPLIT]; oos = ret[ret.index.year >= SPLIT]
    ri = cagr_dd_monthly(isr)[2]; ro = cagr_dd_monthly(oos)[2]
    print(f"  {tag:<26} mo={len(ret):>3} win%={(ret>0).mean()*100:>3.0f} CAGR={c:+5.1f}% DD={dd:4.1f}% "
          f"CAGR/DD={r:5.2f} | IS={ri:4.2f} OOS={ro:4.2f}")
    return r


def main():
    legs = {}
    pos = {}
    print(f"tsmom_basket -- yfinance daily, lookbacks{LOOKBACKS}, vol-tgt {TGT_VOL_M*100:.0f}%/mo, L/S")
    print("  per-instrument vol-targeted TSMOM (long/short):")
    for lab, tk in UNIV.items():
        rm = monthly(tk)
        if rm is None:
            print(f"    {lab:<8} fetch failed"); continue
        leg = tsmom_leg(rm, "longshort")
        legs[lab] = leg
        c, dd, r = cagr_dd_monthly(leg)
        pos[lab] = leg.mean() > 0
        print(f"    {lab:<8}({CLASS[lab]:<5}) mo={len(leg):>3} CAGR/DD={r:>5.2f} meanR/mo={leg.mean()*100:+.2f}% {'+' if pos[lab] else '-'}")
    nclass = len(set(CLASS[k] for k in legs))
    print(f"\n  breadth: {len(legs)} instruments / {nclass} asset classes; "
          f"{sum(pos.values())}/{len(pos)} positive standalone")

    # ---- baskets: full cross-asset vs the weak gold+BTC-only ----
    print("\n  == BASKET standalone (vs the weak 2-asset version) ==")
    full = basket(legs)
    r_full = stat("full cross-asset", full)
    gb = basket({k: legs[k] for k in ("gold", "BTC") if k in legs})
    stat("gold+BTC only (weak)", gb)
    # by class
    for cls in ("EQ", "BOND", "FX", "COMM", "CRYPTO"):
        sub = {k: legs[k] for k in legs if CLASS[k] == cls}
        if sub: stat(f"  class={cls}", basket(sub))

    # ---- correlation with the book (decisive) ----
    Mb = monthly_matrix({k: get_legs()[k] for k in ("gold_bo", "btc_bo_kama")})
    Mb.index = Mb.index.to_period("M")
    book_m = Mb["gold_bo"] * 0.0079 + Mb["btc_bo_kama"] * 0.0121
    def corr(a, b):
        x = pd.concat([a, b], axis=1).dropna()
        return x.iloc[:, 0].corr(x.iloc[:, 1]) if len(x) >= 12 else np.nan
    print(f"\n  == CORRELATION (decisive) ==")
    print(f"  corr(full basket, gold_bo)={corr(full, Mb['gold_bo']):+.2f}  "
          f"(, btc_K)={corr(full, Mb['btc_bo_kama']):+.2f}  (, book)={corr(full, book_m):+.2f}")

    # ---- diversification: add basket to book at ~constant risk ----
    print("\n  == DIVERSIFICATION: add full basket to book (vol-matched, ~constant risk) ==")
    b = full.reindex(book_m.index).dropna()
    bk = book_m.reindex(b.index)
    sc = bk.std() / b.std() if b.std() > 0 else 1.0
    for w in (0.0, 0.25, 0.5):
        blend = (bk + w * sc * b) / (1 + w)
        c, dd, r = cagr_dd_monthly(blend)
        print(f"    book + {w:.2f}*basket: CAGR/DD={r:.2f} (CAGR{c:+.0f}/DD{dd:.0f}){'  <= book alone' if w==0 else ''}")
    print("\n  verdict: breadth lifts CAGR/DD vs 2-asset AND corr<~0.4 AND cuts book DD => real diversifier")
    print("  (then: Vantage-tradeable subset + full gauntlet). Yahoo daily pre-screen only; live-fwd arbitrates.")


if __name__ == "__main__":
    main()
