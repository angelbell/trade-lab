"""orb_asym_exit.py -- the user's structural insight as an H17 upgrade: 'declines are SHARP, rallies are
SLOW' (the up-continue / down-bounce asymmetry that survived regime control). H17 is symmetric (both sides
no-TP, ride to 20:00). If the asymmetry is real, the exits should be ASYMMETRIC:
  LONG  (upside break)  = SLOW grind -> ride to close, no TP  (unchanged, the continuation H17 already rides)
  SHORT (downside break)= SHARP drop that snaps back -> TAKE PROFIT at RR*range before the bounce eats it.

Test honestly vs H17 base (short also no-TP). Same entries/stops -- ONLY the short exit changes -> clean
apples-to-apples. Falsifier (up front): the asym version PASSES only if it beats symmetric H17 on net AND
PF AND CAGR/DD, on IS AND VAL, with a PLATEAU over the short TP RR, and the gain is NOT one-year/one-regime
(per-year spread). Shorts only fire in 1H-downtrends, so a gain that is purely 2025-26 = regime luck, not a
robust exit edge (the user's own beta caveat). cost 2.8p RT (H17). In-sample/val (sealed TEST spent).
  .venv/bin/python research/orb_asym_exit.py
"""
import os, sys, warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
import research.scalp_lab as sl

CFG = dict(asia_start_h=0, asia_end_h=7, bo_start_h=7, bo_end_h=11, force_exit_h=20,
           buf_atr=0.0, sl_buf_atr=0.0, max_range_atr=0.0, min_range_atr=0.0, sl_frac=1.0,
           rsi_max=100.0, box_trend_max=1.0, fade=False, htf_tf="1h", htf_ema=80, htf_slope_k=0,
           cost=1.4, stop_slip=0.0)


def leg(d, side, no_tp, rr):
    p = SimpleNamespace(dir=side, no_tp=no_tp, rr=rr, **CFG)
    dir_, slx, tpx = sl.orb_signals(d, p)
    dir_, slx, tpx = sl.htf_trend_gate(d, dir_, slx, tpx, p)
    return sl.backtest(d, dir_, slx, tpx, p)


def load(split):
    s, e = sl.SPLITS[split]
    d = load_mt5_csv("data/vantage_xauusd_m5.csv").loc[s:e]
    return d.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def stats(t):
    one_r = abs(t.loc[t.pips < 0, "pips"].mean()) if (t.pips < 0).any() else 1.0
    R = t.pips / one_r
    tm = t.t_in.dt.tz_localize(None)
    eq = (1 + 0.01 * R.values).cumprod()
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    span = max((tm.iloc[-1] - tm.iloc[0]).days / 365.25, 0.5)
    cagr = (eq[-1] ** (1 / span) - 1) * 100
    pf = t[t.pips > 0].pips.sum() / abs(t[t.pips < 0].pips.sum())
    return dict(n=len(t), net=t.pips.sum(), pf=pf, win=(t.pips > 0).mean() * 100,
                cdd=cagr / max(dd, 1e-9), dd=dd)


def show(tag, t):
    m = stats(t)
    print(f"  {tag:<26} n={m['n']:>4} net={m['net']:>+7.0f}p win={m['win']:>3.0f}% PF={m['pf']:.2f} "
          f"maxDD={m['dd']:4.1f}% CAGR/DD={m['cdd']:5.2f}")


def byyear(tag, t):
    ty = t.copy(); ty["y"] = ty.t_in.dt.tz_localize(None).dt.year
    print(f"    {tag}: " + "  ".join(f"{int(y)}:{g.pips.sum():+.0f}" for y, g in ty.groupby("y")))


def main():
    for split in ("is", "val"):
        d = load(split)
        L = leg(d, "long", True, 1.0)            # long: ride to close (no TP) -- unchanged
        S_ride = leg(d, "short", True, 1.0)      # short: ride to close (H17 symmetric base)
        print(f"\n######## {split.upper()} {d.index[0].date()}..{d.index[-1].date()} ########")
        print(" -- legs (H17 symmetric = long-ride + short-ride) --")
        show("long ride-to-close", L)
        show("short ride-to-close (base)", S_ride)
        base = pd.concat([L, S_ride]).sort_values("t_in")
        show("H17 base (sym, both ride)", base)
        print(" -- short with TP (capture the SHARP drop); long unchanged -- ")
        for rr in (1.0, 1.5, 2.0, 2.5, 3.0):
            S_tp = leg(d, "short", False, rr)
            show(f"short TP rr{rr}", S_tp)
        print(" -- COMBINED asym book (long-ride + short-TP) vs H17 base --")
        for rr in (1.0, 1.5, 2.0, 2.5, 3.0):
            S_tp = leg(d, "short", False, rr)
            asym = pd.concat([L, S_tp]).sort_values("t_in")
            show(f"asym (short TP rr{rr})", asym)
        # per-year for the mid RR to check regime concentration
        S2 = leg(d, "short", False, 2.0)
        print(" -- per-year (beta check: is the short-TP gain one-regime?) --")
        byyear("H17 base       ", base)
        byyear("asym short-TP2 ", pd.concat([L, S2]).sort_values("t_in"))
    print("\n  verdict: asym PASSES only if it beats H17 base on net+PF+CAGR/DD, on IS AND VAL, plateau over")
    print("  the short RR, and the gain spreads across years (not just a down-regime). Else = regime luck.")


if __name__ == "__main__":
    main()
