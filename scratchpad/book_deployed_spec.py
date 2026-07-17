"""Which number does the user actually get when they trade the Pine files?

`book_spec_fix.build()` -- the machine that produced the README's 8.48 -- silently diverges from the
deployed Pine in two places:
  1. the pullback-limit fill window: Python inherits BASE's fwd=500 bars, the Pine ships fillWin=200
  2. btc15m_S's target: Python inherits BASE's rr=4.0, but the adopted spec (and the Pine) is RR 4.5
Neither is a "bug" in the sense of a wrong result -- they are two different strategies, and only one
of them is the one the user will run. Measure all four corners so the README can quote the deployed one.
Run: .venv/bin/python scratchpad/book_deployed_spec.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from radar_gate_race import BASE
from short_mirror_15m import invert
from book_spec_fix import book, w_trade

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]


def build(fill_win, rr_short):
    """The 6 legs, with the two contested knobs exposed."""
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        gb = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                        "daily_sma": 150, "daily_slope_k": 10}))
        legs = {"gold_bo": pd.Series(gb["R"].values, index=pd.DatetimeIndex(gb["time"]))}

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        bk = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0,
                                                      "fwd": 300}))[["time", "R"]])
        legs["btc_bo_kama"] = pd.Series(bk.R.values, index=pd.DatetimeIndex(bk.time))
        pb = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}),
                                    0.0)[["time", "R"]])
        legs["btc_pull"] = pd.Series(pb.R.values, index=pd.DatetimeIndex(pb.time))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                        "pullback_frac": 0.25, "fill_win": fill_win}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        tL = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5, "fill_win": fill_win}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        RL = (tL["R"].values - 15.0 / tL["risk"].values) * np.where(
            tL["e_px"].values > pdh[d15.index.get_indexer(tL["time"])], 1.0, 0.5)
        legs["btc15m_L"] = pd.Series(RL, index=pd.DatetimeIndex(tL["time"]))

        inv = invert(d15); C = 2 * d15["high"].max()
        ts = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3,
                                         "rr": rr_short, "fill_win": fill_win}))
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        mS = (C - ts["e_px"].values) < pdl[d15.index.get_indexer(ts["time"])]
        legs["btc15m_S"] = pd.Series((ts["R"].values - 15.0 / ts["risk"].values)[mS],
                                     index=pd.DatetimeIndex(ts["time"])[mS])
    return legs


def main():
    print("Pine（実際に運用する物）と Python（READMEの数字を出した物）の差は2箇所:")
    print("  1. 押し目指値の有効期限   Pine fillWin=200 本  /  Python fwd=500 本")
    print("  2. ショート脚の目標        Pine rr=4.5        /  Python は BASE の 4.0 を継承\n")
    print(f"  {'fill_win':>9}{'S の RR':>9}   {'n(L)':>5}{'n(S)':>5}"
          f"{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>10}   備考")
    for fw in (500, 200):
        for rrs in (4.0, 4.5):
            legs = build(fw, rrs)
            cagr, dd, r, n = book(legs, SIX)
            note = ""
            if fw == 500 and rrs == 4.0:
                note = "← README の 8.48 を出した機械（どちらも仕様外）"
            if fw == 200 and rrs == 4.5:
                note = "← **実際に運用する仕様（Pine と一致）**"
            print(f"  {fw:>9}{rrs:>9.1f}   {len(legs['btc15m_L']):>5}{len(legs['btc15m_S']):>5}"
                  f"{cagr:>8.1f}%{dd:>7.2f}%{r:>10.2f}   {note}")

    legs = build(200, 4.5)                      # the deployed spec -- redraw every README table on it
    c0, d0, r0, n0 = book(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)      # the book only exists where every leg exists
    en = min(legs[k].index.max() for k in SIX)
    yrs = (en - st).days / 365.25
    print(f"\n\n運用仕様（fill_win=200 / S の RR=4.5）での脚別\n")
    print(f"  {'leg':<14}{'n':>5}{'本/年':>7}{'勝率':>7}{'PF':>7}{'meanR':>9}{'重み':>8}")
    w = w_trade(legs, SIX)
    for k in SIX:
        s = legs[k]
        pf = s[s > 0].sum() / abs(s[s <= 0].sum())
        y = (s.index[-1] - s.index[0]).days / 365.25
        print(f"  {k:<14}{len(s):>5}{len(s)/y:>7.0f}{100*(s>0).mean():>6.1f}%{pf:>7.2f}"
              f"{s.mean():>+9.3f}{100*w[k]:>7.3f}%")
    print(f"\n  ブック合計: n={n0}  {n0/yrs:.0f}本/年  CAGR {c0:+.1f}%  maxDD {d0:.2f}%  "
          f"CAGR/DD **{r0:.2f}**")

    print(f"\n  1つ抜いたら（leave-one-out）")
    print(f"  {'抜いた leg':<14}{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>10}{'差':>9}")
    print(f"  {'（6本全部）':<14}{c0:>8.1f}%{d0:>7.2f}%{r0:>10.2f}{'':>9}")
    for k in SIX:
        c, d, r, _ = book(legs, [x for x in SIX if x != k])
        print(f"  {k:<14}{c:>8.1f}%{d:>7.2f}%{r:>10.2f}{r-r0:>+9.2f}")

    print("\n  最長連敗（そのレッグだけを見たとき、何本続けて負けたか）")
    for k in SIX:
        v = legs[k].values
        best = cur = 0
        for x in v:
            cur = cur + 1 if x <= 0 else 0
            best = max(best, cur)
        print(f"    {k:<14}{best:>3} 連敗")

    parts = [pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                       index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
             for k in SIX]
    s = pd.concat(parts).sort_index()
    print("\n  年別（口座%・複利）")
    for y, g in s.groupby(s.index.year):
        print(f"    {y}: {100*(np.prod(1+g.values)-1):>+7.1f}%   ({len(g)}本)")

    # btc_bo_kama: is it a diversifier, or just leverage? de-lever the 5-leg book to the SAME DD.
    five = [x for x in SIX if x != "btc_bo_kama"]
    lo = 0.0
    for b in np.linspace(0.005, 0.03, 200):                # total risk budget for the 5-leg book
        wf = w_trade(legs, five, budget=b)
        p5 = pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * wf[k],
                                  index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                        for k in five]).sort_index()
        eq = np.cumprod(1 + p5.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        if dd <= d0:
            lo = (b, (eq[-1] ** (365.25 / (p5.index[-1] - p5.index[0]).days) - 1) * 100, dd)
    print(f"\n  btc_bo_kama は分散か、ただのレバレッジか（5レッグを同じ DD {d0:.2f}% まで減量して比較）")
    print(f"    5レッグ・総リスク {100*lo[0]:.2f}%  →  CAGR {lo[1]:+.1f}%  maxDD {lo[2]:.2f}%")
    print(f"    6レッグ・総リスク 3.00%  →  CAGR {c0:+.1f}%  maxDD {d0:.2f}%   "
          f"差 **{c0-lo[1]:+.1f} ポイント**")


if __name__ == "__main__":
    main()
