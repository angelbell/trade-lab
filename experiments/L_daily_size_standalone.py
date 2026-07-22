"""Standalone-only lever for btc15m_L (structural law 10b: solo != book). The daily-regime size cut
(law 9b: when the DAILY trend is DOWN, a 15m long is counter-drift, its up-legs are short, so size it
DOWN -- do NOT cut the winners, do NOT skip) was measured +9.3 CAGR pt STANDALONE on btc15m_A but
-2.1pt in the book, so the book rejected it. The user has now shelved the book and wants the best
STANDALONE leg. So the rejection reason is gone -- re-open it on the current frozen btc15m_L.

  base       btc15m_L canonical (RR4.5, pull0.3, PDH-soft0.5, 4h-KAMA gate), swap-in.
  lever      multiply the per-trade risk by m on entries taken while the DAILY state is DOWN.
             three daily definitions (KAMA14 / >SMA50 / >SMA150), all read from the CONFIRMED daily
             bar, shift(1), ffill -- no lookahead.
  arbiter    STANDALONE, de-lever each arm to the SAME bootstrapped-median maxDD, compare CAGR
             (f is the dial, so this is leverage-free; law 7.5 neutralised).
  falsify    reversed dummy (size UP on daily-down, m=1.25) must LOSE = mechanism check.
             block bootstrap P(beat base) at 1/3/6/12 mo must RISE with block. per-year spread.
Run: .venv/bin/python experiments/L_daily_size_standalone.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from arb_common import Boot, cd
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
BTC_PCT_YR = 30.0
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "rr": 4.5,
       "fill_win": 200, "fwd": 500}


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    dly = d15.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last"}).dropna()
    kd = kama_adaptive(dly["close"], 14)
    defs = {"日足KAMA14↑": (kd > kd.shift(1)).shift(1),
            "終値>日足SMA50": (dly["close"] > dly["close"].rolling(50).mean()).shift(1),
            "終値>日足SMA150": (dly["close"] > dly["close"].rolling(150).mean()).shift(1)}
    defs = {k: v.reindex(d15.index, method="ffill") for k, v in defs.items()}

    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d15, SimpleNamespace(**CFG))
    ii = d15.index.get_indexer(t["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
    risk = t["risk"].values / w
    R = (t["R"].values * w - 15.0 / risk
         - (BTC_PCT_YR / 365.0 / 100.0) * (t["e_px"].values / risk) * t["hold"].values)
    ti = pd.DatetimeIndex(t["time"])
    base = pd.Series(R, index=ti)

    def arm(defn, m):
        up = defs[defn].values[ii]                 # True=日足↑, False=日足↓
        mult = np.where(up, 1.0, m)
        return pd.Series(R * mult, index=ti)

    MS = [("×0.00 (建てない)", 0.00), ("×0.25", 0.25), ("×0.35", 0.35), ("×0.50", 0.50),
          ("×0.60", 0.60), ("×0.75", 0.75), ("×0.85", 0.85),
          ("×1.00 (現行)", 1.00), ("逆×1.25 (ダミー)", 1.25)]
    bt = Boot(sorted(set(ti.to_period("M"))), nb=1000, k=3)
    D0 = bt.dd_median(base * 0.01)
    cbase = cd((base * 0.01 * bt.equal_dd_cagr(base * 0.01, D0)[1]).values,
               max((ti[-1] - ti[0]).days, 1))[0]
    bp = {kk: Boot(bt.months, nb=800, k=kk) for kk in (1, 3, 6, 12)}
    rbase = {kk: bp[kk].ratios(base * 0.01) for kk in (1, 3, 6, 12)}

    dn = {d: int((~defs[d].values[ii].astype(bool)).sum()) for d in defs}
    print(f"基準 maxDD = {D0:.2f}%（btc15m_L 単独・1%риск・中央値）。全アームを同DDに揃えて CAGR を対比較。")
    print(f"日足↓で建てた本数: " + " / ".join(f"{d} {dn[d]}/{len(ti)}" for d in defs) + "\n")

    for defn in defs:
        print(f"  【日足の定義 = {defn}】")
        print(f"    {'日足↓のサイズ':<18}{'CAGR(同DD)':>11}{'差':>8}"
              f"{'  P(現行に勝つ):':<15}{'1か月':>7}{'3か月':>7}{'6か月':>7}{'12か月':>8}")
        for lab, m in MS:
            s = arm(defn, m) * 0.01
            c = cd((s * bt.equal_dd_cagr(s, D0)[1]).values, max((ti[-1] - ti[0]).days, 1))[0]
            ps = "".join(f"{100*np.mean(bp[kk].ratios(s) > rbase[kk]):>6.0f}%" for kk in (1, 3, 6, 12))
            mk = "  ← 現行" if m == 1.0 else ("  ★" if c > cbase + 0.3 else "")
            print(f"    {lab:<18}{c:>+10.1f}%{c-cbase:>+8.1f}pt{'':<15}{ps}{mk}")
        print()

    print("決定的 null: 『日足↓』を、同数だけランダムに選んだラベルに置換して ×0.50（40seed の中央値）")
    print("  もしランダムでも +数pt 出るなら、利得は日足信号ではなく『どの部分集合でも縮めれば f を買える』機械効果。\n")
    rng = np.random.default_rng(7)
    for defn in defs:
        ndn = int((~defs[defn].values[ii].astype(bool)).sum())
        gains = []
        for _ in range(40):
            lab = np.ones(len(ti)); lab[rng.choice(len(ti), ndn, replace=False)] = 0.50
            s = pd.Series(R * lab, index=ti) * 0.01
            gains.append(cd((s * bt.equal_dd_cagr(s, D0)[1]).values, max((ti[-1]-ti[0]).days, 1))[0] - cbase)
        g = np.array(gains)
        real = {"日足KAMA14↑": 5.5, "終値>日足SMA50": 7.6, "終値>日足SMA150": 15.1}[defn]
        print(f"  {defn:<16} ランダム×0.50 の利得 中央値 {np.median(g):+.1f}pt "
              f"[{np.quantile(g,0.05):+.1f}, {np.quantile(g,0.95):+.1f}]   （実際の日足↓×0.50 = {real:+.1f}pt）")

    print("\n年別R（日足↓を ×0.75 にした時 vs 現行。定義=日足KAMA14↑）")
    d0 = defs["日足KAMA14↑"].values[ii].astype(bool)
    s75 = R * np.where(d0, 1.0, 0.75)
    yr = pd.DataFrame({"cur": R, "cut": s75}, index=ti)
    print(f"  {'年':<7}{'現行':>9}{'↓×0.75':>10}{'差':>9}{'日足↓の本数':>13}")
    for y, g in yr.groupby(yr.index.year):
        mdn = int((~d0[ti.year == y]).sum())
        print(f"  {y:<7}{g['cur'].sum():>+8.0f}R{g['cut'].sum():>+9.0f}R"
              f"{g['cut'].sum()-g['cur'].sum():>+8.1f}R{mdn:>12}")


if __name__ == "__main__":
    main()
