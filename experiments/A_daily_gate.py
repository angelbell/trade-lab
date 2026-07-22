"""The raw-excursion test settled the EXIT question: in a daily downtrend btc15m_A's up-legs are
shorter (MFE median 2.69R -> 1.19R) but the FAR target is still optimal (RR4.5 meanR +0.34; RR1-2
is zero or negative). Cutting winners early kills the fat tail that carries the whole edge.

What the same stratification actually found is a SIZING/GATE lever:
        daily above SMA150 : meanR +1.28      daily below : +0.34      (4x)
So test the lever the lab has actually proved -- WHEN, not the exit. And test both shapes, because
structural law 11 says a with-drift long should SIZE DOWN, not skip:
   G1 hard gate  : no trade when the daily is below its SMA150
   G2 soft size  : 0.5 (and 0.25) when the daily is below
   G3 the user's rule (control, predicted to FAIL): keep the trade, use a NEAR target when daily is down
Judged standalone (this is the leg the user wants to trade alone) AND in the book, per-year, with the
block bootstrap. Frequency is reported everywhere: a gate that halves N is not free.
Run: .venv/bin/python experiments/A_daily_gate.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "fill_win": 200}


def streak(v):
    b = c = 0
    for x in v:
        c = c + 1 if x <= 0 else 0
        b = max(b, c)
    return b


def stat(s, tag, risk=0.01):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    pf = s[s > 0].sum() / abs(s[s <= 0].sum())
    eq = np.cumprod(1 + risk * s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    h = len(s) // 2
    print(f"  {tag:<34}{len(s):>5}{len(s)/yrs:>7.0f}{100*(s>0).mean():>7.1f}%{pf:>7.2f}"
          f"{s.mean():>+9.3f}{s[:h].mean():>+8.3f}{s[h:].mean():>+8.3f}{s.sum()/yrs:>+8.1f}"
          f"{streak(s.values):>6}{cagr:>8.1f}%{dd:>7.1f}%{cagr/max(dd,1e-9):>9.2f}")
    return cagr / max(dd, 1e-9)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    dly = d15["close"].resample("1D").last().dropna()
    up150 = (dly > dly.rolling(150).mean()).shift(1).reindex(d15.index, method="ffill")
    k = kama_adaptive(dly, 14)
    upK = (k > k.shift(1)).shift(1).reindex(d15.index, method="ffill")

    def legA(rr=4.5):
        with contextlib.redirect_stderr(io.StringIO()):
            t = run(d15, SimpleNamespace(**{**CFG, "rr": rr, "fwd": 500}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        ei = d15.index.get_indexer(t["time"])
        ab = t["e_px"].values > pdh[ei]
        R = (t["R"].values - 15.0 / t["risk"].values)[ab]
        ti = pd.DatetimeIndex(t["time"])[ab]
        return pd.Series(R, index=ti), up150.values[ei][ab], upK.values[ei][ab]

    A, u150, uK = legA(4.5)
    print("btc15m_A を単独運用したときの成績（賭け率1%/トレード）。★ 頻度(本/年)を必ず見る\n")
    print(f"  {'':<34}{'n':>5}{'本/年':>7}{'勝率':>8}{'PF':>7}{'meanR':>9}{'IS':>8}{'OOS':>8}"
          f"{'totR/年':>8}{'連敗':>6}{'CAGR':>8}{'maxDD':>7}{'CAGR/DD':>9}")
    r0 = stat(A, "現行（日足を見ない）")
    print()
    print("  G1 ハード・ゲート（日足が下なら建てない）")
    for nm, m in (("日足 SMA150 の上のみ", u150 == True), ("日足 KAMA↑ のみ", uK == True),
                  ("SMA150 ∩ KAMA↑", (u150 == True) & (uK == True))):
        stat(A[m], "  " + nm)
    print()
    print("  G2 ソフトサイズ（日足が下なら小さく張る。構造法則11）")
    for f in (0.75, 0.5, 0.25):
        for nm, m in (("SMA150", u150 == True), ("KAMA", uK == True)):
            w = np.where(m, 1.0, f)
            stat(pd.Series(A.values * w, index=A.index), f"  {nm} 下は ×{f}")
    print()
    print("  G3 【対照・あなたの裁量ルール】日足が下なら近い利確に切り替える")
    for near in (1.5, 2.0, 3.0):
        An, u2, _ = legA(near)
        # 日足↑ は RR4.5、日足↓ は近い RR。時刻で結合（同一足に両方は立たない）
        up_part = A[u150 == True]
        dn_part = An[np.array([up150.reindex([x], method="ffill").iloc[0] == False
                               for x in An.index])] if False else An[
            ~pd.Series(up150.reindex(An.index, method="ffill").values, index=An.index).fillna(True).values]
        mix = pd.concat([up_part, dn_part]).sort_index()
        stat(mix, f"  日足↑=RR4.5 / 日足↓=RR{near}")

    print("\n\n年別（1%リスクの口座%）— 2026年の穴は埋まるか\n")
    arms = {"現行": A, "SMA150 ハード": A[u150 == True],
            "SMA150 下は ×0.5": pd.Series(A.values * np.where(u150 == True, 1.0, 0.5), index=A.index),
            "SMA150 下は ×0.25": pd.Series(A.values * np.where(u150 == True, 1.0, 0.25), index=A.index)}
    yrs = sorted(set(A.index.year))
    print("  " + " " * 20 + "".join(f"{y:>9}" for y in yrs))
    for tag, s in arms.items():
        row = f"  {tag:<20}"
        for y in yrs:
            g = s[s.index.year == y]
            row += f"{100*(np.prod(1+0.01*g.values)-1):>+8.1f}%" if len(g) else f"{'·':>9}"
        print(row)
    print("  " + " " * 20 + "".join(f"{y:>9}" for y in yrs))
    for tag, s in (("  (本数)現行", A), ("  (本数)SMA150ハード", A[u150 == True])):
        row = f"  {tag:<20}"
        for y in yrs:
            row += f"{(s.index.year == y).sum():>9}"
        print(row)


if __name__ == "__main__":
    main()
