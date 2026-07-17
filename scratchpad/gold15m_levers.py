"""gold15m was never optimised -- it inherited its knobs.

btc15m_L got a full sweep of gate-TF, RR and the PDH size rule under the trade-resolution arbiter
and gained most of the book's CAGR/DD. gold15m sits at 46 trades/yr on knobs that were assumed
("it's gold, so use the daily SMA150 like gold_bo") rather than derived. Sweep them.

One lever at a time; the BOOK decides (structural law 10 -- a leg gain that deletes the trades
carrying the book's decorrelation is a loss). Baseline = the deployed 6-leg spec, CAGR/DD 8.28.
Run: .venv/bin/python scratchpad/gold15m_levers.py
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
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import book

ROOT = "/home/angelbell/dev/auto-trade"
CUR = dict(daily_sma=150, daily_slope_k=10, ext_cap=8.0, pullback_frac=0.25, rr=4.0, fill_win=200)


def gold15m(**over):
    cfg = {**BASE, **CUR, **over}
    with contextlib.redirect_stderr(io.StringIO()):
        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**cfg))
    R = t["R"].values - 0.30 / t["risk"].values
    return pd.Series(R, index=pd.DatetimeIndex(t["time"])), t["R"].values


def show(tag, s, gross, r0, legs0):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    pf = s[s > 0].sum() / abs(s[s <= 0].sum())
    pfg = gross[gross > 0].sum() / abs(gross[gross <= 0].sum())
    h = len(s) // 2
    legs = dict(legs0); legs["gold15m"] = s
    c, d, rb, n = book(legs, SIX)
    print(f"  {tag:<28}{len(s):>5}{len(s)/yrs:>7.0f}{100*(s>0).mean():>6.1f}%{pfg:>7.2f}{pf:>7.2f}"
          f"{s.mean():>+9.3f}{s[:h].mean():>+8.3f}{s[h:].mean():>+8.3f}{s.sum()/yrs:>+8.1f}"
          f"{rb:>9.2f}{rb-r0:>+8.2f}{'  ★' if rb > r0 + 0.05 else ''}")


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    r0 = book(legs0, SIX)[2]
    print(f"審判 = 6レッグ・ブックの CAGR/DD（トレード解像度・トレードRσのinv-vol・総リスク3%）。現行 = {r0:.2f}")
    print("1実験1レバー。★ = ブックを +0.05 超えて改善\n")
    hdr = (f"  {'':<28}{'n':>5}{'本/年':>7}{'勝率':>7}{'PF素':>7}{'PF実':>7}"
           f"{'meanR':>9}{'IS':>8}{'OOS':>8}{'totR/年':>8}{'ブック':>9}{'差':>8}")

    print("レバー1: 利確 RR（btc15m_L は 4.0→4.5 で効いた。gold15m は 4.0 のまま）")
    print(hdr)
    for rr in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
        s, g = gold15m(rr=rr)
        show(f"RR {rr}" + ("  ← 現行" if rr == 4.0 else ""), s, g, r0, legs0)

    print("\nレバー2: レジーム・ゲート（btc15m_L は 日足→4時間 で効いた。gold15m は日足SMA150のまま）")
    print(hdr)
    s, g = gold15m()
    show("日足SMA150↑  ← 現行", s, g, r0, legs0)
    for tf in ("240min", "1D"):
        s, g = gold15m(daily_sma=0, daily_slope_k=0, gate_kama=14, gate_kama_tf=tf)
        show(f"KAMA(14)↑ · {tf}", s, g, r0, legs0)
    for tf in ("240min", "1D"):
        s, g = gold15m(gate_kama=14, gate_kama_tf=tf)
        show(f"日足SMA150 ∩ KAMA · {tf}", s, g, r0, legs0)

    print("\nレバー3: 押し目指値の深さ（btc15m_L は 0.30。gold15m は 0.25）")
    print(hdr)
    for fr in (0.0, 0.20, 0.25, 0.30, 0.35, 0.40):
        s, g = gold15m(pullback_frac=fr)
        show(f"押し目 {fr}" + ("  ← 現行" if fr == 0.25 else ""), s, g, r0, legs0)

    print("\nレバー4: ext-cap（ブレイク地点が高値から何%以内か。現行 8%）")
    print(hdr)
    for ec in (0.0, 4.0, 6.0, 8.0, 12.0, 20.0):
        s, g = gold15m(ext_cap=ec)
        show(f"ext-cap {ec}%" + ("  ← 現行" if ec == 8.0 else ""), s, g, r0, legs0)


if __name__ == "__main__":
    main()
