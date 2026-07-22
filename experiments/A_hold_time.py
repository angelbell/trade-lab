"""Did we ever measure the TIME axis? The user asks -- and no, not for an OPEN trade.

Structural law 9 ("trend legs don't age") was measured on the age of the SWING LEG at entry, across
gold/BTC/FX x 4h/1d/weekly x 4 eras. It says nothing about a trade that is ALREADY OPEN and has gone
nowhere. The user's proposal is different: "if the price turns range-like at the 1H level, get out."
A breakout bets that price MOVES. If it does not move, the bet has failed -- that is a coherent story
and it has never been tested here.

Before any exit rule, the diagnostic that decides whether one is even POSSIBLE:
  D1  hold-time distribution, winners vs losers. If losers are held LONGER, time carries information.
  D2  conditional on a trade STILL BEING OPEN at hour h (neither stopped nor at target), what is the
      final R and the remaining excursion? If they decay with h, a time stop can work. If they are
      FLAT in h, no time-based exit can exist -- the same wall that killed the structure exits.
  D3  the same, conditioned on RANGE-LIKE behaviour rather than raw elapsed time: the 1H high-low
      span over the last K hours, measured in units of the trade's own risk. A "range" is a SMALL
      span. Does a small span predict a bad outcome?
No exit rule is proposed until these three say one is possible.
Run: .venv/bin/python experiments/A_hold_time.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "fill_win": 200,
       "rr": 4.5, "fwd": 500}


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**CFG))
    dly = d15["close"].resample("1D").last().dropna()
    upD = (dly > dly.rolling(150).mean()).shift(1).reindex(d15.index, method="ffill")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])
    ab = t["e_px"].values > pdh[ei]
    e = t["e_px"].values[ab]; risk = t["risk"].values[ab]
    W = np.where(upD.values[ei][ab] == True, 1.0, 0.75)
    R = (t["R"].values - 15.0 / t["risk"].values)[ab] * W
    idx = d15.index.get_indexer(pd.DatetimeIndex(t["time"])[ab])
    stop = e - risk; tgt = e + 4.5 * risk
    hi, lo, cl = d15["high"].values, d15["low"].values, d15["close"].values

    # 各トレードの経路を1回だけ走らせて、決着バーと、各時点の状態を記録する
    n = len(e)
    exit_bar = np.zeros(n, int); outcome = np.empty(n, object)
    for i in range(n):
        j0 = idx[i]
        for j in range(j0 + 1, min(j0 + 501, len(cl))):
            if lo[j] <= stop[i]:
                exit_bar[i], outcome[i] = j, "損切り"; break
            if hi[j] >= tgt[i]:
                exit_bar[i], outcome[i] = j, "利確"; break
        else:
            exit_bar[i], outcome[i] = min(j0 + 500, len(cl) - 1), "時間切れ"
    hours = (exit_bar - idx) * 0.25                       # 15分足 -> 時間

    print(f"btc15m_A（日足×0.75込み）  n={n}\n")
    print("D1  保有時間の分布 — 負けトレードは長く持たされているか？\n")
    print(f"  {'決着の仕方':<12}{'n':>5}{'割合':>7}{'保有(時間) 中央値':>18}{'25%点':>8}{'75%点':>8}"
          f"{'最大':>8}{'meanR':>9}")
    for oc in ("利確", "損切り", "時間切れ"):
        m = outcome == oc
        if m.sum() == 0:
            continue
        h = hours[m]
        print(f"  {oc:<12}{m.sum():>5}{100*m.mean():>6.0f}%{np.median(h):>16.1f}h"
              f"{np.percentile(h,25):>8.1f}{np.percentile(h,75):>8.1f}{h.max():>8.0f}{R[m].mean():>+9.3f}")
    print(f"\n  → 勝ちの保有 {np.median(hours[outcome=='利確']):.0f}h vs "
          f"負けの保有 {np.median(hours[outcome=='損切り']):.0f}h")
    print("     負けのほうが**長い**なら、時間に情報がある。**短い**なら、時間ストップは作れない。")

    print("\n\nD2  『h 時間たった時点でまだ生きている』トレードの、その後")
    print("    ★ ここが平ら（時間で劣化しない）なら、どんな時間ベースの退出も存在しない\n")
    print(f"  {'経過':>6}{'まだ生存':>9}{'その後の最終R(平均)':>20}{'その後の最終R(中央値)':>22}"
          f"{'ここから利確に届く率':>20}")
    for H in (2, 6, 12, 24, 48, 96, 168):
        b = int(H * 4)
        alive = (exit_bar - idx) > b
        if alive.sum() < 10:
            continue
        print(f"  {H:>4}h{alive.sum():>9}{R[alive].mean():>+19.3f}{np.median(R[alive]):>+21.3f}"
              f"{100*np.mean(outcome[alive]=='利確'):>19.0f}%")
    print(f"  {'（全部）':>6}{n:>9}{R.mean():>+19.3f}{np.median(R):>+21.3f}"
          f"{100*np.mean(outcome=='利確'):>19.0f}%")

    print("\n\nD3  『1時間足レベルでレンジになった』を測る（あなたの案）")
    print("    直近 K 時間の 1時間足の高安幅を、そのトレードの損切り幅で割る。小さい＝レンジ。\n")
    for K in (6, 12, 24):
        b = int(K * 4)
        print(f"  直近 {K} 時間の値幅 ÷ 損切り幅  （エントリーから {K}h 以上たって、まだ生きている本）")
        alive = (exit_bar - idx) > b
        if alive.sum() < 20:
            print("    n 少なすぎ\n"); continue
        span = np.array([(hi[idx[i]+b-b+1: idx[i]+b+1].max() - lo[idx[i]+1: idx[i]+b+1].min()) / risk[i]
                         for i in np.where(alive)[0]])
        Ra = R[alive]; oc = outcome[alive]
        q = pd.qcut(span, 4, labels=False, duplicates="drop")
        print(f"    {'帯':<6}{'値幅/損切り(中央値)':>20}{'n':>6}{'最終R(平均)':>13}{'利確に届く率':>14}")
        for i in sorted(set(q)):
            m = q == i
            lab = "Q1(最も狭い＝レンジ)" if i == 0 else ("Q4(最も広い)" if i == max(q) else f"Q{i+1}")
            print(f"    {lab:<6}{np.median(span[m]):>18.2f}{m.sum():>6}{Ra[m].mean():>+13.3f}"
                  f"{100*np.mean(oc[m]=='利確'):>13.0f}%")
        print()


if __name__ == "__main__":
    main()
