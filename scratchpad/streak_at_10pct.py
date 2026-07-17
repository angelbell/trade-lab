"""At 10% risk per trade, how many losses in a row can the account take -- and how many will it get?

The user asks two things that must not be confused:
  A. how many consecutive losses does f=10% SURVIVE          -- pure arithmetic, (1-f)^n
  B. how many consecutive losses will the account ACTUALLY MEET -- a property of win% and trade count
The bet is only safe if B is comfortably smaller than A. Almost nobody checks B.

Also: the user floats "RR 6 with a ~50% win rate". Breakeven at RR6 is 1/(1+6) = 14.3%, so a 50%
win rate there is a PF of 6.0. Measure what the book's real entries ACTUALLY do at RR 6 before
sizing anything off a number nobody has produced.
Run: .venv/bin/python scratchpad/streak_at_10pct.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from near_target_repack import legs_at

RNG = np.random.default_rng(20260714)


def max_streak(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best


def kelly(win, rr):
    f = np.linspace(0.001, 0.90, 900)
    g = win * np.log1p(f * rr) + (1 - win) * np.log1p(-f)
    return f[int(np.argmax(g))]


def main():
    print("A. 10% リスクが「耐えられる」連敗回数（ただの複利。エッジとは無関係）\n")
    print(f"  {'連敗':>5}{'資金の残り (f=10%)':>20}{'毀損':>9}   {'f=5%':>10}{'f=3%':>10}{'f=1%':>10}")
    for n in (3, 5, 7, 10, 13, 15, 20, 25):
        row = f"  {n:>5}{100*0.9**n:>17.0f}%{100*(1-0.9**n):>8.0f}%   "
        row += f"{100*(1-0.95**n):>9.0f}%{100*(1-0.97**n):>10.0f}%{100*(1-0.99**n):>10.0f}%"
        print(row)
    print("\n  ※ f=10% は、10連敗で −65%、13連敗で −75%、20連敗で −88%。")
    print("     −65% からの復元には +186% が要る。−88% からは +733%。**負けは複利で効く。**\n")

    print("\nB. その勝率で「実際に食らう」連敗回数（ここを見ない人が破産する）")
    print("   同じ勝率のコインを N 回投げて、その中の最長連敗を 10000 回シミュレート\n")
    print(f"  {'勝率':>6}{'N=100本':>26}{'N=500本':>26}{'N=1000本':>26}")
    print(f"  {'':>6}{'中央値 / 95%点 / 最悪':>26}{'中央値 / 95%点 / 最悪':>26}{'中央値 / 95%点 / 最悪':>26}")
    for win in (0.50, 0.40, 0.30, 0.225):
        row = f"  {100*win:>5.0f}%"
        for N in (100, 500, 1000):
            s = np.array([max_streak(RNG.random(N) < win) for _ in range(10000)])
            row += f"{np.median(s):>12.0f} /{np.percentile(s,95):>6.0f} /{s.max():>6.0f}"
        print(row)
    print("\n  → **勝率50%でも、1000本投げれば連敗10回は「普通に起きる」**（中央値が10）。")
    print("     f=10% なら、その普通の出来事ひとつで **口座は −65%**。")

    print("\n\nC. では RR6・勝率50% は実在するのか（＝PF 6.0。損益分岐は 14.3%）")
    print("   ブックの本物の入口を RR6 で走らせて、実測の勝率を見る\n")
    print(f"  {'leg':<18}{'RR':>4}{'損益分岐':>9}{'実測勝率':>9}{'PF':>7}{'meanR':>9}"
          f"{'最長連敗(実測)':>15}{'Kelly f*':>10}{'ハーフ・ケリー':>14}")
    for rr in (2.0, 3.0, 4.5, 6.0):
        L = legs_at(rr, net=True)
        for k, s in L.items():
            if k not in ("gold_bo (1H)", "btc_bo_kama (4H)", "btc15m_L"):
                continue
            win = (s > 0).mean()
            pf = s[s > 0].sum() / abs(s[s <= 0].sum())
            W = s[s > 0].mean(); Ls = abs(s[s <= 0].mean())
            f = kelly(win, W / Ls)
            print(f"  {k:<18}{rr:>4.1f}{100/(1+W/Ls):>8.1f}%{100*win:>8.1f}%{pf:>7.2f}"
                  f"{s.mean():>+9.3f}{max_streak(s.values):>13}回{100*f:>9.1f}%{100*f/2:>13.1f}%")
        print()

    print("  → 勝率50%が出るRRは **存在しない**。RRを上げるほど勝率は下がる（それが定義）。")
    print("     RR6 の実測勝率は 20〜35%。そこで最長連敗は 15〜20回超になる。**f=10% なら口座は消える。**")


if __name__ == "__main__":
    main()
