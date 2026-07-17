"""How big a bet is too big?  (btc15m_L / L+S, the small-account book)

The user asks whether 5% of the account per trade is fine. "Too risky" is not an opinion -- for a
known R-distribution it is a calculation. Three things decide it:

  1. THE GROWTH-OPTIMAL f (Kelly).  f* maximizes E[log(1 + f*R)]. Above f*, MORE risk produces LESS
     wealth -- the volatility drag eats the edge. Past ~2*f* the expected log-growth turns negative
     and the account trends to zero no matter how good the edge is.
  2. THE OBSERVED LOSING STREAK.  btc15m_L lost 17 in a row (win rate 22.8%). At f, that streak
     alone costs 1 - (1-f)^17.  This already happened; it is not a tail scenario.
  3. THE BACKTEST IS OPTIMISTIC.  Live DD runs 1.5-2x the backtest (lab convention). Any f chosen
     on backtest R's must be divided by roughly that factor.

So: compute f*, then bootstrap the actual trade sequence (circular blocks, to keep losing streaks
intact) and report, per f: median 1yr/3yr wealth multiple, median and 95th-pct maxDD, P(halving),
P(-80%), and the median terminal wealth. The user's own yardstick is the WEALTH-MULTIPLE
DISTRIBUTION, so that is what gets reported -- not Sharpe.
Run: .venv/bin/python scratchpad/bet_size_kelly.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from small_account_sim import legs_with_risk

NDRAW = 4000


def kelly(R):
    """f* maximizing E[log(1+f*R)] -- the growth-optimal fraction, and the f where growth hits 0."""
    fs = np.linspace(0.001, 0.60, 600)
    g = np.array([np.mean(np.log1p(f * R)) if np.all(1 + f * R > 0) else -np.inf for f in fs])
    fstar = fs[int(np.argmax(g))]
    zero = fs[g <= 0]
    fzero = zero[0] if len(zero) and fs[int(np.argmax(g))] < zero[0] else np.nan
    return fstar, np.max(g), fzero


def longest_losing(R):
    best = cur = 0
    for r in R:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def sim(R, f, n):
    eq = np.cumprod(1 + f * R[:n])
    if np.any(eq <= 0):
        return 0.0, 100.0
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    return eq[-1], dd


def main():
    L = legs_with_risk()
    books = {
        "btc15m_L 単独": np.array(L["btc15m_L"][1]),
        "btc15m_L + btc15m_S": np.concatenate([L["btc15m_L"][1], L["btc15m_S"][1]]),
    }
    # keep the L+S pair in TIME order (streaks matter)
    idx = np.concatenate([L["btc15m_L"][0].values, L["btc15m_S"][0].values])
    R2 = np.concatenate([L["btc15m_L"][1], L["btc15m_S"][1]])
    books["btc15m_L + btc15m_S"] = R2[np.argsort(idx)]

    for name, R in books.items():
        n_yr = len(R) / 7.0
        fstar, g, fzero = kelly(R)
        streak = longest_losing(R)
        worst = R.min()
        print(f"\n{'='*92}\n{name}   n={len(R)}  年{n_yr:.0f}本  meanR={R.mean():+.3f}  "
              f"勝率={100*(R>0).mean():.1f}%  最長連敗={streak}回")
        print(f"  成長最適な賭け率（Kelly f*）= **{fstar*100:.1f}%**"
              + (f"   これを超えると成長率は落ち、{fzero*100:.0f}% で成長ゼロ＝口座は増えなくなる"
                 if np.isfinite(fzero) else ""))
        print(f"  ラボの慣行（実DDはbacktestの1.5〜2倍）で割ると、実用上の上限 ≈ "
              f"**{fstar/2*100:.1f}% 〜 {fstar/1.5*100:.1f}%**（ハーフ・ケリー）")
        print(f"  観測された最長連敗 {streak}回 を食らったときの資産の減り方:")
        for f in (0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
            print(f"      f={f*100:>4.0f}%  ->  {100*(1-(1-f)**streak):>5.1f}% の毀損"
                  + ("   ← あなたの案" if abs(f - 0.05) < 1e-9 else ""))

        rng = np.random.default_rng(20260713)
        m = len(R); blk = 20                      # 連敗を壊さないブロック長
        nb = int(np.ceil(m / blk))
        print(f"\n  巡回ブロック・ブートストラップ（{NDRAW}回、ブロック{blk}本＝連敗を保つ）")
        print(f"  ※ 資金倍率は backtest の R をそのまま複利したもの。**実際にはこうならない**")
        print(f"     （指値の約定を楽観・ストップの滑りは未計上・実DDは1.5〜2倍）。**見るべきは maxDD の列。**")
        print(f"  {'f':<6}{'1年の資金倍率(中央値)':>20}{'3年(中央値)':>14}"
              f"{'maxDD 中央値':>13}{'maxDD 95%点':>13}{'P(DD>50%)':>11}{'P(DD>80%)':>11}")
        for f in (0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
            w1, w3, dds = [], [], []
            for _ in range(NDRAW):
                st = rng.integers(0, m, nb)
                k = np.concatenate([(np.arange(s, s + blk) % m) for s in st])[:m]
                Rs = R[k]
                w1.append(sim(Rs, f, int(n_yr))[0])
                w3.append(sim(Rs, f, int(3 * n_yr))[0])
                dds.append(sim(Rs, f, m)[1])
            w1, w3, dds = np.array(w1), np.array(w3), np.array(dds)
            tag = "  ← あなたの案" if abs(f - 0.05) < 1e-9 else ""
            print(f"  {f*100:>3.0f}% {np.median(w1):>16.1f}倍{np.median(w3):>12.0f}倍"
                  f"{np.median(dds):>12.0f}%{np.percentile(dds,95):>12.0f}%"
                  f"{100*np.mean(dds>50):>10.0f}%{100*np.mean(dds>80):>10.0f}%{tag}")


if __name__ == "__main__":
    main()
