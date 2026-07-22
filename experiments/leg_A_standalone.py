"""A = btc15m_L's trades whose breakout closed ABOVE the previous day's high.
Standalone it is the best single strategy in the lab (CAGR/DD 4.72). Before anyone risks money on
it, the things that decide whether it is tradeable rather than merely backtestable:

  1. per-year -- including 2022 (BTC -64%). A leg that only works in bull years is beta.
  2. the bet size -- Kelly f*, and the drawdown the OBSERVED losing streak alone would cause.
  3. what a real drawdown looks like: the backtest DD x1.5-2 is the lab's live convention.
  4. block bootstrap of the wealth multiple, so the answer is a DISTRIBUTION, not one path.
  5. the honest caveat: this is a SUBSET of a leg already in the book, found by slicing it today.
     The PDH rule itself is not new (the book already sizes on it), but nobody had measured this
     half on its own -- so the multiple-comparison discount applies.
Run: .venv/bin/python experiments/leg_A_standalone.py
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
RNG = np.random.default_rng(20260714)


def streak(v):
    b = c = 0
    for x in v:
        c = c + 1 if x <= 0 else 0
        b = max(b, c)
    return b


def kelly(R):
    f = np.linspace(0.001, 0.60, 600)
    g = np.array([np.mean(np.log1p(x * R)) if np.all(1 + x * R > 0) else -np.inf for x in f])
    return f[int(np.argmax(g))]


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ab = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    R = pd.Series((t["R"].values - 15.0 / t["risk"].values)[ab], index=pd.DatetimeIndex(t["time"])[ab])
    btc = d15["close"].resample("YE").last()
    print(f"A: BTC 15分・前日高値の【上】でのブレイク   n={len(R)}\n")

    print("1. 年別（2022年は BTC −64%。ここで死ぬなら、それはベータ）\n")
    print(f"  {'年':<7}{'n':>5}{'勝率':>8}{'PF':>7}{'meanR':>9}{'totR':>8}"
          f"{'口座%(1%risk)':>14}{'BTCの年間騰落':>14}")
    for y in sorted(set(R.index.year)):
        g = R[R.index.year == y]
        if len(g) == 0:
            continue
        pf = g[g > 0].sum() / abs(g[g <= 0].sum()) if (g <= 0).any() else np.inf
        acct = 100 * (np.prod(1 + 0.01 * g.values) - 1)
        try:
            b0 = d15["close"][d15.index.year == y].iloc[0]
            b1 = d15["close"][d15.index.year == y].iloc[-1]
            bch = f"{100*(b1/b0-1):+.0f}%"
        except Exception:
            bch = "—"
        print(f"  {y:<7}{len(g):>5}{100*(g>0).mean():>7.0f}%{pf:>7.2f}{g.mean():>+9.3f}"
              f"{g.sum():>+8.1f}{acct:>+13.1f}%{bch:>14}")

    print("\n\n2. 賭け率（実測の最長連敗は 8回）\n")
    f = kelly(R.values)
    print(f"  成長最適な賭け率 Kelly f* = **{100*f:.1f}%**")
    print(f"  ラボの慣行（実DDはbacktestの1.5〜2倍）で割った実用上限 ≈ "
          f"**{100*f/2:.1f}% 〜 {100*f/1.5:.1f}%**")
    print(f"\n  {'賭け率':>7}{'8連敗での毀損':>14}{'maxDD(実測経路)':>17}{'CAGR':>9}")
    for x in (0.01, 0.02, 0.03, 0.05):
        eq = np.cumprod(1 + x * R.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        yrs = (R.index[-1] - R.index[0]).days / 365.25
        cagr = (eq[-1] ** (1 / yrs) - 1) * 100
        print(f"  {100*x:>6.0f}%{100*(1-(1-x)**8):>13.0f}%{dd:>16.1f}%{cagr:>8.1f}%"
              + ("   ← 想定実DD " + f"{dd*1.5:.0f}〜{dd*2:.0f}%" if x in (0.01, 0.02) else ""))

    print("\n\n3. 巡回ブロック・ブートストラップ（3000回・ブロック10本＝連敗を保つ）")
    print("   1つの経路ではなく、分布で見る\n")
    m, blk = len(R), 10
    nb = int(np.ceil(m / blk))
    yrs = (R.index[-1] - R.index[0]).days / 365.25
    print(f"  {'賭け率':>7}{'1年の資金倍率':>16}{'(中央値)':>10}{'maxDD 中央値':>14}"
          f"{'maxDD 95%点':>13}{'P(DD>30%)':>11}")
    for x in (0.01, 0.02, 0.03):
        w1, dds = [], []
        for _ in range(3000):
            stt = RNG.integers(0, m, nb)
            k = np.concatenate([(np.arange(s, s + blk) % m) for s in stt])[:m]
            Rs = R.values[k]
            e1 = np.cumprod(1 + x * Rs[:int(len(R) / yrs)])
            w1.append(e1[-1])
            eq = np.cumprod(1 + x * Rs); pk = np.maximum.accumulate(eq)
            dds.append(((pk - eq) / pk).max() * 100)
        w1, dds = np.array(w1), np.array(dds)
        print(f"  {100*x:>6.0f}%{np.median(w1):>14.2f}倍"
              f"{f'[{np.percentile(w1,5):.2f}, {np.percentile(w1,95):.2f}]':>12}"
              f"{np.median(dds):>13.0f}%{np.percentile(dds,95):>12.0f}%"
              f"{100*np.mean(dds>30):>10.0f}%")


if __name__ == "__main__":
    main()
