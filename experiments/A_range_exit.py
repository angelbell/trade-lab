"""The 1H range variable SEPARATES: of the trades still alive at hour 6, the narrowest-range quartile
reaches the target 24% of the time, the widest 79%. The user's read is right again -- a range means
the breakout has failed to break out.

But separation is not profit. The narrow-range group still ends at meanR +0.78. Exiting them only
pays if the price you exit at is HIGHER than +0.78R -- and a trade that has gone nowhere for six
hours is, by construction, sitting near zero. So the honest test is the counterfactual, not the AUC:

    for every trade that is still open at hour K and whose last-K-hours 1H span is below `thr` x risk,
    exit at the next 15m open.  R_saved vs R_cut.  Then the equal-maxDD CAGR against just holding.

And the two nulls that killed everything else today:
  - random-exit null: exit the same NUMBER of trades, chosen at random among those alive at hour K.
  - the size lever: hold instead, at a smaller bet. Same drawdown, same edge, less pain.
Run: .venv/bin/python experiments/A_range_exit.py
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
    cost = 15.0 / risk
    R0 = (t["R"].values - 15.0 / t["risk"].values)[ab] * W
    idx = d15.index.get_indexer(pd.DatetimeIndex(t["time"])[ab])
    ti = pd.DatetimeIndex(t["time"])[ab]
    stop = e - risk; tgt = e + 4.5 * risk
    hi, lo, cl, op = (d15["high"].values, d15["low"].values, d15["close"].values, d15["open"].values)
    yrs = (ti[-1] - ti[0]).days / 365.25
    n = len(e)

    def arm(K, thr):
        """K 時間たった時点で、直近 K 時間の値幅が thr×損切り幅 未満なら、次の足の寄付で退出。"""
        b = int(K * 4)
        R = R0.copy(); fired = np.zeros(n, bool)
        for i in range(n):
            j0, jK = idx[i], idx[i] + b
            if jK + 1 >= len(cl):
                continue
            # K時間の間に損切り/利確で決着していたら、この規則は関係ない
            done = False
            for j in range(j0 + 1, jK + 1):
                if lo[j] <= stop[i] or hi[j] >= tgt[i]:
                    done = True; break
            if done:
                continue
            span = (hi[j0 + 1:jK + 1].max() - lo[j0 + 1:jK + 1].min()) / risk[i]
            if span < thr:
                R[i] = ((op[jK + 1] - e[i]) / risk[i] - cost[i]) * W[i]
                fired[i] = True
        return R, fired

    def eq(R, D0):
        lo_, hi_ = 0.0005, 0.30
        for _ in range(70):
            m = (lo_ + hi_) / 2
            q = np.cumprod(1 + m * R); pk = np.maximum.accumulate(q)
            if ((pk - q) / pk).max() * 100 > D0:
                hi_ = m
            else:
                lo_ = m
        q = np.cumprod(1 + lo_ * R)
        return lo_, (q[-1] ** (1 / yrs) - 1) * 100

    q = np.cumprod(1 + 0.01 * R0); pk = np.maximum.accumulate(q)
    D0 = ((pk - q) / pk).max() * 100
    C0 = (q[-1] ** (1 / yrs) - 1) * 100
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    print(f"基準 = btc15m_A ＋日足×0.75。賭け率1%: CAGR {C0:+.1f}%  maxDD {D0:.2f}%  "
          f"CAGR/DD {C0/D0:.2f}  PF {pf(R0):.2f}")
    print(f"→ **全ての腕を maxDD {D0:.2f}% にそろえて CAGR で比べる**\n")
    print(f"  {'ルール':<34}{'発火':>5}{'PF':>7}{'meanR':>9}{'賭け率':>8}{'CAGR':>9}{'現行比':>10}")
    print(f"  {'そのまま持つ':<34}{'—':>5}{pf(R0):>7.2f}{R0.mean():>+9.3f}{1.00:>7.2f}%"
          f"{C0:>8.1f}%{0.0:>+9.1f}pt")
    best = {}
    for K in (6, 12, 24):
        for thr in (0.8, 1.0, 1.2, 1.5):
            R, f = arm(K, thr)
            if f.sum() < 5:
                continue
            bet, c = eq(R, D0)
            print(f"  {f'{K}h・値幅 < {thr}×損切り幅で退出':<34}{f.sum():>5}{pf(R):>7.2f}{R.mean():>+9.3f}"
                  f"{100*bet:>7.2f}%{c:>8.1f}%{c-C0:>+9.1f}pt" + ("  ★" if c > C0 + 1 else ""))
            best[(K, thr)] = (R, f)

    print("\n\n反実仮想の分解（発火した本だけ）: 退出して救ったR vs 切ってしまったR\n")
    for (K, thr), (R, f) in best.items():
        if f.sum() < 8:
            continue
        d = R[f] - R0[f]
        sv, ct = d[d > 0], d[d < 0]
        print(f"  {K}h・{thr}×  発火 {f.sum():>2}本  "
              f"救った {len(sv):>2}本 {sv.sum():>+6.1f}R   切った {len(ct):>2}本 {ct.sum():>+7.1f}R   "
              f"**差引 {d.sum():>+6.1f}R**   （降りた時点の平均R = {R[f].mean():+.2f} / "
              f"持ったら {R0[f].mean():+.2f}）")

    print("\n\nランダム退出null（同じ本数を、6h時点で生存している中からランダムに降ろす。1000回）")
    K, thr = 6, 1.0
    R, f = best[(K, thr)]
    b = int(K * 4)
    alive = np.zeros(n, bool)
    for i in range(n):
        j0, jK = idx[i], idx[i] + b
        if jK + 1 >= len(cl):
            continue
        if not any(lo[j] <= stop[i] or hi[j] >= tgt[i] for j in range(j0 + 1, jK + 1)):
            alive[i] = True
    obs = R0[f].mean()
    pool = np.where(alive)[0]
    nulls = [R0[RNG.choice(pool, f.sum(), replace=False)].mean() for _ in range(1000)]
    nulls = np.array(nulls)
    print(f"  レンジ判定で降ろした本の『持っていたら』meanR = **{obs:+.3f}**")
    print(f"  6h生存の中からランダムに同数: 中央値 {np.median(nulls):+.3f} "
          f"[5% {np.percentile(nulls,5):+.3f}, 95% {np.percentile(nulls,95):+.3f}]")
    print(f"  → **{100*np.mean(obs < nulls):.0f} パーセンタイル**（低いほど「本当に伸びない本を選べている」）")


if __name__ == "__main__":
    main()
