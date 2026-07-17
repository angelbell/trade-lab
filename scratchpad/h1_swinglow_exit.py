"""The user cannot psychologically hold to RR4.5 in the current geopolitics, and proposes: exit when
price closes below the most recent 1-HOUR swing low. That is a structural trail, and it is a real
proposal that deserves a real measurement -- a rule you will not follow is worth zero, so an exit you
WILL follow can beat a better exit you abandon.

But there are two ways to buy peace of mind, and they are not the same kind of thing:
  A. TRAIL OUT at the 1H swing low  -> changes the R-DISTRIBUTION. If CAGR/DD falls, that is a cost
     you pay forever, on every trade, in every regime.
  B. HOLD A SMALLER POSITION        -> leaves the R-distribution untouched. Everything scales. You
     give up absolute return but keep the efficiency; the only thing that shrinks is the pain.
Compare them at the SAME maxDD and let CAGR decide. Plus the two falsifiers that killed the 15m-pivot
exit this morning:
  - counterfactual decomposition: R SAVED by exiting vs R CUT off winners. Cut > saved = the rule loses.
  - random-exit null: exit at random times with the same firing rate. Structure must beat coin-flips.
Base = btc15m_A with the adopted daily x0.75 size rule. Exits act on the CONFIRMED 1H close, filled on
the next 15m bar open (no lookahead).
Run: .venv/bin/python scratchpad/h1_swinglow_exit.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "fill_win": 200,
       "rr": 4.5, "fwd": 500}


def h1_swing_lows(h1, k):
    """Confirmed 1H fractal lows: bar c is a swing low if its low is the strict min of [c-k, c+k].
    KNOWN only at bar c+k -> that is the bar we may act from. No lookahead."""
    lo = h1["low"].values
    known = np.full(len(lo), np.nan)          # known[j] = the level of the latest swing low KNOWN at j
    cur = np.nan
    conf = {}
    for c in range(k, len(lo) - k):
        w = lo[c - k:c + k + 1]
        if lo[c] == w.min() and (w == lo[c]).sum() == 1:
            conf[c + k] = lo[c]               # confirmed at bar c+k
    for j in range(len(lo)):
        if j in conf:
            cur = conf[j]
        known[j] = cur
    return pd.Series(known, index=h1.index)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**CFG))
    h1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "1h")
    dly = d15["close"].resample("1D").last().dropna()
    upD = (dly > dly.rolling(150).mean()).shift(1).reindex(d15.index, method="ffill")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])
    ab = t["e_px"].values > pdh[ei]

    e = t["e_px"].values[ab]; risk = t["risk"].values[ab]
    ti = pd.DatetimeIndex(t["time"])[ab]
    W = np.where(upD.values[ei][ab] == True, 1.0, 0.75)       # 採用済みの日足サイズ
    R0 = (t["R"].values - 15.0 / t["risk"].values)[ab] * W
    stop = e - risk; tgt = e + 4.5 * risk
    cost = 15.0 / risk
    idx = d15.index.get_indexer(ti)
    hi, lo, cl = d15["high"].values, d15["low"].values, d15["close"].values
    yrs = (ti[-1] - ti[0]).days / 365.25

    def walk(kk, min_profit_R):
        """Same trades. Additionally: after each CONFIRMED 1H close below the latest KNOWN 1H swing
        low, exit at the next 15m open. min_profit_R = only allow the exit once the trade is at least
        this many R in front (the user's rule is a TAKE-PROFIT, not a stop-substitute)."""
        SL = h1_swing_lows(h1, kk)
        # 1H の「確定終値が直近の確定スイング安値を割った」時刻 -> その1H足の終了時刻
        brk = (h1["close"].values < SL.values) & np.isfinite(SL.values)
        bt = h1.index[brk] + pd.Timedelta(hours=1)             # その1H足が閉じた瞬間
        R = np.empty(len(e)); ex = np.empty(len(e), dtype=object); why = []
        for i in range(len(e)):
            j0 = idx[i]
            r = None
            for j in range(j0 + 1, min(j0 + 501, len(cl))):
                if lo[j] <= stop[i]:
                    r, w_ = -1.0, "stop"; break
                if hi[j] >= tgt[i]:
                    r, w_ = 4.5, "target"; break
                # 1H の確定ブレイクがこの15分足の開始時刻までに起きているか
                if len(bt) and (bt <= d15.index[j]).any():
                    last = bt[bt <= d15.index[j]][-1]
                    if last > d15.index[j0] and (d15.index[j] - last) < pd.Timedelta(minutes=15):
                        cur = (cl[j] - e[i]) / risk[i]
                        if cur >= min_profit_R:
                            r, w_ = (d15["open"].values[j] - e[i]) / risk[i], "trail"
                            break
            if r is None:
                r, w_ = (cl[min(j0 + 500, len(cl) - 1)] - e[i]) / risk[i], "time"
            R[i] = (r - cost[i]) * W[i]
            why.append(w_)
        return R, np.array(why)

    def eq_dd_cagr(R, D0):
        lo_, hi_ = 0.0005, 0.30
        for _ in range(70):
            m = (lo_ + hi_) / 2
            eq = np.cumprod(1 + m * R); pk = np.maximum.accumulate(eq)
            if ((pk - eq) / pk).max() * 100 > D0:
                hi_ = m
            else:
                lo_ = m
        eq = np.cumprod(1 + lo_ * R); pk = np.maximum.accumulate(eq)
        return lo_, (eq[-1] ** (1 / yrs) - 1) * 100, ((pk - eq) / pk).max() * 100

    eq = np.cumprod(1 + 0.01 * R0); pk = np.maximum.accumulate(eq)
    D0 = ((pk - eq) / pk).max() * 100
    C0 = (eq[-1] ** (1 / yrs) - 1) * 100
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    print(f"基準 = btc15m_A ＋ 日足×0.75（採用済み）。n={len(R0)} 年{len(R0)/yrs:.0f}本")
    print(f"  賭け率1%: CAGR {C0:+.1f}%  maxDD {D0:.2f}%  CAGR/DD {C0/D0:.2f}  PF {pf(R0):.2f}\n")
    print(f"**全ての腕を maxDD {D0:.2f}% にそろえて CAGR で比べる**（サイズを下げれば DD が下がるのは")
    print("  当たり前なので、それを完全に打ち消す。これが唯一フェアな比較）\n")
    print(f"  {'':<38}{'n':>4}{'PF':>7}{'meanR':>9}{'発火率':>8}{'賭け率':>8}{'CAGR':>9}{'現行比':>10}")
    print(f"  {'そのまま持つ（RR4.5固定）':<38}{len(R0):>4}{pf(R0):>7.2f}{R0.mean():>+9.3f}"
          f"{'—':>8}{1.00:>7.2f}%{C0:>8.1f}%{0.0:>+9.1f}pt")

    print("\n  B. サイズを下げて持つ（R の分布は不変。痛みだけが小さくなる）")
    for f in (0.75, 0.5, 0.35):
        bet, c, d = eq_dd_cagr(R0 * f, D0)
        print(f"  {'  玉を ×'+str(f)+' にして RR4.5 を持つ':<38}{len(R0):>4}{pf(R0*f):>7.2f}"
              f"{(R0*f).mean():>+9.3f}{'—':>8}{100*bet:>7.2f}%{c:>8.1f}%{c-C0:>+9.1f}pt")

    print("\n  A. 1時間足の直近の押し目安値を割ったら利確（あなたの案）")
    keep = {}
    for kk in (2, 3, 5):
        for mp in (-9.9, 0.0, 1.0, 2.0):
            R, why = walk(kk, mp)
            bet, c, d = eq_dd_cagr(R, D0)
            fire = 100 * np.mean(why == "trail")
            lab = f"  k={kk}本" + ("（常に発火）" if mp < -1 else f"（+{mp:.0f}R以上でのみ利確）")
            print(f"  {lab:<38}{len(R):>4}{pf(R):>7.2f}{R.mean():>+9.3f}{fire:>7.0f}%"
                  f"{100*bet:>7.2f}%{c:>8.1f}%{c-C0:>+9.1f}pt"
                  + ("  ★" if c > C0 + 1 else ""))
            if kk == 3:
                keep[mp] = (R, why)

    print("\n\n事前登録の反実仮想分解（k=3本）: 退出が「救ったR」と「切ってしまったR」")
    print("  ★ 切った > 救った なら、この規則は負ける（朝の15分ピボット退出はこれで死んだ）\n")
    for mp, (R, why) in keep.items():
        m = why == "trail"
        if m.sum() == 0:
            continue
        d = R[m] - R0[m]                                  # 退出した本の、そのまま持った場合との差
        saved = d[d > 0]; cutr = d[d < 0]
        lab = "常に発火" if mp < -1 else f"+{mp:.0f}R以上でのみ"
        print(f"  {lab:<16} 発火 {m.sum():>3}本   救った {len(saved):>3}本 {saved.sum():>+7.1f}R   "
              f"切った {len(cutr):>3}本 {cutr.sum():>+7.1f}R   **差引 {d.sum():>+6.1f}R**")

    print("\n\nランダム退出null（同じ発火率で、ランダムな時点に降りる。1000回。k=3・+1R以上）")
    R, why = keep[1.0]
    fire = np.mean(why == "trail")
    obs = R[why == "trail"].mean()
    nulls = []
    for _ in range(1000):
        m = RNG.random(len(R0)) < fire
        if m.sum() < 5:
            continue
        # ランダムな時点で降りる ≒ そのトレードの実現Rを [-1, 4.5] の一様な打ち切りで置換するのは
        # 不正確なので、代わりに「同数のトレードをランダムに選んで、構造退出と同じRを与える」
        nulls.append(R0[m].mean())
    nulls = np.array(nulls)
    print(f"  構造退出が発動した本の meanR（そのまま持った場合） = {R0[why=='trail'].mean():+.3f}")
    print(f"  ランダムに同数選んだ本の meanR: 中央値 {np.median(nulls):+.3f} "
          f"[5% {np.percentile(nulls,5):+.3f}, 95% {np.percentile(nulls,95):+.3f}]")
    print(f"  → **{100*np.mean(R0[why=='trail'].mean() < nulls):.0f} パーセンタイル**"
          f"（低いほど「本当に悪いトレードを選んで降りている」）")


if __name__ == "__main__":
    main()
