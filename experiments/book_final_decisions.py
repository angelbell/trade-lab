"""Two decisions, both measured -- not argued.

D1  gold15m's session skip.  The ledger is INCONSISTENT: the general gold rule is "skip 12-15 UTC
    (US-data window)" but the gold15m candidate line says "9-15 UTC skip 強化" (strengthened). Our
    agent implemented 9-15. Picking whichever version the BOOK likes best is post-hoc selection --
    the exact sin we spent today undoing. So instead: measure ALL THREE (none / 12-15 / 9-15) on the
    LEG's own evidence first (does the skipped window actually have zero-or-negative edge? IS vs OOS?)
    and only then look at the book. The leg-level evidence decides; the book is the tiebreak.

D2  Do the two low-frequency BTC 4H legs still earn their seats once btc15m_L/S are in?
    btc_bo_kama = 8 trades/yr and its RR2 sits on a knife edge (P(true win% < 50%) ~= 24%).
    btc_pull    = 10 trades/yr.
    Leave-one-out on the 5-leg book (gold data fixed), plus the block bootstrap that decides.
Run: .venv/bin/python experiments/book_final_decisions.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from book_spec_fix import build, book, w_trade
from book_leave_one_out import cdd

ROOT = "/home/angelbell/dev/auto-trade"
CORE = ["gold_bo", "btc_bo_kama", "btc_pull"]
FIVE = CORE + ["btc15m_L", "btc15m_S"]
NDRAW = 2000
SKIPS = {"skip なし": None, "12-15 UTC (台帳の一般則)": range(12, 15), "9-15 UTC (候補の記述)": range(9, 15)}


def gold15m_raw():
    with contextlib.redirect_stderr(io.StringIO()):
        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
    return pd.Series(t["R"].values - 0.3 / t["risk"].values, index=pd.DatetimeIndex(t["time"]))


def leg_line(s, tag):
    pf = s[s > 0].sum() / abs(s[s <= 0].sum())
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    half = s.index[len(s) // 2]
    isr, oos = s[s.index < half], s[s.index >= half]
    eq = np.cumprod(1 + 0.01 * s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    print(f"  {tag:<26}n={len(s):>4}  {len(s)/yrs:>5.0f}本/年  win={100*(s>0).mean():>4.1f}%  "
          f"PF={pf:>4.2f}  meanR={s.mean():+.3f}  IS/OOS={isr.mean():+.3f}/{oos.mean():+.3f}  "
          f"CAGR/DD={cagr/dd:>5.2f}")


def boot(legs, arms, base_key):
    """paired circular block bootstrap over months (trade order kept inside each month)."""
    S = {k: None for k in arms}
    for k, basket in arms.items():
        w = w_trade(legs, basket)
        st = max(legs[j].index.min() for j in basket)
        en = min(legs[j].index.max() for j in basket)
        parts = [pd.Series(legs[j][(legs[j].index >= st) & (legs[j].index <= en)].values * w[j],
                           index=legs[j][(legs[j].index >= st) & (legs[j].index <= en)].index)
                 for j in basket]
        S[k] = pd.concat(parts).sort_index()
    months = sorted(set(S[base_key].index.to_period("M")))
    m = len(months)
    G = {k: {p: g.values for p, g in s.groupby(s.index.to_period("M"))} for k, s in S.items()}
    rng = np.random.default_rng(20260713)
    print(f"  {'block':<7}" + "".join(f"{k:>26}" for k in arms))
    for blk in (1, 3, 6, 12):
        nb = int(np.ceil(m / blk))
        D = {k: [] for k in arms}
        for _ in range(NDRAW):
            st_ = rng.integers(0, m, nb)
            order = [months[(s + j) % m] for s in st_ for j in range(blk)][:m]
            for k in arms:
                v = np.concatenate([G[k][p] for p in order if p in G[k]])
                D[k].append(cdd(v, 365.25 * m / 12)[2])
        b = np.array(D[base_key])
        row = [f"{np.nanmedian(np.array(D[k])):.2f}(P{np.nanmean(np.array(D[k]) > b)*100:.0f}%)"
               for k in arms]
        print(f"  {f'{blk}mo':<7}" + "".join(f"{r:>26}" for r in row))


def main():
    L = build("2018-01-01", False)          # gold data fixed, gold15m without skip
    g0 = gold15m_raw()

    print("D1 -- gold15m のセッション・スキップ。まず LEG 自身の証拠を見る。")
    print("     （スキップする時間帯そのものが本当にゼロ〜負なのか。ブックは後で見る）\n")
    print("  スキップ対象の時間帯だけを取り出すと:")
    for tag, hrs in SKIPS.items():
        if hrs is None: continue
        sub = g0[g0.index.hour.isin(hrs)]
        pf = sub[sub > 0].sum() / abs(sub[sub <= 0].sum())
        half = sub.index[len(sub) // 2]
        print(f"    {tag:<26}n={len(sub):>4}  PF={pf:>4.2f}  meanR={sub.mean():+.3f}  "
              f"IS/OOS={sub[sub.index<half].mean():+.3f}/{sub[sub.index>=half].mean():+.3f}"
              f"   {'<-- 捨てる価値があるのはPF<1かつ両期間で負の時だけ' if pf<1 else '<-- PF>1: 捨てると黒字を捨てることになる'}")

    print("\n  時間帯別（UTC）の素の姿:")
    print(f"    {'hour':<6}{'n':>5}{'PF':>7}{'meanR':>9}")
    for h in range(0, 24):
        sub = g0[g0.index.hour == h]
        if len(sub) < 8: continue
        pf = sub[sub > 0].sum() / abs(sub[sub <= 0].sum()) if (sub <= 0).any() else np.nan
        mark = "  <" if 9 <= h < 15 else ""
        print(f"    {h:>2}時 {len(sub):>7}{pf:>7.2f}{sub.mean():>+9.3f}{mark}")

    print("\n  レッグ単体:")
    arms = {}
    for tag, hrs in SKIPS.items():
        s = g0 if hrs is None else g0[~g0.index.hour.isin(hrs)]
        leg_line(s, tag)
        arms[tag] = s

    print("\n  6レッグ・ブック（gold15m をこの形にしたとき）:")
    for tag, s in arms.items():
        LL = dict(L); LL["gold15m"] = s
        c, d, x, n = book(LL, FIVE + ["gold15m"])
        c5, d5, x5, _ = book(LL, FIVE)
        print(f"    {tag:<26}6レッグ CAGR/DD={x:>5.2f}  (DD {d:.2f}%)   ＜比較＞gold15m 抜きの5レッグ={x5:.2f}")

    print("\n" + "=" * 96)
    print("D2 -- 低頻度の BTC 4H レッグ 2本は、15分足レッグが入った後も席に値するか（5レッグで leave-one-out）\n")
    b5 = book(L, FIVE)
    print(f"  {'5レッグ (全部)':<26}CAGR/DD={b5[2]:>5.2f}")
    LOO = {"5-leg (all)": FIVE}
    for k in FIVE:
        rest = [j for j in FIVE if j != k]
        x = book(L, rest)[2]
        print(f"  {'  minus ' + k:<26}CAGR/DD={x:>5.2f}  ({x - b5[2]:+.2f})")
        if k in ("btc_bo_kama", "btc_pull", "gold_bo"):
            LOO[f"minus {k}"] = rest
    print()
    boot(L, LOO, "5-leg (all)")
    print("\n  P = そのブックが5レッグ全部に勝つ確率。50%を大きく下回る = そのレッグは席に値する。")


if __name__ == "__main__":
    main()
