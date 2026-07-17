"""FX で「難しい日は入るな」を、方向選択と切り離して振る（2026-07-15）。

これまでの棄権テストは「日足が方向を決める × 曖昧なら棄権」を抱き合わせていた。
だが FX では *方向選択そのものが毒* だと分かっている:
    EURUSD ロング専用   net +0.134 / PF 1.17
    日足に方向を選ばせる  net -0.065 / PF 0.92
→ 方向という毒が、棄権の効果を隠している可能性がある。切り離して測る。

母集団: 狩り(sweep) + MSS + 浅い押し目 f=0.25 + RR4 + NYキルゾーン（ASK基準の指値約定）
サイド: 固定（ロング専用 / ショート専用）。日足には *方向を決めさせない*。
棄権  : 前日の日足の実体比 |C-O|/(H-L) が下位 q% の日は入らない（＝十字線＝迷いの日）
        対照として 日足ER(10) も振る（汎用トレンド強度は台帳で全滅済み＝陰性対照）

審判: ランダム間引き帰無（totR/DD）· 閾値の台地 · 時代別 · プラセボ窓(+8h) ·
      ベータ対照（同じ日にキルゾーンの寄りで成行）· 約定スプレッドのストレス

Run: .venv/bin/python scratchpad/ict_fx_abstain.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk, MODEL
from ict_ablation import build, PIP, BUF
from ict_abstain import daily_frame, join_days, sc, random_drop_null

RNG = np.random.default_rng(20260715)
F, RR = 0.25, 4.0
QS = [0.20, 0.35, 0.50, 0.65]
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]
CELLS = [("eurusd", "long"), ("gbpusd", "long"), ("audusd", "long"), ("usdjpy", "long"),
         ("eurusd", "short"), ("gbpusd", "short"), ("btcusd", "long"), ("gold", "long")]


def side_trades(df, setups, name, side, spread_pips):
    sp = spread_pips * PIP[name]; _, cost = MODEL[name]
    return {d: net for (d, net, g, risk) in walk(df, setups, F, RR, BUF, sp, cost, side)}


def pick(tmap, J, clarcol, q):
    thr = J[clarcol].quantile(q) if q > 0 else -np.inf
    out = []
    for d, row in J.iterrows():
        if q > 0 and (pd.isna(row[clarcol]) or row[clarcol] < thr):
            continue
        if d in tmap:
            out.append((d, tmap[d]))
    return out


def eras_of(tr):
    o = []
    for a, b in ERAS:
        v = [x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b]
        o.append(f"{sum(v):+6.0f}" if v else "   n/a")
    return " ".join(o)


def main():
    print("FX で「難しい日は入るな」— 方向は固定（日足に決めさせない）。棄権だけを測る。")
    print("母集団: 狩り+MSS / 浅0.25 / RR4 / NYキルゾーン。審判=ランダム間引き帰無の totR/DD %ile")
    for name, side in CELLS:
        with contextlib.redirect_stderr(io.StringIO()):
            df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
        df, tarr, dates = prep(df)
        span = int((pd.to_datetime(df["broker_dt"]).dt.year.value_counts() > 5000).sum())
        S0 = build(df, tarr, dates, True, True, "mss", 0)
        S8 = build(df, tarr, dates, True, True, "mss", 8)
        base_spread = 0.3 if name not in ("gold", "btcusd") else (1.5 if name == "gold" else 15.0)
        T0 = side_trades(df, S0, name, side, base_spread)
        T8 = side_trades(df, S8, name, side, base_spread)
        D = daily_frame(df)
        J = join_days(sorted(T0.keys()), D)
        J8 = join_days(sorted(T8.keys()), D)

        base = pick(T0, J, "E4_body", 0.0)
        b = sc(base)
        if b is None:
            continue
        print("\n" + "=" * 124)
        print(f"=== {name} / {side}専用 ===  ({span}年)")
        print(f"  {'棄権ルール':22s} {'棄権':>5} {'n':>5} {'年':>5} {'net':>7} {'PF':>5} {'totR':>7} "
              f"{'DD':>6} {'totR/DD':>8} {'null中央':>8} {'%ile':>5}  時代別 totR")
        print(f"  {'(棄権なし＝ベース)':22s} {'0%':>5} {b['n']:5d} {b['n']/span:5.1f} {b['net']:+7.3f} "
              f"{b['pf']:5.2f} {b['tot']:+7.1f} {b['dd']:6.1f} {b['rdd']:8.2f} {'':8s} {'':5s}  {eras_of(base)}")
        for clarcol, clab in (("E4_body", "前日実体比(十字線)"), ("E1_er", "日足ER(陰性対照)")):
            for q in QS:
                tr = pick(T0, J, clarcol, q)
                s = sc(tr)
                if s is None:
                    continue
                nul = random_drop_null(base, s["n"])
                pc = 100 * (s["rdd"] > nul).mean()
                star = " *" if pc >= 90 else ""
                print(f"  {clab:22s} {int(q*100):4d}% {s['n']:5d} {s['n']/span:5.1f} {s['net']:+7.3f} "
                      f"{s['pf']:5.2f} {s['tot']:+7.1f} {s['dd']:6.1f} {s['rdd']:8.2f} "
                      f"{np.median(nul):8.2f} {pc:4.0f}%  {eras_of(tr)}{star}")
        # プラセボ窓
        print("  --- プラセボ窓(+8h) ---")
        for q in (0.0, 0.35):
            a = sc(pick(T0, J, "E4_body", q)); z = sc(pick(T8, J8, "E4_body", q))
            if a and z:
                print(f"    棄権{int(q*100):3d}%  本物 n={a['n']:4d} net={a['net']:+.3f} PF={a['pf']:.2f} "
                      f"| +8h n={z['n']:4d} net={z['net']:+.3f} PF={z['pf']:.2f} "
                      f"| premium={a['net']-z['net']:+.3f}")


if __name__ == "__main__":
    main()
