"""ICT「discount のときだけ買う」を FX 全ペア + gold/BTC に横展開（2026-07-15）。

EUR/GBP のロング@discount が本物と分かった（GBP: 死んだ母集団 0.02→1.75/96%ile、
EUR: 2.01→3.03/85%ile、ショートは両方死）。これが:
  - FX ロング全般の「discount から買え」（＝ドル安方向の平均回帰）なら頑健で裁量に強い
  - EUR/GBP 固有なら薄い
を決める。gold/BTC は前回ゲートA両サイドで逆極性（トレンド）優位だったので、
ロング単独でも discount は効かない見込み（トレンド銘柄は premium/discount が逆、の裏取り）。

母集団: 狩り(sweep)+MSS + 浅0.25 + RR4 + NYキルゾーン、ASK指値約定。
方向: ロング固定。ゲート: 日足の直近10日レンジ内で close<0.5-band（discount）の時だけ建てる。
審判: base(素ロング) との比較 / ランダム間引き帰無の totR/DD %ile / 巡回ブロック 1/3/6/12mo /
      時代別 / 実コスト(realistic) と 1.5×(stress)。

Run: .venv/bin/python experiments/ict_discount_replicate.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import ict_killzone as K
import ict_v2_mss as V
import ict_ablation as A
from ict_killzone import load_ny
from ict_v2_mss import prep, walk
from ict_ablation import build, BUF
from ict_abstain import join_days, sc
from ict_pd_bias import pd_frame

# nzdusd / usdcad を ICT ローダーに追加（マッピングは自明・データは m15 あり）
for s in ("nzdusd", "usdcad"):
    K.SYMS[s] = f"data/vantage_{s}_m15.csv"
    V.MODEL[s] = (3e-05, 6e-05)      # spread, commission（他の FX メジャーと同じ）
    A.PIP[s] = 0.0001

RNG = np.random.default_rng(20260715)
F, RR = 0.25, 4.0
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]
# realistic RT ≈ spread + commission。FX: 0.3pip spread + 0.6 commission = 0.9pip。gold $0.20。btc $15。
REAL_SPREAD = {"eurusd": 0.3, "gbpusd": 0.3, "audusd": 0.3, "nzdusd": 0.3, "usdcad": 0.3,
               "usdjpy": 0.3, "gold": 2.0, "btcusd": 15.0}
FX = ["eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdjpy"]
ALL = FX + ["gold", "btcusd"]


def eras_of(tr):
    return " ".join(f"{sum(x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b):+5.0f}"
                    if any(a <= pd.Timestamp(x[0]).year <= b for x in tr) else "  n/a"
                    for a, b in ERAS)


def block_boot(tr, months, nrep=2000):
    s = pd.Series([t[1] for t in tr], index=pd.to_datetime([t[0] for t in tr])).sort_index()
    g = [x.values for _, x in s.groupby(s.index.to_period("M"))]
    nb = max(1, len(g) // months)
    bl = [np.concatenate(g[i * months:(i + 1) * months]) for i in range(nb)
          if len(g[i * months:(i + 1) * months])]
    bl = [b for b in bl if len(b)]
    if len(bl) < 4:
        return np.nan
    return 100.0 * sum(1 for _ in range(nrep)
                       if np.concatenate([bl[i] for i in RNG.integers(0, len(bl), len(bl))]).sum() > 0) / nrep


def rdd_null(base, k, nrep=2000):
    net = np.array([t[1] for t in base]); out = []
    for _ in range(nrep):
        x = net[np.sort(RNG.choice(len(net), k, replace=False))]; cum = np.cumsum(x)
        dd = (np.maximum.accumulate(cum) - cum).max()
        out.append(x.sum() / dd if dd > 0 else np.inf)
    return np.array(out)


def long_at_discount(name, spread_pips, band):
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(K.SYMS[name], cut2000=(name == "usdjpy"))
    df, tarr, dates = prep(df)
    span = int((pd.to_datetime(df["broker_dt"]).dt.year.value_counts() > 5000).sum())
    P = pd_frame(df)
    S = build(df, tarr, dates, True, True, "mss", 0)
    sp = spread_pips * A.PIP[name]; _, cost = V.MODEL[name]
    L = {d: net for (d, net, g, risk) in walk(df, S, F, RR, BUF, sp, cost, "long")}
    J = join_days(sorted(L), P)
    base = [(d, L[d]) for d in J.index if d in L]
    disc = [(d, L[d]) for d, r in J.iterrows()
            if not pd.isna(r["pos10"]) and r["pos10"] < 0.5 - band and d in L]
    return span, base, disc


def main():
    print("ICT『discount のときだけ買う』横展開。母集団=狩り+MSS/浅0.25/RR4/NYキルゾーン、ロング固定")
    print("realistic コスト = FX 0.9pip / gold $0.20 / btc $15。band=0.20（discount = 直近10日レンジの下2割）")
    print("=" * 122)
    print(f"  {'銘柄':8s} {'年':>4} {'base n':>6} {'base':>6} | {'disc n':>6} {'年本':>5} {'PF':>5} "
          f"{'net/本':>7} {'totR/DD':>8} {'間引%ile':>7} {'ブロック1/3/6/12':>16} | 時代別 totR(disc)")
    for name in ALL:
        span, base, disc = long_at_discount(name, REAL_SPREAD[name], 0.20)
        b = sc(base); s = sc(disc)
        if b is None or s is None:
            print(f"  {name:8s}  (母集団薄い)"); continue
        nul = rdd_null(base, s["n"]); pc = 100 * (s["rdd"] > nul).mean()
        bbs = "/".join(f"{block_boot(disc, m):.0f}" for m in (1, 3, 6, 12))
        star = " *" if (pc >= 90 and s["rdd"] > b["rdd"]) else ""
        print(f"  {name:8s} {span:4d} {b['n']:6d} {b['rdd']:6.2f} | {s['n']:6d} {s['n']/span:5.1f} "
              f"{s['pf']:5.2f} {s['net']:+7.3f} {s['rdd']:8.2f} {pc:6.0f}% {bbs:>16} | {eras_of(disc)}{star}")

    # band 感度（FX のみ、realistic コスト）
    print("\n" + "=" * 122)
    print("band 感度（discount のしきい: 下 band 分。0=中央より下なら可 / 0.2=下2割のみ）realistic コスト・totR/DD(間引%ile)")
    print("=" * 122)
    print(f"  {'銘柄':8s} {'band0.00':>16} {'band0.10':>16} {'band0.20':>16} {'band0.30':>16}")
    for name in FX:
        cells = []
        for band in (0.0, 0.10, 0.20, 0.30):
            span, base, disc = long_at_discount(name, REAL_SPREAD[name], band)
            s = sc(disc)
            if s is None:
                cells.append("      n/a       "); continue
            nul = rdd_null(base, s["n"]); pc = 100 * (s["rdd"] > nul).mean()
            cells.append(f"{s['rdd']:6.2f}({pc:3.0f}%)n{s['n']:<4d}")
        print(f"  {name:8s} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
