"""GBPUSD の ICT premium/discount バイアスに決定的ストレス（2026-07-15）。

ict_pd_bias.py で GBPUSD だけ光った:
  base（両サイド）totR/DD -0.02 → ゲートA（discount→ロング/premium→ショート）で 0.9〜1.3、
  9セル中6セルが ≥90%ile、台地、逆極性は -0.73/%ile3（符号確認 合格）。
だが 2000-08 に front-load の疑い。潰す:
  1. 年別・時代別 totR（27年。2000-08 のドル・トレンド時代に依存していないか）
  2. 巡回ブロック・ブートストラップ 1/3/6/12か月（別の月の並びでも totR>0 か・伸ばすと上がるか）
  3. プラセボ窓 +4/+8/+12h（本窓が特別か。偽窓でも同じだけ効くならキルゾーン非依存）
  4. 執行コスト 0.3/0.6/1.0 pip（実 RT ≈ 0.9pip）
  5. EURUSD を陰性対照として併走（同じ設定で光らないことを確認）

Run: .venv/bin/python scratchpad/ict_pd_stress.py
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
from ict_abstain import join_days, sc, random_drop_null
from ict_pd_bias import pd_frame, gate_pd, base_bothsides

RNG = np.random.default_rng(20260715)
F, RR = 0.25, 4.0
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]
CELLS = [("pos10", 0.20), ("pos40", 0.00), ("pos20", 0.20)]   # 台地の代表3セル


def pool_side(df, setups, name, spread_pips):
    sp = spread_pips * PIP[name]; _, cost = MODEL[name]
    p = {"long": {}, "short": {}}
    for side in ("long", "short"):
        for (d, net, g, risk) in walk(df, setups, F, RR, BUF, sp, cost, side):
            p[side][d] = net
    return p


def eras_of(tr):
    return " ".join(f"{sum(x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b):+6.0f}"
                    if any(a <= pd.Timestamp(x[0]).year <= b for x in tr) else "   n/a"
                    for a, b in ERAS)


def block_boot(tr, months, nrep=3000):
    s = pd.Series([t[1] for t in tr], index=pd.to_datetime([t[0] for t in tr])).sort_index()
    groups = [g.values for _, g in s.groupby(s.index.to_period("M"))]
    nb = max(1, len(groups) // months)
    blocks = [np.concatenate(groups[i * months:(i + 1) * months]) for i in range(nb)
              if len(groups[i * months:(i + 1) * months])]
    blocks = [b for b in blocks if len(b)]
    if len(blocks) < 4:
        return np.nan
    w = sum(1 for _ in range(nrep)
            if np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))]).sum() > 0)
    return 100.0 * w / nrep


def run_name(name, negctrl=False):
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[name])
    df, tarr, dates = prep(df)
    span = int((pd.to_datetime(df["broker_dt"]).dt.year.value_counts() > 5000).sum())
    P = pd_frame(df)
    S0 = build(df, tarr, dates, True, True, "mss", 0)
    p0 = pool_side(df, S0, name, 0.3)
    J = join_days(sorted(set(list(p0["long"]) + list(p0["short"]))), P)
    base = base_bothsides(p0, J)
    b = sc(base)
    print("\n" + "=" * 104)
    print(f"=== {name}{'（陰性対照）' if negctrl else ''} ===  ({span}年)  "
          f"base両サイド: n={b['n']} totR/DD={b['rdd']:.2f}  時代別 {eras_of(base)}")
    print("=" * 104)

    for poscol, band in CELLS:
        # コスト・ストレス
        print(f"\n  ゲートA {poscol} band={band:.2f}  （discount→ロング / premium→ショート）")
        for sp in (0.3, 0.6, 1.0):
            pc_pool = pool_side(df, S0, name, sp)
            tr = gate_pd(pc_pool, J, poscol, band)
            s = sc(tr)
            if s is None:
                continue
            nul = random_drop_null(base, s["n"])
            pcile = 100 * (s["rdd"] > nul).mean()
            print(f"    cost{sp}pip  n={s['n']:4d} 年{s['n']/span:4.1f} PF={s['pf']:.2f} "
                  f"net={s['net']:+.3f} totR={s['tot']:+6.1f} totR/DD={s['rdd']:5.2f} 間引き%ile={pcile:3.0f}%")
        tr = gate_pd(p0, J, poscol, band)
        yr = pd.Series([t[1] for t in tr]).groupby(pd.to_datetime([t[0] for t in tr]).year).sum()
        print(f"    └ 時代別 totR: {eras_of(tr)}")
        print(f"    └ 年別 totR: " + " ".join(f"{int(y)%100:02d}:{v:+.0f}" for y, v in yr.items()))
        print(f"    └ ブロック・ブートストラップ(totR>0%): " +
              " / ".join(f"{m}か月 {block_boot(tr, m):.0f}%" for m in (1, 3, 6, 12)))

    # プラセボ窓（代表1セル）
    poscol, band = CELLS[0]
    print(f"\n  --- プラセボ窓（{poscol} band={band:.2f} を +4/+8/+12h の偽キルゾーンにも適用） ---")
    for sh in (0, 4, 8, 12):
        Ssh = build(df, tarr, dates, True, True, "mss", sh)
        psh = pool_side(df, Ssh, name, 0.3)
        Jsh = join_days(sorted(set(list(psh["long"]) + list(psh["short"]))), P)
        bs = sc(base_bothsides(psh, Jsh))
        s = sc(gate_pd(psh, Jsh, poscol, band))
        tag = "本窓" if sh == 0 else f"+{sh}h"
        if s and bs:
            print(f"    {tag:5s} base totR/DD={bs['rdd']:6.2f} → ゲートA={s['rdd']:6.2f} "
                  f"(net {bs['net']:+.3f}→{s['net']:+.3f}) 持ち上げ={s['rdd']-bs['rdd']:+.2f}")


def main():
    run_name("gbpusd")
    run_name("eurusd", negctrl=True)


if __name__ == "__main__":
    main()
