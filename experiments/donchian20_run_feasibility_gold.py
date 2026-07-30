"""gold Donchian20 ロングを「実際に回す」前提での実務パラメータ実測（1H vs 4H）。

測るもの:
  1. 建て時点の損切り距離$/oz の分布 → 0.01ロット(=1oz)での1トレード実損$ → 1%リスクに必要な口座残高
  2. 保有期間（暦日）・年別本数
  3. 連敗分布（最大・中央値）とトレード解像度maxDD・ブートストラップ中央値DD
  4. 賭け率を中央値DD 15%に揃えたときの資金倍率・複利年率
出口はトレール（20本安値）でベースラインと同一。コストは apply_cost(0.3, 0.1)。
"""
SCREEN = "donchian20_sar_gold_h1"

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np
import pandas as pd

from donchian20_sar_gold import (
    LENGTH, load_tf, apply_cost, years_span, summarize, max_losing_streak,
)
from donchian20_long_gate_gold import (
    build_daily_gates, align_bool, pf_pct, year_breakdown_pct, daily_pnl_series,
)
from donchian20_fixedtgt_gold import simulate_long_exit, CENTRAL, GUARD, bootstrap_dd_median

RNG_SEED = 20260730
N_BOOT = 400
DD_TARGET = 15.0
USDJPY = 155.0            # 円換算の目安（概算表示のみ）


def stop_dist_at_entry(df, trades):
    """建てた足の20本安値（＝その時点のトレール位置）までの距離$/oz。"""
    ll = df["low"].rolling(LENGTH).max().shift(1)  # placeholder, replaced below
    ll = df["low"].rolling(LENGTH).min().shift(1).values
    out = []
    for t in trades:
        out.append(t["entry_raw"] - ll[t["entry_i"]])
    return np.array(out)


def hold_days(df, trades):
    idx = df.index
    return np.array([(idx[t["exit_i"]] - idx[t["entry_i"]]).total_seconds() / 86400 for t in trades])


def q(a, p):
    return float(np.percentile(a, p))


def main():
    rng = np.random.default_rng(RNG_SEED)
    h1 = load_tf("h1")
    gates = build_daily_gates(h1)

    for tf in ("h1", "4h"):
        df = load_tf(tf)
        yrs = years_span(df)
        dates = pd.DatetimeIndex(sorted(set(df.index.normalize())))
        g = GUARD[tf]
        for gname, garr in (("ゲート無し", None),
                            ("SMA150", align_bool(gates["sma_assigned"], df.index))):
            tr, open_t, _ = simulate_long_exit(df, "trail", gate_arr=garr)
            trc = apply_cost(tr, *CENTRAL)
            s = summarize(trc, yrs)
            if gname == "ゲート無し":
                ok = (s["n"] == g["n"] and abs(s["tot_pct_per_year"] - g["ann_pct"]) < 0.05)
                print(f"\n[番人 {tf}] n={s['n']} 年率={s['tot_pct_per_year']:.2f}% "
                      f"→ {'PASS' if ok else 'FAIL'} (期待 n={g['n']} 年率={g['ann_pct']}%)")
                assert ok

            sd = stop_dist_at_entry(df, tr)
            hd = hold_days(df, tr)
            pnl = np.array([t["pnl_pct"] for t in trc])
            pnl_d = np.array([t["pnl_dollar"] for t in trc])
            daily = daily_pnl_series(trc, dates)
            bdd = float(np.median(bootstrap_dd_median(daily, 3, N_BOOT, rng)))
            f = DD_TARGET / bdd
            mult = float(np.prod(1 + f * pnl / 100))
            comp = (mult ** (1 / yrs) - 1) * 100

            print(f"\n{'='*100}\n### gold {tf} / {gname}  ({yrs:.2f}年, {df.index[0].date()}→{df.index[-1].date()})\n{'='*100}")
            print(f"  n={s['n']} ({s['n_per_year']:.1f}本/年) PF%={pf_pct(trc):.2f} 勝率={s['win_pct']:.1f}% "
                  f"平均R相当={s['mean_pct']:+.3f}% 年率={s['tot_pct_per_year']:.2f}%")
            print(f"  maxDD(トレード解像度)={s['maxdd_pct']:.2f}%  bootDD中央値={bdd:.2f}%  "
                  f"年率/DD={s['tot_pct_per_year']/s['maxdd_pct']:.2f}")
            print(f"  最大連敗={s['max_losing_streak']}本  平均保有={hd.mean():.1f}日 "
                  f"(中央値{np.median(hd):.1f}日 / 90%点{q(hd,90):.1f}日)")
            print(f"  賭け率f={f:.2f}（中央値DDを{DD_TARGET}%に揃える）→ 資金倍率={mult:.2f} 複利年率={comp:.1f}%")
            print(f"  --- 損切り距離 $/oz（＝0.01ロット1枚あたりの実損$） ---")
            print(f"    中央値={np.median(sd):6.2f}  平均={sd.mean():6.2f}  σ={sd.std(ddof=1):6.2f}  "
                  f"25%={q(sd,25):6.2f}  75%={q(sd,75):6.2f}  90%={q(sd,90):6.2f}  最大={sd.max():6.2f}")
            print(f"    円換算(@{USDJPY:.0f}円): 中央値 ¥{np.median(sd)*USDJPY:,.0f} / 90%点 ¥{q(sd,90)*USDJPY:,.0f}")
            for r in (1.0, 2.0):
                need_med = np.median(sd) / (r / 100)
                need_90 = q(sd, 90) / (r / 100)
                print(f"    1トレード{r:.0f}%リスクで0.01ロットを建てるのに必要な口座: "
                      f"中央値ベース ${need_med:,.0f} (¥{need_med*USDJPY:,.0f}) / "
                      f"90%点ベース ${need_90:,.0f} (¥{need_90*USDJPY:,.0f})")
            print(f"  --- 年別 ---")
            for row in year_breakdown_pct(trc):
                print(f"    {row['year']}: n={row['n']:3d} PF%={row['pf_pct']:5.2f} "
                      f"{row['tot_pct']:+7.2f}% 勝率={row['win_pct']:5.1f}%")
            if open_t:
                print(f"  未決済: {open_t['entry_time']} 建て {open_t['entry_raw']:.2f} "
                      f"({open_t['bars_since']}本経過)")


if __name__ == "__main__":
    main()
