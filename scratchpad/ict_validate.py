"""ICT 再設計の結合検算 — 4モジュールを繋いで多銘柄で回し、台帳の横展開と一致するか確認する。

各モジュールの単体自己検査は通った。ここは「exec→population→gates→audit を繋いだ全体」が
台帳（docs/findings/s01_entries.md の ICT v3 項）の記録値を再現するかを一目で見る:
  1. discount ロング横展開（8銘柄、base→discount band0.20 の totR/DD と間引き%ile）
     台帳: eurusd 2.56→3.80(88) / gbpusd 0.25→1.90(95) / audusd 0.19→0.07(49) /
           nzdusd -0.47→-0.83(7) / usdcad 0.66→0.89(73) / usdjpy 1.26→2.18(87) / gold・btc 死
  2. 生存3種(EUR/GBP/JPY)の深掘り: プラセボ窓プレミアム + 巡回ブロック・ブートストラップ

注: gold のスプレッドは MODEL 準拠($0.15+手数料$0.06)。台帳の横展開は $0.20 を使ったが gold は
    どちらでも死なので判定は不変（＝再設計で「スプレッドを1箇所に集約」した副作用、実害なし）。

Run: .venv/bin/python scratchpad/ict_validate.py 2>/dev/null
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import sc
from ict_population import canonical_setups, trade_pool, load_prepped
from ict_gates import pd_frame, join_days, gate_discount_long
from ict_audit import random_drop_null, block_boot, placebo_premium

# 台帳の記録値（base totR/DD → discount band0.20 totR/DD (間引き%ile)）
LEDGER = {
    "eurusd": (2.56, 3.80, 88), "gbpusd": (0.25, 1.90, 95), "audusd": (0.19, 0.07, 49),
    "nzdusd": (-0.47, -0.83, 7), "usdcad": (0.66, 0.89, 73), "usdjpy": (1.26, 2.18, 87),
}
ALL = ["eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdjpy", "gold", "btcusd"]
SURVIVORS = ["eurusd", "gbpusd", "usdjpy"]


def main():
    print("=" * 104)
    print("1. discount ロング横展開（結合パイプライン）band0.20 ―― 台帳と一致するか")
    print("=" * 104)
    print(f"  {'銘柄':8s} {'年':>4} {'base n':>6} {'base':>6} | {'disc n':>6} {'PF':>5} "
          f"{'totR/DD':>8} {'%ile':>5} | {'台帳 base/disc/%ile':>22} {'判定':>6}")
    cache = {}
    for name in ALL:
        df, tarr, dates, span = load_prepped(name)
        S = canonical_setups(df, tarr, dates, 0)
        pool = trade_pool(df, S, name)
        P = pd_frame(df)
        J = join_days(sorted(pool["long"]), P)
        cache[name] = (df, tarr, dates, span, pool, J)
        base = [(d, pool["long"][d]) for d in J.index if d in pool["long"]]
        disc = gate_discount_long(pool["long"], J, "pos10", 0.20)
        b, s = sc(base), sc(disc)
        if b is None or s is None:
            print(f"  {name:8s} (母集団薄い)"); continue
        nul = random_drop_null(base, s["n"]); pc = 100 * (s["rdd"] > nul).mean()
        if name in LEDGER:
            lb, ld, lp = LEDGER[name]
            ok = abs(b["rdd"] - lb) < 0.05 and abs(s["rdd"] - ld) < 0.05
            tag = "○一致" if ok else "×不一致"
            led = f"{lb:+.2f}/{ld:+.2f}/{lp:.0f}"
        else:
            tag = "(死・台帳外)"; led = "gold/btc=死"
        print(f"  {name:8s} {span:4d} {b['n']:6d} {b['rdd']:6.2f} | {s['n']:6d} {s['pf']:5.2f} "
              f"{s['rdd']:8.2f} {pc:4.0f}% | {led:>22} {tag:>8}")

    print("\n" + "=" * 104)
    print("2. 生存3種の深掘り: プラセボ窓プレミアム(本物−+8h) + 巡回ブロック・ブートストラップ(totR>0%)")
    print("   台帳: USDJPY は全4時代+・ブロック 96/96/98/99。窓プレミアムが正なら「窓に実体」")
    print("=" * 104)
    print(f"  {'銘柄':8s} {'本物 totR/DD':>12} {'+8h':>8} {'プレミアム':>10} | {'ブロック 1/3/6/12か月':>22}")
    for name in SURVIVORS:
        df, tarr, dates, span, pool, J = cache[name]
        disc = gate_discount_long(pool["long"], J, "pos10", 0.20)
        pl = placebo_premium(df, tarr, dates, name, "long", span,
                             shifts=(0, 8), gate="discount", poscol="pos10", band=0.20)
        real, p8 = pl[0], pl[8]
        prem = (real["rdd"] - p8["rdd"]) if (real and p8) else np.nan
        bbs = "/".join(f"{block_boot(disc, m):.0f}" for m in (1, 3, 6, 12))
        p8s = f"{p8['rdd']:+.2f}" if p8 else "n/a"
        print(f"  {name:8s} {real['rdd']:12.2f} {p8s:>8} {prem:+10.2f} | {bbs:>22}")


if __name__ == "__main__":
    main()
