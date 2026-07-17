"""ICT 再設計 — フェーズ2 ショート側: クロスペア SMT で死んだ short@premium を蘇生できるか（2026-07-15）。

凍結仕様:
  母集団 = ict_population のショート（売り側流動性の狩り+MSS下+浅0.25+RR4+NYキルゾーン）。ショート固定。
  ゲート = 相方ペアが同じロンドン窓で buyside を掃除しなかった日だけ通す（弱気SMTダイバージェンス）。
           相方は同符号ドルペア: eurusd↔gbpusd、audusd↔nzdusd。確定はロンドン窓終了時＝先読み無し。
  ablation: (a) long-only base（文脈）→ (b) 無ゲート short（＝死を再現）→ (c) SMTゲート short。
  審判: base(無ゲートshort) の totR/DD、ランダム間引き%ile、プラセボ窓(+8h)、ブロック1/3/6/12、時代別。
        SMT が弱いショートを避けて totR/DD をプラスに転じ、帰無90%ile超＋ブロック伸長で上がるか。

Run: .venv/bin/python scratchpad/ict_smt.py 2>/dev/null
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import sc
from ict_population import canonical_setups, trade_pool, load_prepped
from ict_gates import sweep_frame, smt_short_gate
from ict_audit import random_drop_null, block_boot

PAIRS = [("eurusd", "gbpusd"), ("gbpusd", "eurusd"), ("audusd", "nzdusd"), ("nzdusd", "audusd")]
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]


def eras_of(tr):
    return " ".join(f"{sum(x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b):+6.0f}"
                    if any(a <= pd.Timestamp(x[0]).year <= b for x in tr) else "   n/a"
                    for a, b in ERAS)


def load(name):
    df, tarr, dates, span = load_prepped(name)
    S0 = canonical_setups(df, tarr, dates, 0)
    S8 = canonical_setups(df, tarr, dates, 8)
    pool0 = trade_pool(df, S0, name)
    pool8 = trade_pool(df, S8, name)
    return dict(df=df, tarr=tarr, dates=dates, span=span,
                pool0=pool0, pool8=pool8,
                sw0=sweep_frame(df, tarr, dates, 0), sw8=sweep_frame(df, tarr, dates, 8))


def main():
    cache = {}
    for name in set([p for pair in PAIRS for p in pair]):
        cache[name] = load(name)

    print("=" * 122)
    print("フェーズ2 SMT ショート蘇生 ablation ―― (a)long-base文脈 / (b)無ゲートshort=死の再現 / (c)SMTショート")
    print("  相方が buyside を掃除しなかった日だけ通す。審判=ランダム間引き%ile・プラセボ窓+8h・ブロック伸長")
    print("=" * 122)
    hdr = (f"  {'ペア(相方)':16s} {'baseL':>6} | {'baseS n':>7} {'baseS':>6} | {'SMT n':>6} {'年':>4} "
           f"{'PF':>5} {'net':>7} {'SMT rdd':>7} {'間引%ile':>7} {'+8h rdd':>7} {'ブロック1/3/6/12':>15}")
    print(hdr)
    for X, Y in PAIRS:
        cx, cy = cache[X], cache[Y]
        span = cx["span"]
        long_pool = cx["pool0"]["long"]; short_pool = cx["pool0"]["short"]
        base_long = [(d, long_pool[d]) for d in long_pool]
        base_short = [(d, short_pool[d]) for d in short_pool]
        smt = smt_short_gate(short_pool, cy["sw0"])
        smt8 = smt_short_gate(cx["pool8"]["short"], cy["sw8"])
        bL, bS, s = sc(base_long), sc(base_short), sc(smt)
        s8 = sc(smt8)
        if bS is None or s is None:
            print(f"  {X}({Y})  母集団薄い (baseS n={len(base_short)}, SMT n={len(smt)})"); continue
        nul = random_drop_null(base_short, s["n"]); pc = 100 * (s["rdd"] > nul).mean()
        bbs = "/".join(f"{block_boot(smt, m):.0f}" for m in (1, 3, 6, 12))
        s8r = f"{s8['rdd']:+.2f}" if s8 else "n/a"
        print(f"  {X+'('+Y+')':16s} {bL['rdd'] if bL else float('nan'):6.2f} | "
              f"{bS['n']:7d} {bS['rdd']:6.2f} | {s['n']:6d} {s['n']/span:4.1f} {s['pf']:5.2f} "
              f"{s['net']:+7.3f} {s['rdd']:7.2f} {pc:6.0f}% {s8r:>7} {bbs:>15}")

    print("\n  --- 時代別 totR（(b)無ゲートshort → (c)SMTショート）2000-08/2009-16/2017-20/2021-26 ---")
    for X, Y in PAIRS:
        cx, cy = cache[X], cache[Y]
        short_pool = cx["pool0"]["short"]
        base_short = [(d, short_pool[d]) for d in short_pool]
        smt = smt_short_gate(short_pool, cy["sw0"])
        print(f"  {X}({Y})")
        print(f"    baseS: {eras_of(base_short)}")
        print(f"    SMT  : {eras_of(smt)}")


if __name__ == "__main__":
    main()
