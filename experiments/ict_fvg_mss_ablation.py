"""ICT 再設計ライン — 項目4 Step1: FVG(fair value gap) を MSS の displacement 条件に追加。

quant-audit.md 準拠のカノン定義:
  bullish FVG (ロング用): candle3.low > candle1.high -> 帯[candle1.high, candle3.low]
  bearish FVG (ショート用): candle1.low > candle3.high -> 帯[candle3.high, candle1.low]
  size = ギャップ幅 / その日のATR(A)。ブレイク脚（抜かれたフラクタル足 sh/sl 〜 ブレイク足 jm）内を走査。
実装は experiments/ict_population.py の build()/canonical_setups() に use_fvg/fvg_min_atr として追加
（既存の狩り/MSS/執行/コストモデルはそのまま流用。入口(押し目0.25/RR4)は変えない）。

ablation 段: a) MSS-raw(use_fvg=False, 現行) -> b) +FVG存在(min=0) -> c) +FVGサイズ0.15 -> d) 0.25。
ロング/ショートは絶対に別集計。対象: 主軸 eurusd/gbpusd/usdjpy、対照 audusd/nzdusd/usdcad/gold/btcusd。

自己検査(anchor)は ict_population.py 側で担保済み（use_fvg=False で n=1148/PF1.17/net+0.134/totR-DD2.56
を再現、本スクリプトはそのanchorを壊さず新しい列を足すだけ）。

Run: .venv/bin/python experiments/ict_fvg_mss_ablation.py [--smoke]
"""
import sys, io, contextlib, argparse
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, F_CANON, RR_CANON, walk, stats
from ict_population import canonical_setups, load_prepped

STAGES = [
    ("a_MSSraw",   False, 0.00),
    ("b_FVGexist", True,  0.00),
    ("c_FVG0.15",  True,  0.15),
    ("d_FVG0.25",  True,  0.25),
]
PRIMARY = ["eurusd", "gbpusd", "usdjpy"]
SECONDARY = ["audusd", "nzdusd", "usdcad", "gold", "btcusd"]


def fmt_row(sym, side, lab, st):
    if st is None:
        return f"{sym:8s} {side:5s} {lab:11s} {'n<10':>6s}"
    return (f"{sym:8s} {side:5s} {lab:11s} n={st['n']:5d} n/yr={st['npy']:6.1f} "
            f"win%={st['win']:5.1f} meanR={st['net']:+.3f} PF={st['pf']:5.2f} "
            f"totR/DD={st['rdd']:6.2f} IS={st['IS']:+7.0f} OOS={st['OOS']:+7.0f}")


def main(symbols, smoke=False):
    rows = []
    for name in symbols:
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.25):]
        sp, cost = MODEL[name]
        print(f"\n=== {name} (span={span}年, dates={len(dates)}) ===")
        for side in ("long", "short"):
            for lab, uf, fm in STAGES:
                S = canonical_setups(df, tarr, dates, 0, use_fvg=uf, fvg_min_atr=fm)
                tr = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, side)
                st = stats(tr, span)
                print("  " + fmt_row(name, side, lab, st))
                rows.append(dict(sym=name, side=side, stage=lab, stats=st))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    print("FVG-MSS ablation: (a)MSS-raw -> (b)+FVG存在 -> (c)+FVGサイズ0.15 -> (d)0.25")
    print(f"入口固定: f={F_CANON} RR={RR_CANON}（本リポジトリ正典・浅い押し目/遠い固定目標。ICT深OTE/RR2ではない）")
    print("\n########## 主軸 (eurusd/gbpusd/usdjpy) ##########")
    main(PRIMARY, smoke=args.smoke)
    print("\n########## 対照 (audusd/nzdusd/usdcad/gold/btcusd) ##########")
    main(SECONDARY, smoke=args.smoke)
