"""適応閾値の対立仮説つぶし: 順位窓が長すぎた（250-2000本＝10日〜3か月）だけではないか。

前段（atr_spike_adaptive_k.py）で適応閾値は3銘柄とも固定倍率に負けた。却下する前に、
「短い窓ならレジーム適応として働いたはず」という対立仮説を測る。
あわせて【本数をそろえた直接対決】を出す（本数が違うと総/DD の比較が賭け金の比較になる）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_adaptive_k import prep, trig, run, SYMS   # noqa: E402


def gate_of(d, gk):
    if gk != "wk30":
        return None
    wc = d["close"].resample("W").last().dropna()
    return (wc > wc.rolling(30).mean()).shift(1).reindex(
        d.index, method="ffill").fillna(False).to_numpy()


if __name__ == "__main__":
    chk = {}
    for nm, path, utc, st, cost, we, gk in SYMS:
        d, o, h, l, c, ap, pdh = prep(path, utc, st)
        g = gate_of(d, gk)
        base = run(d, o, l, c, trig(d, o, h, l, c, ap, pdh, "FIX", 2.0, 0, we, g), cost)
        chk[nm] = base
        print(f"\n===== {nm}  短い順位窓（RANKA = 実体/ATR の順位）  各セル: PF / 本数 / 総÷DD")
        print(f"  {'N本':>6} " + " ".join(f"{'P='+str(p):>16}" for p in (90, 95, 97, 98, 99)))
        for win in (50, 100, 150, 250):
            row = []
            for p in (90, 95, 97, 98, 99):
                r = run(d, o, l, c, trig(d, o, h, l, c, ap, pdh, "RANKA", win, p, we, g), cost)
                row.append("      --      " if r is None else
                           f"{r['pf']:4.2f}/{r['n']:4d}/{r['score']:5.1f}")
            print(f"  {win:>6} " + " ".join(f"{x:>16}" for x in row))

        # 本数をそろえた直接対決
        print(f"  --- 本数をそろえた対決（基準 FIX k=2.0: N={base['n']}）")
        print(f"    {'引き金':<22} {'N':>5} {'勝率':>7} {'PF':>6} {'平均%':>9} "
              f"{'DD%':>7} {'総/DD':>7} {'黒字年':>8}")
        cands = [("FIX k=2.0", "FIX", 2.0, 0)]
        for kind in ("RANK", "RANKA"):
            best, bd = None, None
            for win in (50, 100, 150, 250, 500, 1000, 2000):
                for p in (90, 95, 97, 98, 99, 99.5):
                    n = len(trig(d, o, h, l, c, ap, pdh, kind, win, p, we, g))
                    if bd is None or abs(n - base["n"]) < bd:
                        bd, best = abs(n - base["n"]), (win, p)
            cands.append((f"{kind} {best[0]}本/P{best[1]}", kind, best[0], best[1]))
        for lab, kind, a, b in cands:
            r = run(d, o, l, c, trig(d, o, h, l, c, ap, pdh, kind, a, b, we, g), cost)
            print(f"    {lab:<22} {r['n']:5d} {r['win']:6.1f}% {r['pf']:6.2f} "
                  f"{r['mean']:+9.3f} {r['dd']:7.1f} {r['score']:7.2f} "
                  f"{r['pos']:4d}/{r['ny']:<3d}")

    assert chk["BTC"]["n"] > 250 and chk["USDJPY"]["n"] > 250, (chk["BTC"]["n"], chk["USDJPY"]["n"])
    print(f"\nOK: 基準の再現 BTC N={chk['BTC']['n']} PF={chk['BTC']['pf']:.2f} / "
          f"USDJPY N={chk['USDJPY']['n']} PF={chk['USDJPY']['pf']:.2f}")
