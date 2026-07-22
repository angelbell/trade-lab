"""BOOK-PIPELINE TIE-BACK — research/book.py must reproduce the frozen evidence
scripts exactly: every leg series identical to book_deployed_spec.build(200, 4.5),
the verdict identical to book_spec_fix.book, and BASE identical to the frozen
radar_gate_race.BASE. Plus the adopted anchor (CAGR/DD 7.88) as a value check.

Run: .venv/bin/python invariants/book_tieback.py 2>&1 | tee experiments/out_book_tieback.txt
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))   # 番人は invariants/ にあるが、
# 比較対象の実装は experiments/ にある（素の import を解決するため）

from src.engine.presets import BASE as BASE_NEW
from radar_gate_race import BASE as BASE_OLD
from research.book import get_book_legs, book, w_trade, SIX
from book_deployed_spec import build as build_old
from book_spec_fix import book as book_old, w_trade as w_old

oks = []


def chk(name, ok):
    oks.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def main():
    chk("presets.BASE == radar_gate_race.BASE", BASE_NEW == BASE_OLD)

    legs_new = get_book_legs(200, 4.5)
    legs_old = build_old(200, 4.5)
    for k in SIX:
        a, b = legs_new[k], legs_old[k]
        chk(f"leg {k}: series identical (n={len(a)})",
            len(a) == len(b) and a.index.equals(b.index)
            and np.array_equal(a.values, b.values))

    vn = book(legs_new, SIX)
    vo = book_old(legs_old, SIX)
    chk(f"book() verdict identical {tuple(round(x, 6) if isinstance(x, float) else x for x in vn)}",
        vn == vo)
    wn, wo = w_trade(legs_new, SIX), w_old(legs_old, SIX)
    chk("w_trade weights identical", np.allclose(wn[SIX].values, wo[SIX].values, rtol=0, atol=0))

    c0, d0, r0, n0 = vn
    chk(f"anchor: CAGR/DD {r0:.2f} == 7.88 (採用値)", round(r0, 2) == 7.88)
    chk(f"anchor: maxDD {d0:.2f} == 7.74", round(d0, 2) == 7.74)
    chk(f"anchor: CAGR {c0:.1f} == 61.0", round(c0, 1) == 61.0)

    print(f"\n===== {sum(oks)}/{len(oks)} PASS =====")
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
