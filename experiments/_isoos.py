import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0,"."); sys.path.insert(0,"experiments")
from rr_with_swap import leg
b = leg("btc15m_L")[0]; a = leg("btc15m_L", fwd=96)[0]
def half(s):
    yr = sorted(s.index.year.unique()); m = yr[len(yr)//2]
    return s[s.index.year < m].mean(), s[s.index.year >= m].mean(), m
for nm, s in (("現行(打切り無し)", b), ("1日で強制決済", a)):
    i,o,m = half(s)
    print(f"{nm:<16} n={len(s):>4} meanR={s.mean():+.3f}  IS(<{m})={i:+.3f}  OOS(>={m})={o:+.3f}")
print(f"\n{'年':<6}{'現行 n':>8}{'現行 totR':>11}{'1日 n':>8}{'1日 totR':>11}{'差':>9}")
for y in sorted(set(b.index.year) | set(a.index.year)):
    bb, aa = b[b.index.year==y], a[a.index.year==y]
    print(f"{y:<6}{len(bb):>8}{bb.sum():>+11.1f}{len(aa):>8}{aa.sum():>+11.1f}{aa.sum()-bb.sum():>+9.1f}")
