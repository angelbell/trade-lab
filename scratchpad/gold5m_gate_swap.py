"""B4: gold 5m breakout+pullback -- gate-swap + dead-window improvement test.
Death cause after real-cost recompute: DD(R)60 / IS+0.24>>OOS+0.08 (bleed years 2021/2023).
Lever 1: regime gate granularity swap (daily SMA150+slope -> kama4h / kama4h&stack4h / kama1d),
         exactly the C1 lesson ("finer gate cuts bleed years") applied to gold 5m.
Lever 2: 9-15 server-hour dead-window skip (canon gold enhancement; +0.05 on market already).
Machinery: exec-split of pullback_5m_realcost.py (canonical walk incl. target-first-miss),
gate swapped by replacing the module-global `up_full` before build(). ext-cap8 (SMA-based)
held fixed across all variants so ONLY the regime gate varies. All net $0.3 + slip 0.27ATR.
PASS bar (pre-registered in proposals B4): ret/DD >= 5.5 with balanced IS/OOS, else final-close."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from breakout_wave import kama_adaptive

src = open("scratchpad/pullback_5m_realcost.py").read()
ns = {}
exec(src.split("\nB5r4  = build")[0], ns)          # defs + full-file load, no driver
build, eval_pull, eval_market = ns["build"], ns["eval_pull"], ns["eval_market"]
stats, row, HDR, full = ns["stats"], ns["row"], ns["HDR"], ns["full"]

dck4 = full["close"].resample("240min").last().dropna()
km4 = kama_adaptive(dck4, 14)
kama4h_up = (km4 > km4.shift(1)).shift(1)
c4 = full["close"].resample("240min").last().dropna()
f4 = c4.ewm(span=20, adjust=False).mean(); s4 = c4.ewm(span=50, adjust=False).mean()
stack_up = ((np.sign(c4-f4)+np.sign(f4-s4)+np.sign(s4-s4.shift(10))) > 0).shift(1)
dck1 = full["close"].resample("1D").last().dropna()
km1 = kama_adaptive(dck1, 14)
kama1d_up = (km1 > km1.shift(1)).shift(1)

GATES = {"SMA150d(canon)": ns["up_full"],
         "kama1d": kama1d_up,
         "kama4h": kama4h_up,
         "kama4h&stack4h": (kama4h_up.reindex(full.index, method="ffill")
                            & stack_up.reindex(full.index, method="ffill"))}

def window_skip(B):
    B2 = dict(B)
    B2["entries"] = [en for en in B["entries"] if not (9 <= B["d"].index[en[0]].hour < 15)]
    return B2

print(HDR)
for tag, g in GATES.items():
    ns["up_full"] = g                       # build() reads module-global up_full
    B = build(None, 4.0)
    span = (B["d"].index[-1] - B["d"].index[0]).days / 365.25
    tr, ms = eval_pull(B, lambda e, s, H: e - 0.25 * (e - s), 0.3, stop_slip=0.27)
    row(f"{tag}", tr, span, ms)
    tr2, ms2 = eval_pull(window_skip(B), lambda e, s, H: e - 0.25 * (e - s), 0.3, stop_slip=0.27)
    row(f"{tag}+win", tr2, span, ms2)
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    print("      per-year: " + "  ".join(f"{y}:{R[yr==y].sum():+.0f}" for y in np.unique(yr)))
