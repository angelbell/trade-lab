"""gold 5m breakout: the RR curve at REAL cost ($0.3 + slip0.27). Question: does a lower RR
(2, 2.5, 3) work at 5m? Mechanism prediction: cost/risk ~9-17% is a FIXED toll per trade;
gross meanR scales with RR (runner economics) -> low RR falls below the toll. Market + frac0.25."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open("experiments/pullback_5m_realcost.py").read()
ns = {}
exec(src.split("\nB5r4  = build")[0], ns)
build, eval_market, eval_pull, row, HDR = (ns[k] for k in
    ("build", "eval_market", "eval_pull", "row", "HDR"))
print(HDR)
for rr in (2.0, 2.5, 3.0, 4.0, 5.0):
    B = build(None, rr)
    span = (B["d"].index[-1] - B["d"].index[0]).days / 365.25
    be = 1.0 / (1.0 + rr) * 100
    row(f"mkt RR{rr} (be{be:.0f}%)", eval_market(B, 0.3, stop_slip=0.27), span)
    tr, ms = eval_pull(B, lambda e, s, H: e - 0.25 * (e - s), 0.3, stop_slip=0.27)
    row(f"f.25 RR{rr}", tr, span, ms)
