"""fx_structure_screen.py -- CHEAP root screen: does low-TF USDJPY carry ANY mechanical raw material
(mean-reversion OR momentum) at the bar level, before building a new entry family?

session-breakout/fade is dead (2 majors x 2 feeds: vol-without-direction). The user wants a DIFFERENT low-TF
FX mechanism. Per the playbook (cheap-screen-first, all-signals base first): before mechanizing any entry
family, measure the bar-level STRUCTURE -- if returns are a pure random walk, NO entry family can extract an
edge; if they mean-revert (neg autocorr / VR<1) a fade has raw material; if they trend (pos autocorr / VR>1)
continuation does. Real Vantage USDJPY h1 (16yr) -- no bridge needed; this IS the trade feed.

Reports per TF: lag-1..3 return autocorr (|val|>~2/sqrt(n) = notable), variance-ratio VR(q) (Lo-MacKinlay;
<1 mean-revert, >1 trend, =1 random walk), and the ALL-SIGNALS BASE meanR of the two raw mechanisms (fade
the last bar's move / continue it), gross AND after a realistic ~0.6bp cost. A base that is <=0 gross =>
no raw material => kill the family cheaply. Descriptive; in-sample; Vantage is the feed.
  .venv/bin/python research/fx_structure_screen.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample


def variance_ratio(ret, q):
    ret = ret.dropna().values
    n = len(ret)
    if n < q * 4:
        return np.nan
    mu = ret.mean()
    var1 = np.sum((ret - mu) ** 2) / n
    # overlapping q-period returns
    rq = np.convolve(ret, np.ones(q), "valid")
    varq = np.sum((rq - q * mu) ** 2) / (n - q + 1)
    return (varq / q) / var1 if var1 > 0 else np.nan


def base_mech(d, cost_bp=0.6):
    """all-signals base for the two raw bar-level mechanisms (no filter, no selection)."""
    c = d["close"].values
    ret = np.diff(np.log(c))                                  # bar log-returns
    prev, nxt = ret[:-1], ret[1:]                             # prev bar move, this bar move
    cost = cost_bp * 1e-4
    # FADE: bet next = -sign(prev). pnl = -sign(prev)*nxt
    fade = -np.sign(prev) * nxt
    cont = np.sign(prev) * nxt                                # CONTINUE: bet next = sign(prev)
    return {
        "fade_gross_bp": fade.mean() * 1e4, "fade_net_bp": (fade.mean() - cost) * 1e4,
        "cont_gross_bp": cont.mean() * 1e4, "cont_net_bp": (cont.mean() - cost) * 1e4,
        "fade_win%": (fade > 0).mean() * 100, "n": len(fade),
        "bar_move_bp": np.abs(ret).mean() * 1e4,              # avg |move| -- vs cost, the breakeven scale
    }


def screen(name, d):
    c = d["close"]
    ret = np.log(c / c.shift(1))
    n = len(ret.dropna())
    se = 2 / np.sqrt(n)                                       # ~2-sigma band for autocorr
    ac = [ret.autocorr(k) for k in (1, 2, 3)]
    vr = [variance_ratio(ret, q) for q in (2, 4, 8)]
    b = base_mech(d)
    flag = lambda v: "*" if abs(v) > se else " "
    print(f"\n== {name}  (n={n}, |move|={b['bar_move_bp']:.1f}bp, autocorr 2sig band=±{se*1e4:.0f}e-4) ==")
    print(f"  autocorr  lag1={ac[0]:+.3f}{flag(ac[0])} lag2={ac[1]:+.3f}{flag(ac[1])} lag3={ac[2]:+.3f}{flag(ac[2])}"
          f"   (neg=mean-revert, pos=trend; * = beyond 2sigma)")
    print(f"  var-ratio VR2={vr[0]:.3f} VR4={vr[1]:.3f} VR8={vr[2]:.3f}   (<1 mean-revert, >1 trend, =1 random walk)")
    print(f"  all-signals base (bp/bar): FADE gross={b['fade_gross_bp']:+.3f} net={b['fade_net_bp']:+.3f} "
          f"(win{b['fade_win%']:.1f}%) | CONT gross={b['cont_gross_bp']:+.3f} net={b['cont_net_bp']:+.3f}")
    return ac[0], vr[0], b


def main():
    h1 = load_mt5_csv("data/vantage_usdjpy_h1.csv")
    print("USDJPY low-TF bar-level structure (Vantage feed = the trade feed). cost~0.6bp/side modeled.")
    tfs = [("1h", h1), ("2h", resample(h1, "2h")), ("4h", resample(h1, "4h")), ("8h", resample(h1, "8h"))]
    # try m5 too if the short m1 file exists
    try:
        m1 = load_mt5_csv("data/vantage_usdjpy_m1.csv")
        tfs = [("5m", resample(m1, "5min")), ("15m", resample(m1, "15min"))] + tfs
    except Exception as e:
        print(f"  (m1 unavailable: {e})")
    for name, d in tfs:
        screen(f"USDJPY {name}", d)
    print("\n  read: a notable NEG lag-1 autocorr or VR<1 = mean-reversion raw material (fade family worth a")
    print("        real test); POS/VR>1 = momentum; ~0/VR~1 + base<=0 net = random walk, NO low-TF family can")
    print("        extract an edge after cost. The base meanR must clear cost just to be worth mechanizing.")


if __name__ == "__main__":
    main()
