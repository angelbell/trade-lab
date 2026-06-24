"""squeeze_breakout.py -- INDEPENDENT re-implementation + falsification of the user's
BTC 4H Squeeze Breakout claim (n=263, PF1.64, meanR+0.42, OOS+73.6 @RR3, long+short).

Written from the spec ONLY (no reuse of any external strategy code), so the reproduction is
a genuine check. Spec:
  - BTC H1 -> 4H. ATR(14)=SMA(TR,14). squeeze = ATR percentile-rank over L=120 of the PRIOR bar <=0.25.
  - break: close>Donchian-High(30) [rolling(30).max().shift(1)] long / close<Donchian-Low(30) short.
  - fill at the break bar's CLOSE. risk=1*ATR. SL=entry-+1ATR. TP=entry-+RR*ATR. SL priority intrabar.
  - time stop 60 bars (~10d) -> MTM. cost 0.1% round-trip (deducted in R). long+short.

Falsifiers (stated up front): (1) reproduce the headline numbers; (2) lookahead audit -- a causal vs
a deliberately-cheating (no-shift) variant must DIFFER and the causal one is the real number; (3)
per-year / rolling stability of FIXED params (no curve-fit); (4) robustness surface (ATR def, squeeze
thr, Donchian len, L, RR) must be a PLATEAU not a spike; (5) cost stress; (6) GOLD must stay dead
(BTC-specific claim). In-sample; live-forward arbitrates.

  .venv/bin/python research/squeeze_breakout.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import resample

SPLIT = 2022


def atr(d, n=14, wilder=False):
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean() if wilder else tr.rolling(n).mean()


def run(d, rr=3.0, atr_n=14, L=120, sqz=0.25, don=30, fwd=60, cost=0.001,
        wilder=False, side="both", no_overlap=True, cheat=False, hi_atr=False):
    """returns trades DataFrame[time,side,R]. cheat=True removes the causal shifts (lookahead probe).
    hi_atr=True selects the TOP sqz-quantile ATR (EXPANSION = what the user's inverted metric did)."""
    d = d.copy()
    a = atr(d, atr_n, wilder)
    rank = a.rolling(L).rank(pct=True)                      # percentile rank of CURRENT atr in trailing L
    gate = (rank >= 1 - sqz) if hi_atr else (rank <= sqz)   # hi_atr=expansion / else=squeeze(low atr)
    sq = gate if cheat else gate.shift(1).fillna(False)      # on PRIOR bar (causal)
    if cheat:
        dHi = d["high"].rolling(don).max(); dLo = d["low"].rolling(don).min()   # includes current bar = lookahead
    else:
        dHi = d["high"].rolling(don).max().shift(1); dLo = d["low"].rolling(don).min().shift(1)
    longsig = sq & (d["close"] > dHi)
    shortsig = sq & (d["close"] < dLo)
    H, Lw, C = d["high"].values, d["low"].values, d["close"].values
    av = a.values
    ls, ss = longsig.values, shortsig.values
    rows = []; last_exit = -1
    for i in range(len(d)):
        if i + 1 >= len(d) or np.isnan(av[i]) or av[i] <= 0:
            continue
        if no_overlap and i <= last_exit:
            continue
        is_long = ls[i] and side in ("both", "long")
        is_short = ss[i] and side in ("both", "short")
        if not (is_long or is_short):
            continue
        e = C[i]; risk = av[i]
        if is_long:
            sl, tp = e - risk, e + rr * risk
        else:
            sl, tp = e + risk, e - rr * risk
        end = min(i + 1 + fwd, len(d)); R = None
        for j in range(i + 1, end):
            if is_long:
                if Lw[j] <= sl: R = -1; break
                if H[j] >= tp: R = rr; break
            else:
                if H[j] >= sl: R = -1; break
                if Lw[j] <= tp: R = rr; break
        if R is None:
            R = ((C[end - 1] - e) if is_long else (e - C[end - 1])) / risk
            j = end - 1
        R -= cost * e / risk                               # 0.1% round-trip in R units
        rows.append((d.index[i], "L" if is_long else "S", R))
        last_exit = j
    return pd.DataFrame(rows, columns=["time", "side", "R"])


def stats(t):
    if len(t) == 0:
        return dict(n=0)
    w = t[t.R > 0].R.sum(); ll = -t[t.R < 0].R.sum()
    isr = t[t.time.dt.year < SPLIT].R.sum(); oos = t[t.time.dt.year >= SPLIT].R.sum()
    return dict(n=len(t), win=(t.R > 0).mean() * 100, meanR=t.R.mean(), totR=t.R.sum(),
                PF=w / max(ll, 1e-9), IS=isr, OOS=oos)


def line(tag, t):
    s = stats(t)
    if s["n"] == 0:
        print(f"  {tag:<34} n=0"); return
    print(f"  {tag:<34} n={s['n']:>4} win%={s['win']:>3.0f} meanR={s['meanR']:+5.2f} "
          f"totR={s['totR']:>+6.1f} PF={s['PF']:4.2f} | IS={s['IS']:+6.1f} OOS={s['OOS']:+6.1f}")


def main():
    d = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    g = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "4h")
    print(f"BTC 4h {d.index.min().date()}->{d.index.max().date()} ({len(d)} bars)")

    print("\n== 1. REPRODUCTION (RR3, long+short, cost0.1%) -- claim: n=263 PF1.64 meanR+0.42 OOS+73.6 ==")
    t = run(d, rr=3.0)
    line("BTC RR3 both (no-overlap)", t)
    line("  long only", t[t.side == "L"]); line("  short only", t[t.side == "S"])
    t_ov = run(d, rr=3.0, no_overlap=False)
    line("BTC RR3 both (allow overlap)", t_ov)

    print("\n== 2. LOOKAHEAD AUDIT -- causal vs deliberately-cheating (no shift). Cheat MUST be >> causal ==")
    line("causal (shifts ON)", run(d, rr=3.0))
    line("CHEAT (no shift=lookahead)", run(d, rr=3.0, cheat=True))

    print("\n== 3. PER-YEAR stability of FIXED params (both sides) ==")
    by = t.groupby([t.time.dt.year, t.side]).R.sum().unstack(fill_value=0)
    by["tot"] = by.sum(axis=1); by["n"] = t.groupby(t.time.dt.year).size()
    print(by.round(1).to_string())

    print("\n== 4. ROBUSTNESS surface (each varies ONE knob; real edge=plateau, not spike) ==")
    print(" RR sweep:");      [line(f"  RR{rr}", run(d, rr=rr)) for rr in (2, 2.5, 3, 3.5, 4)]
    print(" squeeze thr:");   [line(f"  sqz<={s}", run(d, rr=3, sqz=s)) for s in (0.15, 0.20, 0.25, 0.30, 0.40)]
    print(" Donchian len:");  [line(f"  don={n}", run(d, rr=3, don=n)) for n in (20, 25, 30, 40, 55)]
    print(" L window:");      [line(f"  L={x}", run(d, rr=3, L=x)) for x in (80, 100, 120, 150, 200)]
    print(" ATR def:");       line("  SMA(14)", run(d, rr=3)); line("  Wilder(14)", run(d, rr=3, wilder=True))
    print(" no-squeeze ctrl (plain Donchian breakout, NO squeeze gate):")
    t_ng = run(d, rr=3, sqz=1.0); line("  squeeze OFF (sqz<=1.0)", t_ng)

    print("\n== 5. COST STRESS (RR3 both) ==")
    [line(f"  cost={c*100:.2f}%", run(d, rr=3, cost=c)) for c in (0.001, 0.002, 0.004)]

    print("\n== 6. GOLD control -- claim says squeeze must be DEAD on gold (BTC-specific) ==")
    line("GOLD RR3 both", run(g, rr=3.0))
    line("GOLD squeeze OFF (plain BO)", run(g, rr=3.0, sqz=1.0))


if __name__ == "__main__":
    main()
