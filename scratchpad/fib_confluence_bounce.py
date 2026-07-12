"""fib_confluence_bounce.py -- mechanization of the user's discretionary method:
HH/LL structure lines + ZigZag-impulse Fib retracement; where they OVERLAP (confluence),
follow the HTF trend / follow the bounce after a reaction-confirmation close.

Event (causal, long side; mirror short):
  impulse   : confirmed ZigZag L0->H1 (upward), H1 confirm bar = start of watch window
  fib zone  : H1 - f*(H1-L0), f in {0.382, 0.5, 0.618}
  structure : any PRIOR confirmed swing low within tol*ATR of the fib level -> confluence
  HTF trend : 4h EMA-stack direction (stack>0 long / <0 short), confirmed bar, ffill
  trigger   : price's low touches the zone, then a bar CLOSES back above it within the
              watch window (reaction confirmation) -> event at that close bar
STEP1: +-1 ATR barrier race from the event close vs SAME-HOUR beta. Cells: fib-only /
confluence / structure-only, HTF-trend on/off. CONTROL: same code on gold 15m (bounce
edge exists there; if confluence adds nothing on gold it adds nothing anywhere).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag
from radar_gate_race import comps_tf

FIBS = (0.382, 0.5, 0.618)
TOL = 0.5          # structure-confluence tolerance in ATRs
WATCH = 96         # bars to wait for touch+reclaim after H1 confirm
K_RACE = 96


def screen(name, d):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c)
    span = (d.index[-1] - d.index[0]).days / 365.25
    up_l, dn_l = c + atr, c - atr
    t_up = np.full(n, K_RACE + 1, np.int32)
    t_dn = np.full(n, K_RACE + 1, np.int32)
    for k in range(1, K_RACE + 1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n - k], ls[:n - k] = h[k:], l[k:]
        hs[n - k:], ls[n - k:] = -np.inf, np.inf
        t_up = np.where((t_up > K_RACE) & (hs >= up_l), k, t_up)
        t_dn = np.where((t_dn > K_RACE) & (ls <= dn_l), k, t_dn)
    win_long = (np.minimum(t_up, t_dn) <= K_RACE) & (t_up < t_dn)
    win_short = (np.minimum(t_up, t_dn) <= K_RACE) & (t_dn < t_up)
    valid = ~np.isnan(atr) & (np.arange(n) < n - K_RACE)
    hours = d.index.hour.values
    bL = {hh: win_long[valid & (hours == hh)].mean() for hh in range(24)}
    bS = {hh: win_short[valid & (hours == hh)].mean() for hh in range(24)}

    C4 = comps_tf(d, "240min")
    trend_up = C4["stack"] > 0
    trend_dn = C4["stack"] < 0

    sw = swings_zigzag(h, l, np.where(np.isnan(atr), np.nanmean(atr), atr), 2.0)
    lows = [(cc, pp) for cc, ii, pp, kk in sw if kk == -1]   # (confirm_bar, price)
    highs = [(cc, pp) for cc, ii, pp, kk in sw if kk == 1]

    def collect(side):
        """side='long': up-impulse L0->H1, fade-follow the pullback bounce."""
        evs = []
        for t in range(1, len(sw)):
            cH, iH, pH, kH = sw[t]
            cL, iL, pL, kL = sw[t - 1]
            if side == "long" and not (kH == 1 and kL == -1 and pH > pL):
                continue
            if side == "short" and not (kH == -1 and kL == 1 and pH < pL):
                continue
            if cH >= n - K_RACE or np.isnan(atr[cH]) or atr[cH] <= 0:
                continue
            rng_ = abs(pH - pL)
            pool = lows if side == "long" else highs
            prior = [p for cc, p in pool if cc < cH]
            for f in FIBS:
                z = pH - f * rng_ if side == "long" else pH + f * rng_
                conf = any(abs(p - z) <= TOL * atr[cH] for p in prior[-30:])
                touched = False
                for j in range(cH + 1, min(cH + WATCH, n - K_RACE)):
                    if side == "long":
                        if l[j] <= z:
                            touched = True
                        if touched and c[j] > z:
                            evs.append((j, conf, f)); break
                        if l[j] < (pL if side == "long" else pH):
                            break               # impulse origin broken -> setup void
                    else:
                        if h[j] >= z:
                            touched = True
                        if touched and c[j] < z:
                            evs.append((j, conf, f)); break
                        if h[j] > pH:
                            break
        return evs

    print(f"\n===== {name} ({span:.1f}yr) fib-bounce reaction, race +-1ATR {K_RACE} bars =====")
    for side in ("long", "short"):
        evs = collect(side)
        if len(evs) < 30:
            print(f"  [{side}] n={len(evs)} too few"); continue
        idx = np.array([e[0] for e in evs])
        cf = np.array([e[1] for e in evs])
        win = win_long if side == "long" else win_short
        bmap = bL if side == "long" else bS
        tr_ok = (trend_up if side == "long" else trend_dn)[idx]
        for tag, m in [("fib全体", np.ones(len(idx), bool)),
                       ("合流のみ (fib∩構造線)", cf), ("fib単独 (合流なし)", ~cf),
                       ("合流∩4hトレンド順", cf & tr_ok),
                       ("fib全体∩4hトレンド順", tr_ok)]:
            if m.sum() < 30:
                print(f"  [{side}] {tag:<22} n={m.sum()} too few"); continue
            w = win[idx[m]]
            b = np.mean([bmap[hh] for hh in hours[idx[m]]])
            print(f"  [{side}] {tag:<22} n={m.sum():5d} N/yr={m.sum()/span:4.0f}  win={w.mean()*100:4.1f}%"
                  f"  beta={b*100:4.1f}%  diff={(w.mean()-b)*100:+4.1f}pt")


def main():
    jp = load_mt5_csv("data/vantage_usdjpy_m15.csv")
    cnt = jp.groupby(jp.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= 80]
    screen("USDJPY 15m", jp[jp.index.date >= ok.index[0]])
    gd = load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":]
    screen("GOLD 15m (control)", gd.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())


if __name__ == "__main__":
    main()
