"""Ceiling diagnostics (NO new execution modes) for the 4 proposed execution improvements
to the two 15m breakout legs. Bounds them BEFORE implementing anything.

  D1 MISS  (案1 split / 案7 chase ceiling): in the frac-limit walk, trades whose price ran to
           target before the limit filled = the runaway winners we skip. What would they have
           earned at MARKET entry?  = the upside a split/chase could reclaim.
  D2 BUSY  (案2 re-entry / 案3 time-stop ceiling): entries discarded by the one-position
           busy_until lock. Their STANDALONE market meanR = the N we throw away to occupancy.
  D3 LAG   (案3 core): among TAKEN market trades, those NOT at +1R within K bars -> final meanR.
           ~0 = time-stop is free turnover; <0 = it also lifts PF; >0 = late bloomers, keep.

Faithful port of the two canonical walks (pullback_fixedtgt.py / btc15m_pullback_gauntlet.py).
GROSS (cost muddies a ceiling). market base must match canon or nothing is trusted.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
BO, FWD = 20, 500


def build_gold():
    d = load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg(AGG).dropna()
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    dc = d["close"].resample("1D").last().dropna(); sma = dc.rolling(150).mean()
    up = ((dc > sma) & (sma > sma.shift(10))).shift(1)
    reg = up.reindex(d.index, method="ffill").fillna(False).values
    ext = ((dc - sma) / sma * 100.0).shift(1); ext_arr = ext.reindex(d.index, method="ffill").values
    sw = swings_zigzag(h, l, a, 2.0)

    def fb(level, after):
        for j in range(after, min(after + BO, len(c))):
            if c[j] > level: return j
        return None
    E = []
    for t in range(2, len(sw)):
        (cL2, iL2, pL2, kL2), (cH1, iH1, pH1, kH1), (cL0, iL0, pL0, kL0) = sw[t], sw[t-1], sw[t-2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1): continue
        if pL2 <= pL0 or pH1 - pL0 <= 0: continue
        if not np.isnan(es[cL2]) and pH1 < es[cL2]: continue
        e_i = fb(pH1, cL2 + 1)
        if e_i is None: continue
        if not reg[e_i]: continue
        if not np.isnan(ext_arr[e_i]) and ext_arr[e_i] > 8: continue
        e = c[e_i]; stop = pL2; risk = e - stop
        if risk <= 0: continue
        tgt = e + 4.0 * risk
        if tgt <= e: continue
        E.append((e_i, e, stop, tgt))
    return dedupe(E), d.index, h, l, c


def build_btc():
    d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    dck = d["close"].resample("1D").last().dropna(); kmg = kama_adaptive(dck, 14)
    kreg = ((kmg > kmg.shift(1)).shift(1)).reindex(d.index, method="ffill").fillna(False).values
    sw = swings_zigzag(h, l, a, 2.0)

    def fb(level, after):
        for j in range(after, min(after + BO, len(c))):
            if c[j] > level: return j
        return None
    E = []
    for t in range(2, len(sw)):
        (cL2, iL2, pL2, kL2), (cH1, iH1, pH1, kH1), (cL0, iL0, pL0, kL0) = sw[t], sw[t-1], sw[t-2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1): continue
        if pL2 <= pL0 or pH1 - pL0 <= 0: continue
        if not np.isnan(es[cL2]) and pH1 < es[cL2]: continue
        e_i = fb(pH1, cL2 + 1)
        if e_i is None: continue
        if not kreg[e_i]: continue
        e = c[e_i]; stop = pL2; risk = e - stop
        if risk <= 0: continue
        tgt = e + 4.0 * risk
        E.append((e_i, e, stop, tgt))
    return dedupe(E), d.index, h, l, c


def dedupe(E):
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U


def market_R(e, stop, tgt, e_i, h, l, c):
    """standalone market outcome from bar e_i (enter at e=close of e_i, resolve i+1..)."""
    risk = e - stop; reward = tgt - e
    exit_j = min(e_i + FWD, len(c) - 1); R = None
    for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
        if l[j] <= stop: R = -1.0; exit_j = j; break
        if h[j] >= tgt: R = reward / risk; exit_j = j; break
    if R is None: R = (c[exit_j] - e) / risk
    return R, exit_j


def reached_1R_by(e, stop, e_i, h, l, K):
    """did high reach e+1*risk within K bars, before hitting stop? return final-R-classifier need."""
    risk = e - stop; one = e + risk
    for j in range(e_i + 1, min(e_i + 1 + K, len(h))):
        if l[j] <= stop: return False   # stopped before +1R
        if h[j] >= one: return True
    return False


def run(name, E, idx, h, l, c, frac):
    span = (idx[-1] - idx[0]).days / 365.25
    # ---- MARKET walk: taken trades + busy-skipped entries ----
    busy = -1; taken = []; skipped = []
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy:
            R, _ = market_R(e, stop, tgt, e_i, h, l, c)
            skipped.append((idx[e_i], R)); continue
        R, exit_j = market_R(e, stop, tgt, e_i, h, l, c)
        taken.append((idx[e_i], R, e, stop, tgt, e_i)); busy = exit_j
    tk = np.array([r for _, r, *_ in taken])
    print(f"\n===== {name} =====  span={span:.2f}yr  entries(armed)={len(E)}")
    print(f"  MARKET base: N={len(taken)} N/yr={len(taken)/span:.0f} win={ (tk>0).mean()*100:.0f}% "
          f"meanR={tk.mean():+.3f} totR={tk.sum():+.0f}  <-- must match canon")

    # ---- D1 MISS (frac-limit runaway winners) ----
    busy = -1; miss_mktR = []
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        lim = e - frac * (e - stop)
        if lim <= stop or lim >= e:
            continue
        fill_j = None; ran = False
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if h[j] >= tgt: ran = True; break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None:
            # this entry is either a runaway (ran to tgt) or never filled+never tgt within FWD
            R, exit_j = market_R(e, stop, tgt, e_i, h, l, c)
            if ran:  # missed a would-be runner
                miss_mktR.append(R)
            # advance busy the way the frac walk would (skip, no position) -> busy unchanged
            continue
        # filled: advance busy to its exit like the frac walk
        risk = lim - stop
        exit_j = min(fill_j + FWD, len(c) - 1)
        for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
            if l[j] <= stop: exit_j = j; break
            if h[j] >= tgt: exit_j = j; break
        busy = exit_j
    mm = np.array(miss_mktR)
    if len(mm):
        print(f"  D1 MISS  (frac{frac}): missed_runners={len(mm)}  ={len(mm)/(len(mm)+len(taken))*100:.0f}% of taken+miss "
              f"| market meanR={mm.mean():+.2f} totR={mm.sum():+.0f} ({mm.sum()/span:+.1f}/yr) win={(mm>0).mean()*100:.0f}%")
    else:
        print(f"  D1 MISS  (frac{frac}): none")

    # ---- D2 BUSY (occupancy) ----
    sk = np.array([r for _, r in skipped])
    if len(sk):
        yrs = pd.Series([t.year for t, _ in skipped])
        print(f"  D2 BUSY  skipped_by_occupancy={len(sk)} ({len(sk)/span:.0f}/yr) | standalone market "
              f"meanR={sk.mean():+.3f} totR={sk.sum():+.0f} win={(sk>0).mean()*100:.0f}%")
    else:
        print(f"  D2 BUSY  skipped=0")

    # ---- D3 LAG (laggard among taken) ----
    for K in (20, 50, 100):
        not_reached = [(r) for (_, r, e, stop, tgt, e_i) in taken
                       if not reached_1R_by(e, stop, e_i, h, l, K)]
        reached = [(r) for (_, r, e, stop, tgt, e_i) in taken
                   if reached_1R_by(e, stop, e_i, h, l, K)]
        nr = np.array(not_reached); rr = np.array(reached)
        print(f"  D3 LAG   K={K:>3}b: reached+1R n={len(rr)} meanR={rr.mean() if len(rr) else 0:+.2f} | "
              f"NOT-reached n={len(nr)} ({len(nr)/len(taken)*100:.0f}%) meanR={nr.mean() if len(nr) else 0:+.3f} "
              f"totR={nr.sum() if len(nr) else 0:+.0f}")


if __name__ == "__main__":
    Eg, ig, hg, lg, cg = build_gold()
    run("GOLD 15m (frac0.25)", Eg, ig, hg, lg, cg, 0.25)
    Eb, ib, hb, lb, cb = build_btc()
    run("BTC 15m (frac0.30)", Eb, ib, hb, lb, cb, 0.30)
