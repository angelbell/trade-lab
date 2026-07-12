"""だまし回避 / liquidity-hunt: the pL2 stop sits on the obvious higher-low where everyone's
stops pool -> hunted. Move the stop OFF that level. entry(lim) & target UNCHANGED (breakout
structure intact); ONLY the stop moves: pL2 (baseline) / pL2 - buf*ATR (cushion) / pL0 (deeper
structural). All trades kept (design change, NOT an n-trimming filter). Compare at fixed risk%
via ret/DD (unit-invariant); meanR is in each variant's own risk so read win/ret-DD/per-year.
Sweep the buffer for a plateau. Cost = realistic ($0.6 gold, $15 BTC)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
BO, FWD = 20, 500


def build_gold():
    d = load_mt5_csv("data/vantage_xauusd_m5.csv").resample("15min").agg(AGG).dropna()
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    dc = d["close"].resample("1D").last().dropna(); sma = dc.rolling(150).mean()
    up = ((dc > sma) & (sma > sma.shift(10))).shift(1)
    reg = up.reindex(d.index, method="ffill").fillna(False).values
    ext = ((dc - sma) / sma * 100.0).shift(1); ea = ext.reindex(d.index, method="ffill").values
    return _entries(d.index, o, h, l, c, a, es, swings_zigzag(h, l, a, 2.0), reg, ea)


def build_btc():
    d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    dck = d["close"].resample("1D").last().dropna(); kmg = kama_adaptive(dck, 14)
    reg = ((kmg > kmg.shift(1)).shift(1)).reindex(d.index, method="ffill").fillna(False).values
    return _entries(d.index, o, h, l, c, a, es, swings_zigzag(h, l, a, 2.0), reg, None)


def _entries(idx, o, h, l, c, a, es, sw, reg, ea):
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
        if ea is not None and not np.isnan(ea[e_i]) and ea[e_i] > 8: continue
        e = c[e_i]; risk = e - pL2
        if risk <= 0: continue
        E.append((e_i, e, pL2, pL0, e + 4.0 * risk, a[e_i]))
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U, idx, o, h, l, c


def play(e_i, e, pL2, stop, tgt, frac, sp, h, l, c):
    """entry limit uses pL2 (breakout risk); stop/risk uses `stop`. returns (R_net,exit_j,was_stop) or None."""
    lim = e - frac * (e - pL2)
    if lim <= stop: return None
    fill_j = None
    for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
        if h[j] >= tgt: return None
        if l[j] <= lim: fill_j = j; break
    if fill_j is None: return None
    u = lim - stop; rew = tgt - lim
    if l[fill_j] <= stop: return (-1.0 - sp / u, fill_j, True)
    xj = min(fill_j + FWD, len(c) - 1); R = None; ws = False
    for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
        if l[j] <= stop: R = -1.0; xj = j; ws = True; break
        if h[j] >= tgt: R = rew / u; xj = j; break
    if R is None: R = (c[xj] - lim) / u
    return (R - sp / u, xj, ws)


def walk(E, idx, o, h, l, c, frac, sp, stopmode, buf):
    busy = -1; tr = []; nstop = 0
    for (e_i, e, pL2, pL0, tgt, atre) in E:
        if e_i <= busy: continue
        if stopmode == "pL2": stop = pL2
        elif stopmode == "buf": stop = pL2 - buf * atre
        elif stopmode == "pL0": stop = pL0
        r = play(e_i, e, pL2, stop, tgt, frac, sp, h, l, c)
        if r is None: continue
        R, xj, ws = r; tr.append((idx[e_i], R)); busy = xj
        if ws: nstop += 1
    return tr, nstop


def st(tr, span):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
    return dict(N=len(R), win=(R > 0).mean()*100, pf=pf, meanR=R.mean(), totR=R.sum(),
                IS=R[yr < half].mean(), OOS=R[yr >= half].mean(),
                retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))


def peryr(tr):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    return " ".join(f"{y}:{R[yr==y].sum():+.0f}" for y in sorted(set(yr)))


def report(name, E, idx, o, h, l, c, frac, sp):
    span = (idx[-1] - idx[0]).days / 365.25
    print(f"\n===== {name}  frac={frac} cost=${sp} =====")
    print(f"  {'stop':>14}{'N':>5}{'stop%':>6}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'IS/OOS':>13}{'ret/DD':>8}{'grn':>6}")
    variants = [("pL2 (baseline)", "pL2", 0.0)] + \
               [(f"pL2-{b}ATR", "buf", b) for b in (0.25, 0.5, 1.0, 1.5)] + \
               [("pL0 (structural)", "pL0", 0.0)]
    out = {}
    for lab, mode, b in variants:
        tr, nstop = walk(E, idx, o, h, l, c, frac, sp, mode, b)
        s = st(tr, span); out[lab] = tr
        io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"; stoppct = nstop / s['N'] * 100 if s['N'] else 0
        print(f"  {lab:>14}{s['N']:>5}{stoppct:>5.0f}%{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}"
              f"{s['totR']:>+7.0f}{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")
    print(f"    per-year totR  baseline pL2: {peryr(out['pL2 (baseline)'])}")
    best = max(out, key=lambda k: st(out[k], span)['retdd'])
    print(f"    per-year totR  best({best}): {peryr(out[best])}")


if __name__ == "__main__":
    Eg, ig, og, hg, lg, cg = build_gold()
    report("GOLD 15m", Eg, ig, og, hg, lg, cg, 0.25, 0.6)
    Eb, ib, ob, hb, lb, cb = build_btc()
    report("BTC 15m", Eb, ib, ob, hb, lb, cb, 0.30, 15.0)
