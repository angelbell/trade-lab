"""案2 SHAKEOUT RE-ENTRY. After a pullback-limit trade STOPS OUT (-1R), if within M bars price
re-reclaims the SAME break level (confirmed close > pH1), re-enter via the SAME pullback-limit
(same structural stop pL2, same fixed target level). Sequential (slot already freed) -> no added
concurrency. kaizenan kill condition: the re-entry SUBSET meanR < 0 = dead. Report subset first.
Build faithfully duplicates canon (verified: gold N354/market+0.342, BTC N657/+0.175)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas_ta as ta
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
    return _entries(d.index, o, h, l, c, es, swings_zigzag(h, l, a, 2.0), reg, ea)


def build_btc():
    d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    dck = d["close"].resample("1D").last().dropna(); kmg = kama_adaptive(dck, 14)
    reg = ((kmg > kmg.shift(1)).shift(1)).reindex(d.index, method="ffill").fillna(False).values
    return _entries(d.index, o, h, l, c, es, swings_zigzag(h, l, a, 2.0), reg, None)


def _entries(idx, o, h, l, c, es, sw, reg, ea):
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
        e = c[e_i]; stop = pL2; risk = e - stop
        if risk <= 0: continue
        E.append((e_i, e, stop, e + 4.0 * risk, pH1))
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U, idx, o, h, l, c


def play_limit(e_i, e, stop, tgt, frac, sp, h, l, c):
    """pullback-limit attempt. returns (R_net, exit_j, was_stop, entry_j) or None (missed/no-fill)."""
    lim = e - frac * (e - stop)
    if lim <= stop or lim >= e: return None
    fill_j = None
    for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
        if h[j] >= tgt: return None                 # ran to tgt without pulling back = miss
        if l[j] <= lim: fill_j = j; break
    if fill_j is None: return None
    u = lim - stop; rew = tgt - lim
    if l[fill_j] <= stop: return (-1.0 - sp / u, fill_j, True, fill_j)
    xj = min(fill_j + FWD, len(c) - 1); R = None; was_stop = False
    for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
        if l[j] <= stop: R = -1.0; xj = j; was_stop = True; break
        if h[j] >= tgt: R = rew / u; xj = j; break
    if R is None: R = (c[xj] - lim) / u
    return (R - sp / u, xj, was_stop, fill_j)


def walk(E, idx, o, h, l, c, frac, sp, M):
    """M=None -> pure-limit baseline. Else add one shakeout re-entry per stopped-out trade."""
    busy = -1; allt = []; ret = []
    for (e_i, e, stop, tgt, pH1) in E:
        if e_i <= busy: continue
        r = play_limit(e_i, e, stop, tgt, frac, sp, h, l, c)
        if r is None: continue
        R, xj, was_stop, ej = r
        allt.append((idx[ej], R)); busy = xj
        if M is None or not was_stop: continue
        # shakeout re-entry: first confirmed re-reclaim of pH1 within M bars after the stopout
        y = None
        for j in range(xj + 1, min(xj + 1 + M, len(c))):
            if c[j] > pH1: y = j; break
        if y is None or y <= busy: continue
        r2 = play_limit(y, c[y], stop, tgt, frac, sp, h, l, c)   # SAME stop & fixed tgt level
        if r2 is None: continue
        R2, xj2, _, ej2 = r2
        allt.append((idx[ej2], R2)); ret.append((idx[ej2], R2)); busy = xj2
    return allt, ret


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
    return " ".join(f"{y}:{R[yr==y].sum():+.0f}(n{(yr==y).sum()})" for y in sorted(set(yr)))


def run(name, E, idx, o, h, l, c, frac, sp):
    span = (idx[-1] - idx[0]).days / 365.25
    print(f"\n===== {name}  frac={frac} cost=${sp} =====")
    print(f"  {'variant':>18}{'N':>5}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'IS/OOS':>13}{'ret/DD':>8}{'grn':>6}")
    def prow(tag, tr):
        s = st(tr, span); io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
        print(f"  {tag:>18}{s['N']:>5}{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}{s['totR']:>+7.0f}"
              f"{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")
    base, _ = walk(E, idx, o, h, l, c, frac, sp, None); prow("pure-limit", base)
    for M in (20, 50, 100):
        allt, ret = walk(E, idx, o, h, l, c, frac, sp, M)
        prow(f"+reentry M={M}", allt)
        if ret:
            R = np.array([r for _, r in ret])
            print(f"      RE-ENTRY SUBSET M={M}: n={len(R)} win={ (R>0).mean()*100:.0f}% meanR={R.mean():+.3f} "
                  f"totR={R.sum():+.0f}  per-yr: {peryr(ret)}")


if __name__ == "__main__":
    Eg, ig, og, hg, lg, cg = build_gold()
    run("GOLD 15m", Eg, ig, og, hg, lg, cg, 0.25, 0.6)
    Eb, ib, ob, hb, lb, cb = build_btc()
    run("BTC 15m", Eb, ib, ob, hb, lb, cb, 0.30, 15.0)
