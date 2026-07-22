"""BREAK the chase (案7) result. Rebuild WITH open[] for realistic next-bar-open fill.
Checks: (1) close-fill vs next-open fill (remove entry optimism), (2) CHASED-SUBSET standalone
meanR + per-year (kaizenan kill condition: chased subset meanR<0 = dead), (3) cost stress.
Build faithfully duplicates the canon (already verified: gold N354/market+0.342, BTC N657/+0.175)."""
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
    sw = swings_zigzag(h, l, a, 2.0)
    return _entries(d.index, o, h, l, c, es, sw, reg, ea)


def build_btc():
    d = load_mt5_csv("data/vantage_btcusd_m15.csv").loc["2018-10-01":]
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    dck = d["close"].resample("1D").last().dropna(); kmg = kama_adaptive(dck, 14)
    reg = ((kmg > kmg.shift(1)).shift(1)).reindex(d.index, method="ffill").fillna(False).values
    sw = swings_zigzag(h, l, a, 2.0)
    return _entries(d.index, o, h, l, c, es, sw, reg, None)


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
        E.append((e_i, e, stop, e + 4.0 * risk))
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U, idx, o, h, l, c


def chase(E, idx, o, h, l, c, frac, sp, nextopen=True):
    """CAUSAL: scan e_i+1.. in TIME ORDER. Whichever comes FIRST decides the trade:
       - low<=lim  -> limit fills (pullback trade), OR
       - close>break-close -> momentum continuation, CHASE at next open (decision made NOW,
         no knowledge of whether tgt is later reached; the trade can still reverse to stop).
       - high>=tgt before either (single-bar runaway) -> never entered, skip.
    returns (all_trades, chased_trades)."""
    busy = -1; allt = []; chst = []
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        lim = e - frac * (e - stop)
        ep = None; eb = None; is_chase = False
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if l[j] <= lim:                         # pullback fills the limit first
                ep, eb = lim, j; break
            if c[j] > c[e_i]:                       # higher close first -> chase (CAUSAL)
                if j + 1 >= len(c): break
                ep, eb, is_chase = o[j + 1], j + 1, True; break
            if h[j] >= tgt:                         # ran to tgt before either -> not entered
                break
        if ep is None: continue
        u = ep - stop
        if u <= 0: continue
        rew = tgt - ep; xj = min(eb + FWD, len(c) - 1); R = None
        for j in range(eb + 1, min(eb + 1 + FWD, len(c))):
            if l[j] <= stop: R = -1.0; xj = j; break
            if h[j] >= tgt: R = rew / u; xj = j; break
        if R is None: R = (c[xj] - ep) / u
        Rn = R - sp / u
        allt.append((idx[eb], Rn)); busy = xj
        if is_chase: chst.append((idx[eb], Rn))
    return allt, chst


def limit_only(E, idx, o, h, l, c, frac, sp):
    busy = -1; tr = []
    for (e_i, e, stop, tgt) in E:
        if e_i <= busy: continue
        lim = e - frac * (e - stop)
        if lim <= stop: continue
        fill_j = None
        for j in range(e_i + 1, min(e_i + 1 + FWD, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        u = lim - stop; rew = tgt - lim
        if l[fill_j] <= stop: R = -1.0; xj = fill_j
        else:
            xj = min(fill_j + FWD, len(c) - 1); R = None
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                if l[j] <= stop: R = -1.0; xj = j; break
                if h[j] >= tgt: R = rew / u; xj = j; break
            if R is None: R = (c[xj] - lim) / u
        tr.append((idx[fill_j], R - sp / u)); busy = xj
    return tr


def st(tr, span):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
    return dict(N=len(R), npy=len(R)/span, win=(R > 0).mean()*100, pf=pf, meanR=R.mean(),
                totR=R.sum(), IS=R[yr < half].mean(), OOS=R[yr >= half].mean(),
                retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))


def peryr(tr):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    return " ".join(f"{y}:{R[yr==y].sum():+.0f}(n{(yr==y).sum()})" for y in sorted(set(yr)))


def run(name, E, idx, o, h, l, c, frac, sp):
    span = (idx[-1] - idx[0]).days / 365.25
    print(f"\n===== {name}  frac={frac} cost=${sp} =====")
    print(f"  {'variant':>22}{'N':>5}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'IS/OOS':>13}{'ret/DD':>8}{'grn':>6}")
    def prow(tag, tr):
        s = st(tr, span); io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
        print(f"  {tag:>22}{s['N']:>5}{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}{s['totR']:>+7.0f}"
              f"{io:>13}{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")
    prow("limit only", limit_only(E, idx, o, h, l, c, frac, sp))
    an, chn = chase(E, idx, o, h, l, c, frac, sp); prow("chase (causal,next-open)", an)
    if chn:
        print(f"  CHASED SUBSET (causal): n={len(chn)}  meanR={np.mean([r for _,r in chn]):+.3f}  "
              f"win={np.mean([r>0 for _,r in chn])*100:.0f}%  totR={np.sum([r for _,r in chn]):+.0f}")
        print(f"    per-year: {peryr(chn)}")


if __name__ == "__main__":
    Eg, ig, og, hg, lg, cg = build_gold()
    for sp in (0.6, 1.2):
        run("GOLD 15m", Eg, ig, og, hg, lg, cg, 0.25, sp)
    Eb, ib, ob, hb, lb, cb = build_btc()
    for sp in (15.0, 25.0):
        run("BTC 15m", Eb, ib, ob, hb, lb, cb, 0.30, sp)
