"""'5分で見てる' version: signal on 15m, but the pullback-limit FILL and the PRIOR-BAR-LOW TRAIL
are resolved on the underlying 5-MINUTE bars (trail = prior 5m bar low, ratchet up; exit when a
5m bar takes out the prior 5m low). Finer/tighter than the 15m-bar trail. Compares, on 5m
resolution: baseline (L2 + RR4) vs 5m-prevlow-trail (no tgt / + RR4 cap). Gold & BTC.
Cost $0.6 / $15. Conservative: prior-5m-low known before the bar; stop fills at the trail level."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
BO = 20
FWD5 = 1500     # ~5 trading days of 5m bars to resolve a trade


def prep(csv, gate, frac, sp, start=None):
    d5 = load_mt5_csv(csv)
    if start: d5 = d5.loc[start:]
    i5 = d5.index.values
    o5, h5, l5, c5 = (d5[x].values for x in ("open", "high", "low", "close"))
    d = d5.resample("15min").agg(AGG).dropna()
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    if gate == "sma":
        dc = d["close"].resample("1D").last().dropna(); sma = dc.rolling(150).mean()
        up = ((dc > sma) & (sma > sma.shift(10))).shift(1)
        reg = up.reindex(d.index, method="ffill").fillna(False).values
        ext = ((dc - sma) / sma * 100.0).shift(1); ea = ext.reindex(d.index, method="ffill").values
    else:
        dck = d["close"].resample("1D").last().dropna(); kmg = kama_adaptive(dck, 14)
        reg = ((kmg > kmg.shift(1)).shift(1)).reindex(d.index, method="ffill").fillna(False).values
        ea = None
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
        if ea is not None and not np.isnan(ea[e_i]) and ea[e_i] > 8: continue
        e = c[e_i]; risk = e - pL2
        if risk <= 0: continue
        E.append((d.index[e_i], e, pL2, e + 4.0 * risk))
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U, i5, o5, h5, l5, c5, frac, sp


def walk(prepped, mode, use_tgt):
    E, i5, o5, h5, l5, c5, frac, sp = prepped
    step = np.timedelta64(15, "m"); busy_t = np.datetime64("1900-01-01"); tr = []
    for (t_e, e, pL2, tgt) in E:
        if np.datetime64(t_e) <= busy_t: continue
        s = int(np.searchsorted(i5, np.datetime64(t_e) + step))   # first 5m bar after the 15m break bar
        lim = e - frac * (e - pL2)
        if lim <= pL2: continue
        # pullback-limit fill on 5m
        fill = None
        for j in range(s, min(s + FWD5, len(c5))):
            if h5[j] >= tgt: fill = None; break
            if l5[j] <= lim: fill = j; break
        if fill is None: continue
        u = lim - pL2; cur = pL2; trailing = (mode == "immediate"); R = None; xj = min(fill + FWD5, len(c5) - 1)
        if l5[fill] <= pL2:
            R = -1.0; xj = fill
        else:
            for k in range(fill + 1, min(fill + FWD5, len(c5))):
                if mode == "after1R" and not trailing and h5[k] >= lim + u:
                    trailing = True
                if trailing:
                    cur = max(cur, l5[k - 1])
                if l5[k] <= cur:
                    R = (cur - lim) / u; xj = k; break
                if use_tgt and h5[k] >= tgt:
                    R = (tgt - lim) / u; xj = k; break
            if R is None:
                R = (c5[xj] - lim) / u
        tr.append((pd.Timestamp(i5[fill]), R - sp / u)); busy_t = i5[xj]
    return tr


def st(tr, span):
    R = np.array([r for _, r in tr]); yr = np.array([t.year for t, _ in tr])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    eq = np.cumprod(1 + 0.01 * R); ddp = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]; grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
    return dict(N=len(R), win=(R > 0).mean()*100, pf=pf, meanR=R.mean(), totR=R.sum(),
                IS=R[yr < half].mean(), OOS=R[yr >= half].mean(), ddp=ddp, mx=R.max(),
                retdd=R.sum()/dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))


def report(name, prepped):
    span = (prepped[1][-1] - prepped[1][0]) / np.timedelta64(1, "D") / 365.25
    print(f"\n===== {name}  (15m signal, 5m fill+trail) span={span:.1f}yr =====")
    print(f"  {'variant':>24}{'N':>5}{'win':>5}{'PF':>6}{'meanR':>8}{'totR':>7}{'maxR':>6}{'IS/OOS':>13}{'maxDD%':>7}{'ret/DD':>8}{'grn':>6}")
    for tag, mode, ut in [("baseline L2+RR4 (5m)", "baseline", True),
                          ("5m prevlow trail, notgt", "immediate", False),
                          ("5m prevlow trail + RR4", "immediate", True),
                          ("5m trail after +1R", "after1R", False)]:
        s = st(walk(prepped, mode, ut), span); io = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"
        print(f"  {tag:>24}{s['N']:>5}{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}{s['totR']:>+7.0f}"
              f"{s['mx']:>6.1f}{io:>13}{s['ddp']:>6.1f}%{s['retdd']:>8.2f}{s['grn']:>3}/{s['ny']}")


if __name__ == "__main__":
    report("GOLD 15m", prep("data/vantage_xauusd_m5.csv", "sma", 0.25, 0.6))
    report("BTC 15m", prep("data/vantage_btcusd_m5.csv", "kama", 0.30, 15.0, start="2018-10-01"))
