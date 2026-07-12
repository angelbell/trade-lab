"""GOLD 5m breakout + pullback-limit rescue gauntlet.
Question: does the pullback-limit lever (fixed stop L2 + fixed far tgt RR4, entry lowered)
rescue a NATIVE-5m version of the gold 15m breakout canon (Pattern B / zz-k2 / trend-ema80 /
daily SMA150+slope10 / ext-cap8)? Prior verdict "gold 5m dead" predates this lever.
Faithful port of scratchpad/pullback_fixedtgt.py logic (exact canon reproduction verified).
Span restricted to TRUE M5 density (2018-09-13+; before that the file holds H1/daily bars).
No lookahead: confirmed close entry / limit fills intrabar / walk starts at i+1.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

AGG = {"open":"first","high":"max","low":"min","close":"last"}
BO, FWD = 20, 500
START = "2018-09-14"   # true M5 density begins 2018-09-13 21:20

full = load_mt5_csv("data/vantage_xauusd_m5.csv")
# daily gate from the FULL file (daily closes are valid back to 2007 -> SMA150 warm at START)
dc_full = full["close"].resample("1D").last().dropna()
sma_full = dc_full.rolling(150).mean()
up_full = ((dc_full > sma_full) & (sma_full > sma_full.shift(10))).shift(1)
ext_full = ((dc_full - sma_full) / sma_full * 100.0).shift(1)

def build(tf, rr):
    """returns dict with exec arrays + entry list [(i, e, stop, tgt, H1lvl)] on the true-M5 span."""
    d = full.loc[START:]
    if tf is not None:
        d = d.resample(tf).agg(AGG).dropna()
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], 14).values
    es = d["close"].ewm(span=80, adjust=False).mean().values
    reg = up_full.reindex(d.index, method="ffill").fillna(False).values
    ext_arr = ext_full.reindex(d.index, method="ffill").values
    sw = swings_zigzag(h, l, a, 2.0)
    def first_breakout(level, after):
        for j in range(after, min(after + BO, len(c))):
            if c[j] > level: return j
        return None
    entries = []
    for t in range(2, len(sw)):
        (cL2,iL2,pL2,kL2),(cH1,iH1,pH1,kH1),(cL0,iL0,pL0,kL0) = sw[t], sw[t-1], sw[t-2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1): continue
        if pL2 <= pL0 or pH1 - pL0 <= 0: continue
        if not np.isnan(es[cL2]) and pH1 < es[cL2]: continue
        e_i = first_breakout(pH1, cL2 + 1)
        if e_i is None: continue
        if not reg[e_i]: continue
        if not np.isnan(ext_arr[e_i]) and ext_arr[e_i] > 8: continue
        e = c[e_i]; stop = pL2; risk = e - stop
        if risk <= 0: continue
        tgt = e + rr * risk
        if tgt <= e: continue
        entries.append((e_i, e, stop, tgt, pH1))
    entries.sort(key=lambda x: x[0])
    seen = set(); uniq = []
    for en in entries:
        if en[0] in seen: continue
        seen.add(en[0]); uniq.append(en)
    return dict(d=d, h=h, l=l, c=c, atr=a, entries=uniq)

def stats(tr, span):
    if len(tr) < 10: return None
    R = np.array([r for _, r in tr]); ts = [t for t, _ in tr]
    yr = np.array([t.year for t in ts])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    grn = sum(1 for y in yrs if R[yr == y].sum() > 0)
    return dict(N=len(R), npy=len(R) / span, win=(R > 0).mean() * 100, pf=pf, meanR=R.mean(),
                totR=R.sum(), dd=dd, IS=R[yr < half].mean(), OOS=R[yr >= half].mean(),
                retdd=R.sum() / dd if dd > 0 else np.inf, grn=grn, ny=len(yrs))

def eval_market(B, spread, stop_slip=0.0):
    h, l, c, d, a = B["h"], B["l"], B["c"], B["d"], B["atr"]
    busy = -1; tr = []
    for (i, e, stop, tgt, _) in B["entries"]:
        if i <= busy: continue
        risk = e - stop; reward = tgt - e; exit_j = min(i + FWD, len(c) - 1); R = None
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if l[j] <= stop: R = -1.0 - stop_slip * a[i] / risk; exit_j = j; break
            if h[j] >= tgt: R = reward / risk; exit_j = j; break
        if R is None: R = (c[exit_j] - e) / risk
        R -= spread / risk; tr.append((d.index[i], R)); busy = exit_j
    return tr

def eval_pull(B, limfn, spread, stop_slip=0.0):
    """limfn(e, stop, H1lvl) -> limit price. stop=L2 & tgt fixed at market levels."""
    h, l, c, d, a = B["h"], B["l"], B["c"], B["d"], B["atr"]
    busy = -1; tr = []; miss = 0
    for (i, e, stop, tgt, H1) in B["entries"]:
        if i <= busy: continue
        lim = limfn(e, stop, H1)
        if lim <= stop or lim >= e: miss += 1; continue
        fill_j = None
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if h[j] >= tgt: break            # ran to target first = MISSED (adverse selection)
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: miss += 1; continue
        risk = lim - stop; reward = tgt - lim
        if risk <= 0: miss += 1; continue
        if l[fill_j] <= stop: R = -1.0 - stop_slip * a[i] / risk; exit_j = fill_j
        else:
            exit_j = min(fill_j + FWD, len(c) - 1); R = None
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                if l[j] <= stop: R = -1.0 - stop_slip * a[i] / risk; exit_j = j; break
                if h[j] >= tgt: R = reward / risk; exit_j = j; break
            if R is None: R = (c[exit_j] - lim) / risk
        R -= spread / risk; tr.append((d.index[fill_j], R)); busy = exit_j
    return tr, miss

def row(label, tr, span, miss=None):
    s = stats(tr, span)
    if s is None:
        print(f"  {label:<14} too few trades ({len(tr)})"); return None
    mp = f"{miss/(miss+len(tr))*100:>4.0f}%" if miss is not None else "   - "
    print(f"  {label:<14}{s['N']:>5}{s['npy']:>6.0f}{mp:>6}{s['win']:>5.0f}%{s['pf']:>6.2f}"
          f"{s['meanR']:>+8.3f}{s['IS']:>+7.2f}/{s['OOS']:>+.2f}{s['grn']:>4}/{s['ny']}"
          f"{s['dd']:>7.1f}{s['retdd']:>7.2f}")
    return s

HDR = f"  {'config':<14}{'N':>5}{'N/yr':>6}{'miss%':>6}{'win':>6}{'PF':>6}{'meanR':>8}{'IS/OOS':>13}{'grn':>6}{'DD(R)':>7}{'ret/DD':>7}"

B5r4  = build(None,    4.0)
B5r2  = build(None,    2.0)
B15r4 = build("15min", 4.0)
span5  = (B5r4["d"].index[-1]  - B5r4["d"].index[0]).days / 365.25
span15 = (B15r4["d"].index[-1] - B15r4["d"].index[0]).days / 365.25
print(f"span: {B5r4['d'].index[0].date()} -> {B5r4['d'].index[-1].date()} ({span5:.2f} yr, true-M5 only)")
print(f"raw signal count: 5m RR4={len(B5r4['entries'])}  15m RR4={len(B15r4['entries'])}")

# ---- cost/risk distribution: the crux (how much does the stop shrink at 5m?) ----
for nm, B in (("5m", B5r4), ("15m", B15r4)):
    rk = np.array([e - s for (_, e, s, _, _) in B["entries"]])
    for sp in (0.6,):
        cr = sp / rk
        print(f"  {nm:>4} stop-width$: med={np.median(rk):.2f} p10={np.percentile(rk,10):.2f}  "
              f"cost/risk @$0.6: med={np.median(cr)*100:.1f}% p90={np.percentile(cr,90)*100:.1f}%")

print("\n==== A. BASE market execution (gross + net absolute $/rt) ====")
print(HDR)
for sp in (0.0, 0.2, 0.3, 0.4, 0.3):
    row(f"5m RR4 ${sp}", eval_market(B5r4, sp), span5)
for sp in (0.0, 0.3):
    row(f"5m RR2 ${sp}", eval_market(B5r2, sp), span5)
for sp in (0.0, 0.3):
    row(f"15m RR4 ${sp}", eval_market(B15r4, sp), span15)

print("\n==== B. PULLBACK-LIMIT rescue @ $0.3 (real RAW cost) (frac sweep + structural anchor = broken-level retest) ====")
FRACS = (0.2, 0.25, 0.3, 0.38, 0.5)
print(HDR)
res5 = {}
for f in FRACS:
    tr, ms = eval_pull(B5r4, lambda e, s, H, f=f: e - f * (e - s), 0.3)
    res5[f"frac{f}"] = tr; row(f"5m frac{f}", tr, span5, ms)
trH, msH = eval_pull(B5r4, lambda e, s, H: H, 0.3)
res5["H1lvl"] = trH
dep = [(e - H) / (e - s) for (_, e, s, _, H) in B5r4["entries"]]
row("5m lvl-retest", trH, span5, msH)
print(f"  (broken-level retest depth as frac of risk: med={np.median(dep):.2f})")
print("  -- 15m reference on the SAME span --")
for f in (0.25, 0.3):
    tr, ms = eval_pull(B15r4, lambda e, s, H, f=f: e - f * (e - s), 0.3)
    row(f"15m frac{f}", tr, span15, ms)
tr, ms = eval_pull(B15r4, lambda e, s, H: H, 0.3)
row("15m lvl-retest", tr, span15, ms)

print("\n==== C. COST LADDER for 5m (meanR / ret-DD) ====")
print(f"  {'$/rt':>6}{'market':>16}{'frac0.25':>16}{'frac0.3':>16}{'lvl-retest':>16}")
for sp in (0.0, 0.2, 0.3, 0.4, 0.3):
    cells = [stats(eval_market(B5r4, sp), span5)]
    for fn in (lambda e, s, H: e - 0.25 * (e - s), lambda e, s, H: e - 0.3 * (e - s), lambda e, s, H: H):
        cells.append(stats(eval_pull(B5r4, fn, sp)[0], span5))
    line = f"  {sp:>6.1f}"
    for s_ in cells:
        line += f"   {s_['meanR']:+.3f}/{s_['retdd']:>6.2f}" if s_ else "        n/a    "
    print(line)

print("\n==== D. STOP-SLIP sensitivity (loss = -1 - 0.27*ATR(entry)/risk) @ $0.3 (real RAW cost) ====")
print(HDR)
row("5m mkt +slip", eval_market(B5r4, 0.3, stop_slip=0.27), span5)
tr, ms = eval_pull(B5r4, lambda e, s, H: e - 0.25 * (e - s), 0.3, stop_slip=0.27)
row("5m f.25 +slip", tr, span5, ms)
tr, ms = eval_pull(B5r4, lambda e, s, H: H, 0.3, stop_slip=0.27)
row("5m lvl +slip", tr, span5, ms)

print("\n==== E. PER-YEAR meanR(n) ====")
def peryear(tr, label):
    R = np.array([r for _, r in tr]); ts = [t for t, _ in tr]
    yr = np.array([t.year for t in ts]); ys = sorted(set(yr))
    print(f"  {label:<12}" + " ".join(f"{y}:{R[yr==y].mean():+.2f}({(yr==y).sum()})" for y in ys))
peryear(eval_market(B5r4, 0.3), "5m market")
peryear(res5["frac0.25"], "5m frac0.25")
peryear(res5["H1lvl"], "5m lvl-rt")

print("\n==== F. BETA NULL: same-frequency random long @ $0.3 (real RAW cost) (200 trials) ====")
from research.edge_harness import _walk
def beta_null(B, real_tr, rr, spread, trials=200, seed=0):
    h, l, c, a, d = B["h"], B["l"], B["c"], B["atr"], B["d"]
    Nreal = len(real_tr); real_mean = np.mean([r for _, r in real_tr])
    katr = np.median([(e - s) / a[i] for (i, e, s, _, _) in B["entries"] if a[i] > 0 and not np.isnan(a[i])])
    rng = np.random.default_rng(seed); lo = 200; hi = len(c) - FWD - 2
    means = []
    for _ in range(trials):
        idx = np.sort(rng.choice(np.arange(lo, hi), size=Nreal * 3, replace=False))
        e = c[idx]; risk = katr * a[idx]
        ok = risk > 0
        idx, e, risk = idx[ok], e[ok], risk[ok]
        R = _walk(idx + 1, np.ones(len(idx)), e, e - risk, e + rr * risk, risk,
                  h, l, c, spread, 0.0, FWD)
        R = R[~np.isnan(R)][:Nreal]
        means.append(R.mean())
    means = np.array(means)
    pct = (means < real_mean).mean() * 100
    print(f"    real meanR={real_mean:+.3f} vs null med={np.median(means):+.3f} sd={means.std():.3f}"
          f" -> real at {pct:.0f}%ile of random-long")
print("  5m market:"); beta_null(B5r4, eval_market(B5r4, 0.3), 4.0, 0.3)
print("  5m frac0.25:"); beta_null(B5r4, res5["frac0.25"], 4.0, 0.3)
print("  5m lvl-retest:"); beta_null(B5r4, res5["H1lvl"], 4.0, 0.3)

print("\n==== G. HOUR-OF-DAY (server clock) meanR(n), 5m market @ $0.3 (real RAW cost) + drop 9-15 effect ====")
tr = eval_market(B5r4, 0.3); R = np.array([r for _, r in tr]); hh = np.array([t.hour for t, _ in tr])
print("  " + " ".join(f"{H:02d}:{R[hh==H].mean():+.2f}({(hh==H).sum()})" for H in range(24) if (hh == H).sum() > 0))
keep = ~((hh >= 9) & (hh < 15))
print(f"  drop 9-15: meanR {R.mean():+.3f} -> {R[keep].mean():+.3f} (n {len(R)}->{keep.sum()})")

print("\n==== H. OVERLAP with the 15m candidate (independence check) ====")
t5 = pd.DatetimeIndex([t for t, _ in res5["frac0.25"]])
tr15, _ = eval_pull(B15r4, lambda e, s, H: e - 0.25 * (e - s), 0.3)
t15 = pd.DatetimeIndex([t for t, _ in tr15])
if len(t5) and len(t15):
    near = sum(1 for t in t5 if ((t15 >= t - pd.Timedelta("2h")) & (t15 <= t + pd.Timedelta("2h"))).any())
    print(f"  5m frac0.25 fills within +/-2h of a 15m frac0.25 fill: {near}/{len(t5)} = {near/len(t5)*100:.0f}%")
