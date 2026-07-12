"""(B) Two-instrument 15m parallel book:
  gold15m = gold 15m Pattern-B breakout + daily SMA150+slope gate + ext-cap8% + RR4,
            pullback-limit frac0.25, net abs spread $0.6  (machinery: scratchpad/pullback_fixedtgt.py)
  btc15m  = BTC 15m Pattern-B breakout + daily KAMA(14)-rising gate + RR4,
            pullback-limit frac0.3, net $15  (machinery: scratchpad/btc15m_pullback_gauntlet.py)
Both regenerated via exec of the canonical setup (parity), with EXIT TIMES recorded for
overlap/day accounting. Outputs: anchor checks, annual+monthly R correlation, composite
trade-level equity (risk f per trade: single / equal-risk / inverse-vol), monthly-bootstrap
1-year fund-multiple distribution at f=1% and 2%, and same-day position-overlap share.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# ---------- regenerate GOLD 15m entries (canonical setup) ----------
gsrc = open("/home/angelbell/dev/auto-trade/scratchpad/pullback_fixedtgt.py").read()
gns = {}
exec(gsrc.split("m=eval_market()")[0], gns)   # entries, d, h, l, c, FWD, stats, eval_market ...

# ---------- regenerate BTC 15m entries (canonical setup) ----------
bsrc = open("/home/angelbell/dev/auto-trade/scratchpad/btc15m_pullback_gauntlet.py").read()
bns = {}
exec(bsrc.split("TFS = [")[0], bns)           # base, build, evaluate, net, stats, FWD, SPAN_YRS
bcell = bns["build"](bns["base"], 4.0, True)  # (df, E, h, l, c) 15m RR4 KAMA-gated


def eval_pull_ext(dfi, E, h, l, c, frac, spread, FWD):
    """Faithful copy of the canonical pullback walk (target-first = miss, same-bar stop),
    additionally recording exit time. Returns [(fill_time, exit_time, netR)]."""
    busy = -1; tr = []
    for tup in E:
        i, e, stop, tgt = tup[0], tup[1], tup[2], tup[3]
        if i <= busy: continue
        lim = e - frac * (e - stop)
        if lim <= stop or lim >= e: continue
        fill_j = None
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        risk = lim - stop; reward = tgt - lim
        if risk <= 0: continue
        if l[fill_j] <= stop: R = -1.0; exit_j = fill_j
        else:
            exit_j = min(fill_j + FWD, len(c) - 1); R = None
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt: R = reward / risk; exit_j = j; break
            if R is None: R = (c[exit_j] - lim) / risk
        tr.append((dfi.index[fill_j], dfi.index[exit_j], R - spread / risk))
        busy = exit_j
    return tr


# ---------- anchor checks ----------
def brief(tr, label):
    R = np.array([r for _, _, r in tr]); ts = [t for t, _, _ in tr]
    span = (ts[-1] - ts[0]).days / 365.25
    pf = R[R > 0].sum() / abs(R[R <= 0].sum())
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yr = np.array([t.year for t in ts]); yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    print(f"  {label:<26} n={len(R)} N/yr={len(R)/span:.1f} win={(R>0).mean()*100:.1f}% PF={pf:.2f} "
          f"meanR={R.mean():+.3f} IS/OOS={R[yr<half].mean():+.2f}/{R[yr>=half].mean():+.2f} ret/DD={R.sum()/dd:.2f}")
    return R, ts

print("ANCHOR CHECKS")
m = gns["eval_market"]()   # gold market baseline, COST=0.001 (relative) -- ledger anchor
sm = gns["stats"](m)
print(f"  gold market(COST.001)      n={sm['N']} win={sm['win']:.0f}% meanR={sm['meanR']:+.3f} ret/DD={sm['retdd']:.2f}"
      f"   [anchor: n354 / 30% / +0.133 / 1.56]")
gold = eval_pull_ext(gns["d"], gns["entries"], gns["h"], gns["l"], gns["c"], 0.25, 0.6, gns["FWD"])
brief(gold, "gold15m frac0.25 $0.6")
print(f"                             [anchor: meanR ~ +0.47]")
dfb, Eb, hb, lb, cb = bcell
btc = eval_pull_ext(dfb, Eb, hb, lb, cb, 0.3, 15.0, bns["FWD"])
brief(btc, "btc15m frac0.3 $15")
print(f"                             [anchor: N614 / 22.3% / +0.322 / PF1.37 / ret-DD5.25]")

legs = {"gold15m": gold, "btc15m": btc}

# ---------- common span ----------
t0 = max(min(t for t, _, _ in tr) for tr in legs.values())
t1 = min(max(t for _, t, _ in tr) for tr in legs.values())
yrs_common = (t1 - t0).days / 365.25
print(f"\nCOMMON SPAN: {t0.date()} -> {t1.date()}  ({yrs_common:.2f} yr)")
legs_c = {k: [x for x in v if t0 <= x[0] <= t1] for k, v in legs.items()}
for k, v in legs_c.items():
    print(f"  {k}: {len(v)} trades in common span")

# ---------- correlations (annual + monthly total R, zero-filled) ----------
def rser(tr, freq):
    s = pd.Series([r for _, _, r in tr], index=pd.DatetimeIndex([t for t, _, _ in tr]))
    return s.groupby(s.index.to_period(freq)).sum()

for freq, name in (("Y", "annual"), ("M", "monthly")):
    a, b = rser(legs_c["gold15m"], freq), rser(legs_c["btc15m"], freq)
    idx = pd.period_range(t0, t1, freq=freq)
    a, b = a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)
    print(f"  {name} R corr (gold15m vs btc15m): {np.corrcoef(a, b)[0,1]:+.2f}  (n={len(idx)})")
a, b = rser(legs_c["gold15m"], "Y"), rser(legs_c["btc15m"], "Y")
idx = pd.period_range(t0, t1, freq="Y")
print("  annual R  gold15m: " + " ".join(f"{p}:{v:+.0f}" for p, v in a.reindex(idx, fill_value=0.0).items()))
print("  annual R  btc15m : " + " ".join(f"{p}:{v:+.0f}" for p, v in b.reindex(idx, fill_value=0.0).items()))

# ---------- composite equity (trade-level, R applied at EXIT time) ----------
def combo_stats(weighted, f, label, show=True):
    """weighted = [(exit_time, w*R)] sorted by exit; risk f per unit weight."""
    weighted = sorted(weighted, key=lambda x: x[0])
    wr = np.array([r for _, r in weighted])
    eq = np.cumprod(1 + f * wr)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100
    cagr = (eq[-1] ** (1 / yrs_common) - 1) * 100
    pf = wr[wr > 0].sum() / abs(wr[wr <= 0].sum())
    if show:
        print(f"  {label:<28} PF={pf:.2f}  N/yr={len(wr)/yrs_common:5.1f}  CAGR={cagr:6.2f}%  "
              f"maxDD={dd:5.2f}%  CAGR/DD={cagr/dd:5.2f}")
    return weighted

# inverse-vol weights from per-leg monthly-R vol, scaled so w_gold + w_btc = 2 (same total as equal-risk)
vols = {}
for k, v in legs_c.items():
    mr = rser(v, "M").reindex(pd.period_range(t0, t1, freq="M"), fill_value=0.0)
    vols[k] = mr.std()
iv = {k: 1.0 / v for k, v in vols.items()}
sc = 2.0 / sum(iv.values())
iv = {k: v * sc for k, v in iv.items()}
print(f"\n  monthly-R vol: gold15m {vols['gold15m']:.2f}  btc15m {vols['btc15m']:.2f}"
      f"   -> inv-vol weights (sum=2): gold {iv['gold15m']:.2f} / btc {iv['btc15m']:.2f}")

print(f"\nCOMPOSITE (trade-level equity, common span, risk f=1%/trade per unit weight)")
streams = {}
streams["gold15m alone (w=1)"] = combo_stats([(e, r) for _, e, r in legs_c["gold15m"]], 0.01, "gold15m alone (w=1)")
streams["btc15m alone (w=1)"] = combo_stats([(e, r) for _, e, r in legs_c["btc15m"]], 0.01, "btc15m alone (w=1)")
streams["2-leg equal (1,1)"] = combo_stats(
    [(e, r) for _, e, r in legs_c["gold15m"]] + [(e, r) for _, e, r in legs_c["btc15m"]],
    0.01, "2-leg equal risk (1,1)")
streams["2-leg inv-vol"] = combo_stats(
    [(e, iv["gold15m"] * r) for _, e, r in legs_c["gold15m"]] + [(e, iv["btc15m"] * r) for _, e, r in legs_c["btc15m"]],
    0.01, f"2-leg inv-vol ({iv['gold15m']:.2f},{iv['btc15m']:.2f})")

# ---------- 1-year fund-multiple distribution (monthly bootstrap, 1000 draws) ----------
print("\n1-YEAR FUND MULTIPLE (monthly bootstrap x1000; months from common span, zero-filled)")
print(f"  {'stream':<28} {'f/tr':>5} {'median':>7} {'sd':>6} {'P(>=2x)':>8} {'P(<=0.5x)':>9} {'p10':>6} {'p90':>6}")
rng = np.random.default_rng(11)
midx = pd.period_range(t0, t1, freq="M")
for name, weighted in streams.items():
    s = pd.Series([r for _, r in weighted], index=pd.DatetimeIndex([t for t, _ in weighted]))
    for f in (0.01, 0.02):
        mret = s.groupby(s.index.to_period("M")).apply(lambda x: np.prod(1 + f * x.values) - 1)
        mret = mret.reindex(midx, fill_value=0.0).values
        draws = rng.choice(mret, size=(1000, 12), replace=True)
        mult = np.prod(1 + draws, axis=1)
        print(f"  {name:<28} {f*100:4.0f}% {np.median(mult):7.2f} {mult.std():6.2f} "
              f"{(mult >= 2).mean()*100:7.1f}% {(mult <= 0.5).mean()*100:8.1f}% "
              f"{np.percentile(mult,10):6.2f} {np.percentile(mult,90):6.2f}")

# ---------- same-day position overlap ----------
def posdays(tr):
    days = set()
    for fill, ex, _ in tr:
        for dts in pd.date_range(fill.normalize(), ex.normalize(), freq="D"):
            days.add(dts.date())
    return days

dg, db = posdays(legs_c["gold15m"]), posdays(legs_c["btc15m"])
both = dg & db; either = dg | db
cal = (t1 - t0).days + 1
print(f"\nPOSITION-DAY OVERLAP (common span, {cal} calendar days)")
print(f"  gold15m in-position days: {len(dg)} ({len(dg)/cal*100:.0f}%)   btc15m: {len(db)} ({len(db)/cal*100:.0f}%)")
print(f"  BOTH simultaneously: {len(both)} days = {len(both)/cal*100:.1f}% of calendar days"
      f" / {len(both)/len(either)*100:.1f}% of days-either-in-position")
