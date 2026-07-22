"""Full falsification gauntlet for the BTC 4h deep-Fib bounce STANDALONE leg.
Reuses the faithful mechanization (bounce_fib logic). Reports numbers only; human decides.
Checklist: 1 plateau(frac) 2 win-vs-breakeven excess 3 IS/OOS(2022) 4 per-year 5 cost/slip stress
6 overfit(DSR/PBO/boot/null via edge_harness.audit) 7 BETA NULL (random-long same-RR same-regime)
8 corr vs book legs. Compare on CAGR/DD (portfolio_kama.cagr_dd), report PF+N+N/yr+maxDD every time."""
import sys, os; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive
from research.portfolio_kama import cagr_dd, get_legs
from research.edge_harness import audit
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
W, FWD = 40, 300

# ---- load + build gated uptrend impulses once ----
DF = load_mt5_csv("data/vantage_btcusd_h1.csv").resample("240min").agg(AGG).dropna()
h, l, c = DF["high"].values, DF["low"].values, DF["close"].values
a = ta.atr(DF["high"], DF["low"], DF["close"], 14).values
es = DF["close"].ewm(span=80, adjust=False).mean().values
dck = DF["close"].resample("1D").last().dropna(); kmg = kama_adaptive(dck, 14)
G = ((kmg > kmg.shift(1)).shift(1)).reindex(DF.index, method="ffill").fillna(False).values
SW = swings_zigzag(h, l, a, 2.0)
IMPS = []
for t in range(1, len(SW)):
    cH, iH, pH, kH = SW[t]; cL, iL, pL, kL = SW[t - 1]
    if kH != +1 or kL != -1 or pH - pL <= 0: continue
    if es is not None and not np.isnan(es[cH]) and pH < es[cH]: continue
    IMPS.append((cH, pL, pH))
N_BARS = len(c)
IDX = DF.index


def bounce(frac=0.786, spread=15.0, sslip=0.5):
    """returns DataFrame(time,R) of the deep-Fib bounce trades (net of cost+slip)."""
    busy = -1; rows = []
    for (cH, L0, H1) in IMPS:
        if cH <= busy: continue
        lim = H1 - frac * (H1 - L0)
        if lim <= L0 or not G[min(cH, N_BARS - 1)]: continue
        fj = None
        for j in range(cH + 1, min(cH + 1 + W, N_BARS)):
            if l[j] <= L0: break
            if l[j] <= lim: fj = j; break
        if fj is None: continue
        entry = lim; risk = entry - L0; reward = H1 - entry
        if risk <= 0 or reward <= 0: continue
        xj = min(fj + FWD, N_BARS - 1); r = None
        for j in range(fj + 1, min(fj + 1 + FWD, N_BARS)):
            if l[j] <= L0: r = -1.0 - sslip * (L0 - l[j]) / risk; xj = j; break
            if h[j] >= H1: r = reward / risk; xj = j; break
        if r is None: r = (c[xj] - entry) / risk
        r -= spread / risk
        rows.append((IDX[fj], r, risk, frac / (1 - frac)))  # keep risk$ + RR for the null
        busy = xj
    return pd.DataFrame(rows, columns=["time", "R", "risk$", "RR"])


def card(name, t):
    n = len(t); R = t.R.values
    win = (R > 0).mean() * 100
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    yrs = t.time.dt.year
    span = (t.time.iloc[-1] - t.time.iloc[0]).days / 365.25
    cg, dd, cdd, ret = cagr_dd(t.assign(R=R))
    # IS/OOS at 2022
    is_ = R[yrs < 2022]; oos = R[yrs >= 2022]
    gy = t.groupby(yrs)["R"].sum()
    green = (gy > 0).sum()
    print(f"  {name}: N={n} N/yr={n/span:.1f} PF={pf:.2f} win={win:.0f}% meanR={R.mean():+.3f} "
          f"IS/OOS(2022)={is_.mean():+.2f}/{oos.mean():+.2f} green={green}/{len(gy)} "
          f"maxDD={dd:.1f}% CAGR={cg:+.0f}% CAGR/DD={cdd:.2f}")
    return dict(n=n, R=R, gy=gy, cdd=cdd, dd=dd, span=span)


print("=" * 70)
print("VERDICT CARDS (net cost $15 + slip0.5)")
print("=" * 70)
info = {}
for fr in (0.7, 0.786):
    info[fr] = card(f"frac={fr}", bounce(fr))

FLAG = 0.786
BT = bounce(FLAG)
print(f"\n[flagship frac={FLAG}: RR={FLAG/(1-FLAG):.2f}, breakeven={1/(1+FLAG/(1-FLAG))*100:.1f}%]")

# ---- 2. win vs RR-breakeven excess per depth (gross, to isolate entry quality) ----
print("\n--- (2) win% vs RR-breakeven EXCESS per depth (net) ---")
print(f"  {'frac':>6}{'RR':>7}{'win%':>7}{'brkevn':>8}{'excess':>8}{'N':>6}")
for fr in (0.5, 0.55, 0.6, 0.618, 0.65, 0.7, 0.75, 0.786, 0.82, 0.85):
    t = bounce(fr)
    if len(t) < 12: continue
    rr = fr / (1 - fr); be = 1 / (1 + rr) * 100; win = (t.R.values > 0).mean() * 100
    print(f"  {fr:>6.3f}{rr:>7.2f}{win:>7.1f}{be:>8.1f}{win-be:>+8.1f}{len(t):>6}")

# ---- 4. per-year ----
print("\n--- (4) per-year R (flagship 0.786) ---")
gy = info[FLAG]["gy"]
for y in gy.index: print(f"  {y}: {gy[y]:+6.1f}R")
print(f"  green {(gy>0).sum()}/{len(gy)}")

# ---- 5. cost+slip stress ----
print("\n--- (5) COST+SLIP STRESS (tight L0 stop = the vulnerability) ---")
print(f"  {'cost$':>6}{'slip':>6}  " + "".join(f"frac{fr}".rjust(11) for fr in (0.7, 0.786, 0.85)))
for cost, slip in [(15, 0.5), (20, 1.0), (25, 1.0), (25, 1.5)]:
    cells = []
    for fr in (0.7, 0.786, 0.85):
        t = bounce(fr, spread=cost, sslip=slip)
        cg, dd, cdd, _ = cagr_dd(t)
        cells.append(f"{t.R.mean():+.2f}/{cdd:.1f}")
    print(f"  {cost:>6}{slip:>6}  " + "".join(cell.rjust(11) for cell in cells) + "   (meanR/CAGRDD)")

# ---- 6. overfit audit (frac family) ----
print("\n--- (6) OVERFIT AUDIT (frac family) ---")
cfgs = {f"f{fr}": [(r.time, r.R) for r in bounce(fr).itertuples()] for fr in (0.6, 0.65, 0.7, 0.75, 0.786, 0.82, 0.85)}
audit(cfgs, flagship=f"f{FLAG}", trials=2000)

# ---- 7. BETA NULL: random-long, same regime(gated uptrend), same risk$ dist, same RR, same N ----
print("\n--- (7) BETA NULL: random-long same-RR same-gated-regime, same N ---")
gate_bars = np.where(G)[0]
gate_bars = gate_bars[(gate_bars > 5) & (gate_bars < N_BARS - FWD - 2)]
real = BT
RRv = FLAG / (1 - FLAG)
risks = real["risk$"].values  # sample real stop distances (holds risk geometry)
Nreal = len(real)


def sim_random(rng):
    bars = rng.choice(gate_bars, Nreal, replace=True)
    rk = rng.choice(risks, Nreal, replace=True)
    Rs, ts = [], []
    for b, risk in zip(bars, rk):
        entry = c[b]; stop = entry - risk; targ = entry + RRv * risk
        r = None
        for j in range(b + 1, min(b + 1 + FWD, N_BARS)):
            if l[j] <= stop: r = -1.0 - 0.5 * (stop - l[j]) / risk; break
            if h[j] >= targ: r = RRv; break
        if r is None: r = (c[min(b + FWD, N_BARS - 1)] - entry) / risk
        r -= 15.0 / risk
        Rs.append(r); ts.append(IDX[b])
    return pd.DataFrame({"time": ts, "R": Rs})


rng = np.random.default_rng(0)
obs_mr = real.R.mean(); obs_cdd = cagr_dd(real)[2]; obs_pf = real.R[real.R > 0].sum() / abs(real.R[real.R <= 0].sum())
nmr, ncdd, npf = [], [], []
for _ in range(600):
    s = sim_random(rng)
    nmr.append(s.R.mean())
    try: ncdd.append(cagr_dd(s)[2])
    except Exception: ncdd.append(0.0)
    neg = abs(s.R[s.R <= 0].sum()); npf.append(s.R[s.R > 0].sum() / neg if neg > 0 else 9.99)
nmr, npf = np.array(nmr), np.array(npf)
ncdd = np.array(ncdd); ncdd = ncdd[np.isfinite(ncdd)]
print(f"  REAL:  meanR={obs_mr:+.3f}  PF={obs_pf:.2f}  CAGR/DD={obs_cdd:.2f}   (N={Nreal})")
print(f"  NULL meanR: med={np.median(nmr):+.3f} std={nmr.std():.3f}  REAL pctile={(nmr<obs_mr).mean()*100:.0f}%")
print(f"  NULL PF:    med={np.median(npf):.2f}                REAL pctile={(npf<obs_pf).mean()*100:.0f}%")
print(f"  NULL CAGRDD:med={np.median(ncdd):+.2f} std={ncdd.std():.2f}  REAL pctile={(ncdd<obs_cdd).mean()*100:.0f}%  (n_valid={len(ncdd)})")

# BTC buy&hold CAGR/DD over the same span
close_d = DF["close"]
span = (close_d.index[-1] - close_d.index[0]).days / 365.25
bh_ret = close_d.iloc[-1] / close_d.iloc[0]
bh_cagr = (bh_ret ** (1 / span) - 1) * 100
eqbh = close_d / close_d.iloc[0]
bh_dd = ((eqbh.cummax() - eqbh) / eqbh.cummax()).max() * 100
print(f"  BTC buy&hold same span: CAGR={bh_cagr:+.0f}% maxDD={bh_dd:.0f}% CAGR/DD={bh_cagr/bh_dd:.2f}")

# ---- 8. correlation vs adopted book legs (context only) ----
print("\n--- (8) annual-R correlation vs adopted book (context only) ---")
legs = get_legs()
def ann(t): return t.groupby(pd.to_datetime(t.time).dt.year)["R"].sum()
cols = {"bounce": ann(real), **{k: ann(v) for k, v in legs.items()}}
al = pd.concat(cols, axis=1).fillna(0.0)
print(al.corr().round(2).to_string())
