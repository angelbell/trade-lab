"""BTC 15m breakout + pullback-limit execution: FULL falsification gauntlet.
Machinery = faithful copy of scratchpad/pullback_btc.py (the canonical STEP2 script,
reproduced 2026-07-02 against the ledger). Pattern B / ZigZag zz-k2 / trend-ema80 /
BO window 20 / FWD 500 bars / no-overlap / entry at confirmed close / intrabar SL-TP.
Data: vantage_btcusd_m15.csv sliced to TRUE 15m density (2018-10-01 ->).
Stages: base gross market (15m/30m/1h x RR2/4) -> +KAMA daily gate -> net $15/$25 +
cost/risk dist -> frac sweep {0.2,0.25,0.3,0.38,0.5} -> beta null + random-drop null
-> per-year tables. Fixed target mechanism: stop & tgt at MARKET levels, only entry lowered.
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
BO, FWD = 20, 500
START = "2018-10-01"

base = load_mt5_csv("data/vantage_btcusd_m15.csv").loc[START:]
SPAN_YRS = (base.index[-1] - base.index[0]).days / 365.25
print(f"data: {base.index[0]} -> {base.index[-1]}  ({SPAN_YRS:.2f} yr, {len(base)} bars)")


def build(df, RR, gate):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = ta.atr(df["high"], df["low"], df["close"], 14).values
    es = df["close"].ewm(span=80, adjust=False).mean().values
    if gate:
        dck = df["close"].resample("1D").last().dropna()
        kmg = kama_adaptive(dck, 14)
        kreg = ((kmg > kmg.shift(1)).shift(1)).reindex(df.index, method="ffill").fillna(False).values
    else:
        kreg = np.ones(len(c), bool)
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
        tgt = e + RR * risk
        E.append((e_i, e, stop, tgt, pH1))
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return df, U, h, l, c


def evaluate(df, E, h, l, c, frac):
    """frac=None -> market. Returns trades [(time, gross_R, risk$)], miss count.
    Pullback: stop/tgt FIXED at market levels; entry limit = e - frac*(e-stop)."""
    busy = -1; tr = []; miss = 0
    for (i, e, stop, tgt, H1) in E:
        if i <= busy: continue
        if frac is None:
            risk = e - stop; reward = tgt - e; exit_j = min(i + FWD, len(c) - 1); R = None
            for j in range(i + 1, min(i + 1 + FWD, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt: R = reward / risk; exit_j = j; break
            if R is None: R = (c[exit_j] - e) / risk
            tr.append((df.index[i], R, risk)); busy = exit_j; continue
        lim = e - frac * (e - stop)
        if lim <= stop or lim >= e: miss += 1; continue
        fill_j = None
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: miss += 1; continue
        risk = lim - stop; reward = tgt - lim
        if l[fill_j] <= stop: R = -1.0; exit_j = fill_j
        else:
            exit_j = min(fill_j + FWD, len(c) - 1); R = None
            for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt: R = reward / risk; exit_j = j; break
            if R is None: R = (c[exit_j] - lim) / risk
        tr.append((df.index[fill_j], R, risk)); busy = exit_j
    return tr, miss


def net(tr, sp):
    return [(t, R - sp / risk) for (t, R, risk) in tr]


def stats(trn):
    R = np.array([r for _, r in trn]); ts = [t for t, _ in trn]
    yr = np.array([t.year for t in ts])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    ann = {int(y): R[yr == y].sum() for y in yrs}
    green = np.mean([v > 0 for v in ann.values()]) * 100
    return dict(N=len(R), npy=len(R) / SPAN_YRS, win=(R > 0).mean() * 100, pf=pf,
                meanR=R.mean(), IS=R[yr < half].mean(), OOS=R[yr >= half].mean(),
                maxDD=dd, retdd=R.sum() / dd if dd > 0 else np.inf, green=green, ann=ann)


TFS = [("15m", None), ("30m", "30min"), ("1h", "60min")]
RRS = (2.0, 4.0)
FRACS = (0.2, 0.25, 0.3, 0.38, 0.5)

cells = {}  # (tf,RR,gate) -> (df,E,h,l,c)
for tf, fr in TFS:
    df = base if fr is None else base.resample(fr).agg(AGG).dropna()
    for RR in RRS:
        for gate in (False, True):
            cells[(tf, RR, gate)] = build(df, RR, gate)

HDR = f"  {'cell':<22}{'N':>5}{'N/yr':>6}{'win':>5}{'PF':>6}{'meanR':>8}{'IS/OOS':>13}{'maxDD_R':>8}{'ret/DD':>7}{'grn%':>5}"


def row(name, trn, miss=None, extra=""):
    s = stats(trn)
    m = f" miss={miss/(miss+s['N'])*100:.0f}%" if miss is not None else ""
    print(f"  {name:<22}{s['N']:>5}{s['npy']:>6.1f}{s['win']:>4.0f}%{s['pf']:>6.2f}{s['meanR']:>+8.3f}"
          f"{s['IS']:>+6.2f}/{s['OOS']:>+.2f}{s['maxDD']:>8.1f}{s['retdd']:>7.2f}{s['green']:>4.0f}%{m}{extra}")
    return s


print("\n================ STAGE 2: BASE (all-signals, NO gate, GROSS, market) ================")
print(HDR)
base_gross = {}
for tf, _ in TFS:
    for RR in RRS:
        df, E, h, l, c = cells[(tf, RR, False)]
        tr, _ = evaluate(df, E, h, l, c, None)
        base_gross[(tf, RR)] = tr
        row(f"{tf} RR{RR:.0f} nogate gross", net(tr, 0.0))

print("\n================ STAGE 3: +daily KAMA(14)-rising gate (GROSS, market) ================")
print(HDR)
gated_market = {}
for tf, _ in TFS:
    for RR in RRS:
        df, E, h, l, c = cells[(tf, RR, True)]
        tr, _ = evaluate(df, E, h, l, c, None)
        gated_market[(tf, RR)] = tr
        d = np.mean([r for _, r in net(tr, 0)]) - np.mean([r for _, r in net(base_gross[(tf, RR)], 0)])
        row(f"{tf} RR{RR:.0f} KAMA gross", net(tr, 0.0), extra=f"  d_meanR_vs_nogate={d:+.3f}")

print("\n================ STAGE 4: NET cost, gated market ($15 rt; stress $25) ================")
for tf, _ in TFS:
    for RR in RRS:
        tr = gated_market[(tf, RR)]
        risks = np.array([risk for _, _, risk in tr])
        cr15 = 15.0 / risks
        print(f"\n  {tf} RR{RR:.0f}  cost/risk($15): med={np.median(cr15):.3f}R  p90={np.percentile(cr15,90):.3f}R"
              f"   (stop width $ med={np.median(risks):.0f} p90={np.percentile(risks,10):.0f}@p10)")
        print(HDR)
        row(f"{tf} RR{RR:.0f} net$15", net(tr, 15.0))
        row(f"{tf} RR{RR:.0f} net$25", net(tr, 25.0))

print("\n================ STAGE 5: PULLBACK frac sweep (gated, stop/tgt fixed at market levels) ================")
pull = {}
for tf, _ in TFS:
    for RR in RRS:
        df, E, h, l, c = cells[(tf, RR, True)]
        print(f"\n--- {tf} RR{RR:.0f} ---  (market net$15 first, then fracs; meanR cols: net$15 shown, gross/net25 appended)")
        print(HDR)
        trm = gated_market[(tf, RR)]
        g = np.mean([r for _, r in net(trm, 0)]); n25 = np.mean([r for _, r in net(trm, 25)])
        row("market", net(trm, 15.0), extra=f"  gross={g:+.3f} net25={n25:+.3f}")
        for f_ in FRACS:
            tr, miss = evaluate(df, E, h, l, c, f_)
            pull[(tf, RR, f_)] = (tr, miss)
            g = np.mean([r for _, r in net(tr, 0)]); n25 = np.mean([r for _, r in net(tr, 25)])
            row(f"frac{f_}", net(tr, 15.0), miss=miss, extra=f"  gross={g:+.3f} net25={n25:+.3f}")
