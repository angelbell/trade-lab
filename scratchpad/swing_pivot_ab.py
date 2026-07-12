"""swing detector A/B: amplitude-only ZigZag (canon) vs time-gated N-bar fractal pivot.
User hypothesis (2026-07-12): the 15m BTC ZigZag looks 'unclean' (no time concept in legs);
a fractal-pivot swing closer to human wave perception may detect trend structure better.

PRIMARY  BTC 15m long (btc15m_L canon: Pattern B / trend-ema80 / RR4 / daily-KAMA14-rising /
         BO20 / FWD500 / net $15 abs; market + frac0.3 pullback-limit). Machinery = faithful
         copy of scratchpad/btc15m_pullback_gauntlet.py build/evaluate (which reproduced the
         ledger canon 2026-07-02), with ONLY the swing detector parameterized.
         Tie-back required: zigzag k2 market meanR≈+0.175/n657, frac0.3 ≈+0.322/n614.
CONTROLS gold 1h (canon CLI config via breakout_wave.run: SMA150+slope10, RR3, cost frac 0.001)
         and gold 15m (RR4 + ext-cap 8% + SMA150+slope10) — INTERNAL A/B only (same args both
         arms; no absolute tie-back claimed for gold15m).
ARMS     swing=zigzag k in {1.5, 2.0, 2.5}   vs   swing=pivot n in {3, 5, 7, 10}
DIAG     (a) leg-duration distributions zigzag-k2 vs pivot-n5 on BTC 15m (median/p90/%<=2bars)
         (b) zigzag-arm trades split by pattern min-leg <=2 bars vs >2 (is the user's visual
             complaint visible in PnL?)
VERDICT  (pre-registered) PASS = pivot beats the zigzag-k2 canon on BOTH totR/yr and ret/DD
         with an n-plateau (neighbors agree), direction consistent on >=2 of 3 cells.
Run: .venv/bin/python scratchpad/swing_pivot_ab.py [--smoke]
"""
import sys, os, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from types import SimpleNamespace
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, swings_pivot, kama_adaptive, run

ROOT = "/home/angelbell/dev/auto-trade"
BO, FWD = 20, 500
START = "2018-10-01"
COST_BTC = 15.0

# ---------------- BTC 15m machinery (faithful copy of btc15m_pullback_gauntlet.py; swing_fn param) --
def build(df, RR, swing_mode, param):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = ta.atr(df["high"], df["low"], df["close"], 14).values
    es = df["close"].ewm(span=80, adjust=False).mean().values
    dck = df["close"].resample("1D").last().dropna()
    kmg = kama_adaptive(dck, 14)
    kreg = ((kmg > kmg.shift(1)).shift(1)).reindex(df.index, method="ffill").fillna(False).values
    sw = swings_zigzag(h, l, a, param) if swing_mode == "zigzag" else swings_pivot(h, l, param)

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
        minleg = min(iH1 - iL0, iL2 - iH1)          # pattern leg durations (bars)
        E.append((e_i, e, stop, e + RR * risk, pH1, minleg))
    E.sort(key=lambda x: x[0]); seen = set(); U = []
    for en in E:
        if en[0] in seen: continue
        seen.add(en[0]); U.append(en)
    return U, h, l, c, sw


def evaluate(df, E, h, l, c, frac):
    busy = -1; tr = []; miss = 0
    for (i, e, stop, tgt, H1, minleg) in E:
        if i <= busy: continue
        if frac is None:
            risk = e - stop; reward = tgt - e; exit_j = min(i + FWD, len(c) - 1); R = None
            for j in range(i + 1, min(i + 1 + FWD, len(c))):
                if l[j] <= stop: R = -1.0; exit_j = j; break
                if h[j] >= tgt: R = reward / risk; exit_j = j; break
            if R is None: R = (c[exit_j] - e) / risk
            tr.append((df.index[i], R, risk, minleg)); busy = exit_j; continue
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
        tr.append((df.index[fill_j], R, risk, minleg)); busy = exit_j
    return tr, miss


def stats(trn, span_yr):
    R = np.array([r for _, r, *_ in trn])
    if len(R) < 5: return None
    yr = np.array([t.year for t, *_ in trn]); yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    green = np.mean([R[yr == y].sum() > 0 for y in yrs]) * 100
    return dict(N=len(R), npy=len(R) / span_yr, win=(R > 0).mean() * 100, pf=pf,
                meanR=R.mean(), med=np.median(R), sd=R.std(), totyr=R.sum() / span_yr,
                maxDD=dd, retdd=R.sum() / dd if dd > 0 else np.inf,
                IS=R[yr < half].mean(), OOS=R[yr >= half].mean(), green=green)


def fmt(tag, s):
    if s is None: return f"  {tag:<22} n<5"
    return (f"  {tag:<22} N={s['N']:>4} N/yr={s['npy']:>5.1f} win={s['win']:>4.0f}% PF={s['pf']:>5.2f} "
            f"meanR={s['meanR']:>+.3f}(med{s['med']:>+.2f}/sd{s['sd']:.2f}) totR/yr={s['totyr']:>+6.2f} "
            f"maxDD={s['maxDD']:>5.1f}R ret/DD={s['retdd']:>5.2f} IS/OOS={s['IS']:>+.3f}/{s['OOS']:>+.3f} "
            f"grn={s['green']:>3.0f}%")


def leg_durations(sw):
    piv = sorted(i for _, i, _, _ in sw)
    d = np.diff(piv)
    return d[d > 0]


# ---------------- gold cells via breakout_wave.run() -------------------------------------------
GBASE = dict(pattern="B", sl_mode="line", sl_buf=0.25, sl_b="swinglow", sl_b_k=1.5,
             swing="zigzag", zz_k=2.0, pivot_n=5, renko_k=2.0, mom_fast=12, mom_slow=26,
             trend_ema=80, bo_window=20, tp_mode="rr", rr=3.0, atr=14, cost=0.001, swap_pct=0.0,
             fwd=500, peryear=False, start=None, end=None, daily_sma=150, daily_slope_k=10,
             gate_tf="1D", risk=0.01, gate_kama=0, gate_kama_tf="1D", gate_kama_tf2="",
             ext_cap=0.0, retest=0, retest_tol=0.10, pullback_frac=0.0, max_pos=1, exec_split=0,
             exit_kama=0, exit_kama_tf="1D", tp1_frac=0.0, tp1_rr=1.0, tp1_be=1,
             wave="all", dump_trades=False, tf="", csv="")

def gold_cell(d, span_yr, swing_mode, param, rr, ext_cap):
    args = SimpleNamespace(**{**GBASE, "swing": swing_mode, "rr": rr, "ext_cap": ext_cap,
                              "zz_k": param if swing_mode == "zigzag" else 2.0,
                              "pivot_n": param if swing_mode == "pivot" else 5})
    with contextlib.redirect_stdout(io.StringIO()):
        t = run(d, args)
    if t is None or len(t) < 5: return None
    return stats([(r.time, r.R, 0, 0) for r in t.itertuples()], span_yr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ARMS = [("zigzag", 1.5), ("zigzag", 2.0), ("zigzag", 2.5),
            ("pivot", 3), ("pivot", 5), ("pivot", 7), ("pivot", 10)]
    if a.smoke: ARMS = [("zigzag", 2.0), ("pivot", 5)]

    # ---- PRIMARY: BTC 15m ----
    with contextlib.redirect_stderr(io.StringIO()):
        btc = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (btc.index[-1] - btc.index[0]).days / 365.25
    print(f"=== PRIMARY BTC 15m (Pattern B / ema80 / RR4 / KAMA-D gate / ${COST_BTC:.0f} net) "
          f"{btc.index[0].date()}->{btc.index[-1].date()} {span:.1f}yr ===")
    res = {}
    for mode, p in ARMS:
        E, h, l, c, sw = build(btc, 4.0, mode, p)
        for tag, frac in (("mkt", None), ("frac0.3", 0.3)):
            tr, miss = evaluate(btc, E, h, l, c, frac)
            trn = [(t, R - COST_BTC / risk, risk, ml) for (t, R, risk, ml) in tr]
            s = stats(trn, span)
            res[(mode, p, tag)] = s
            print(fmt(f"{mode}-{p} {tag}", s) + (f" miss={miss}" if frac else ""))
            if mode == "zigzag" and p == 2.0:
                ref = 0.175 if frac is None else 0.322
                if s and abs(s["meanR"] - ref) > 0.02:
                    print(f"    !! TIE-BACK DIVERGENCE vs canon {ref:+.3f}")
                elif s:
                    print(f"    tie-back OK (canon {ref:+.3f})")
            # diag (b): zigzag k2 market, split by pattern min-leg
            if mode == "zigzag" and p == 2.0 and frac is None and s:
                sp = [x for x in trn if x[3] <= 2]; lg = [x for x in trn if x[3] > 2]
                for nm, sub in (("minleg<=2bars", sp), ("minleg>2bars", lg)):
                    ss = stats(sub, span)
                    if ss: print(f"    [diag] {nm:<14} N={ss['N']:>4} meanR={ss['meanR']:+.3f} PF={ss['pf']:.2f}")

    # diag (a): leg durations
    _, _, _, _, swz = build(btc, 4.0, "zigzag", 2.0)
    _, _, _, _, swp = build(btc, 4.0, "pivot", 5)
    for nm, sw in (("zigzag-k2", swz), ("pivot-n5", swp)):
        d = leg_durations(sw)
        print(f"  [legs] {nm}: n={len(d)} median={np.median(d):.0f} p90={np.percentile(d,90):.0f} "
              f"bars, %<=2bars={100*(d<=2).mean():.1f}%")

    if a.smoke:
        print("(smoke only)"); return

    # ---- CONTROLS: gold 1h / gold 15m ----
    for name, fn, rr, cap in (("gold 1h (RR3, SMA150+slope)", "vantage_xauusd_h1.csv", 3.0, 0.0),
                              ("gold 15m (RR4, ext-cap8%)", "vantage_xauusd_m15.csv", 4.0, 0.08)):
        with contextlib.redirect_stderr(io.StringIO()):
            d = load_mt5_csv(os.path.join(ROOT, "data", fn))
        syr = (d.index[-1] - d.index[0]).days / 365.25
        print(f"\n=== CONTROL {name} (internal A/B, cost frac 0.001) ===")
        for mode, p in ARMS:
            print(fmt(f"{mode}-{p}", gold_cell(d, syr, mode, p, rr, cap)))

    # ---- pre-registered verdict ----
    print("\n=== PRE-REGISTERED CHECK (PRIMARY, market + frac0.3) ===")
    for tag in ("mkt", "frac0.3"):
        base = res[("zigzag", 2.0, tag)]
        beats = [(p, res[("pivot", p, tag)]) for p in (3, 5, 7, 10) if res.get(("pivot", p, tag))]
        wins = [p for p, s in beats if s["totyr"] > base["totyr"] and s["retdd"] > base["retdd"]]
        print(f"  {tag}: zigzag-k2 totR/yr={base['totyr']:+.2f} ret/DD={base['retdd']:.2f} | "
              f"pivot arms beating BOTH: {wins if wins else 'none'}")

if __name__ == "__main__":
    main()
