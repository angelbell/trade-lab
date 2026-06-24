"""liq_sweep_fade.py -- BTC counter-trend "fade the liquidation sweep" (Hiropi heatmap idea), falsified.

The mechanizable, CAUSAL core of the heatmap idea (the predictive heatmap itself repaints / is not
backtestable): stops & liquidations pile up just beyond recent swing extremes; price sweeps the extreme
then REVERSES. So: new-L-bar extreme + rejection wick + close back inside (reclaim) -> FADE it, structural
stop beyond the swept wick, RR exit, intrabar SL/TP, no-overlap, cost. (Heatmap-free price-structure proxy.)
Positioning is added via the FREE funding-rate proxy (data/btc_funding.csv): crowded longs (high +funding)
=> downside liquidation hunt => the LONG fade (buy the swept dip) is favored; crowded shorts (neg funding)
=> upside squeeze => the SHORT fade. (run research/btc_funding_fetch.py first.)

Falsifier (pre-registered, ALL required to be a LEAD):
  - base meanR>0 & MFE/MAE>1.1 on >=1 TF, win% clears RR breakeven (not random)
  - beats a RANDOM-equal-entry null >=90%ile on CAGR/DD (not just meanR = the n-trim trap)
  - IS~OOS (same sign, OOS>=0); per-year majority green (not one-era beta)
  - survives realistic cost; funding gate must ADD over the no-gate base (else funding inert)
  - annual-R corr < ~0.5 with every book leg (a real MR diversifier, not redundant trend re-derivation)
In-sample; live-forward arbitrates.
  .venv/bin/python research/liq_sweep_fade.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from research.volume_reversal_screen import resample, atr
from research.portfolio_kama import cagr_dd, get_legs

SPLIT = 2022   # BTC h1 2017-2026 -> rough half by trade count
FUND_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "btc_funding.csv")


def sweep_trades(d, side="both", reclaim=True, L=10, wick_k=1.0, rr=2.0, buf=0.2, fwd=24, cost=0.0005):
    """Fade a swept swing extreme. reclaim=True requires a rejection wick + close back inside (the
    confirmed reversal); reclaim=False = naked poke (fade the break itself, no close confirmation)."""
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = atr(d).values
    hiL = pd.Series(h).rolling(L).max().shift(1).values   # prior-L extreme (no self-inclusion)
    loL = pd.Series(l).rolling(L).min().shift(1).values
    uwick = h - np.maximum(o, c); lwick = np.minimum(o, c) - l
    rng = (h - l)
    rows = []; last = -1
    for i in range(L + 1, len(d) - 1):
        if i <= last or not np.isfinite(a[i]) or a[i] <= 0 or rng[i] <= 0:
            continue
        if reclaim:
            top = (h[i] >= hiL[i]) and (uwick[i] >= wick_k * a[i]) and (c[i] < h[i] - 0.25 * rng[i])
            bot = (l[i] <= loL[i]) and (lwick[i] >= wick_k * a[i]) and (c[i] > l[i] + 0.25 * rng[i])
        else:
            top = (h[i] >= hiL[i]); bot = (l[i] <= loL[i])
        if top and bot:            # ambiguous outside bar -> skip
            continue
        if not (top or bot):
            continue
        isS = top
        if (isS and side == "long") or ((not isS) and side == "short"):
            continue
        e = o[i + 1]
        stop = (h[i] + buf * a[i]) if isS else (l[i] - buf * a[i])
        risk = (stop - e) if isS else (e - stop)
        if risk <= 0:
            continue
        tgt = (e - rr * risk) if isS else (e + rr * risk)
        R = None; mfe = mae = 0.0; end = min(i + 1 + fwd, len(d))
        for j in range(i + 1, end):
            mfe = max(mfe, ((e - l[j]) if isS else (h[j] - e)) / risk)
            mae = max(mae, ((h[j] - e) if isS else (e - l[j])) / risk)
            if (h[j] >= stop) if isS else (l[j] <= stop):
                R = -1.0; break
            if (l[j] <= tgt) if isS else (h[j] >= tgt):
                R = rr; break
        if R is None:
            R = ((e - c[end - 1]) if isS else (c[end - 1] - e)) / risk; j = end - 1
        R -= cost * e / risk
        rows.append((d.index[i], "S" if isS else "L", R, mfe, mae)); last = j
    return pd.DataFrame(rows, columns=["time", "tside", "R", "mfe", "mae"])


def summ(tag, t, rr):
    if len(t) < 12:
        print(f"  {tag:<26} n={len(t):>4} (too few)"); return None
    be = 100 / (1 + rr)
    _, dd, cdd, tot = cagr_dd(t[["time", "R"]])
    yrs = t.time.dt.year
    is_ = t[yrs < SPLIT].R.mean(); oos = t[yrs >= SPLIT].R.mean()
    mm = t.mfe.mean() / max(t.mae.mean(), 1e-9)
    print(f"  {tag:<26} n={len(t):>4} win%={(t.R>0).mean()*100:>3.0f}(BE{be:.0f}) meanR={t.R.mean():+5.2f} "
          f"totR={t.R.sum():>+6.0f} MFE/MAE={mm:4.2f} CAGR/DD={cdd:5.2f} | IS={is_:+.2f} OOS={oos:+.2f}")
    return cdd


def peryear(t):
    g = t.groupby(t.time.dt.year).R.agg(["count", "sum", "mean"])
    cells = " ".join(f"{y}:{r['sum']:+.0f}({int(r['count'])})" for y, r in g.iterrows())
    green = (g["sum"] > 0).sum()
    print(f"      per-year green {green}/{len(g)}: {cells}")


def join_funding(t):
    f = pd.read_csv(FUND_CSV)
    f["time"] = pd.to_datetime(f["time"], utc=True, format="ISO8601").dt.tz_localize(None) + pd.Timedelta(hours=3)  # UTC->broker
    f["fundingRate"] = f["fundingRate"].shift(1)                          # safety lag: never peek
    f = f.dropna().sort_values("time")
    t = t.copy()
    if getattr(t["time"].dt, "tz", None) is not None:
        t["time"] = t["time"].dt.tz_localize(None)   # both now broker-wall-clock naive
    t = t.sort_values("time")
    m = pd.merge_asof(t, f, on="time", direction="backward", allow_exact_matches=False)
    return m.dropna(subset=["fundingRate"])


def funding_gate(t, rr, side):
    m = join_funding(t)
    if len(m) < 12:
        print("  funding gate: too few joined"); return
    fr = m["fundingRate"]
    q = fr.quantile([1/3, 2/3]).values
    print(f"  funding coverage n={len(m)}/{len(t)}  rate p33={q[0]*100:+.4f}% p67={q[1]*100:+.4f}%")
    # aligned gate: LONG fade wants crowded longs (high +funding); SHORT fade wants crowded shorts (low/neg)
    if side == "long":
        gated = m[fr >= q[1]]; lo = m[fr <= q[0]]
        summ("LONG | funding HIGH(top33)", gated, rr); summ("LONG | funding LOW(bot33)", lo, rr)
    elif side == "short":
        gated = m[fr <= q[0]]; hi = m[fr >= q[1]]
        summ("SHORT| funding LOW(bot33)", gated, rr); summ("SHORT| funding HIGH(top33)", hi, rr)


def null_pctile(d, real_cdd, side, rr, n_entries, fwd, cost, iters=400, seed=0):
    """Random-equal-entry null: n_entries random bars, same side/exit, CAGR/DD distribution."""
    rng = np.random.default_rng(seed)
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    a = atr(d).values
    idx = np.arange(L_FOR_NULL + 1, len(d) - fwd - 1)
    idx = idx[np.isfinite(a[idx]) & (a[idx] > 0)]
    isS = (side == "short")
    cdds = []
    for _ in range(iters):
        pick = rng.choice(idx, size=min(n_entries, len(idx)), replace=False)
        rows = []
        for i in pick:
            e = o[i + 1]; buf = 0.2
            stop = (h[i] + buf * a[i]) if isS else (l[i] - buf * a[i])
            risk = (stop - e) if isS else (e - stop)
            if risk <= 0:
                continue
            tgt = (e - rr * risk) if isS else (e + rr * risk)
            R = None; end = min(i + 1 + fwd, len(d))
            for j in range(i + 1, end):
                if (h[j] >= stop) if isS else (l[j] <= stop):
                    R = -1.0; break
                if (l[j] <= tgt) if isS else (h[j] >= tgt):
                    R = rr; break
            if R is None:
                R = ((e - c[end - 1]) if isS else (c[end - 1] - e)) / risk
            rows.append((d.index[i], R - cost * e / risk))
        nt = pd.DataFrame(rows, columns=["time", "R"])
        if len(nt) > 5:
            cdds.append(cagr_dd(nt)[2])
    cdds = np.array(cdds)
    pct = (cdds < real_cdd).mean() * 100
    print(f"  random-entry null CAGR/DD: median={np.median(cdds):.2f} p90={np.percentile(cdds,90):.2f} "
          f"-> real {real_cdd:.2f} at {pct:.0f}%ile")


def book_corr(t):
    legs = get_legs()
    ty = t.groupby(t.time.dt.year).R.sum()
    out = []
    for name, lt in legs.items():
        ly = lt.groupby(lt.time.dt.year).R.sum()
        j = pd.concat([ty, ly], axis=1).dropna()
        cc = j.iloc[:, 0].corr(j.iloc[:, 1]) if len(j) >= 3 else float("nan")
        out.append(f"{name}={cc:+.2f}(yrs{len(j)})")
    print(f"  annual-R corr w/ book: {'  '.join(out)}")


L_FOR_NULL = 10

def main():
    raw = load_mt5_csv("data/vantage_btcusd_h1.csv")
    rr = 2.0; fwd = 24; cost = 0.0005
    print(f"=== BASE sweep-reclaim fade (RR{rr:.0f}, cost {cost*100:.2f}%, reclaim=wick+close-back) ===")
    best = None
    for tf in ["1h", "2h", "4h", "8h", "1d"]:
        d = resample(raw, tf)
        for side in ["long", "short"]:
            t = sweep_trades(d, side=side, reclaim=True, rr=rr, fwd=fwd, cost=cost)
            cdd = summ(f"{tf:>3} {side:<5} reclaim", t, rr)
            if cdd is not None and t.R.mean() > 0 and (t.mfe.mean()/max(t.mae.mean(),1e-9)) > 1.0:
                if best is None or cdd > best[0]:
                    best = (cdd, tf, side, t, d)

    if best is None:
        print("\nNo base config has meanR>0 & MFE/MAE>1 -> floor DEAD. Stop.")
        return
    _, tf, side, t, d = best
    print(f"\n=== DEEP DIVE on best base: {tf} {side} ===")
    peryear(t)
    print("  --- reclaim vs naked poke (confirmed-close law) ---")
    summ(f"{tf} {side} reclaim", t, rr)
    summ(f"{tf} {side} poke", sweep_trades(d, side=side, reclaim=False, rr=rr, fwd=fwd, cost=cost), rr)
    print("  --- funding positioning gate ---")
    funding_gate(t, rr, side)
    print("  --- random-entry null + book correlation + cost stress ---")
    null_pctile(d, best[0], side, rr, len(t), fwd, cost)
    book_corr(t)
    for cst in [0.0010, 0.0020]:
        summ(f"{tf} {side} cost{cst*100:.1f}%", sweep_trades(d, side=side, reclaim=True, rr=rr, fwd=fwd, cost=cst), rr)


if __name__ == "__main__":
    main()
