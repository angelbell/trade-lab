"""FROZEN exit-rule trial (pre-registered 2026-07-12, fired on user GO):

RULE      At the decision point (trade touched >= lim+2R intrabar, later first dips to <= lim+1R
          at bar d): if taker_4h (mean of Binance 5m taker L/S vol ratio over (t_d - 4h, t_d],
          data available at bar start = causal) < 1.10 -> EXIT at bar d+1 OPEN.
          Otherwise, and for trades without flow coverage (< 2020-09) or d == exit bar: unchanged.
          Threshold 1.10 frozen (round number between diagnostic medians 1.124/1.065);
          sensitivity rows 1.08 / 1.12 reported as PLATEAU CHECK, not selection.
POPULATION canon btc15m_L frac0.3 (zigzag k2 / RR4 / KAMA-D / BO20 / FWD500 / net $15),
          FIXED trade set (no re-arm modeling; conservative for early exit -- same convention
          as the 9-variant price-exit test).
JUDGE     leg totR/yr, maxDD(R), totR/DD vs base; RANDOM-INTERVENTION null = same COUNT of
          intervened trades drawn uniformly from the eligible decision-point set, same exit
          prices, 2000 draws -> %ile of rule's totR/DD and totR/yr. IS/OOS halves by time.
PASS      totR/DD > base AND >= 90%ile vs null on totR/DD AND IS/OOS same direction.
KILL      below base on totR/DD, or < 90%ile (= a random same-size intervention does as well).
Run: .venv/bin/python experiments/inhold_taker_exit_test.py
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from src.data_loader import load_mt5_csv
from swing_pivot_ab import build, FWD, START, ROOT

FRAC, RR, COST = 0.3, 4.0, 15.0
THRESH = [1.08, 1.10, 1.12]          # 1.10 = the frozen primary
CACHE = os.path.join(ROOT, "data/ext_btc_oi_metrics.csv")
RNG = np.random.default_rng(20260712)


def replay_full(df, E, h, l, c, o):
    """Every canon frac trade, with decision-point info + counterfactual early-exit R."""
    busy = -1; out = []
    for (i, e, stop, tgt, H1, minleg) in E:
        if i <= busy: continue
        lim = e - FRAC * (e - stop)
        if lim <= stop or lim >= e: continue
        fill_j = None
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        risk = lim - stop; reward = tgt - lim
        lv2, lv1 = lim + 2 * risk, lim + 1 * risk
        if l[fill_j] <= stop:
            out.append(dict(t=df.index[fill_j], R=-1.0, risk=risk, d=None, t_d=None, cf=None)); busy = fill_j; continue
        exit_j = min(fill_j + FWD, len(c) - 1); R = None; t2 = None; d = None
        for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
            if l[j] <= stop: R = -1.0; exit_j = j; break
            if t2 is None and h[j] >= lv2: t2 = j
            if t2 is not None and d is None and j > t2 and l[j] <= lv1: d = j
            if h[j] >= tgt: R = reward / risk; exit_j = j; break
        if R is None: R = (c[exit_j] - lim) / risk
        cf = None
        if d is not None and d < exit_j and d + 1 <= exit_j:
            cf = (o[d + 1] - lim) / risk                 # counterfactual: exit at next bar open
        out.append(dict(t=df.index[fill_j], R=R, risk=risk, d=d,
                        t_d=df.index[d] if d is not None else None, cf=cf))
        busy = exit_j
    return out


def metrics(times, R, span_yr):
    R = np.asarray(R, float)
    order = np.argsort(times); R = R[order]
    cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
    yrs = np.array([t.year for t in np.asarray(times)[order]])
    uy = np.unique(yrs); half = uy[len(uy) // 2]
    return dict(tot=R.sum(), totyr=R.sum() / span_yr, dd=dd,
                retdd=R.sum() / dd if dd > 0 else np.inf,
                IS=R[yrs < half].sum(), OOS=R[yrs >= half].sum(),
                meanR=R.mean(), pf=R[R > 0].sum() / abs(R[R <= 0].sum()))


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    o = df["open"].values
    E, h, l, c, _ = build(df, RR, "zigzag", 2.0)
    tr = replay_full(df, E, h, l, c, o)
    print(f"canon trades n={len(tr)} (fixed set, no re-arm)")

    m = pd.read_csv(CACHE, index_col=0); m.index = pd.to_datetime(m.index, utc=True)
    m = m.sort_index(); m = m[~m.index.duplicated(keep="last")]
    m = m[m["sum_open_interest"] > 0]
    taker = m["sum_taker_long_short_vol_ratio"]

    T0 = pd.Timestamp("2020-09-01", tz="UTC")
    elig = []
    for k, t in enumerate(tr):
        if t["cf"] is None or t["t_d"] is None or t["t_d"] < T0: continue
        tk = taker.loc[t["t_d"] - pd.Timedelta("4h"):t["t_d"]].mean()
        if np.isfinite(tk): elig.append((k, tk))
    print(f"eligible decision-point trades (flow-covered, d<exit): n={len(elig)}")

    times = [t["t"] for t in tr]
    Rnet = np.array([t["R"] - COST / t["risk"] for t in tr])
    base = metrics(times, Rnet, span)
    print(f"\nBASE      totR/yr={base['totyr']:+6.2f} maxDD={base['dd']:5.1f}R totR/DD={base['retdd']:5.2f} "
          f"PF={base['pf']:.2f} IS/OOS={base['IS']:+.0f}/{base['OOS']:+.0f}R")

    cf_net = {k: tr[k]["cf"] - COST / tr[k]["risk"] for k, _ in elig}

    def apply(idxs):
        Rm = Rnet.copy()
        for k in idxs: Rm[k] = cf_net[k]
        return metrics(times, Rm, span)

    res_primary = None
    for th in THRESH:
        hit = [k for k, tk in elig if tk < th]
        s = apply(hit)
        tag = " <= FROZEN PRIMARY" if th == 1.10 else " (plateau row)"
        print(f"th<{th:.2f}  n_int={len(hit):>3}  totR/yr={s['totyr']:+6.2f} maxDD={s['dd']:5.1f}R "
              f"totR/DD={s['retdd']:5.2f} PF={s['pf']:.2f} IS/OOS={s['IS']:+.0f}/{s['OOS']:+.0f}R{tag}")
        if th == 1.10: res_primary, n_int = s, len(hit)

    # random-intervention null (same count, uniform over eligible)
    keys = [k for k, _ in elig]
    null_retdd, null_totyr = [], []
    for _ in range(2000):
        pick = RNG.choice(keys, size=n_int, replace=False)
        s = apply(pick)
        null_retdd.append(s["retdd"]); null_totyr.append(s["totyr"])
    null_retdd = np.array(null_retdd); null_totyr = np.array(null_totyr)
    p_retdd = (res_primary["retdd"] > null_retdd).mean() * 100
    p_totyr = (res_primary["totyr"] > null_totyr).mean() * 100
    print(f"\nNULL (random same-count intervention, 2000 draws): "
          f"totR/DD median={np.median(null_retdd):.2f}  rule %ile={p_retdd:.1f}   "
          f"totR/yr median={np.median(null_totyr):+.2f}  rule %ile={p_totyr:.1f}")

    is_dir = res_primary["IS"] >= base["IS"]; oos_dir = res_primary["OOS"] >= base["OOS"]
    print(f"\nPRE-REGISTERED CHECK: totR/DD>base={res_primary['retdd'] > base['retdd']}  "
          f">=90%ile={p_retdd >= 90}  IS/OOS same direction={is_dir == oos_dir} (IS {'+' if is_dir else '-'}, OOS {'+' if oos_dir else '-'})")

if __name__ == "__main__":
    main()
