"""ROBUSTNESS sweep of the taker-exit rule (user GO 2026-07-12):
  (1) window length x threshold PLATEAU (is 4h/1.10 a lone spike or a plateau?)
  (2) bootstrap CI of the improvement (is the size distinguishable from zero?)
  (3) random-intervention null at every sweep cell (does selection beat n-trimming everywhere?)
  (4) both flow sources: OFFICIAL taker ratio (5m metrics cache) and PROXY r_up
      (up-bar volume / down-bar volume, the Pine-computable one), same grid.

Windows {1,2,4,6,8,12,24}h x thresholds = the {35,45,55,65,75}th percentile of that window's
own eligible-point distribution (percentile framing keeps the intervention COUNT comparable
across windows so cells are read against a like-for-like null, and avoids re-tuning an absolute
number per window). The frozen primary (official 4h, abs 1.10) sits at the 59th %ile.
Trade set FIXED (no re-arm; conservative for early exits, matches the frozen trial).
Proxy inputs: one 24h 1m-kline fetch per decision point -> all windows derived, cached to
scratchpad/cache_taker_windows_24h.csv (re-runs free).
Run: .venv/bin/python scratchpad/taker_exit_sweep.py
"""
import sys, os, time, json, urllib.request
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from src.data_loader import load_mt5_csv
from swing_pivot_ab import build, FWD, START, ROOT

FRAC, RR, COST = 0.3, 4.0, 15.0
CACHE = os.path.join(ROOT, "data/ext_btc_oi_metrics.csv")
CACHE24 = os.path.join(ROOT, "scratchpad/cache_taker_windows_24h.csv")
T0 = pd.Timestamp("2020-09-01", tz="UTC")
WINDOWS = [1, 2, 4, 6, 8, 12, 24]
PCTS = [35, 45, 55, 65, 75]
RNG = np.random.default_rng(20260712)


def fetch_24h(t_d):
    """one fetch per decision point: 1440 1m bars ending at t_d -> per-window aggregates."""
    end = int(t_d.timestamp() * 1000)
    out = {}
    chunks = []
    for part in range(2):                       # 2 x 720m (limit-safe)
        e = end - part * 720 * 60 * 1000
        s = e - 720 * 60 * 1000
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m"
               f"&startTime={s}&endTime={e - 1}&limit=720")
        with urllib.request.urlopen(url, timeout=30) as r:
            chunks.append(json.load(r))
        time.sleep(0.12)
    k = sorted([x for ch in chunks for x in ch], key=lambda x: x[0])
    if not k: return None
    a = np.array([[float(x[1]), float(x[4]), float(x[5]), float(x[9])] for x in k])
    o, c, v, tb = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    n = len(k)
    for w in WINDOWS:
        need = w * 60
        sl = slice(max(0, n - need), n)
        oo, cc, vv, tt = o[sl], c[sl], v[sl], tb[sl]
        sell = (vv - tt).sum()
        out[f"kline_{w}h"] = tt.sum() / sell if sell > 0 else np.nan
        dn = vv[cc < oo].sum()
        out[f"rup_{w}h"] = vv[cc > oo].sum() / dn if dn > 0 else np.nan
    return out


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    o = df["open"].values
    E, h, l, c, _ = build(df, RR, "zigzag", 2.0)

    # ---- canon walk (fixed trade set) + decision points ----
    busy = -1; trades = []; dps = []
    for (i, e, stop0, tgt, H1, ml) in E:
        if i <= busy: continue
        lim = e - FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fj = None
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fj = j; break
        if fj is None: continue
        u = lim - stop0; reward = tgt - lim; lv2, lv1 = lim + 2 * u, lim + 1 * u
        if l[fj] <= stop0:
            trades.append([df.index[fj], -1.0 - COST / u]); busy = fj; continue
        R = None; exit_j = min(fj + FWD, len(c) - 1); t2 = None; d = None
        for j in range(fj + 1, min(fj + 1 + FWD, len(c))):
            if l[j] <= stop0: R = -1.0; exit_j = j; break
            if t2 is None and h[j] >= lv2: t2 = j
            if t2 is not None and d is None and j > t2 and l[j] <= lv1: d = j
            if h[j] >= tgt: R = reward / u; exit_j = j; break
        if R is None: R = (c[exit_j] - lim) / u
        k = len(trades)
        trades.append([df.index[fj], R - COST / u])
        if d is not None and d < exit_j and d + 1 < len(c) and df.index[d] >= T0:
            dps.append(dict(k=k, t_d=df.index[d], cf=(o[d + 1] - lim) / u - COST / u))
        busy = exit_j
    times = np.array([t[0] for t in trades]); Rbase = np.array([t[1] for t in trades])
    print(f"canon: n={len(trades)} trades, {len(dps)} flow-era decision points, span {span:.1f}yr")

    # ---- official taker: all windows, computed locally ----
    m = pd.read_csv(CACHE, index_col=0); m.index = pd.to_datetime(m.index, utc=True)
    m = m.sort_index(); m = m[~m.index.duplicated(keep="last")]; m = m[m["sum_open_interest"] > 0]
    tkr = m["sum_taker_long_short_vol_ratio"]
    for d in dps:
        for w in WINDOWS:
            d[f"off_{w}h"] = tkr.loc[d["t_d"] - pd.Timedelta(hours=w):d["t_d"]].mean()

    # ---- proxy: 24h fetch per decision point (cached) ----
    if os.path.exists(CACHE24):
        cw = pd.read_csv(CACHE24, index_col=0, parse_dates=[0])
    else:
        cw = pd.DataFrame()
    new = 0
    for d in dps:
        key = d["t_d"]
        if len(cw) and key in cw.index:
            for col in cw.columns: d[col] = cw.loc[key, col]
        else:
            got = fetch_24h(key)
            if got is None: continue
            for col, v in got.items():
                d[col] = v; cw.loc[key, col] = v
            new += 1
    if new: cw.sort_index().to_csv(CACHE24)
    print(f"proxy windows: {new} newly fetched, {len(cw)} cached\n")

    def run_rule(feat, th):
        Rm = Rbase.copy(); hit = []
        for d in dps:
            v = d.get(feat, np.nan)
            if np.isfinite(v) and v < th:
                Rm[d["k"]] = d["cf"]; hit.append(d["k"])
        return Rm, hit

    def sc(R):
        cum = np.cumsum(R); dd = (np.maximum.accumulate(cum) - cum).max()
        return R.sum() / span, R.sum() / dd if dd > 0 else np.inf, \
               R[R > 0].sum() / abs(R[R <= 0].sum())

    base_totyr, base_retdd, base_pf = sc(Rbase)
    print(f"BASE totR/yr={base_totyr:+.2f} totR/DD={base_retdd:.2f} PF={base_pf:.2f}\n")

    keys = [d["k"] for d in dps]
    def null_pct(Rm, n_int, metric_idx):
        val = sc(Rm)[metric_idx]
        draws = []
        for _ in range(400):
            R2 = Rbase.copy()
            for k in RNG.choice(keys, size=n_int, replace=False):
                cf = next(d["cf"] for d in dps if d["k"] == k)
                R2[k] = cf
            draws.append(sc(R2)[metric_idx])
        return (val > np.array(draws)).mean() * 100

    for src, pref in (("OFFICIAL taker", "off"), ("PROXY r_up (Pine)", "rup")):
        print(f"=== {src}: totR/DD  [null %ile]   (base {base_retdd:.2f}) ===")
        print("  win  " + "".join(f"{p:>14}%ile" for p in PCTS))
        for w in WINDOWS:
            feat = f"{pref}_{w}h"
            vals = np.array([d.get(feat, np.nan) for d in dps], float)
            if not np.isfinite(vals).sum(): continue
            row = f"  {w:>2}h  "
            for p in PCTS:
                th = np.nanpercentile(vals, p)
                Rm, hit = run_rule(feat, th)
                _, retdd, _ = sc(Rm)
                pc = null_pct(Rm, len(hit), 1) if len(hit) else 0
                row += f"{retdd:>9.2f}[{pc:>3.0f}]"
            print(row)
        print()

    # ---- bootstrap CI of the improvement (primary: official 4h @ 1.10; proxy 4h @ 55%ile) ----
    print("=== bootstrap CI of the improvement in totR/DD (2000 resamples of trades) ===")
    for label, feat, th in (("official 4h < 1.10", "off_4h", 1.10),
                            ("proxy 4h < 55%ile", "rup_4h",
                             np.nanpercentile([d.get("rup_4h", np.nan) for d in dps], 55))):
        Rm, hit = run_rule(feat, th)
        d_pf, d_retdd, d_totyr = [], [], []
        n = len(Rbase)
        for _ in range(2000):
            idx = RNG.integers(0, n, n)
            b0, r0, p0 = sc(Rbase[idx]); b1, r1, p1 = sc(Rm[idx])
            d_totyr.append(b1 - b0); d_retdd.append(r1 - r0); d_pf.append(p1 - p0)
        q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 50), np.percentile(a, 97.5),
                       (np.array(a) > 0).mean() * 100)
        for nm, arr in (("totR/yr", d_totyr), ("totR/DD", d_retdd), ("PF", d_pf)):
            lo, md, hi, pos = q(arr)
            print(f"  {label:<20} d{nm:<8} median={md:+6.2f}  95%CI[{lo:+6.2f},{hi:+6.2f}]  P(>0)={pos:.0f}%")
        print(f"    (n_intervened={len(hit)})")

if __name__ == "__main__":
    main()
