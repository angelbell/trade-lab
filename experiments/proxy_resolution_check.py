"""Implementation-gap check before writing the Pine (2026-07-12): the validated proxy r_up was
computed from 1-MINUTE Binance bars aggregated over the trailing 4h. A 15m-chart Pine naturally
computes up/down volume from 15-MINUTE bars (16 of them). Does the coarser classification keep
the effect? (If not, the Pine must pull 1m via request.security_lower_tf -- heavier but faithful.)

Compares, at the same 128 decision points, on the fixed canon trade set:
  rup_1m   (validated)      : sum(vol | close>open) / sum(vol | close<open) over 240 1m bars
  rup_15m  (Pine-natural)   : same but over 16 15m bars
  rup_5m   (middle ground)  : same over 48 5m bars
Reports correlation between resolutions, AUC on recover/collapse, and the frozen-trial numbers
(totR/yr, totR/DD, PF) + random-intervention null at the 55th-percentile threshold.
Run: .venv/bin/python experiments/proxy_resolution_check.py
"""
import sys, os, time, json, urllib.request
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from src.data_loader import load_mt5_csv
from swing_pivot_ab import build, FWD, START, ROOT

FRAC, RR, COST = 0.3, 4.0, 15.0
T0 = pd.Timestamp("2020-09-01", tz="UTC")
CACHE_RES = os.path.join(ROOT, "experiments/cache_proxy_resolutions.csv")
RNG = np.random.default_rng(20260712)


def fetch(t_d, interval, nbars):
    end = int(t_d.timestamp() * 1000)
    per = {"1m": 60, "5m": 300, "15m": 900}[interval] * 1000
    start = end - nbars * per
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval={interval}"
           f"&startTime={start}&endTime={end - 1}&limit={nbars}")
    with urllib.request.urlopen(url, timeout=30) as r:
        k = json.load(r)
    if not k: return np.nan
    a = np.array([[float(x[1]), float(x[4]), float(x[5])] for x in k])
    o, c, v = a[:, 0], a[:, 1], a[:, 2]
    dn = v[c < o].sum()
    return v[c > o].sum() / dn if dn > 0 else np.nan


def rank_auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 8 or len(b) < 8: return np.nan
    r = pd.Series(np.concatenate([a, b])).rank().values
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    o = df["open"].values
    E, h, l, c, _ = build(df, RR, "zigzag", 2.0)

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
            dps.append(dict(k=k, t_d=df.index[d], cf=(o[d + 1] - lim) / u - COST / u,
                            rec=(R > 1.5)))
        busy = exit_j
    Rbase = np.array([t[1] for t in trades])
    print(f"canon n={len(trades)}, decision points={len(dps)}")

    cw = pd.read_csv(CACHE_RES, index_col=0, parse_dates=[0]) if os.path.exists(CACHE_RES) else pd.DataFrame()
    new = 0
    for d in dps:
        key = d["t_d"]
        if len(cw) and key in cw.index and cw.loc[key].notna().all():
            for col in cw.columns: d[col] = cw.loc[key, col]
        else:
            for col, iv, nb in (("rup_1m", "1m", 240), ("rup_5m", "5m", 48), ("rup_15m", "15m", 16)):
                v = fetch(key, iv, nb); d[col] = v; cw.loc[key, col] = v
                time.sleep(0.1)
            new += 1
    if new: cw.sort_index().to_csv(CACHE_RES)
    print(f"fetched {new} new decision windows ({len(cw)} cached)\n")

    dd = pd.DataFrame(dps).dropna(subset=["rup_1m", "rup_5m", "rup_15m"])
    rec = dd["rec"].values.astype(bool)
    print("resolution agreement (Pearson / Spearman vs rup_1m) and separation:")
    for col in ("rup_1m", "rup_5m", "rup_15m"):
        pr = dd["rup_1m"].corr(dd[col]); sp = dd["rup_1m"].corr(dd[col], method="spearman")
        auc = rank_auc(dd[col][rec], dd[col][~rec])
        print(f"  {col:<8} pearson={pr:>5.2f} spearman={sp:>5.2f}  AUC={auc:.2f}")

    def sc(R):
        cum = np.cumsum(R); dd_ = (np.maximum.accumulate(cum) - cum).max()
        return R.sum() / span, R.sum() / dd_, R[R > 0].sum() / abs(R[R <= 0].sum())
    b = sc(Rbase)
    print(f"\nBASE totR/yr={b[0]:+.2f} totR/DD={b[1]:.2f} PF={b[2]:.2f}")
    keys = [int(x) for x in dd["k"]]
    cfmap = {int(r.k): r.cf for r in dd.itertuples()}
    print(f"{'variant':<10}{'thr':>7}{'n_int':>6}{'totR/yr':>9}{'totR/DD':>9}{'PF':>6}{'null%ile':>10}")
    for col in ("rup_1m", "rup_5m", "rup_15m"):
        th = dd[col].quantile(0.55)
        hit = [int(r.k) for r in dd.itertuples() if getattr(r, col) < th]
        Rm = Rbase.copy()
        for k in hit: Rm[k] = cfmap[k]
        s = sc(Rm)
        draws = []
        for _ in range(500):
            R2 = Rbase.copy()
            for k in RNG.choice(keys, size=len(hit), replace=False): R2[k] = cfmap[k]
            draws.append(sc(R2)[1])
        pc = (s[1] > np.array(draws)).mean() * 100
        print(f"{col:<10}{th:>7.3f}{len(hit):>6}{s[0]:>+9.2f}{s[1]:>9.2f}{s[2]:>6.2f}{pc:>9.0f}%")

if __name__ == "__main__":
    main()
