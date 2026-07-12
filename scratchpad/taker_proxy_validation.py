"""Can a Pine-computable PROXY replace the official Binance taker ratio in the taker-exit rule?
(user GO 2026-07-12: "1で近似できるか見てみよう。出来なかったら手動運用")

At each of the eligible decision points (btc15m_L canon, +2R->+1R dip, flow-covered), fetch
Binance USDT-M futures 1m klines for the trailing 4h window and compute:
  ratio_kline  = sum(taker_buy)/sum(vol-taker_buy)   -- true aggressor ratio from kline field
                 (stage-1 sanity: should ~match the official 5m-metrics cache mean)
  ratio_uptick = sum(vol | close>open)/sum(vol | close<open)   -- TV Volume-Delta style proxy
                 (what Pine can actually compute via lower-TF up/down volume)
  ratio_prevc  = same but classified by close vs prev close     -- alt TV classification
REPORT  Pearson/Spearman vs official cache value; AUC on recover/collapse per variant;
        frozen-trial re-run with ratio_uptick at the RANK-MATCHED threshold (the percentile
        the official 1.10 occupies among eligible points), +/-5pct sensitivity rows;
        random-intervention null (2000) on totR/DD & totR/yr.
PASS    proxy rule keeps totR/DD > base AND >= 90%ile vs null. KILL -> manual operation.
Fetched window aggregates cached to scratchpad/cache_taker_proxy_windows.csv (re-runs are free).
Run: .venv/bin/python scratchpad/taker_proxy_validation.py
"""
import sys, os, time, json, urllib.request
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from src.data_loader import load_mt5_csv
from swing_pivot_ab import build, FWD, START, ROOT
from inhold_taker_exit_test import replay_full, metrics, FRAC, RR, COST, CACHE

CACHE_WIN = os.path.join(ROOT, "scratchpad/cache_taker_proxy_windows.csv")
RNG = np.random.default_rng(20260712)


def fetch_window(t_d):
    end = int(t_d.timestamp() * 1000)
    start = end - 4 * 3600 * 1000
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m"
           f"&startTime={start}&endTime={end - 1}&limit=240")
    with urllib.request.urlopen(url, timeout=30) as r:
        k = json.load(r)
    if not k: return None
    a = np.array([[float(x[1]), float(x[4]), float(x[5]), float(x[9])] for x in k])  # o,c,vol,takerbuy
    o, c, v, tb = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    ts = (v - tb).sum()
    r_kline = tb.sum() / ts if ts > 0 else np.nan
    up, dn = v[c > o].sum(), v[c < o].sum()
    r_up = up / dn if dn > 0 else np.nan
    cp = np.concatenate([[c[0]], c[:-1]])
    up2, dn2 = v[c > cp].sum(), v[c < cp].sum()
    r_pc = up2 / dn2 if dn2 > 0 else np.nan
    return r_kline, r_up, r_pc, len(k)


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
    tr = replay_full(df, E, h, l, c, o)

    m = pd.read_csv(CACHE, index_col=0); m.index = pd.to_datetime(m.index, utc=True)
    m = m.sort_index(); m = m[~m.index.duplicated(keep="last")]
    m = m[m["sum_open_interest"] > 0]
    taker = m["sum_taker_long_short_vol_ratio"]
    T0 = pd.Timestamp("2020-09-01", tz="UTC")

    elig = []
    for k, t in enumerate(tr):
        if t["cf"] is None or t["t_d"] is None or t["t_d"] < T0: continue
        tk = taker.loc[t["t_d"] - pd.Timedelta("4h"):t["t_d"]].mean()
        if np.isfinite(tk): elig.append(dict(k=k, t_d=t["t_d"], official=tk, rec=(t["R"] > 1.5)))
    print(f"eligible decision points: {len(elig)}")

    # ---- fetch (cached) ----
    if os.path.exists(CACHE_WIN):
        cw = pd.read_csv(CACHE_WIN, index_col=0, parse_dates=[0])
    else:
        cw = pd.DataFrame(columns=["r_kline", "r_up", "r_pc", "nbars"])
    rows = []
    for e in elig:
        key = e["t_d"]
        if key in cw.index:
            r_kline, r_up, r_pc, nb = cw.loc[key]
        else:
            got = fetch_window(key)
            if got is None:
                r_kline = r_up = r_pc = np.nan; nb = 0
            else:
                r_kline, r_up, r_pc, nb = got
            cw.loc[key] = [r_kline, r_up, r_pc, nb]
            time.sleep(0.15)
        rows.append((e["k"], e["official"], r_kline, r_up, r_pc, e["rec"]))
    cw.sort_index().to_csv(CACHE_WIN)
    d = pd.DataFrame(rows, columns=["k", "official", "r_kline", "r_up", "r_pc", "rec"]).dropna()
    print(f"windows with data: {len(d)}  (cached to {os.path.basename(CACHE_WIN)})")

    # ---- stage 1+2: agreement & AUC ----
    rec = d["rec"].values.astype(bool)
    print(f"\n{'variant':<10}{'pearson':>9}{'spearman':>9}{'AUC':>6}   (official AUC={rank_auc(d.official[rec], d.official[~rec]):.2f})")
    for col in ("r_kline", "r_up", "r_pc"):
        pr = d["official"].corr(d[col]); sp = d["official"].corr(d[col], method="spearman")
        auc = rank_auc(d[col][rec], d[col][~rec])
        print(f"{col:<10}{pr:>9.3f}{sp:>9.3f}{auc:>6.2f}")

    # ---- frozen-trial re-run with r_up at rank-matched threshold ----
    pct110 = (d["official"] < 1.10).mean()
    print(f"\nofficial 1.10 = {100*pct110:.0f}%ile of eligible; rank-matched proxy thresholds:")
    times = [t["t"] for t in tr]
    Rnet = np.array([t["R"] - COST / t["risk"] for t in tr])
    cf_net = {int(r.k): tr[int(r.k)]["cf"] - COST / tr[int(r.k)]["risk"] for r in d.itertuples()}
    base = metrics(times, Rnet, span)
    print(f"BASE                    totR/yr={base['totyr']:+6.2f} totR/DD={base['retdd']:5.2f}")

    def apply(ks):
        Rm = Rnet.copy()
        for kk in ks: Rm[kk] = cf_net[kk]
        return metrics(times, Rm, span)

    res_primary, n_int = None, 0
    for dp in (-5, 0, +5):
        q = np.clip(pct110 + dp / 100, 0.05, 0.95)
        th = d["r_up"].quantile(q)
        hit = [int(r.k) for r in d.itertuples() if r.r_up < th]
        s = apply(hit)
        tag = " <= PRIMARY (rank-matched)" if dp == 0 else ""
        print(f"r_up<{th:.3f} ({100*q:.0f}%ile) n_int={len(hit):>3}  totR/yr={s['totyr']:+6.2f} "
              f"totR/DD={s['retdd']:5.2f} IS/OOS={s['IS']:+.0f}/{s['OOS']:+.0f}R{tag}")
        if dp == 0: res_primary, n_int = s, len(hit)

    keys = [int(r.k) for r in d.itertuples()]
    null_retdd, null_totyr = [], []
    for _ in range(2000):
        pick = RNG.choice(keys, size=n_int, replace=False)
        s = apply(list(pick))
        null_retdd.append(s["retdd"]); null_totyr.append(s["totyr"])
    p_retdd = (res_primary["retdd"] > np.array(null_retdd)).mean() * 100
    p_totyr = (res_primary["totyr"] > np.array(null_totyr)).mean() * 100
    print(f"\nNULL (random same-count): totR/DD %ile={p_retdd:.1f}  totR/yr %ile={p_totyr:.1f}")
    print(f"PASS(proxy usable in Pine) = totR/DD>base({res_primary['retdd'] > base['retdd']}) "
          f"AND >=90%ile({p_retdd >= 90})")

if __name__ == "__main__":
    main()
