"""In-hold flow/volume DIAGNOSTIC (no exit rule, measurement only), user request 2026-07-12.

QUESTION  At the exact decision point where the price-path ratchet failed -- trade touched +2R
          then fell back to +1R -- do NON-PRICE in-hold variables (Binance taker flow, open
          interest, broker tick-volume) separate "recovers to target" from "collapses"?
          Price-path alone could not (57% of target winners dip below +1R = no separation plane).

POPULATION canon btc15m_L (Pattern B / zigzag k2 / ema80 / RR4 / daily-KAMA gate / frac0.3
          pullback-limit; machinery = faithful copy of btc15m_pullback_gauntlet.py via
          swing_pivot_ab.py, tie-back verified there). Trades that touch >= lim+2R intrabar
          before the stop, then later dip to <= lim+1R (the decision bar d).
OUTCOME   recover = the trade's canonical walk hits the full target after d; collapse = stop/timeout.
FEATURES at bar d (all trailing/causal):
  price-path controls: bars fill->2R, bars 2Rtouch->d
  broker tick-volume : down/up-bar volume ratio last 24 bars; vol trend (8-bar / 96-bar mean)
  Binance 5m cache   : taker L/S vol ratio mean 4h & 24h; dOI 24h; dOI since fill
          (flow features only for fills >= 2020-09-01; cache deduped, OI=0 rows dropped)
REPORT    per feature: median/IQR per group, rank-AUC, quartile-bucket recovery table.
          Pre-registered read: AUC ~0.5 with overlapping IQRs = same wall as price -> close the axis.
Run: .venv/bin/python scratchpad/inhold_flow_diag.py
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import io, contextlib
from src.data_loader import load_mt5_csv
from swing_pivot_ab import build, BO, FWD, START, ROOT   # canon machinery (tie-back proven)

FRAC = 0.3
RR = 4.0
CACHE = os.path.join(ROOT, "data/ext_btc_oi_metrics.csv")


def replay(df, E, h, l, c, vol):
    """canon frac-limit walk, extended to record the full path per trade."""
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
        risk = lim - stop
        lv2, lv1 = lim + 2 * risk, lim + 1 * risk
        if l[fill_j] <= stop:
            busy = fill_j; continue                      # stopped on fill bar; never touched 2R
        exit_j = min(fill_j + FWD, len(c) - 1); res = "timeout"
        t2 = None; d = None
        for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
            if l[j] <= stop: res = "stop"; exit_j = j; break
            if t2 is None and h[j] >= lv2: t2 = j
            if t2 is not None and d is None and j > t2 and l[j] <= lv1: d = j
            if h[j] >= tgt: res = "tgt"; exit_j = j; break
        busy = exit_j
        if t2 is None or d is None or d >= exit_j and res == "tgt" and d == exit_j: pass
        if t2 is None or d is None: continue             # never touched 2R, or never dipped to 1R
        out.append(dict(fill_j=fill_j, t2=t2, d=d, exit_j=exit_j, res=res,
                        t_fill=df.index[fill_j], t_d=df.index[d]))
    return out


def rank_auc(a, b):
    """P(random A > random B) via rank sum. a = recover group values, b = collapse."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 8 or len(b) < 8: return np.nan, len(a), len(b)
    allv = np.concatenate([a, b])
    r = pd.Series(allv).rank().values
    ra = r[:len(a)].sum()
    auc = (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return auc, len(a), len(b)


def bucket_table(vals, rec, q=4):
    v = np.asarray(vals, float); r = np.asarray(rec, bool)
    ok = np.isfinite(v)
    v, r = v[ok], r[ok]
    if len(v) < 24: return "  (n too small)"
    edges = np.quantile(v, np.linspace(0, 1, q + 1))
    rows = []
    for k in range(q):
        m = (v >= edges[k]) & (v <= edges[k + 1] if k == q - 1 else v < edges[k + 1])
        rows.append(f"Q{k+1} n={m.sum():>3} rec={100*r[m].mean():>4.0f}%")
    return "  " + " | ".join(rows)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    raw = pd.read_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv"))
    raw["time"] = pd.to_datetime(raw["time"], format="%Y.%m.%d %H:%M", utc=True)
    vol = raw.set_index("time")["tick_volume"].reindex(df.index).ffill()

    E, h, l, c, _ = build(df, RR, "zigzag", 2.0)
    tr = replay(df, E, h, l, c, vol)
    rec = np.array([t["res"] == "tgt" for t in tr])
    print(f"population: fills touching +2R then dipping to +1R: n={len(tr)}  "
          f"base recovery-to-target rate={100*rec.mean():.1f}%  "
          f"(stop {sum(t['res']=='stop' for t in tr)}, timeout {sum(t['res']=='timeout' for t in tr)})")

    # ---- features ----
    m = pd.read_csv(CACHE, index_col=0)
    m.index = pd.to_datetime(m.index, utc=True)
    m = m.sort_index(); m = m[~m.index.duplicated(keep="last")]
    m = m[m["sum_open_interest"] > 0]
    taker = m["sum_taker_long_short_vol_ratio"]; oi = m["sum_open_interest"]

    vv = vol.values
    feats = {k: [] for k in ("bars_fill_2R", "bars_2R_d", "voldnup_6h", "voltrend",
                             "taker_4h", "taker_24h", "dOI_24h", "dOI_fill")}
    has_flow = []
    for t in tr:
        d = t["d"]; td = t["t_d"]
        feats["bars_fill_2R"].append(t["t2"] - t["fill_j"])
        feats["bars_2R_d"].append(d - t["t2"])
        w = slice(max(0, d - 23), d + 1)
        cl = c[w]; op = df["open"].values[w]; vw = vv[w]
        dn = vw[cl < op].sum(); up = vw[cl > op].sum()
        feats["voldnup_6h"].append(dn / up if up > 0 else np.nan)
        feats["voltrend"].append(vv[max(0, d-7):d+1].mean() / vv[max(0, d-95):d+1].mean())
        if td >= pd.Timestamp("2020-09-01", tz="UTC"):
            has_flow.append(True)
            feats["taker_4h"].append(taker.loc[td - pd.Timedelta("4h"):td].mean())
            feats["taker_24h"].append(taker.loc[td - pd.Timedelta("24h"):td].mean())
            o_now = oi.loc[:td].iloc[-1] if len(oi.loc[:td]) else np.nan
            o_24 = oi.loc[:td - pd.Timedelta("24h")]
            feats["dOI_24h"].append(o_now / o_24.iloc[-1] - 1 if len(o_24) else np.nan)
            o_f = oi.loc[:t["t_fill"]]
            feats["dOI_fill"].append(o_now / o_f.iloc[-1] - 1 if len(o_f) else np.nan)
        else:
            has_flow.append(False)
            for k in ("taker_4h", "taker_24h", "dOI_24h", "dOI_fill"): feats[k].append(np.nan)

    print(f"flow-feature coverage: {sum(has_flow)}/{len(tr)} decision points >= 2020-09\n")
    print(f"{'feature':<14}{'AUC':>6}{'nR/nC':>10}   recover med[IQR]      collapse med[IQR]   quartile recovery")
    for k, v in feats.items():
        v = np.array(v, float)
        a, b = v[rec], v[~rec]
        auc, na, nb = rank_auc(a, b)
        qa = lambda x: f"{np.nanmedian(x):+.3f}[{np.nanquantile(x,.25):+.3f},{np.nanquantile(x,.75):+.3f}]"
        print(f"{k:<14}{auc:>6.2f}{f'{na}/{nb}':>10}   {qa(a):<21} {qa(b):<21}")
        print(bucket_table(v, rec))

if __name__ == "__main__":
    main()
