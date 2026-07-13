"""FROZEN SCREEN (pre-registered 2026-07-12, defence only): does NON-PRICE information predict
the remaining path of a btc15m_L trade WHILE IT IS OPEN?

Context: the earlier "flow exit" green verdict was a LOOKAHEAD ARTIFACT -- Binance (UTC) joined
onto Vantage labels (broker time EET/EEST) made a "trailing 4h" window cover (t-2h, t+2h].
Fixed, the separation vanished (AUC 0.69 -> 0.52). That test also probed ONE cell (decision point
= +2R->+1R giveback, variable = 4h volume ratio) on n=123 points from 2020-09.
This screen probes the SURFACE instead, once, with the multiple comparisons paid for up front.

POPULATION  btc15m_L canonical, Pine-faithful (ZigZag k2 / EMA80 / Pattern-B / pullback limit
            frac 0.30 / RR4 / PDH soft 0.5 / daily-KAMA gate / $15 RT), 2018-10 -> 2026-06.
            Ratchet OFF for the diagnostic: it truncates paths, and we want the untruncated
            label. Any rule that survives must then be re-tested ON TOP of the live form (ratchet
            ON) before it can be believed.

GRID (5 decision points x 6 variables x 5 windows = 150 cells)
  D1  first touch of +1R (running winner, not yet +2R)
  D2  giveback to +1R after touching +2R      <- the cell that died; kept as a NEGATIVE CONTROL
  D3  giveback to breakeven (fill price) after touching +1R
  D4  every 24h in hold (pooled)
  D5  adverse: price reaches lim - 0.5u (halfway to the stop)
  V1  taker buy/sell ratio, PERP        V2  taker buy/sell ratio, SPOT  (independent venue)
  V3  up/down volume ratio, SPOT        (the only one TradingView/Pine can compute)
  V4  taker-imbalance ACCELERATION      (imbalance of the 2nd half of the window minus the 1st)
  V5  open-interest change, PERP        V6  price-down x OI-up  (new shorts entering)
  W   30m / 1h / 2h / 4h / 12h, all ending at the decision bar's CLOSE (past-only, (t-W, t])

LABEL   did the trade eventually finish positive (target/positive time-exit) from that point?
        AUC > 0.5 means "high value of the variable -> the trade recovers".

FALSIFIERS (all pre-registered; a cell is believed only if it clears ALL of them)
  1. MAX-STATISTIC NULL: circularly shift the flow series (preserves its autocorrelation, breaks
     the alignment) 500x and take the BEST |AUC-0.5| over the whole 150-cell grid each time.
     The observed best must exceed the 95th percentile of THAT distribution. Single-cell p-values
     are not looked at -- this is what makes 150 tries honest.
  2. CROSS-VENUE REPLICATION: perp (V1) and spot (V2) must agree in sign at the winning cell.
     (Their 4h taker ratios correlate only 0.44 -> this is a real replication, not a copy.)
  3. IS/OOS: time-split in half; the threshold is fixed on IS and applied to OOS.
  4. PRICE-CONTROL WALL: the same AUC is computed for price-path features (speed to the point,
     giveback speed, ATR percentile). Flow that does not beat the wall is not information.
  PASS = null 95%ile cleared AND V1/V2 same sign AND OOS AUC >= 0.58 AND intervention EV >= +0.3R.

Run: .venv/bin/python scratchpad/inhold_flow_screen.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P
from btc15m_gate_ab import build_entries, kama_rising

ROOT = "/home/angelbell/dev/auto-trade"
START = "2018-10-01"
WINDOWS = {"30m": 2, "1h": 4, "2h": 8, "4h": 16, "12h": 48}      # in 15m bars
NPERM = 500
RNG = np.random.default_rng(20260712)


# ---------------------------------------------------------------- flow features
def load_flow(path):
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index, utc=True, format="mixed")
    d = d[~d.index.duplicated(keep="first")].sort_index()
    # THE FIX: Binance stamps are UTC; Vantage bar labels are broker wall-clock (EET/EEST).
    d.index = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    return d.sort_index()


def build_features(idx):
    """15m-aligned feature series. Every value at bar t uses only bars (t-W, t] -> past-only."""
    spot = load_flow(os.path.join(ROOT, "data/ext_btc_5m_flow_spot.csv"))
    perp = load_flow(os.path.join(ROOT, "data/ext_btc_5m_flow.csv"))
    oi = pd.read_csv(os.path.join(ROOT, "data/ext_btc_oi_metrics.csv"), index_col=0)
    oi.index = pd.to_datetime(oi.index, utc=True, format="mixed")
    oi = oi[~oi.index.duplicated(keep="first")].sort_index()
    oi = oi[oi["sum_open_interest"] > 0]
    oi.index = oi.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")

    S = spot[["taker_buy", "taker_sell", "up_vol", "dn_vol"]].resample("15min").sum().reindex(idx)
    F = perp[["taker_buy", "taker_sell"]].resample("15min").sum().reindex(idx)
    O = oi["sum_open_interest"].resample("15min").last().reindex(idx)
    px = None  # price change comes from the Vantage frame, passed in by the caller

    feats = {}
    for wn, wb in WINDOWS.items():
        def roll(s):
            return s.rolling(wb, min_periods=max(1, wb // 2)).sum()
        sb, ss = roll(S["taker_buy"]), roll(S["taker_sell"])
        fb, fs = roll(F["taker_buy"]), roll(F["taker_sell"])
        uv, dv = roll(S["up_vol"]), roll(S["dn_vol"])
        feats[("V1 taker perp", wn)] = (fb / fs).replace([np.inf, -np.inf], np.nan)
        feats[("V2 taker spot", wn)] = (sb / ss).replace([np.inf, -np.inf], np.nan)
        feats[("V3 up/dn vol (TV)", wn)] = (uv / dv).replace([np.inf, -np.inf], np.nan)
        # V4: is the selling ACCELERATING?  imbalance(2nd half) - imbalance(1st half)
        half = max(1, wb // 2)
        imb = ((S["taker_buy"] - S["taker_sell"]) / (S["taker_buy"] + S["taker_sell"]))
        recent = imb.rolling(half, min_periods=1).mean()
        older = imb.shift(half).rolling(half, min_periods=1).mean()
        feats[("V4 taker accel", wn)] = recent - older
        feats[("V5 dOI perp", wn)] = (O / O.shift(wb) - 1.0).replace([np.inf, -np.inf], np.nan)
        feats[("V6 down x OI-up", wn)] = np.nan       # filled by the caller (needs price)
    return feats, O


# ---------------------------------------------------------------- path replay
def replay(df, E):
    """Mirror of pine_replica_btc15m.walk (ratchet OFF) but recording the decision points."""
    h, l, c, o = (df[k].values for k in ("high", "low", "close", "open"))
    busy = -1
    trades = []
    for (i, e, stop0, tgt, w) in E:
        if i <= busy: continue
        lim = e - P.FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fill = None
        for j in range(i + 1, min(i + 1 + P.FILLWIN, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill = j; break
        if fill is None: continue
        u = lim - stop0
        if l[fill] <= stop0:
            trades.append(dict(fill=fill, exit=fill, R=-1.0, pts={})); busy = fill; continue
        R = None; exit_j = min(fill + P.FWD, len(c) - 1)
        hit1 = hit2 = False
        pts = {}
        for j in range(fill + 1, min(fill + 1 + P.FWD, len(c))):
            if l[j] <= stop0: R = -1.0; exit_j = j; break
            if h[j] >= tgt: R = (tgt - lim) / u; exit_j = j; break
            if hit1 and "D3" not in pts and l[j] <= lim: pts["D3"] = j
            if hit2 and "D2" not in pts and l[j] <= lim + u: pts["D2"] = j
            if "D5" not in pts and l[j] <= lim - 0.5 * u: pts["D5"] = j
            if not hit1 and h[j] >= lim + u:
                hit1 = True; pts["D1"] = j
            if not hit2 and h[j] >= lim + 2 * u: hit2 = True
            if (j - fill) % 96 == 0:                   # every 24h in hold (pooled)
                pts.setdefault("D4", []).append(j)
        if R is None: R = (c[exit_j] - lim) / u
        trades.append(dict(fill=fill, exit=exit_j, R=R, pts=pts))
        busy = exit_j
    return trades


def auc(x, y):
    """AUC of separating y==1 from y==0 using x (rank based). NaNs dropped."""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 < 10 or n0 < 10: return np.nan
    r = pd.Series(x).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    idx = df.index
    gD = kama_rising(df, "1D")
    E = build_entries(df, gD)
    trades = replay(df, E)
    print(f"population: {len(trades)} trades, {sum(t['R'] > 0 for t in trades)} finished positive "
          f"({100*np.mean([t['R'] > 0 for t in trades]):.0f}%), span {idx[0].date()} -> {idx[-1].date()}")

    feats, O = build_features(idx)
    # V6 needs price: down-move x OI-up, both measured over the window
    for wn, wb in WINDOWS.items():
        ret = df["close"] / df["close"].shift(wb) - 1.0
        doi = (O / O.shift(wb) - 1.0).replace([np.inf, -np.inf], np.nan)
        z = lambda s: (s - s.mean()) / s.std()
        feats[("V6 down x OI-up", wn)] = -z(ret) * z(doi)          # high = price fell AND OI rose
    # price-path controls (the known wall)
    atr = ta.atr(df["high"], df["low"], df["close"], 14)
    ctrl = {"C1 ATR%ile": (atr / df["close"]).rolling(2000, min_periods=200).rank(pct=True),
            "C2 ret 4h": df["close"] / df["close"].shift(16) - 1.0,
            "C3 ret 24h": df["close"] / df["close"].shift(96) - 1.0}

    # ---- decision-point tables -------------------------------------------------
    D = {}
    for d in ("D1", "D2", "D3", "D4", "D5"):
        rows = []
        for t in trades:
            p = t["pts"].get(d)
            if p is None: continue
            for b in (p if isinstance(p, list) else [p]):
                rows.append((b, 1 if t["R"] > 0 else 0))
        if rows:
            D[d] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    print("\ndecision points (n, recovery rate = P(trade still finishes positive from here)):")
    for d, (b, y) in D.items():
        print(f"  {d}: n={len(b):>4}  recovery {100*y.mean():>4.0f}%")

    # ---- the 150-cell grid ------------------------------------------------------
    names = ["V1 taker perp", "V2 taker spot", "V3 up/dn vol (TV)", "V4 taker accel",
             "V5 dOI perp", "V6 down x OI-up"]
    fvals = {k: v.values if isinstance(v, pd.Series) else np.asarray(v) for k, v in feats.items()}
    print(f"\nAUC grid (>0.5 = high value -> recovers). "
          f"|AUC-0.5| >= 0.10 would be a real separator.")
    obs = {}
    for d, (b, y) in D.items():
        print(f"\n  --- {d} (n={len(b)}) ---")
        print(f"    {'variable':<20}" + "".join(f"{w:>8}" for w in WINDOWS))
        for v in names:
            row = []
            for w in WINDOWS:
                a = auc(fvals[(v, w)][b], y)
                obs[(d, v, w)] = a
                row.append(a)
            print(f"    {v:<20}" + "".join(f"{x:>8.3f}" if np.isfinite(x) else f"{'--':>8}"
                                           for x in row))
        cr = [f"{c} {auc(s.values[b], y):.3f}" for c, s in ctrl.items()]
        print(f"    {'(price controls)':<20}" + "  ".join(cr))

    best = max((k for k in obs if np.isfinite(obs[k])), key=lambda k: abs(obs[k] - 0.5))
    print(f"\nbest cell: {best} AUC={obs[best]:.3f}  (|AUC-0.5|={abs(obs[best]-0.5):.3f})")

    # ---- FALSIFIER 1: max-statistic null over the whole grid --------------------
    n = len(idx)
    maxstat = []
    keys = [k for k in obs if np.isfinite(obs[k])]
    for _ in range(NPERM):
        s = int(RNG.integers(96 * 7, n - 96 * 7))          # circular shift >= 7 days
        m = 0.0
        for (d, v, w) in keys:
            x = np.roll(fvals[(v, w)], s)
            b, y = D[d]
            a = auc(x[b], y)
            if np.isfinite(a): m = max(m, abs(a - 0.5))
        maxstat.append(m)
    maxstat = np.array(maxstat)
    thr95 = np.percentile(maxstat, 95)
    print(f"\nFALSIFIER 1 -- max-statistic null ({NPERM} circular shifts of the flow series):")
    print(f"  null best-|AUC-0.5| over the 150-cell grid: median {np.median(maxstat):.3f}, "
          f"95th {thr95:.3f}, max {maxstat.max():.3f}")
    print(f"  observed best: {abs(obs[best]-0.5):.3f}  -> "
          f"{'PASS (beats the null)' if abs(obs[best]-0.5) > thr95 else 'FAIL (inside the noise)'}"
          f"   [p = {(maxstat >= abs(obs[best]-0.5)).mean():.3f}]")

    # ---- FALSIFIER 2: cross-venue replication at the best cell ------------------
    d, v, w = best
    b, y = D[d]
    a1, a2 = auc(fvals[("V1 taker perp", w)][b], y), auc(fvals[("V2 taker spot", w)][b], y)
    print(f"\nFALSIFIER 2 -- cross-venue replication at {d}/{w}: "
          f"perp AUC {a1:.3f} | spot AUC {a2:.3f}  -> "
          f"{'same sign' if np.isfinite(a1) and np.isfinite(a2) and (a1-0.5)*(a2-0.5) > 0 else 'DISAGREE'}")

    # ---- FALSIFIER 3: IS/OOS at the best cell -----------------------------------
    half = idx[len(idx) // 2]
    tb = np.array([idx[i] for i in b])
    is_m, oos_m = tb < half, tb >= half
    print(f"FALSIFIER 3 -- IS/OOS at the best cell: "
          f"IS AUC {auc(fvals[best[1:][0] if False else (v, w)][b][is_m], y[is_m]):.3f} "
          f"(n={is_m.sum()}) | OOS AUC {auc(fvals[(v, w)][b][oos_m], y[oos_m]):.3f} "
          f"(n={oos_m.sum()})")


if __name__ == "__main__":
    main()
