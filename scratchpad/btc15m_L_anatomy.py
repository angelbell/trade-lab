"""btc15m_L_anatomy.py -- DIAGNOSTIC decomposition of the btc15m_L adopted-candidate leg.

No strategy changes, no parameter mining. Reproduces the EXACT construction from
scratchpad/pwh_adoption.py lines ~123-132 (data density guard, run_bo signal, net-R,
PDH soft weight) and then decomposes it: per-year, drawdown anatomy, loser/winner
excursion anatomy (MFE/MAE reconstructed from the raw 15m bars, causal), hold time,
context splits, ONE pre-registered pullback-vs-market-fill comparison, and monthly
lumpiness.

Causal discipline: every indicator here (ATR14, daily KAMA14) is shift(1)'d before
use; the MFE/MAE replay walks bars strictly AFTER the entry bar and (to avoid same-
bar stop/target ambiguity) excludes the bar that actually resolves the trade from
the excursion accumulation.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # scratchpad/
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample, kama_adaptive
from radar_gate_race import BASE

pd.set_option("display.width", 160)

RR = BASE["rr"]          # 4.0 -- fixed reward:risk used by tp_mode="rr" (BASE default)
FWD = BASE["fwd"]        # 500 bars max hold
COST = 15.0              # BTC round-trip $ cost per lot-risk-unit (live canon)


# --------------------------------------------------------------------------- construction
def build_data():
    b = load_mt5_csv("data/vantage_btcusd_m15.csv")
    cnt = b.groupby(b.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    d15 = resample(b[b.index.date >= okd.index[0]], "15min")
    return d15


def run_leg(d15, pullback_frac=0.3):
    kw = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min"}
    if pullback_frac and pullback_frac > 0:
        kw["pullback_frac"] = pullback_frac
    return run_bo(d15, SimpleNamespace(**kw))


def pdh_weight(d15, tb):
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ix = d15.index.get_indexer(tb["time"])
    ab = tb["e_px"].values > pdh[ix]
    w = np.where(ab, 1.0, 0.5)
    return w, ab, ix


# --------------------------------------------------------------------------- summary line
def summary(R, times, label=""):
    R = np.asarray(R, dtype=float)
    n = len(R)
    if n == 0:
        print(f"{label:<10} n=0"); return
    span_yr = max((times.max() - times.min()).days / 365.25, 1e-9)
    win = (R > 0).mean() * 100
    wins = R[R > 0].sum(); losses = -R[R < 0].sum()
    PF = wins / losses if losses > 0 else np.inf
    meanR = R.mean()
    totR = R.sum()
    eq = np.cumsum(R)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq).max()
    totR_dd = totR / dd if dd > 0 else np.inf
    print(f"  {label:<10} n={n:4d}  span={span_yr:5.2f}y  N/yr={n/span_yr:5.1f}  win%={win:5.1f}  "
          f"PF={PF:5.2f}  meanR={meanR:+.3f}  totR/yr={totR/span_yr:+7.1f}  maxDD(R)={dd:6.1f}  "
          f"totR/DD={totR_dd:6.2f}")


# --------------------------------------------------------------------------- MFE/MAE replay
def replay(d15, tb, prog_bars=(8, 16, 32)):
    """Reconstruct each trade's holding window WITHOUT re-simulating stop/target
    triggers (an earlier version of this script did that and it was WRONG for
    pullback-limit fills: the code fixes the target at the ORIGINAL market level,
    e_market + RR*risk_market, not at e_px + RR*risk_fill -- since risk shrinks
    and the reward grows once filled on the pullback, e_px+RR*risk_fill sits far
    BELOW the true target and falsely triggers an early "win" on ~26% of trades,
    confirmed by comparing the wrong-target replay's R against tb['R'] (205/789
    mismatched). tb has no column for the original market e/stop/tgt, so the
    target can't be reconstructed directly.

    Instead: tb['hold'] is the EXACT (days) delta between the real entry and real
    exit bar timestamps, already computed by run_bo's own walk. entry_time+hold
    lands on the real exit bar (to within float64 ns-precision), so
    get_indexer(..., method='nearest') recovers the true exit bar with no
    re-simulation and no lookahead (we only use tb's own already-computed outcome
    to locate WHERE it happened, then read the raw h/l/c bars in between).

    mfe/mae: running max/min excursion in R units over bars STRICTLY BETWEEN entry
    and the resolving bar (excludes the resolving bar -- avoids conflating a same-
    bar stop-vs-target ordering ambiguity). mfe_full/mae_full: same but INCLUDING
    the resolving bar (used where "how far did price actually get" matters, not
    "before the stop"). prog{b}: mark-to-market close-vs-entry R at +b bars if the
    trade is still open then, else the realized final R (position already closed)."""
    h, l, c = d15["high"].values, d15["low"].values, d15["close"].values
    n = len(c)
    idx = d15.index
    pos = idx.get_indexer(tb["time"])
    exit_time = tb["time"] + pd.to_timedelta(tb["hold"].values, unit="D")
    exit_pos = idx.get_indexer(exit_time, method="nearest")
    snap_err = np.abs((idx[exit_pos] - exit_time).values.astype("timedelta64[s]").astype(float))
    out = []
    for k in range(len(tb)):
        i = int(pos[k]); j_exit = int(exit_pos[k])
        e_px = float(tb["e_px"].values[k])
        risk = float(tb["risk"].values[k])
        stop = e_px - risk
        R = float(tb["R"].values[k])
        j_exit = min(max(j_exit, i), n - 1)
        pre = range(i + 1, j_exit) if j_exit > i else range(0)
        mfe = max([(h[j] - e_px) / risk for j in pre], default=np.nan)
        mae = min([(l[j] - e_px) / risk for j in pre], default=np.nan)
        full = range(i + 1, j_exit + 1) if j_exit > i else range(0)
        mfe_full = max([(h[j] - e_px) / risk for j in full], default=np.nan)
        mae_full = min([(l[j] - e_px) / risk for j in full], default=np.nan)
        bars_half = next((j - i for j in full if (h[j] - e_px) / risk >= 0.5), np.nan)
        prog = {}
        for b in prog_bars:
            if j_exit <= i + b:
                prog[b] = R
            elif i + b < n:
                prog[b] = (c[i + b] - e_px) / risk
            else:
                prog[b] = np.nan
        out.append(dict(entry_bar=i, exit_bar=j_exit, bars_held=j_exit - i, R_replay=R,
                         mfe=mfe, mae=mae, mfe_full=mfe_full, mae_full=mae_full,
                         bars_to_half_r=bars_half, snap_err_s=snap_err[k],
                         **{f"prog{b}": prog[b] for b in prog_bars}))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- DD episodes
def dd_episodes(R, times, top=3):
    eq = np.cumsum(R)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    episodes = []
    i = 0
    n = len(dd)
    while i < n:
        if dd[i] <= 1e-12:
            i += 1; continue
        j = i
        while j < n and dd[j] > 1e-12:
            j += 1
        seg = dd[i:j]
        depth = seg.max()
        trough_rel = int(np.argmax(seg))
        trough_idx = i + trough_rel
        peak_idx = i - 1 if i > 0 else 0     # last index where dd==0 before the episode
        episodes.append((peak_idx, trough_idx, j - 1, depth))
        i = j
    episodes.sort(key=lambda e: -e[3])
    out = []
    for peak_idx, trough_idx, end_idx, depth in episodes[:top]:
        seg_R = R[peak_idx + 1: trough_idx + 1]
        n_trades = trough_idx - peak_idx
        # max consecutive losers within peak->trough
        best = cur = 0
        for r in seg_R:
            if r < 0:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        months = sorted(set(pd.DatetimeIndex(times[peak_idx + 1: trough_idx + 1]).to_period("M").astype(str)))
        out.append(dict(peak_date=times[peak_idx], trough_date=times[trough_idx], depth=depth,
                         n_trades=n_trades, max_losers_in_row=best, months=months,
                         recovered=(end_idx < n - 1) or dd[-1] <= 1e-12))
    return out


def main():
    d15 = build_data()
    tb = run_leg(d15, pullback_frac=0.3)
    Rn = tb["R"].values - COST / tb["risk"].values
    w, ab, ix = pdh_weight(d15, tb)
    Rw = Rn * w
    times = pd.DatetimeIndex(tb["time"])

    print("=" * 100)
    print("ANCHOR -- reproduce btc15m_L exactly (ledger: n~=238, meanR~=+0.94 unweighted)")
    print("=" * 100)
    summary(Rn, times, "Rn (raw)")
    summary(Rw, times, "Rw (live)")

    # ---------------------------------------------------------------- 1. PER-YEAR
    print("\n" + "=" * 100)
    print("1. PER-YEAR (live form Rw)")
    print("=" * 100)
    dfy = pd.DataFrame({"time": times, "Rw": Rw, "w5": (w == 0.5)})
    dfy["year"] = dfy["time"].dt.year
    print(f"  {'year':>6} {'n':>5} {'win%':>6} {'PF':>6} {'totR':>8} {'maxDD-yr(R)':>12} {'share w=0.5':>12}")
    for y, g in dfy.groupby("year"):
        R = g["Rw"].values
        win = (R > 0).mean() * 100
        wins = R[R > 0].sum(); losses = -R[R < 0].sum()
        PF = wins / losses if losses > 0 else np.inf
        eq = np.cumsum(R); peak = np.maximum.accumulate(eq); ddy = (peak - eq).max()
        print(f"  {y:>6} {len(g):>5} {win:>6.1f} {PF:>6.2f} {R.sum():>8.1f} {ddy:>12.1f} {g['w5'].mean()*100:>11.1f}%")

    # ---------------------------------------------------------------- 2. DD ANATOMY
    print("\n" + "=" * 100)
    print("2. DD ANATOMY (top 3 deepest episodes, cumulative Rw)")
    print("=" * 100)
    for e in dd_episodes(Rw, times, top=3):
        print(f"  peak={e['peak_date']}  trough={e['trough_date']}  depth={e['depth']:.1f}R  "
              f"n_trades={e['n_trades']}  max_losers_in_row={e['max_losers_in_row']}  "
              f"recovered_by_end={e['recovered']}")
        print(f"    months involved: {', '.join(e['months'])}")

    # ---------------------------------------------------------------- replay for 3 & 4
    rep = replay(d15, tb)
    bad_snap = (rep["snap_err_s"].values > 1.0).sum()   # exit bar located via time+hold, method='nearest'
    print(f"\n  [replay validation: exit-bar located via entry_time+hold snapped to the nearest real bar; "
          f"{bad_snap}/{len(tb)} trades snapped >1s away from an exact match "
          f"(max snap error {rep['snap_err_s'].max():.3f}s) -- an EARLIER version of this reconstruction "
          f"re-simulated stop/target and was WRONG for pullback-fills (205/789 mismatched R); this "
          f"hold-based method reads the real exit bar directly, no re-simulation]")

    # ---------------------------------------------------------------- 3. LOSER / WINNER ANATOMY
    print("\n" + "=" * 100)
    print("3. LOSER ANATOMY (MFE before stop, R units) / WINNER mirror (MAE before target)")
    print("=" * 100)
    is_loss = Rw < 0
    is_win = Rw > 0
    mfe_l = rep.loc[is_loss, "mfe"].values
    mae_w = rep.loc[is_win, "mae"].values
    print(f"  losers n={is_loss.sum()}  (of which MFE available: {np.isfinite(mfe_l).sum()})")
    mfe_l_f = mfe_l[np.isfinite(mfe_l)]
    for th in (0.3, 0.5, 1.0, 2.0):
        print(f"    reached >= +{th:.1f}R before stop: {(mfe_l_f >= th).mean()*100:5.1f}%")
    print(f"    median={np.median(mfe_l_f):+.3f}R  std={np.std(mfe_l_f):.3f}R")
    print(f"  winners n={is_win.sum()}  (of which MAE available: {np.isfinite(mae_w).sum()})")
    mae_w_f = mae_w[np.isfinite(mae_w)]
    for th in (-0.3, -0.5):
        print(f"    dipped <= {th:.1f}R before target: {(mae_w_f <= th).mean()*100:5.1f}%")
    print(f"    median={np.median(mae_w_f):+.3f}R  std={np.std(mae_w_f):.3f}R")

    # ---------------------------------------------------------------- 4. HOLD TIME
    print("\n" + "=" * 100)
    print("4. HOLD TIME (bars, 15m each)")
    print("=" * 100)
    bh = rep["bars_held"].values
    for tag, mask in [("all", np.ones(len(bh), bool)), ("winners", is_win), ("losers", is_loss)]:
        v = bh[mask]
        print(f"  {tag:<8} n={mask.sum():4d}  median={np.median(v):6.1f}  p90={np.percentile(v,90):6.1f}")

    # ---------------------------------------------------------------- 5. CONTEXT SPLITS
    print("\n" + "=" * 100)
    print("5. CONTEXT SPLITS (meanR + n, live form Rw, no thresholds tuned)")
    print("=" * 100)
    atr15 = ta.atr(d15["high"], d15["low"], d15["close"], 14).shift(1)
    atr_at_e = atr15.values[ix]
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    dist_atr = (tb["e_px"].values - pdh[ix]) / atr_at_e
    stopw_atr = tb["risk"].values / atr_at_e

    dck = d15["close"].resample("1D").last().dropna()
    kmg = kama_adaptive(dck, 14)
    krise = (kmg > kmg.shift(1)).shift(1).reindex(d15.index, method="ffill").fillna(False)
    kama_dir = np.where(krise.values[ix], "rising", "falling")

    yrs = sorted(dfy["year"].unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    ishalf = np.where(dfy["year"].values < (half if half is not None else -1), "IS", "OOS")

    def bucket_report(name, keys):
        print(f"  ({name})")
        d = pd.DataFrame({"key": keys, "Rw": Rw})
        for k, g in d.groupby("key"):
            print(f"    {str(k):<14} n={len(g):4d}  meanR={g['Rw'].mean():+.3f}")

    bucket_report("a. PDH label", np.where(ab, "above_PDH", "below_PDH"))
    try:
        qd = pd.qcut(dist_atr, 4, duplicates="drop")
        bucket_report("b. dist(e_px-PDH)/ATR quartile", qd.astype(str))
    except Exception as ex:
        print(f"  (b. dist quartile failed: {ex})")
    hourblk = (times.hour // 4 * 4)
    bucket_report("c. hour-of-day (4h blocks, UTC/broker time)", [f"{h:02d}-{h+4:02d}" for h in hourblk])
    bucket_report("d. weekday", times.day_name())
    bucket_report("e. daily KAMA(14) direction at entry", kama_dir)
    try:
        qs = pd.qcut(stopw_atr, 4, duplicates="drop")
        bucket_report("f. stop width / ATR quartile", qs.astype(str))
    except Exception as ex:
        print(f"  (f. stop-width quartile failed: {ex})")
    bucket_report("g. entry-year half (IS/OOS)", ishalf)

    # ---------------------------------------------------------------- 6. UNFILLED LIMITS
    print("\n" + "=" * 100)
    print("6. UNFILLED LIMITS -- ONE pre-registered comparison: pullback_frac=0.3 vs market-at-signal (0)")
    print("=" * 100)
    tb0 = run_leg(d15, pullback_frac=0.0)
    Rn0 = tb0["R"].values - COST / tb0["risk"].values
    times0 = pd.DatetimeIndex(tb0["time"])
    summary(Rn, times, "pullback(0.3)")
    summary(Rn0, times0, "market(0.0)")
    print(f"  n diff (market - pullback) = {len(tb0) - len(tb)}")

    # Match: the pullback trade table only stores the FILL bar/time, not the original
    # SIGNAL bar (that's internal to run_bo's entries list and isn't returned), so exact
    # signal-to-signal matching isn't reconstructable without editing breakout_wave.py.
    # Best available proxy: each pullback FILL must occur within [signal_time,
    # signal_time+fwd*15min] of its own (unknown) originating market-run signal, and a
    # market-run signal is entered immediately (time = signal time itself) -- so backward
    # asof-match each pullback fill-time to the NEAREST market signal at or before it,
    # within that same fwd tolerance, one-to-one (drop a market row once claimed).
    tm = pd.DataFrame({"time": times0, "Rn": Rn0}).sort_values("time").reset_index(drop=True)
    tm["tm_idx"] = tm.index
    tp_t = pd.DataFrame({"time": pd.Series(times).sort_values().reset_index(drop=True)})
    tol = pd.Timedelta(minutes=15 * FWD)
    merged = pd.merge_asof(tp_t, tm, on="time", direction="backward", tolerance=tol)
    claimed = merged["tm_idx"].dropna()
    dup = int(claimed.duplicated().sum())         # >0 if two fills asof-matched the same market row
    claimed_u = claimed.drop_duplicates()
    matched_mask = tm["tm_idx"].isin(claimed_u)
    skipped = tm.loc[~matched_mask]
    print(f"  proxy match (backward asof, tol={FWD}bars, one-to-one; NOT exact -- tb has no signal-bar "
          f"column for pullback fills, only the fill bar): {matched_mask.sum()} matched / {len(tm)} market "
          f"trades  ({dup} asof-collisions dropped)  vs naive n-diff {len(tb0)-len(tb)}")
    print(f"  net-R sum of the UNMATCHED (skipped-by-limit) market trades: "
          f"{skipped['Rn'].sum():+.1f}R  over {len(skipped)} trades  (mean {skipped['Rn'].mean():+.3f}R)"
          if len(skipped) else "  (no unmatched market trades)")

    # ---------------------------------------------------------------- 7. LUMPINESS
    print("\n" + "=" * 100)
    print("7. WEEKLY/MONTHLY LUMPINESS (monthly Rw sums)")
    print("=" * 100)
    mon = pd.Series(Rw, index=times).groupby(times.to_period("M")).sum()
    print(f"  months n={len(mon)}  median={mon.median():+.2f}  sd={mon.std():.2f}  "
          f"% negative={ (mon<0).mean()*100:.1f}%")
    worst = mon.sort_values().head(3)
    print("  worst 3 months:")
    for p, v in worst.items():
        print(f"    {p}: {v:+.2f}R")

    # ---------------------------------------------------------------- 8. RANGE-LABEL DECOMPOSITION
    print("\n" + "=" * 100)
    print("8. RANGE-LABEL DECOMPOSITION (SPEC EXTENSION -- user hypothesis: losses cluster in ranges)")
    print("=" * 100)

    def class_stats(labels, R, title, extra_note=""):
        print(f"  ({title}){extra_note}")
        d = pd.DataFrame({"lab": labels, "R": R})
        for k, g in d.groupby("lab"):
            R_ = g["R"].values
            win = (R_ > 0).mean() * 100
            wins = R_[R_ > 0].sum(); losses = -R_[R_ < 0].sum()
            PF = wins / losses if losses > 0 else np.inf
            print(f"    {str(k):<24} n={len(g):4d}  win%={win:5.1f}  PF={PF:5.2f}  "
                  f"meanR={R_.mean():+.3f}  totR={R_.sum():+7.1f}")

    # -- 8a. EX-POST market label (uses future data -- descriptive ONLY, NOT a gate/filter) --
    print("  *** 8a is EX-POST (uses +5d of future daily closes around entry) -- descriptive, NOT a gate ***")
    daily_close = d15["close"].resample("1D").last().dropna()

    def er_window(center_date):
        seg = daily_close.loc[center_date - pd.Timedelta(days=5): center_date + pd.Timedelta(days=5)]
        if len(seg) < 4:
            return np.nan
        path = seg.diff().abs().sum()
        return abs(seg.iloc[-1] - seg.iloc[0]) / path if path > 0 else np.nan

    entry_dates = times.normalize()
    er_post = np.array([er_window(d) for d in entry_dates])
    valid = ~np.isnan(er_post)
    lbl_post = np.full(len(er_post), "NA", dtype=object)
    terc = pd.qcut(er_post[valid], 3, labels=["range(low ER)", "mixed", "trend(high ER)"])
    lbl_post[valid] = terc.astype(str)
    class_stats(lbl_post, Rw, "8a. ex-post ER[-5d,+5d] tercile x class, live Rw")

    # -- 8b. ENTRY-TIME causal features (shift-safe), quartile-split --
    print("\n  *** 8b entry-time CAUSAL features (all shift(1)'d before entries can see them) ***")

    def rolling_er(close, N):
        net = close.diff(N).abs()
        path = close.diff().abs().rolling(N).sum()
        return (net / path).clip(0, 1).shift(1)

    er24 = rolling_er(d15["close"], 96).values[ix]     # trailing 24h (96x15m)
    er72 = rolling_er(d15["close"], 288).values[ix]     # trailing 72h

    d4 = resample(d15, "240min")
    adx4 = ta.adx(d4["high"], d4["low"], d4["close"], 14)["ADX_14"].shift(1)
    adx4_al = adx4.reindex(d15.index, method="ffill").values[ix]

    def bb_width_pct100(close):
        bb = ta.bbands(close, 20)
        up = [c for c in bb.columns if c.startswith("BBU")][0]
        lo = [c for c in bb.columns if c.startswith("BBL")][0]
        mid = [c for c in bb.columns if c.startswith("BBM")][0]
        width = (bb[up] - bb[lo]) / bb[mid]
        vals = width.values; n = len(vals)
        pct = np.full(n, np.nan)
        for k in range(100, n):
            if np.isnan(vals[k]):
                continue
            pct[k] = np.nanmean(vals[k - 100:k] < vals[k])
        return pd.Series(pct, index=width.index).shift(1)

    bbw_pct = bb_width_pct100(d4["close"]).reindex(d15.index, method="ffill").values[ix]

    dc4 = d15["close"].resample("240min").last().dropna()
    kmg4 = kama_adaptive(dc4, 14)
    krise4 = (kmg4 > kmg4.shift(1)).shift(1).reindex(d15.index, method="ffill").fillna(False)
    kama4_dir = krise4.values[ix]
    kama_agree = np.where(kama4_dir & krise.values[ix], "both_rising",
                  np.where((~kama4_dir) & (~krise.values[ix]), "both_falling", "split"))

    def days_since_high(daily_high, window=20):
        vals = daily_high.values; n = len(vals)
        out = np.full(n, np.nan)
        for i2 in range(window - 1, n):
            seg = vals[i2 - window + 1:i2 + 1]
            out[i2] = (window - 1) - np.argmax(seg)
        return pd.Series(out, index=daily_high.index).shift(1)

    dh_daily = d15["high"].resample("1D").max().dropna()
    dl_daily = d15["low"].resample("1D").min().dropna()
    dsh = days_since_high(dh_daily, 20).reindex(d15.index, method="ffill").values[ix]

    hi20 = dh_daily.rolling(20).max().shift(1)
    lo20 = dl_daily.rolling(20).min().shift(1)
    hi20_al = hi20.reindex(d15.index, method="ffill").values[ix]
    lo20_al = lo20.reindex(d15.index, method="ffill").values[ix]
    range_pos = (tb["e_px"].values - lo20_al) / (hi20_al - lo20_al)

    feat8b = {
        "ER trailing 24h": er24, "ER trailing 72h": er72, "4h ADX(14)": adx4_al,
        "4h BBwidth pct-of-past100": bbw_pct, "daily-KAMAxr4h-KAMA agree": kama_agree,
        "days since last 20d high": dsh, "range pos (20d lo-hi)": range_pos,
        "(e_px-PDH)/ATR [see sec.5b]": dist_atr,
    }
    for name, vals in feat8b.items():
        if name == "daily-KAMAxr4h-KAMA agree":
            class_stats(vals, Rw, f"8b. {name}")
            continue
        try:
            q = pd.qcut(pd.Series(vals), 4, duplicates="drop")
            class_stats(q.astype(str).values, Rw, f"8b. {name} quartile")
        except Exception as ex:
            print(f"  (8b. {name} quartile failed: {ex})")

    # -- 8c. cross: ex-post RANGE vs TREND tercile -- median(+-std) of each 8b feature --
    print("\n  *** 8c. cross: median(+-std) of each entry-time feature, ex-post RANGE vs TREND tercile ***")
    is_range = lbl_post == "range(low ER)"
    is_trend = lbl_post == "trend(high ER)"
    for name, vals in feat8b.items():
        if name == "daily-KAMAxr4h-KAMA agree":
            continue
        v = np.asarray(vals, dtype=float)
        vr, vt = v[is_range], v[is_trend]
        vr, vt = vr[~np.isnan(vr)], vt[~np.isnan(vt)]
        print(f"    {name:<28} RANGE med={np.median(vr):+8.3f} sd={np.std(vr):7.3f} (n{len(vr)})  |  "
              f"TREND med={np.median(vt):+8.3f} sd={np.std(vt):7.3f} (n{len(vt)})")

    # ---------------------------------------------------------------- 9. EXIT/RUNNER DISCRIMINATION
    print("\n" + "=" * 100)
    print("9. EXIT / RUNNER DISCRIMINATION (SPEC EXTENSION)")
    print("=" * 100)
    mfe_full = rep["mfe_full"].values
    mfe_full_ok = mfe_full[~np.isnan(mfe_full)]

    print("  9a. outcome ladder P(MFE_full >= X R), whole leg")
    for th in (0.5, 1, 2, 3, 4):
        print(f"    >= {th:.1f}R : {(mfe_full_ok >= th).mean()*100:5.1f}%   (n={len(mfe_full_ok)})")

    print("\n  9b. P(reach >= 2R) by each 8b feature's quartile")
    reach2 = (mfe_full >= 2.0).astype(float)
    for name, vals in feat8b.items():
        if name == "daily-KAMAxr4h-KAMA agree":
            d = pd.DataFrame({"lab": vals, "r2": reach2})
            print(f"    ({name})")
            for k, g in d.groupby("lab"):
                print(f"      {str(k):<14} n={len(g):4d}  P(>=2R)={g['r2'].mean()*100:5.1f}%")
            continue
        try:
            q = pd.qcut(pd.Series(vals), 4, duplicates="drop")
            d = pd.DataFrame({"lab": q.astype(str), "r2": reach2})
            print(f"    ({name} quartile)")
            for k, g in d.groupby("lab"):
                print(f"      {str(k):<24} n={len(g):4d}  P(>=2R)={g['r2'].mean()*100:5.1f}%")
        except Exception as ex:
            print(f"    ({name} quartile failed: {ex})")

    print("\n  9c. within-trade early progress (mark-to-market R at +N bars) -> final Rw")
    for b in (8, 16, 32):
        prog = rep[f"prog{b}"].values
        ok = ~np.isnan(prog)
        try:
            q = pd.qcut(pd.Series(prog[ok]), 4, duplicates="drop")
            d = pd.DataFrame({"q": q.astype(str), "Rw": Rw[ok]})
            print(f"    progress at +{b} bars quartile -> P(final Rw>0) / mean final Rw:")
            for k, g in d.groupby("q"):
                print(f"      {str(k):<24} n={len(g):4d}  P(final>0)={ (g['Rw']>0).mean()*100:5.1f}%  "
                      f"meanFinalRw={g['Rw'].mean():+.3f}")
        except Exception as ex:
            print(f"    (progress +{b} bars failed: {ex})")

    print("\n  9d. STALL metric: bars to first reach +0.5R, eventual winners vs losers")
    bth = rep["bars_to_half_r"].values
    for tag, mask in [("winners", is_win), ("losers", is_loss)]:
        v = bth[mask]
        ok = ~np.isnan(v)
        print(f"    {tag:<8} reached +0.5R: {ok.mean()*100:5.1f}% of {mask.sum()}  |  "
              f"among those: median={np.median(v[ok]):6.1f} bars  std={np.std(v[ok]):6.1f} bars")


if __name__ == "__main__":
    main()
