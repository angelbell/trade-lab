"""regime_discriminator.py — "effく場面/効かない場面" を見分ける汎用ツール.

哲学（ユーザー 2026-07-05）: エッジを全天候にする必要はない。「この方法のベータが乗る
場面」を因果的に見分け、そこだけ張れれば良い（=正しくベータに乗る）。これはこのラボの
最大レバー（WHEN to deploy > entry; KAMA-rising / 週足サイクル位相が唯一生き残った例）。

だが「効く場面」の物語は最も自己欺瞞に陥りやすい（後知恵で年を選べば必ず綺麗になる）。
∴ このツールの価値は見分けと同時に「その見分けが後知恵・ランダムでない」ことの検証にある:
  1. per-trade R を、確定足で計算した文脈フィーチャの分位でバケット化（どの帯にRが集中？）
  2. IS で閾値を決め OOS に適用 → OOS でも base を超えるか（後知恵でないか）
  3. random-drop null → 同数をランダムに削るのに勝てるか（単なるN間引きでないか）
  4. 分位を跨ぐ meanR がプラトーか（スパイク＝過学習を排除）
  5. 年別 ON% → 悪い年を消し良い年を残すか

使い方: profile(ladder_name, signal, features=FEATURES, rr=..., exit_mode=...).
signal は edge_harness の因果契約（sig[i] は bar i の CLOSE まで、i+1 OPEN 入場）。
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from pandas.tseries.frequencies import to_offset
import research.edge_harness as EH
from research.edge_harness import LADDERS, AGG, _prep_and_walk, HORIZON
from research.overfit_audit import cdd_R

# ---------- causal context features (value usable at bar i CLOSE) ----------
def _htf(df, fr, fn):
    """map a higher-TF series onto entry bars using only the LAST CLOSED HTF bar."""
    h = df.resample(fr).agg(AGG).dropna()
    v = pd.Series(np.asarray(fn(h), float), index=h.index).shift(1)   # last closed HTF bar
    m = pd.merge_asof(pd.DataFrame({"t": df.index.values}),
                      pd.DataFrame({"t": h.index.values, "v": v.values}).sort_values("t"),
                      on="t", direction="backward")
    return m["v"].values

def f_er(df):                       # efficiency ratio (entry TF trendiness)
    c = df["close"]; return ((c - c.shift(20)).abs() / (c - c.shift(1)).abs().rolling(20).sum()).values
def f_adx(df):                      # ADX (entry TF)
    return ta.adx(df["high"], df["low"], df["close"], 14)["ADX_14"].values
def f_kama_d(df):                   # daily KAMA(14) slope (the surviving breakout gate)
    return _htf(df, "1440min", lambda h: (ta.kama(h["close"], 14) - ta.kama(h["close"], 14).shift(3)).values)
def f_cyc_wk(df):                   # weekly cycle phase: (close - 30wSMA)/close (below=early recovery)
    return _htf(df, "1W", lambda h: ((h["close"] - h["close"].rolling(30).mean()) / h["close"]).values)
def f_dsma_d(df):                   # daily SMA150 slope (gold breakout gate)
    return _htf(df, "1440min", lambda h: (h["close"].rolling(150).mean() - h["close"].rolling(150).mean().shift(10)).values)
def f_atr_reg(df):                  # vol regime: ATR / its 100-median
    a = ta.atr(df["high"], df["low"], df["close"], 14); return (a / a.rolling(100).median()).values
def f_stretch(df):                  # distance from EMA200 in daily-ATR units
    e = df["close"].ewm(span=200, adjust=False).mean().values
    d = _htf(df, "1440min", lambda h: ta.atr(h["high"], h["low"], h["close"], 14).values)
    return (df["close"].values - e) / np.where(d > 0, d, np.nan)

FEATURES = {"ER20": f_er, "ADX": f_adx, "KAMAd_slope": f_kama_d, "cycle_wk": f_cyc_wk,
            "dSMA150_slope": f_dsma_d, "ATR_regime": f_atr_reg, "stretch_ema200": f_stretch}

# ---------- core: per-trade R + feature values at entry ----------
def _trades(name, signal, tf, rr, katr, exit_mode, cost, stop_slip):
    csv, dfl_cost, tfs = LADDERS[name]
    cost = dfl_cost if cost is None else cost
    base = EH.load_mt5_csv(csv)
    fr = dict(tfs).get(tf)
    df = base if fr is None else base.resample(fr).agg(AGG).dropna()
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr = ta.atr(df["high"], df["low"], df["close"], 14).values
    mean = df["close"].rolling(20).mean().values
    sig = np.asarray(signal(df))
    idx = np.where(sig != 0)[0]; idx = idx[idx + 1 < len(c)]
    sides = sig[idx].astype(np.int64)
    R = _prep_and_walk(idx, sides, atr, mean, rr, katr, exit_mode, 0.0, cost, stop_slip, o, h, l, c, idx + 1, HORIZON)
    ok = ~np.isnan(R)
    et = df.index[idx + 1][ok]
    return df, idx[ok], R[ok], np.array([t for t in et])

def _cdd(R, times):
    yrs = max(1e-9, (pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).days / 365.25)
    return cdd_R(R, yrs)[2], yrs

def _rand_drop_pctile(R, keep_mask, times, trials=2000, seed=0):
    _, yrs = _cdd(R, times)
    kn = int(keep_mask.sum())
    if kn < 12: return float("nan")
    obs = cdd_R(R[keep_mask], yrs)[2]
    rng = np.random.default_rng(seed)
    nul = np.array([cdd_R(rng.choice(R, kn, replace=False), yrs)[2] for _ in range(trials)])
    return (nul < obs).mean() * 100

def profile(name, signal, features=FEATURES, tf="4h", rr=2.0, katr=1.0,
            exit_mode="rr", cost=None, stop_slip=0.5, min_keep=20):
    df, idx, R, times = _trades(name, signal, tf, rr, katr, exit_mode, cost, stop_slip)
    yr = np.array([t.year for t in times]); span_med = np.median(np.unique(yr))
    base_cdd, yrs = _cdd(R, times)
    print(f"\n########## DISCRIMINATOR  {name} @ {tf}  (N={len(R)}, {yrs:.1f}yr, "
          f"base meanR={R.mean():+.3f}, CAGR/DD={base_cdd:+.2f}) ##########")
    print(f"  {'feature':<15}{'terciles meanR (lo|mid|hi)':<30}{'best-side':>9}{'OOSgate':>9}{'OOSbase':>8}"
          f"{'randDrop':>9}{'ON%bad':>7}  verdict")
    rows = []
    for fname, ffn in features.items():
        f = np.asarray(ffn(df), float)[idx]
        m = ~np.isnan(f)
        if m.sum() < 3 * min_keep:
            print(f"  {fname:<15}(too few valued)"); continue
        Rf, ff, tf_, yrf = R[m], f[m], times[m], yr[m]
        # tercile profile
        q1, q2 = np.quantile(ff, [1/3, 2/3])
        buckets = [ff <= q1, (ff > q1) & (ff <= q2), ff > q2]
        tmeans = [Rf[b].mean() if b.any() else np.nan for b in buckets]
        # choose eff-side + threshold on IS only, apply to OOS (no hindsight)
        ism = yrf < span_med; oosm = ~ism
        grid = [0.3, 0.4, 0.5, 0.6, 0.7]; best = (None, None, -1e9)
        for side in ("hi", "lo"):
            for qv in grid:
                thr = np.quantile(ff[ism], qv)
                kis = (ff[ism] >= thr) if side == "hi" else (ff[ism] <= thr)
                if kis.sum() < min_keep: continue
                cd = cdd_R(Rf[ism][kis], max(1e-9, (pd.Timestamp(tf_[ism][-1]) - pd.Timestamp(tf_[ism][0])).days/365.25))[2]
                if cd > best[2]: best = (side, thr, cd)
        side, thr, _ = best
        if side is None:
            print(f"  {fname:<15}{'|'.join(f'{x:+.2f}' for x in tmeans):<30}(no IS gate)"); continue
        keep_oos = (ff[oosm] >= thr) if side == "hi" else (ff[oosm] <= thr)
        keep_all = (ff >= thr) if side == "hi" else (ff <= thr)
        oos_gate = cdd_R(Rf[oosm][keep_oos], max(1e-9,(pd.Timestamp(tf_[oosm][-1])-pd.Timestamp(tf_[oosm][0])).days/365.25))[2] if keep_oos.sum() >= 12 else float("nan")
        oos_base = cdd_R(Rf[oosm], max(1e-9,(pd.Timestamp(tf_[oosm][-1])-pd.Timestamp(tf_[oosm][0])).days/365.25))[2]
        rdrop = _rand_drop_pctile(Rf, keep_all, tf_)
        # year ON%: of the base-negative years, what fraction does the gate turn OFF
        badyrs = [y for y in np.unique(yrf) if Rf[yrf == y].sum() < 0]
        on_bad = np.mean([keep_all[yrf == y].mean() for y in badyrs]) * 100 if badyrs else float("nan")
        passed = (oos_gate > max(oos_base, 0)) and (rdrop >= 90)
        verdict = "PASS" if passed else ("weak" if (oos_gate > oos_base and rdrop >= 75) else "fail")
        print(f"  {fname:<15}{'|'.join(f'{x:+.2f}' for x in tmeans):<30}{side+f'>{thr:.2f}' if side=='hi' else side+f'<{thr:.2f}':>9}"
              f"{oos_gate:>9.2f}{oos_base:>8.2f}{rdrop:>8.0f}%{on_bad:>6.0f}%  {verdict}")
        rows.append((fname, oos_gate - oos_base, rdrop, verdict))
    rows.sort(key=lambda r: (r[3] == "PASS", r[1]), reverse=True)
    print("  ---- ranked by (OOS gate−base CAGR/DD) ----")
    for fn_, lift, rd, vd in rows:
        print(f"    {fn_:<15} OOSlift={lift:+.2f}  randDrop={rd:.0f}%  {vd}")
    print("  (PASS = OOS-gate>base AND randDrop>=90; ON%bad = gate keeps this % of BASE-NEGATIVE years' trades → want LOW)")
    return rows


if __name__ == "__main__":
    # ---- POSITIVE CONTROL: BTC breakout (KAMA-rising is the KNOWN eff-zone) ----
    #   the tool should SURFACE KAMAd_slope / ER / trend features as PASS here.
    print("=== POSITIVE CONTROL: BTC Donchian breakout @ 4h (KAMA gate known to help) ===")
    profile("BTC", EH.demo_breakout, tf="4h", rr=2.0, stop_slip=0.5)

    # ---- ACTUAL QUESTION: SAR x EMA on USDJPY (dead gross; expect NO eff-zone survives) ----
    print("\n=== SUBJECT: SAR x 200EMA trend-follow, USDJPY 1h (does ANY eff-zone survive?) ===")
    def sar_ema(df):
        c = df["close"].values
        e10 = df["close"].ewm(span=10, adjust=False).mean().values
        e25 = df["close"].ewm(span=25, adjust=False).mean().values
        e200 = df["close"].ewm(span=200, adjust=False).mean().values
        ps = ta.psar(df["high"], df["low"], df["close"], af0=0.02, af=0.02, max_af=0.2)
        ld = ps.filter(like="PSARl").iloc[:, 0].notna().values
        sd = ps.filter(like="PSARs").iloc[:, 0].notna().values
        cu = np.zeros(len(c), bool); cd = np.zeros(len(c), bool)
        cu[1:] = (e10[1:] > e25[1:]) & (e10[:-1] <= e25[:-1]); cd[1:] = (e10[1:] < e25[1:]) & (e10[:-1] >= e25[:-1])
        sig = np.zeros(len(c)); sig[cu & (c > e200) & ld] = 1; sig[cd & (c < e200) & sd] = -1; sig[:210] = 0
        return sig
    profile("USDJPY", sar_ema, tf="1h", rr=2.0, stop_slip=0.5)
