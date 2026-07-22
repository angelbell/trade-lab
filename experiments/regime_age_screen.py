"""regime_age_screen.py -- FROZEN SPEC: test the REGIME-AGE hypothesis on the book's own legs.

HYPOTHESIS (print at top, verbatim from the spec):
  週足サイクルゲート（勝った唯一のWHENレバー）の本質は『回復初期＝レジームが若いほど良い』の可能性。
  各レッグ自身のゲートについて『ON転換からの経過』でトレードを分位分割し、若い>古いの単調傾きが
  2レッグ以上で出るかを見る。転換の検出は不能でも転換後の若さは因果に測れる。

Legs + regime clocks (all causal: every gate below is the EXACT shift(1)+ffill boolean the
strategy itself already filters entries on -- age is a descriptive re-read of that same
causal series, not a new lookahead channel):
  1. gold_bo       -- days since daily (close>SMA150 AND SMA150 rising) gate flipped OFF->ON
  2. btc_bo_kama   -- days since daily KAMA(14)-rising gate flipped OFF->ON
  3. btc15m_L      -- days since the 4h-KAMA(14) gate (the ACTUAL filter) flipped OFF->ON;
                      + context-only 2nd clock: days since DAILY KAMA(14) also flipped ON
  4. btc_pull      -- weeks since price last crossed BELOW 1.10x weekly-30SMA (cycle gate ON)

Stage 1: descriptive quartile buckets (n/win%/PF/meanR/totR/N-yr) per leg x clock, no
thresholds. Stage 2 (PRE-REGISTERED, run only if >=2 legs show monotone-declining meanR
with age): tilt weight = 1.0 if age<=leg median else 0.5; test vs a 4000-draw permuted-
weight null on totR/DD (need >=90%ile); then swap qualifying leg(s) into the 6-leg book
(exact section-E construction from pwh_adoption.py) and compare CAGR/DD base vs tilted.

PRE-REGISTERED PREDICTIONS: 本命=フラットか記述止まり(PWH型); 傾きが出るなら『若い>古い』方向;
逆傾き(古いほど良い)が出たらサイクルゲートの解釈自体を見直す発見。

  .venv/bin/python experiments/regime_age_screen.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ROOT)

from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample, kama_adaptive
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG, at
from research.regime_adaptive import kama
from research.portfolio_kama import PB

rng_global = np.random.default_rng(7)


def invert(d):
    """Price-inversion helper (inlined, NOT imported from short_mirror_15m.py -- that
    module runs full backtests at import time with no __main__ guard; importing it would
    silently re-run an unrelated GOLD+BTC short study)."""
    C = 2 * d["high"].max()
    return pd.DataFrame({"open": C - d["open"], "high": C - d["low"],
                         "low": C - d["high"], "close": C - d["close"]}, index=d.index)


# --------------------------------------------------------------------------- regime clock
def flip_age(gate_bool):
    """gate_bool: pd.Series[bool], the CAUSAL (already shift(1)+possibly-ffilled) series a
    strategy actually applies to filter entries. Returns age (in calendar DAYS, regardless
    of the index's own bar spacing) since the most recent False->True flip; NaN where the
    gate is False (never happens at a kept trade's OWN entry bar, since these are exactly
    the filters that produced the trade) or before any flip is observable in-sample."""
    g = gate_bool.fillna(False).astype(bool)
    idx = g.index
    prev = g.shift(1).fillna(False)
    flip_on = (g & (~prev)).values
    ts = idx.to_series()                # tz-aware if idx is tz-aware
    flip_ts = ts.where(flip_on)         # .where keeps dtype/tz, fills NaT elsewhere
    last_flip = flip_ts.ffill()
    age_days = (ts - last_flip).dt.total_seconds() / 86400.0
    return age_days.where(g)


def age_at(age_series, times):
    """ffill the gate-frequency age series to (possibly higher-resolution) trade entry
    times -- identical causal lookup pattern to breakout_wave's own `reg.reindex(d.index,
    method='ffill')` gate application."""
    return age_series.reindex(pd.DatetimeIndex(times), method="ffill").values


# --------------------------------------------------------------------------- leg builders
def build_gold_bo():
    d = resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h")
    t = run_bo(d, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                      "daily_sma": 150, "daily_slope_k": 10}))[["time", "R"]].copy()
    dc = d["close"].resample("1D").last().dropna()
    sma = dc.rolling(150).mean()
    up = (dc > sma) & (sma > sma.shift(10))
    gate = up.shift(1)                         # exactly breakout_wave.run's `reg` pre-ffill
    age = flip_age(gate)
    t["age"] = age_at(age, t["time"])
    return t.reset_index(drop=True)


def build_btc_bo_kama():
    d4 = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    btc = run_bo(d4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))[["time", "R"]]
    dc = d4["close"].resample("1D").last().dropna()
    km = kama(dc, 14)
    gate = (km > km.shift(1)).shift(1)         # exactly portfolio_kama.kama_gate_btc's gate
    t = btc[at(gate, btc["time"])].copy()
    age = flip_age(gate)
    t["age"] = age_at(age, t["time"])
    return t.reset_index(drop=True)


def build_btc15m_L():
    b = load_mt5_csv("data/vantage_btcusd_m15.csv")
    cnt = b.groupby(b.index.date).size()
    okd = cnt[cnt.rolling(30).median() >= 80]
    d15 = resample(b[b.index.date >= okd.index[0]], "15min")
    from radar_gate_race import BASE
    kw = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3}
    tb = run_bo(d15, SimpleNamespace(**kw))
    Rn = tb["R"].values - 15.0 / tb["risk"].values
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ix = d15.index.get_indexer(tb["time"])
    ab = tb["e_px"].values > pdh[ix]
    w = np.where(ab, 1.0, 0.5)
    Rw = Rn * w
    out = pd.DataFrame({"time": pd.DatetimeIndex(tb["time"]), "R": Rw})

    # clock 1 (the ACTUAL filtering gate): 4h-KAMA(14) rising
    dck4 = d15["close"].resample("240min").last().dropna()
    kmg4 = kama_adaptive(dck4, 14)
    gate4 = (kmg4 > kmg4.shift(1)).shift(1)
    age4 = flip_age(gate4)
    out["age"] = age_at(age4, out["time"])

    # clock 2 (context only, NOT a filter -- coverage <100%, only defined when the daily
    # KAMA also happens to be rising at that entry)
    dck1 = d15["close"].resample("1D").last().dropna()
    kmg1 = kama_adaptive(dck1, 14)
    gate1 = (kmg1 > kmg1.shift(1)).shift(1)
    age1 = flip_age(gate1)
    out["age_daily_ctx"] = age_at(age1, out["time"])
    return out, d15


def build_btc_pull():
    d4 = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h")
    pb = run_pb(d4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)[["time", "R"]]
    maxext, cyclelen = 0.10, 30
    w30 = d4["close"].resample("1W").last().rolling(cyclelen).mean().shift(1)
    ceil = (1 + maxext) * w30.reindex(d4.index, method="ffill")
    gate = d4["close"] <= ceil                 # exactly portfolio_kama.cycle_gate_pull's gate
    t = pb[at(gate, pb["time"])].copy()
    age_days = flip_age(gate)
    t["age_weeks"] = age_at(age_days, t["time"]) / 7.0
    return t.reset_index(drop=True)


# --------------------------------------------------------------------------- stage 1 report
def report_leg(leg_name, clock_label, times, R, age):
    times = pd.DatetimeIndex(times)
    R = np.asarray(R, dtype=float)
    age = np.asarray(age, dtype=float)
    ok = ~np.isnan(age)
    print(f"\n--- {leg_name}  |  clock = {clock_label} ---")
    print(f"  n_total={len(R)}  n_valid_age={ok.sum()} ({ok.mean()*100:.0f}% coverage)")
    if ok.sum() < 8:
        print("  (too few trades with a valid age -- skip)")
        return None
    t2, R2, age2 = times[ok], R[ok], age[ok]
    span_yr = max((t2.max() - t2.min()).days / 365.25, 1e-9)
    med, sd = np.median(age2), np.std(age2)
    print(f"  age distribution: median={med:.1f}  std={sd:.1f}  (units of the clock above)")
    try:
        bins, edges = pd.qcut(age2, 4, duplicates="drop", retbins=True, labels=False)
    except Exception as ex:
        print(f"  (qcut failed: {ex})")
        return None
    means = []
    print(f"  {'bucket':<22} {'n':>5} {'N/yr':>6} {'win%':>6} {'PF':>6} {'meanR':>8} {'totR':>8}")
    for k in sorted(np.unique(bins)):
        m = bins == k
        Rv = R2[m]
        n = len(Rv)
        win = (Rv > 0).mean() * 100
        wins = Rv[Rv > 0].sum(); losses = -Rv[Rv < 0].sum()
        PF = wins / losses if losses > 0 else np.inf
        meanR = Rv.mean(); totR = Rv.sum()
        nyr = n / span_yr
        lo, hi = edges[k], edges[k + 1]
        label = f"Q{k+1} [{lo:.1f},{hi:.1f}]"
        print(f"  {label:<22} {n:5d} {nyr:6.1f} {win:6.1f} {PF:6.2f} {meanR:+8.3f} {totR:+8.1f}")
        means.append(meanR)
    diffs = np.diff(means)
    if all(d_ < -1e-9 for d_ in diffs):
        verdict = "MONOTONE-DECLINING (young > old)"
    elif all(d_ > 1e-9 for d_ in diffs):
        verdict = "MONOTONE-RISING (old > young)"
    else:
        verdict = "FLAT/MIXED"
    print(f"  VERDICT: {verdict}   bucket meanR = [{', '.join(f'{m:+.3f}' for m in means)}]")
    return dict(times=t2, R=R2, age=age2, median_age=med, means=means, verdict=verdict)


# --------------------------------------------------------------------------- stage 2: tilt
def totdd(R):
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(eq) - eq).max()
    return R.sum() / max(dd, 1e-9)


def tilt_test(name, R, age, n_draws=4000, seed=7):
    rng = np.random.default_rng(seed)
    med = np.median(age)
    w = np.where(age <= med, 1.0, 0.5)
    n_half = int((w == 0.5).sum())
    base = totdd(R)
    tilt = totdd(R * w)
    idxs = np.arange(len(R))
    null = np.empty(n_draws)
    for i in range(n_draws):
        half_idx = rng.choice(idxs, size=n_half, replace=False)
        wperm = np.ones(len(R)); wperm[half_idx] = 0.5
        null[i] = totdd(R * wperm)
    pct = (null < tilt).mean() * 100
    passed = pct >= 90
    print(f"  {name:<16} base totR/DD={base:6.2f}  tilt totR/DD={tilt:6.2f}  "
          f"null(n={n_draws}) med={np.median(null):5.2f} sd={np.std(null):5.2f}  "
          f"tilt %ile={pct:5.1f}%  -> {'PASS' if passed else 'FAIL'}")
    return dict(base=base, tilt=tilt, pct=pct, passed=passed, weight=w)


def book_cagr(legs_dict, total_risk=0.03, n_boot=4000, seed=7):
    mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in legs_dict.items()}
    start = max(s.index.min() for s in mon.values())
    end = min(s.index.max() for s in mon.values())
    midx = pd.period_range(start, end, freq="M")
    M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
    wgt = 1.0 / M.std(); wgt = wgt / wgt.sum() * total_risk
    port = (M * wgt).sum(axis=1).values
    rng = np.random.default_rng(seed)
    mult = np.array([np.prod(1 + port[rng.integers(0, len(port), 12)]) for _ in range(n_boot)])
    eq = np.cumprod(1 + port)
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    cagr = (eq[-1] ** (12 / len(port)) - 1) * 100
    return dict(cagr=cagr, dd=dd, cdd=cagr / max(dd, 1e-9), mult=mult)


def main():
    print("=" * 100)
    print("HYPOTHESIS: 週足サイクルゲート（勝った唯一のWHENレバー）の本質は『回復初期＝レジームが若いほど")
    print("良い』の可能性。各レッグ自身のゲートについて『ON転換からの経過』でトレードを分位分割し、")
    print("若い>古いの単調傾きが2レッグ以上で出るかを見る。転換の検出は不能でも転換後の若さは因果に測れる。")
    print("PRE-REGISTERED PREDICTIONS: 本命=フラットか記述止まり(PWH型); 傾きが出るなら『若い>古い』方向;")
    print("逆傾き(古いほど良い)が出たらサイクルゲートの解釈自体を見直す発見。")
    print("=" * 100)

    # ---------------------------------------------------------- smoke-run each leg first
    print("\n[smoke] building legs ...")
    gold = build_gold_bo()
    print(f"  gold_bo       n={len(gold)}  age-cov={gold['age'].notna().mean()*100:.0f}%  "
          f"range {gold.time.min().date()}..{gold.time.max().date()}")
    btck = build_btc_bo_kama()
    print(f"  btc_bo_kama   n={len(btck)}  age-cov={btck['age'].notna().mean()*100:.0f}%  "
          f"range {btck.time.min().date()}..{btck.time.max().date()}")
    b15, d15_btc = build_btc15m_L()
    print(f"  btc15m_L      n={len(b15)}  age-cov(4h)={b15['age'].notna().mean()*100:.0f}%  "
          f"age-cov(daily-ctx)={b15['age_daily_ctx'].notna().mean()*100:.0f}%  "
          f"range {b15.time.min().date()}..{b15.time.max().date()}")
    pull = build_btc_pull()
    print(f"  btc_pull      n={len(pull)}  age-cov={pull['age_weeks'].notna().mean()*100:.0f}%  "
          f"range {pull.time.min().date()}..{pull.time.max().date()}")

    # ================================================================= STAGE 1
    print("\n" + "=" * 100)
    print("STAGE 1 -- descriptive quartile buckets (no thresholds)")
    print("=" * 100)

    res = {}
    res["gold_bo"] = report_leg("gold_bo", "days since daily SMA150(+slope) gate flipped ON",
                                 gold.time, gold.R, gold.age)
    res["btc_bo_kama"] = report_leg("btc_bo_kama", "days since daily KAMA(14)-rising gate flipped ON",
                                     btck.time, btck.R, btck.age)
    res["btc15m_L"] = report_leg("btc15m_L", "days since 4h-KAMA(14) gate flipped ON (the ACTUAL filter)",
                                  b15.time, b15.R, b15.age)
    res["btc15m_L_dailyctx"] = report_leg(
        "btc15m_L (context)", "days since DAILY KAMA(14) also flipped ON (supplementary, not a filter)",
        b15.time, b15.R, b15.age_daily_ctx)
    res["btc_pull"] = report_leg("btc_pull", "weeks since price crossed BELOW 1.10x weekly-30SMA (cycle gate ON)",
                                  pull.time, pull.R, pull.age_weeks)

    # ================================================================= STAGE 1 -- verdict tally
    primary_legs = ["gold_bo", "btc_bo_kama", "btc15m_L", "btc_pull"]
    declining = [k for k in primary_legs if res[k] and res[k]["verdict"].startswith("MONOTONE-DECLINING")]
    rising = [k for k in primary_legs if res[k] and res[k]["verdict"].startswith("MONOTONE-RISING")]
    print("\n" + "-" * 100)
    print(f"STAGE 1 TALLY (4 primary legs; btc15m_L context clock is supplementary, not counted):")
    for k in primary_legs:
        v = res[k]["verdict"] if res[k] else "(skipped -- too few)"
        print(f"  {k:<14} {v}")
    print(f"  MONOTONE-DECLINING count = {len(declining)}/4  -> {declining}")
    if rising:
        print(f"  MONOTONE-RISING (reverse) legs = {rising}  -- 逆傾き、サイクルゲートの解釈を見直す材料")

    # ================================================================= STAGE 2 (gated on stage1)
    print("\n" + "=" * 100)
    if len(declining) < 2:
        print(f"STAGE 2 SKIPPED (frozen rule: needs >=2 monotone-declining legs, got {len(declining)}).")
        print("Reading: 本命どおりフラット/記述止まり(PWH型) -- サイクルゲートの『若さ』仮説はこの4レッグでは再現せず。")
        print("=" * 100)
        return
    print(f"STAGE 2 -- pre-registered tilt test (qualifying legs: {declining})")
    print("=" * 100)

    leg_frames = {"gold_bo": gold, "btc_bo_kama": btck, "btc15m_L": b15, "btc_pull": pull}
    age_cols = {"gold_bo": "age", "btc_bo_kama": "age", "btc15m_L": "age", "btc_pull": "age_weeks"}

    print("\n-- per-leg tilt (w=1.0 if age<=median else 0.5) vs 4000-draw permuted-weight null --")
    tilt_results = {}
    for k in declining:
        df = leg_frames[k].dropna(subset=[age_cols[k]])
        tilt_results[k] = tilt_test(k, df["R"].values, df[age_cols[k]].values)

    qualifying = [k for k, v in tilt_results.items() if v["passed"]]
    print(f"\n  legs beating the null at >=90%ile: {qualifying if qualifying else '(none)'}")

    # ---- book judgment ----
    print("\n-- BOOK judgment: 6-leg book (pwh_adoption.py section E construction), base vs tilted --")
    legs_base = {"gold_bo": pd.Series(gold.R.values, index=pd.DatetimeIndex(gold.time)),
                 "btc_bo_kama": pd.Series(btck.R.values, index=pd.DatetimeIndex(btck.time)),
                 "btc_pull": pd.Series(pull.R.values, index=pd.DatetimeIndex(pull.time)),
                 "btc15m_L": pd.Series(b15.R.values, index=pd.DatetimeIndex(b15.time))}

    g = resample(load_mt5_csv("data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    from radar_gate_race import BASE
    tg = run_bo(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                       "ext_cap": 8.0, "pullback_frac": 0.25}))
    legs_base["gold15m"] = pd.Series(tg["R"].values - 0.3 / tg["risk"].values,
                                      index=pd.DatetimeIndex(tg["time"]))

    inv = invert(d15_btc)
    ts_ = run_bo(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
    Rs = ts_["R"].values - 15.0 / ts_["risk"].values
    pdl = d15_btc["low"].resample("1D").min().dropna().shift(1).reindex(d15_btc.index, method="ffill").values
    C = 2 * d15_btc["high"].max()
    mS = (C - ts_["e_px"].values) < pdl[d15_btc.index.get_indexer(ts_["time"])]
    legs_base["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])

    legs_tilt = dict(legs_base)
    for k in qualifying:
        df = leg_frames[k].dropna(subset=[age_cols[k]]).reset_index(drop=True)
        w = tilt_results[k]["weight"]
        legs_tilt[k] = pd.Series(df["R"].values * w, index=pd.DatetimeIndex(df["time"]))

    if not qualifying:
        print("  (no leg passed the null at >=90%ile -- book test run anyway on the RAW tilt weights")
        print("   of the declining-but-not-passing legs, for completeness; read as exploratory only)")
        for k in declining:
            df = leg_frames[k].dropna(subset=[age_cols[k]]).reset_index(drop=True)
            w = tilt_results[k]["weight"]
            legs_tilt[k] = pd.Series(df["R"].values * w, index=pd.DatetimeIndex(df["time"]))

    b_base = book_cagr(legs_base)
    b_tilt = book_cagr(legs_tilt)
    for tag, b in [("base (untilted)", b_base), ("tilted", b_tilt)]:
        print(f"  {tag:<18} CAGR={b['cagr']:5.1f}%  maxDD={b['dd']:4.1f}%  CAGR/DD={b['cdd']:5.2f} | "
              f"1yr mult med={np.median(b['mult']):.2f} sd={b['mult'].std():.2f} "
              f"p10={np.percentile(b['mult'],10):.2f} p90={np.percentile(b['mult'],90):.2f}")


if __name__ == "__main__":
    main()
