"""instrument_character.py -- instrument "character card" decomposer (spec card 8,
scratchpad/spec_instrument_character.md). Measures structural law 2 ("the instrument's
character decides the method") on a common yardstick across every instrument on hand,
so method choice is read off numbers instead of vibes.

7 axes, per instrument, daily (d1) primary / weekly (w1) secondary, log-returns:
  1. drift            -- annualised mean log-return + bootstrap CI/t-stat, B&H CAGR/maxDD,
                          long/short asymmetry (up-day vs down-day mean, tail asymmetry).
  2. trend/mean-revert -- Lo-MacKinlay variance ratio VR(q) for q in {2,5,10,20,60}
                          + Hurst (R/S), both reused from research/regime_statedet.py.
  3. periodicity       -- top spectral peaks of detrended log-price and of returns,
                          + long-lag monthly autocorrelation. Tested against a
                          phase-shuffle null (multiple-comparison-safe: null is the
                          MAX power across all frequencies, not per-frequency).
  4. seasonality       -- monthly (12) and weekday (up to 7, FX/gold have 5) mean
                          return, bootstrap CI per bucket, flagged both raw (95%) and
                          Bonferroni-corrected for the number of buckets tested.
  5. concentration     -- share of total |return| (and of total signed return)
                          contributed by the top 1/5/10% |return| days; % of days
                          beyond 2 sigma.
  6. vol structure     -- annualised vol, ARCH signature (autocorr of |r| and r^2 at
                          lag 1/5/20), up-day-vol vs down-day-vol, a leverage-effect
                          proxy corr(r_t, |r_{t+1}|).
  7. distribution      -- skew, excess kurtosis, tail asymmetry (5% vs 95% pctile),
                          1%/99% tail ratio.

Reuses (does not reimplement): src.data_loader.load_mt5_csv, breakout_wave.resample,
research.regime_statedet.{hurst_rs, variance_ratio}.

No lookahead concern here: these are full-sample DESCRIPTIVE statistics of a fixed
instrument's character, not a tradeable signal -- there is nothing to leak into.

Run:
  .venv/bin/python scratchpad/instrument_character.py --smoke
  .venv/bin/python scratchpad/instrument_character.py                  (all instruments)
  .venv/bin/python scratchpad/instrument_character.py --only gold,BTC,USDJPY
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv, GOLD_H1_START          # noqa: E402
from breakout_wave import resample                                # noqa: E402
from research.regime_statedet import hurst_rs, variance_ratio     # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ---------------------------------------------------------------------------
# instrument registry: how to build a d1 close series and (if available) a w1
# close series for each instrument, plus the data-trap truncations mandated by
# CLAUDE.md / the spec card. "h1" entries have no d1 file on disk -> resample.
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "gold":   dict(h1=f"{DATA}/vantage_xauusd_h1.csv", h1_start=GOLD_H1_START,
                    w1=f"{DATA}/vantage_xauusd_w1.csv"),
    "BTC":    dict(h1=f"{DATA}/vantage_btcusd_h1.csv", h1_start=None,
                    w1=f"{DATA}/vantage_btcusd_w1.csv"),
    "USDJPY": dict(d1=f"{DATA}/vantage_usdjpy_d1.csv",
                    w1=f"{DATA}/vantage_usdjpy_w1.csv", w1_start="1973-01-01"),
    "EURUSD": dict(d1=f"{DATA}/vantage_eurusd_d1.csv",
                    w1=f"{DATA}/vantage_eurusd_w1.csv", w1_start="1999-01-01"),
    "GBPUSD": dict(d1=f"{DATA}/vantage_gbpusd_d1.csv", w1=f"{DATA}/vantage_gbpusd_w1.csv"),
    "AUDUSD": dict(d1=f"{DATA}/vantage_audusd_d1.csv", w1=f"{DATA}/vantage_audusd_w1.csv"),
    "NZDUSD": dict(d1=f"{DATA}/vantage_nzdusd_d1.csv", w1=f"{DATA}/vantage_nzdusd_w1.csv"),
    "USDCAD": dict(d1=f"{DATA}/vantage_usdcad_d1.csv", w1=f"{DATA}/vantage_usdcad_w1.csv"),
    "nas100": dict(h1=f"{DATA}/vantage_nas100.r_h1.csv", h1_start=None),
    "ger40":  dict(h1=f"{DATA}/vantage_ger40.r_h1.csv", h1_start=None),
    "us2000": dict(h1=f"{DATA}/vantage_us2000.r_h1.csv", h1_start=None),
    "xagusd": dict(h1=f"{DATA}/vantage_xagusd_h1.csv", h1_start=None),
    "usousd": dict(h1=f"{DATA}/vantage_usousd_h1.csv", h1_start=None),
}
SMOKE_SET = ["gold", "BTC", "USDJPY"]
SHORT_SAMPLE_YEARS = 10.0   # below this, flag "short / uncertain" for cyclic axes


# =============================================================================
# data loading
# =============================================================================
def span_info(idx, label):
    years = (idx[-1] - idx[0]).days / 365.25
    return dict(label=label, start=str(idx[0].date()), end=str(idx[-1].date()),
                years=round(years, 2), n=len(idx))


def load_instrument(name, cfg):
    """Returns (d1_close: pd.Series, w1_close: pd.Series|None, spans: dict)."""
    spans = {}
    if "d1" in cfg:
        d = load_mt5_csv(cfg["d1"])
        d1 = d["close"]
    else:
        h = load_mt5_csv(cfg["h1"])
        if cfg.get("h1_start"):
            h = h.loc[cfg["h1_start"]:]
        d = resample(h, "d1")
        d1 = d["close"]
    spans["d1"] = span_info(d1.index, "d1")

    w1 = None
    if "w1" in cfg:
        w = load_mt5_csv(cfg["w1"])
        if cfg.get("w1_start"):
            w = w.loc[cfg["w1_start"]:]
        w1 = w["close"]
        spans["w1"] = span_info(w1.index, "w1")
    return d1, w1, spans


def log_returns(close):
    return np.log(close).diff().dropna()


def periods_per_year(index):
    years = (index[-1] - index[0]).days / 365.25
    return len(index) / years if years > 0 else np.nan


# =============================================================================
# generic bootstrap helpers
# =============================================================================
def bootstrap_mean_ci(x, n_boot, seed, alpha=0.05):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return np.nan, np.nan, np.array([])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = x[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, samples


def block_bootstrap_stat(x, block, stat_fn, n_boot, seed):
    """Circular moving-block bootstrap (law 7: path-dependent stats need blocks,
    not iid resampling). Returns array of stat_fn values across n_boot draws."""
    x = np.asarray(x, float)
    n = len(x)
    if n < block * 4:
        return np.array([])
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=nblocks)
        pieces = [x[np.arange(s, s + block) % n] for s in starts]
        xs = np.concatenate(pieces)[:n]
        out[b] = stat_fn(xs)
    return out


# =============================================================================
# axis 1: drift
# =============================================================================
def axis_drift(close, ppy, n_boot, seed):
    r = log_returns(close).values
    mean_r = r.mean()
    ann_drift = mean_r * ppy
    se = r.std(ddof=1) / np.sqrt(len(r))
    tstat = mean_r / se if se > 0 else np.nan
    lo, hi, _ = bootstrap_mean_ci(r, n_boot, seed)
    years = (close.index[-1] - close.index[0]).days / 365.25
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
    cum = close / close.iloc[0]
    dd = 1 - cum / cum.cummax()
    maxdd = dd.max()
    up, down = r[r > 0], r[r < 0]
    up_mean = up.mean() if len(up) else np.nan
    down_mean = down.mean() if len(down) else np.nan
    up_pct = (r > 0).mean() * 100
    p5, p95 = np.percentile(r, 5), np.percentile(r, 95)
    return dict(ann_drift=ann_drift, ci_lo=lo * ppy, ci_hi=hi * ppy, tstat=tstat,
                cagr=cagr, maxdd=maxdd, up_mean=up_mean, down_mean=down_mean,
                up_pct=up_pct, p5=p5, p95=p95,
                tail_ratio=abs(p5) / p95 if p95 != 0 else np.nan, n=len(r))


# =============================================================================
# axis 2: trend persistence vs mean reversion (VR + Hurst; reused functions)
# =============================================================================
def axis_persistence(close, n_boot, seed, block):
    r = log_returns(close).values
    qs = (2, 5, 10, 20, 60)
    vr = {q: variance_ratio(r, q) for q in qs}
    h = hurst_rs(r)
    # block-bootstrap CI on the two headline stats (VR20, Hurst) -- path-dependent,
    # so iid resampling would be wrong; block ~ 1 quarter of data.
    vr20_boot = block_bootstrap_stat(r, block, lambda x: variance_ratio(x, 20), n_boot, seed)
    h_boot = block_bootstrap_stat(r, block, hurst_rs, n_boot, seed + 1)
    vr20_ci = (np.percentile(vr20_boot, 5), np.percentile(vr20_boot, 95)) if len(vr20_boot) else (np.nan, np.nan)
    h_ci = (np.percentile(h_boot, 5), np.percentile(h_boot, 95)) if len(h_boot) else (np.nan, np.nan)
    return dict(vr=vr, hurst=h, vr20_ci90=vr20_ci, hurst_ci90=h_ci)


# =============================================================================
# axis 3: periodicity (spectral peaks + monthly ACF), with shuffle null
# =============================================================================
def _periodogram(x):
    n = len(x)
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs[1:], power[1:]   # drop DC


def _top_peaks(freqs, power, k=3, min_sep=0.05):
    order = np.argsort(power)[::-1]
    peaks, used = [], []
    for i in order:
        f = freqs[i]
        if all(abs(f - uf) / uf > min_sep for uf in used):
            peaks.append((1.0 / f, power[i]))   # (period_in_bars, power)
            used.append(f)
        if len(peaks) >= k:
            break
    return peaks


def _shuffle_null_maxpower(x, n_reps, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(n_reps)
    for i in range(n_reps):
        xs = rng.permutation(x)
        _, power = _periodogram(xs)
        out[i] = power.max()
    return out


def axis_periodicity(close, n_null, seed, bar_days):
    lp = np.log(close.values)
    t = np.arange(len(lp))
    coef = np.polyfit(t, lp, 1)
    detrended = lp - np.polyval(coef, t)
    r = log_returns(close).values
    r_dm = r - r.mean()

    fP, pP = _periodogram(detrended)
    fR, pR = _periodogram(r_dm)
    peaks_price = _top_peaks(fP, pP, k=3)
    peaks_ret = _top_peaks(fR, pR, k=3)

    null_price = _shuffle_null_maxpower(detrended, n_null, seed)
    null_ret = _shuffle_null_maxpower(r_dm, n_null, seed + 1)
    top_price_power = peaks_price[0][1] if peaks_price else np.nan
    top_ret_power = peaks_ret[0][1] if peaks_ret else np.nan
    pct_price = (null_price < top_price_power).mean() * 100 if len(null_price) else np.nan
    pct_ret = (null_ret < top_ret_power).mean() * 100 if len(null_ret) else np.nan

    peaks_price_days = [(p * bar_days, pw) for p, pw in peaks_price]
    peaks_ret_days = [(p * bar_days, pw) for p, pw in peaks_ret]
    span_days = len(close) * bar_days
    # A price/cum-return periodogram is spurious-periodicity-prone (Slutsky-Yule):
    # ANY nonlinear trend curvature in a non-stationary series produces a strong
    # low-frequency "peak" at period ~= span/2, span/3, ... (harmonics of one hump),
    # which a shuffle-null WILL mark as beating noise (permutation kills all
    # autocorrelation, so it can't tell "one hump" from "true repeating cycle").
    # A peak is only good evidence of genuine cyclicality if the sample contains
    # several repetitions of it (period << span). Flag peaks failing that test.
    price_peaks_reps = [span_days / p if p > 0 else 0 for p, _ in peaks_price_days]
    price_peaks_reliable = [reps >= 3 for reps in price_peaks_reps]

    # long-lag monthly return autocorrelation
    m = close.resample("ME").last()
    mr = np.log(m).diff().dropna()
    max_lag = max(1, min(48, len(mr) // 3))
    acf = {lag: mr.autocorr(lag=lag) for lag in range(1, max_lag + 1)}
    # ~95% white-noise band for ACF under iid null: +-1.96/sqrt(n); with max_lag
    # lags tested, expect ~0.05*max_lag false positives by chance (uncorrected).
    acf_band = 1.96 / np.sqrt(len(mr)) if len(mr) > 0 else np.nan

    return dict(peaks_price_days=peaks_price_days, peaks_ret_days=peaks_ret_days,
                price_beats_null_pct=pct_price, ret_beats_null_pct=pct_ret,
                price_peaks_reps=price_peaks_reps, price_peaks_reliable=price_peaks_reliable,
                monthly_acf=acf, acf_band95=acf_band, n_months=len(mr), max_lag=max_lag)


# =============================================================================
# axis 4: seasonality (monthly / weekday), bootstrap + Bonferroni flag
# =============================================================================
def axis_seasonality(close, n_boot, seed):
    r = log_returns(close)
    df = pd.DataFrame({"ret": r.values}, index=r.index)
    df["month"] = df.index.month
    df["dow"] = df.index.dayofweek

    def bucket_stats(groups, n_tests):
        out = {}
        alpha_bonf = 0.05 / max(n_tests, 1)
        for key, x in groups:
            x = x.values
            if len(x) < 20:
                out[key] = dict(mean=np.nan, n=len(x), sig95=False, sig_bonf=False)
                continue
            lo95, hi95, _ = bootstrap_mean_ci(x, n_boot, seed + int(key), alpha=0.05)
            lob, hib, _ = bootstrap_mean_ci(x, n_boot, seed + int(key) + 1000, alpha=alpha_bonf)
            out[key] = dict(mean=x.mean(), n=len(x),
                             ci95=(lo95, hi95), sig95=(lo95 > 0 or hi95 < 0),
                             ci_bonf=(lob, hib), sig_bonf=(lob > 0 or hib < 0))
        return out

    month_stats = bucket_stats(df.groupby("month")["ret"], n_tests=12)
    dow_groups = list(df.groupby("dow")["ret"])
    dow_stats = bucket_stats(dow_groups, n_tests=len(dow_groups))
    return month_stats, dow_stats


# =============================================================================
# axis 5: return concentration
# =============================================================================
def axis_concentration(close):
    r = log_returns(close).values
    absr = np.abs(r)
    n = len(r)
    order = np.argsort(absr)[::-1]
    total_var = absr.sum()
    total_signed = r.sum()
    out = {}
    for pct in (1, 5, 10):
        k = max(1, int(round(n * pct / 100)))
        idx = order[:k]
        out[pct] = dict(
            var_share=absr[idx].sum() / total_var * 100 if total_var > 0 else np.nan,
            signed_share=r[idx].sum() / total_signed * 100 if total_signed != 0 else np.nan,
            k=k)
    mu, sd = r.mean(), r.std(ddof=1)
    out["frac_2sigma"] = (np.abs(r - mu) > 2 * sd).mean() * 100
    return out


# =============================================================================
# axis 6: vol structure
# =============================================================================
def axis_vol(close, ppy):
    r = log_returns(close)
    ann_vol = r.std(ddof=1) * np.sqrt(ppy)
    absr, sq = r.abs(), r ** 2
    arch_abs = {lag: absr.autocorr(lag=lag) for lag in (1, 5, 20)}
    arch_sq = {lag: sq.autocorr(lag=lag) for lag in (1, 5, 20)}
    prior_up = r.shift(1) > 0
    prior_down = r.shift(1) < 0
    up_vol = r[prior_up].std(ddof=1)
    down_vol = r[prior_down].std(ddof=1)
    leverage_corr = r.corr(absr.shift(-1))
    return dict(ann_vol=ann_vol, arch_abs=arch_abs, arch_sq=arch_sq,
                up_vol=up_vol, down_vol=down_vol, leverage_corr=leverage_corr)


# =============================================================================
# axis 7: distribution
# =============================================================================
def axis_distribution(close):
    r = log_returns(close).values
    p1, p5, p95, p99 = np.percentile(r, [1, 5, 95, 99])
    return dict(skew=sps.skew(r), kurt=sps.kurtosis(r),   # excess (Fisher)
                p5=p5, p95=p95, p1=p1, p99=p99,
                tail_ratio_5_95=abs(p5) / p95 if p95 != 0 else np.nan,
                tail_ratio_1_99=abs(p1) / p99 if p99 != 0 else np.nan)


# =============================================================================
# method-fit tagging layer
# =============================================================================
def method_fit_tags(card):
    tags = []
    # method-fit is MULTI-FACTOR, not daily-VR alone. Daily mean-reversion (VR20<1)
    # is near-universal (overnight reversal / bid-ask bounce) and does NOT preclude a
    # trend-following edge: gold/indices are "grind-up" trends (reliable positive drift
    # + concentrated tails + slow-horizon persistence, yet daily VR<1). The repo's book
    # exploits exactly this via SMA/KAMA gate + PULLBACK entry (buy the daily dip in an
    # up-drift), NOT daily momentum -- so keying "fade" off VR20<1 mislabels every
    # trending asset. Discriminate on drift-significance + concentration + slow persistence;
    # daily VR only sub-types the trend-long entry (momentum vs pullback).
    d = card["drift"]
    drift = d["ann_drift"]; tstat = d["tstat"]
    vr = card["persistence"]["vr"]
    vr20 = vr.get(20, np.nan); vr60 = vr.get(60, np.nan)
    hurst = card["persistence"]["hurst"]
    w1p = card.get("w1_persistence")
    vr20w = w1p["vr"].get(20, np.nan) if w1p else np.nan
    sig_drift = np.isfinite(tstat) and tstat > 1.8          # drift reliably != 0
    daily_mom = np.isfinite(vr20) and vr20 > 1.0
    slow_persist = ((np.isfinite(vr20w) and vr20w > 1.0) or
                    (np.isfinite(vr60) and vr60 > 1.0) or
                    (np.isfinite(hurst) and hurst > 0.55))
    # trend-long = (a) statistically reliable drift, OR (b) daily momentum, OR
    # (c) MATERIALLY positive drift (>5%/yr) with slow-horizon persistence even when the
    # t-stat misses the bar. (c) catches high-vol trenders (silver/Russell) whose drift is
    # economically clear but statistically noisy -- without it they'd wrongly fall to
    # "managed", contradicting e.g. silver's 0.81 annual-R correlation with gold.
    strong_trend = drift > 0 and (sig_drift or daily_mom)
    weak_trend = drift > 0.05 and slow_persist
    if strong_trend or weak_trend:
        wk = "" if sig_drift or daily_mom else " [weak/high-vol: drift+ but t<1.8]"
        if daily_mom:
            tags.append(f"trend-long-fit / MOMENTUM (drift+, VR20={vr20:.2f}>1): breakout-follow ok")
        else:
            tags.append(f"trend-long-fit / GRIND-UP (drift+ t={tstat:.1f}, VR20={vr20:.2f}<1): "
                        f"PULLBACK entry, not daily-momentum (law4){wk}")
    elif (not np.isfinite(tstat)) or abs(tstat) < 1.5 or abs(drift) < 0.03:
        if slow_persist:
            tags.append("managed / trend-only-in-eras (drift~0 but slow-horizon persistence): "
                        "needs L/S + policy-era gate, not long-only")
        else:
            tags.append(f"fade/mean-revert-fit (drift~0 t={tstat:.1f}, VR20={vr20:.2f}<1, no slow persistence)")
    else:
        tags.append(f"mixed (drift={drift*100:+.0f}%/yr t={tstat:.1f}, VR20={vr20:.2f}) -- inspect")

    up_m, down_m = card["drift"]["up_mean"], card["drift"]["down_mean"]
    if np.isfinite(up_m) and np.isfinite(down_m) and down_m != 0:
        ratio = abs(up_m) / abs(down_m)
        if ratio > 1.25:
            tags.append(f"long-skew (up/down |mean| ratio={ratio:.2f}) -> law11 long-favoured")
        elif ratio < 0.8:
            tags.append(f"short-skew (up/down |mean| ratio={ratio:.2f}) -> law11 short-favoured")

    per = card["periodicity"]
    # NOTE: only the return-series peak counts here. The price/cum-return
    # periodogram is spurious-periodicity-prone (harmonics of one nonlinear-trend
    # hump beat a shuffle null too -- see axis_periodicity docstring note), so it
    # is reported for transparency but excluded from the tagging rule.
    if per["ret_beats_null_pct"] is not None and np.isfinite(per["ret_beats_null_pct"]):
        if per["ret_beats_null_pct"] >= 95:
            tags.append("cycle-gate-candidate (return-series spectral peak beats shuffle null)")

    top5 = card["concentration"][5]["var_share"]
    if np.isfinite(top5) and top5 >= 30:
        tags.append(f"far-target+frequency (top5%|ret| days = {top5:.0f}% of variation)")

    arch = card["vol"]["arch_abs"]
    if any(np.isfinite(v) and v > 0.10 for v in arch.values()):
        tags.append("vol-clustering -> regime-conditioning has room")

    return tags


# =============================================================================
# sanity check (device self-test)
# =============================================================================
def sanity_check(cards):
    out = {}
    if "gold" in cards:
        c = cards["gold"]
        d = c["drift"]; vr20 = c["persistence"]["vr"][20]; hurst = c["persistence"]["hurst"]
        top5 = c["concentration"][5]["var_share"]
        vr20w = c["w1_persistence"]["vr"][20] if c.get("w1_persistence") else np.nan
        vr60 = c["persistence"]["vr"].get(60, np.nan)
        slow = (np.isfinite(vr20w) and vr20w > 1) or (np.isfinite(vr60) and vr60 > 1) or hurst > 0.55
        # CORRECTED prior: gold is a GRIND-UP trend, not a daily-momentum asset. Its
        # trend character = reliable positive drift + concentrated tails + slow-horizon
        # persistence, WITH daily VR<1 (mean-reverting at 20d, hence pullback entries).
        # The old "gold VR20(d1)>1" criterion was wrong -- it tested daily momentum, which
        # gold (like nearly every asset except BTC) does not have.
        ok = d["ann_drift"] > 0 and d["tstat"] > 1.8 and slow
        out["gold: trend-long grind-up (sig drift+ & slow persistence; daily VR<1 expected)"] = (
            ok, f"drift={d['ann_drift']*100:+.2f}%/yr t={d['tstat']:.2f}, VR20(d1)={vr20:.3f}"
                f"(<1 = grind-up, pullback), Hurst={hurst:.3f}, VR20(w1)={vr20w:.3f}, "
                f"VR60(d1)={vr60:.3f}, top5%={top5:.1f}%")
    if "BTC" in cards:
        c = cards["BTC"]
        vr20, drift = c["persistence"]["vr"][20], c["drift"]["ann_drift"]
        out["BTC: VR20>1 & drift>0"] = (vr20 > 1 and drift > 0,
                                         f"VR20={vr20:.3f} drift={drift*100:+.2f}%/yr")
        per = c["periodicity"]
        all_peaks = per["peaks_price_days"] + per["peaks_ret_days"]
        near_4yr_any = any(abs(p - 1460) < 365 for p, _ in all_peaks)
        # reliable = price-detrend peak near 4yr AND represents >=3 repetitions
        # in-sample (not just a harmonic of one hump) -- BTC has only ~9.1y of
        # history = ~2.3 halvings, so this can at best be "consistent with",
        # never "confirms", a true repeating 4yr cycle.
        near_4yr_reliable = any(
            abs(p - 1460) < 365 and reps >= 3
            for (p, _), reps in zip(per["peaks_price_days"], per["price_peaks_reps"]))
        out["BTC: ~4yr cycle peak present (any peak, incl. price-detrend artifact-prone)"] = (
            near_4yr_any, f"top periods(days)={[round(p) for p, _ in all_peaks]}")
        out["BTC: ~4yr cycle, reliability-filtered (>=3 reps in-sample)"] = (
            near_4yr_reliable,
            f"reps={[round(float(r),1) for r in per['price_peaks_reps']]} -- only ~9.1y history = "
            f"~2.3 halvings observed; CANNOT strongly confirm regardless of this flag")
    if "USDJPY" in cards:
        c = cards["USDJPY"]
        vr20, drift = c["persistence"]["vr"][20], c["drift"]["ann_drift"]
        near_zero = abs(drift) < 0.03          # < 3%/yr = "managed", arbitrary but stated
        vr_ok = vr20 <= 1.15                   # near 1 or below
        out["USDJPY: drift~0/managed & VR20<=~1"] = (
            near_zero and vr_ok, f"VR20={vr20:.3f} drift={drift*100:+.2f}%/yr")
    return out


# =============================================================================
# printing
# =============================================================================
def fmt_pct(x):
    return f"{x*100:+.2f}%" if np.isfinite(x) else "nan"


def print_card(name, spans, card, short_flag):
    print(f"\n{'='*90}\n### {name}  {'[SHORT/UNCERTAIN SAMPLE]' if short_flag else ''}")
    for tf, s in spans.items():
        print(f"  span[{tf}]: {s['start']} -> {s['end']}  ({s['years']}y, n={s['n']})")

    d = card["drift"]
    print(f"  1.drift   ann={fmt_pct(d['ann_drift'])}  boot95%CI=[{fmt_pct(d['ci_lo'])},{fmt_pct(d['ci_hi'])}]"
          f"  t={d['tstat']:.2f}  CAGR={fmt_pct(d['cagr'])}  maxDD={d['maxdd']*100:.1f}%")
    print(f"            up-day mean={d['up_mean']*100:+.3f}%  down-day mean={d['down_mean']*100:+.3f}%"
          f"  up%={d['up_pct']:.1f}%  p5={d['p5']*100:+.2f}%  p95={d['p95']*100:+.2f}%"
          f"  tail|p5|/p95={d['tail_ratio']:.2f}")

    p = card["persistence"]
    vr_str = "  ".join(f"q{q}={v:.3f}" for q, v in p["vr"].items())
    print(f"  2.persist VR: {vr_str}")
    print(f"            Hurst(R/S)={p['hurst']:.3f}  [90%CI block-boot: VR20={p['vr20_ci90'][0]:.3f}"
          f"-{p['vr20_ci90'][1]:.3f}  Hurst={p['hurst_ci90'][0]:.3f}-{p['hurst_ci90'][1]:.3f}]")

    per = card["periodicity"]
    pk_p = ", ".join(f"{d:.0f}d(pwr%ile={per['price_beats_null_pct']:.0f},reps={reps:.1f})"
                      for (d, _), reps in zip(per["peaks_price_days"], per["price_peaks_reps"]))
    pk_r = ", ".join(f"{d:.0f}d(pwr%ile={per['ret_beats_null_pct']:.0f})" for d, _ in per["peaks_ret_days"])
    print(f"  3.cycle   price-detrend peaks(days): {pk_p}")
    print(f"            [CAVEAT: price-detrend spectrum is spurious-periodicity-prone -- a single")
    print(f"             nonlinear trend hump beats the shuffle null too; only trust a peak with")
    print(f"             reps>=3 (period << sample span). tag layer ignores this series entirely.]")
    print(f"            return-series peaks(days) [tag-eligible, artifact-resistant]: {pk_r}")
    sig_acf = {l: a for l, a in per["monthly_acf"].items() if np.isfinite(a) and abs(a) > per["acf_band95"]}
    exp_false = 0.05 * per["max_lag"]
    print(f"            monthly ACF: n_months={per['n_months']} max_lag={per['max_lag']}  "
          f"|sig(>{per['acf_band95']:.2f} band, UNCORRECTED)| lags="
          f"{sorted(sig_acf.keys())[:8]}{'...' if len(sig_acf) > 8 else ''}  "
          f"(expect ~{exp_false:.1f} false positives by chance at this lag count)")

    ms, ds = card["seasonality"]
    sig_m95 = [k for k, v in ms.items() if v.get("sig95")]
    sig_m_bonf = [k for k, v in ms.items() if v.get("sig_bonf")]
    sig_d95 = [k for k, v in ds.items() if v.get("sig95")]
    sig_d_bonf = [k for k, v in ds.items() if v.get("sig_bonf")]
    print(f"  4.season  months sig95={sig_m95} sig_Bonferroni={sig_m_bonf}  "
          f"(n_tests=12, out of {len(ms)} with data)")
    dow_note = "  [continuous market: 7 buckets incl. Sat/Sun, not the spec's 5]" if len(ds) != 5 else ""
    print(f"            weekdays(0=Mon) sig95={sig_d95} sig_Bonferroni={sig_d_bonf}  "
          f"(n_buckets={len(ds)}){dow_note}")

    cc = card["concentration"]
    print(f"  5.concen  top1%|ret|={cc[1]['var_share']:.1f}% top5%={cc[5]['var_share']:.1f}% "
          f"top10%={cc[10]['var_share']:.1f}%  (share of total |log-ret|)  "
          f">2sigma days={cc['frac_2sigma']:.2f}%")

    v = card["vol"]
    arch_s = "  ".join(f"lag{l}={a:.3f}" for l, a in v["arch_abs"].items())
    print(f"  6.vol     ann.vol={v['ann_vol']*100:.1f}%  ARCH(|r|) {arch_s}")
    print(f"            up-day vol={v['up_vol']*100:.3f}%  down-day vol={v['down_vol']*100:.3f}%  "
          f"leverage corr(r_t,|r_t+1|)={v['leverage_corr']:.3f}")

    dist = card["distribution"]
    print(f"  7.dist    skew={dist['skew']:+.2f}  excess-kurt={dist['kurt']:+.2f}  "
          f"|p5|/p95={dist['tail_ratio_5_95']:.2f}  |p1|/p99={dist['tail_ratio_1_99']:.2f}")

    if "w1_persistence" in card:
        wd, wp = card["w1_drift"], card["w1_persistence"]
        vr_str_w = "  ".join(f"q{q}={v:.3f}" for q, v in wp["vr"].items())
        print(f"  [w1 aux]  drift={fmt_pct(wd['ann_drift'])} (CI [{fmt_pct(wd['ci_lo'])},{fmt_pct(wd['ci_hi'])}])"
              f"  VR: {vr_str_w}  Hurst={wp['hurst']:.3f}")

    print(f"  TAGS: {card['tags']}")


def print_comparison_table(cards):
    print(f"\n{'='*90}\n### comparison table (all instruments, d1)")
    hdr = f"{'instrument':10s} {'ann.drift':>10s} {'VR(20)':>8s} {'Hurst':>7s} {'top5%|r|':>9s} {'ann.vol':>9s} {'skew':>7s}  tags"
    print(hdr)
    for name, c in cards.items():
        d = c["drift"]["ann_drift"]
        vr20 = c["persistence"]["vr"][20]
        h = c["persistence"]["hurst"]
        top5 = c["concentration"][5]["var_share"]
        vol = c["vol"]["ann_vol"]
        sk = c["distribution"]["skew"]
        tagstr = "; ".join(t.split(" (")[0].split(" ->")[0] for t in c["tags"])
        print(f"{name:10s} {fmt_pct(d):>10s} {vr20:8.3f} {h:7.3f} {top5:8.1f}% {fmt_pct(vol):>9s} {sk:+6.2f}  {tagstr}")


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="comma list of instrument names")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--n-boot-season", type=int, default=1500)
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args()

    if args.smoke:
        names = SMOKE_SET
        n_boot, n_boot_season, n_null = 500, 400, 100
    elif args.only:
        names = [x.strip() for x in args.only.split(",")]
        n_boot, n_boot_season, n_null = args.n_boot, args.n_boot_season, args.n_null
    else:
        names = list(INSTRUMENTS.keys())
        n_boot, n_boot_season, n_null = args.n_boot, args.n_boot_season, args.n_null

    print(f"instrument_character.py -- n_boot={n_boot} n_boot_season={n_boot_season} "
          f"n_null={n_null} seed={args.seed} instruments={names}")

    cards = {}
    for i, name in enumerate(names):
        if name not in INSTRUMENTS:
            print(f"  !! unknown instrument '{name}', skipping")
            continue
        cfg = INSTRUMENTS[name]
        d1, w1, spans = load_instrument(name, cfg)
        years = spans["d1"]["years"]
        short_flag = years < SHORT_SAMPLE_YEARS
        ppy = periods_per_year(d1.index)
        avg_days_per_bar = (d1.index[-1] - d1.index[0]).days / len(d1)
        seed = args.seed + i * 97

        card = dict(
            drift=axis_drift(d1, ppy, n_boot, seed),
            persistence=axis_persistence(d1, n_boot, seed + 1, block=63),
            periodicity=axis_periodicity(d1, n_null, seed + 2, avg_days_per_bar),
            seasonality=axis_seasonality(d1, n_boot_season, seed + 3),
            concentration=axis_concentration(d1),
            vol=axis_vol(d1, ppy),
            distribution=axis_distribution(d1),
        )
        # w1 auxiliary (spec: "d1中心・w1補助") -- drift + persistence only, at the
        # weekly horizon, as a cross-check when the daily-bar read is ambiguous.
        if w1 is not None and len(w1) >= 60:
            ppy_w1 = periods_per_year(w1.index)
            card["w1_drift"] = axis_drift(w1, ppy_w1, n_boot, seed + 10)
            card["w1_persistence"] = axis_persistence(w1, n_boot, seed + 11, block=13)
        card["tags"] = method_fit_tags(card)
        if short_flag:
            card["tags"].append("SHORT-SAMPLE: cycle/seasonality reads are uncertain")
        cards[name] = card
        print_card(name, spans, card, short_flag)

    print_comparison_table(cards)

    print(f"\n{'='*90}\n### sanity check (device self-test)")
    checks = sanity_check(cards)
    all_pass = True
    for k, (ok, detail) in checks.items():
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}  -- {detail}")
    if checks:
        print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    else:
        print("  (no known-instrument checks available in this run -- include gold/BTC/USDJPY to self-test)")


if __name__ == "__main__":
    main()
