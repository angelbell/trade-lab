"""regime_statedet.py -- price-regime STATE detectors (HMM / Hurst / variance-ratio) as daily deploy
gates on the breakout legs. Tests the richer classifiers the user named against the documented null.

regime_headroom.py declared the price-feature gate space a NULL (slope/ER/vol/extension/ADX all failed)
with daily-KAMA-rising the one survivor. HMM (latent state + transition persistence), Hurst, and
variance-ratio (trending vs mean-reverting) are mechanistically richer, so they earn a real test --
but the bar is KAMA-rising (not always-on), on BOTH gold 1H and BTC 4H breakout, with a plateau, and
they must not just duplicate KAMA (a gate helps only when it supplies context the strategy LACKS).

Dependency-free: hand-rolled 2-state Gaussian HMM (Baum-Welch + causal forward-filter, FIT ON IS ONLY,
no refit OOS = the overfit guard) and pure-numpy Hurst (R/S) + Lo-MacKinlay variance-ratio. All daily,
shift(1) before gating = no lookahead. Entries are FIXED (same breakout); only the deploy gate changes.

  .venv/bin/python research/regime_statedet.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG, SPLIT, metrics, at
from research.regime_adaptive import kama

RNG = np.random.default_rng(7)


# ============================== detectors (all causal / IS-fit where noted) ==============================
def hurst_rs(x):
    """rescaled-range Hurst exponent of a 1d array (H>0.5 trending, <0.5 mean-reverting)."""
    x = np.asarray(x, float); n = len(x)
    if n < 16:
        return np.nan
    ns = [n // k for k in (8, 4, 2, 1) if n // k >= 8]
    rs = []
    for m in sorted(set(ns)):
        vals = []
        for i in range(0, n - m + 1, m):
            seg = x[i:i + m]
            z = seg - seg.mean(); c = np.cumsum(z)
            R = c.max() - c.min(); S = seg.std()
            if S > 0:
                vals.append(R / S)
        if vals:
            rs.append((np.log(m), np.log(np.mean(vals))))
    if len(rs) < 2:
        return np.nan
    a = np.array(rs)
    return np.polyfit(a[:, 0], a[:, 1], 1)[0]


def rolling_hurst(r, W):
    return pd.Series(r).rolling(W).apply(hurst_rs, raw=True)


def variance_ratio(r, q):
    """Lo-MacKinlay VR(q): var of q-period returns / (q * var of 1-period). >1 trending, <1 MR."""
    r = np.asarray(r, float)
    if len(r) < q + 2:
        return np.nan
    v1 = np.var(r, ddof=1)
    agg = np.convolve(r, np.ones(q), "valid")
    vq = np.var(agg, ddof=1) / q
    return vq / v1 if v1 > 0 else np.nan


def rolling_vr(r, W, q):
    return pd.Series(r).rolling(W).apply(lambda s: variance_ratio(s, q), raw=True)


def gaussian_hmm_fit(x, iters=40, seed=0):
    """compact 2-state Gaussian HMM via Baum-Welch on a 1d series. Returns (mu, var, A, pi)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); T = len(x)
    mu = np.array([x.mean() - x.std(), x.mean() + x.std()])
    var = np.array([x.var(), x.var()]) + 1e-9
    A = np.array([[0.9, 0.1], [0.1, 0.9]]); pi = np.array([0.5, 0.5])
    for _ in range(iters):
        B = np.stack([np.exp(-0.5 * (x - mu[k]) ** 2 / var[k]) / np.sqrt(2 * np.pi * var[k])
                      for k in range(2)], 1) + 1e-300
        # forward-backward (scaled)
        al = np.zeros((T, 2)); be = np.zeros((T, 2)); c = np.zeros(T)
        al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
        for t in range(1, T):
            al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
        be[-1] = 1.0
        for t in range(T - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])); be[t] /= c[t + 1]
        g = al * be; g /= g.sum(1, keepdims=True)
        xi = np.zeros((2, 2))
        for t in range(T - 1):
            m = (al[t][:, None] * A) * (B[t + 1] * be[t + 1])[None, :]
            xi += m / m.sum()
        pi = g[0]; A = xi / xi.sum(1, keepdims=True)
        mu = (g * x[:, None]).sum(0) / g.sum(0)
        var = (g * (x[:, None] - mu) ** 2).sum(0) / g.sum(0) + 1e-9
    return mu, var, A, pi


def hmm_causal_state(x, mu, var, A, pi):
    """causal forward-filter posterior P(state=k | data up to t) -- uses NO future bars."""
    x = np.asarray(x, float); T = len(x)
    al = np.zeros((T, 2))
    B = np.stack([np.exp(-0.5 * (x - mu[k]) ** 2 / var[k]) / np.sqrt(2 * np.pi * var[k])
                  for k in range(2)], 1) + 1e-300
    al[0] = pi * B[0]; al[0] /= al[0].sum()
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * B[t]; al[t] /= al[t].sum()
    return al  # posterior per state


# ============================== reporting ==============================
def show(name, t, ref=None):
    m = metrics(t)
    if m is None:
        print(f"    {name:<24} (too few)"); return None
    flag = ""
    if ref is not None and m["cdd"] > ref:
        flag = "  > KAMA"
    print(f"    {name:<24} n={m['n']:>4} CAGR={m['cagr']:+5.1f}% DD={m['dd']:4.1f}% "
          f"CAGR/DD={m['cdd']:5.2f} | IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | yr+ {m['py']}{flag}")
    return m["cdd"]


def overlap(g1, g2):
    """fraction of days the two daily gates agree (both reindexed/ffilled already)."""
    a, b = g1.align(g2, join="inner")
    a = a.fillna(False).astype(bool); b = b.fillna(False).astype(bool)
    return (a == b).mean()


def run_instrument(csv, tf, rr, fwd, kama_n=14):
    d = resample(load_mt5_csv(csv), tf)
    t = run(d, SimpleNamespace(**{**CFG, "csv": csv, "tf": tf, "rr": rr, "fwd": fwd}))
    t = t.sort_values("time").reset_index(drop=True)
    dc = d["close"].resample("1D").last().dropna()
    r = np.log(dc).diff().fillna(0.0)
    a14 = dc.diff().abs().rolling(14).mean()
    s150 = dc.rolling(150).mean()

    # --- references ---
    km = kama(dc, kama_n); kama_up = (km > km.shift(1))
    mech = (dc > s150) & (s150 > s150.shift(10))
    oracle = (dc.shift(-20) > dc)

    print(f"\n{'='*84}\n=== {os.path.basename(csv)} {tf} breakout (entries FIXED) RR{rr}  "
          f"days={len(dc)}  IS<{SPLIT} ===")
    print("  -- references (bar to beat = KAMA-rising) --")
    show("always-on", t)
    show("mech SMA150+slope", t[at(mech.shift(1), t.time)])
    kama_cdd = show("KAMA-rising", t[at(kama_up.shift(1), t.time)])
    show("ORACLE fwd20 (cheat)", t[at(oracle, t.time)])

    # --- HMM (fit IS only, causal apply) ---
    is_mask = dc.index.year < SPLIT
    mu, var, A, pi = gaussian_hmm_fit(r[is_mask].values)
    bull = int(np.argmax(mu))                      # higher-mean-return state = deploy
    post = hmm_causal_state(r.values, mu, var, A, pi)
    hmm_on = pd.Series(post[:, bull] > 0.5, index=dc.index)
    persist = A[bull, bull]
    print(f"  -- HMM 2-state (IS-fit, causal): bull-state mu={mu[bull]:+.4f} stay-prob={persist:.2f} "
          f"on%={hmm_on.mean()*100:.0f} overlap(KAMA)={overlap(hmm_on, kama_up):.2f} --")
    show("HMM bull-state", t[at(hmm_on.shift(1), t.time)], kama_cdd)

    # --- Hurst plateau ---
    print("  -- Hurst R/S > 0.5 (trending) : window plateau --")
    for W in (30, 60, 90, 120):
        h = rolling_hurst(r.values, W); hs = pd.Series(h.values > 0.5, index=dc.index)
        show(f"Hurst W={W} (ov{overlap(hs,kama_up):.2f})", t[at(hs.shift(1), t.time)], kama_cdd)

    # --- Variance-ratio plateau ---
    print("  -- Variance-ratio VR(q) > 1 (trending), window=60 : q plateau --")
    for q in (5, 10, 20):
        vr = rolling_vr(r.values, 60, q); vs = pd.Series(vr.values > 1.0, index=dc.index)
        show(f"VR q={q} (ov{overlap(vs,kama_up):.2f})", t[at(vs.shift(1), t.time)], kama_cdd)
    return kama_cdd


def hmm_sanity():
    """synthetic 2-regime series: fitted states should recover the truth well above chance."""
    rng = np.random.default_rng(1)
    seg = []
    truth = []
    for _ in range(20):
        n = rng.integers(20, 60)
        if rng.random() < 0.5:
            seg.append(rng.normal(+0.01, 0.005, n)); truth.append(np.zeros(n))    # calm-up
        else:
            seg.append(rng.normal(-0.01, 0.02, n)); truth.append(np.ones(n))      # volatile-down
    x = np.concatenate(seg); tr = np.concatenate(truth)
    mu, var, A, pi = gaussian_hmm_fit(x)
    bull = int(np.argmax(mu))
    post = hmm_causal_state(x, mu, var, A, pi)
    pred_calm = (post[:, bull] > 0.5).astype(int)        # bull = calm-up = truth 0
    acc = max((pred_calm == (1 - tr)).mean(), (pred_calm == tr).mean())
    print(f"  [HMM sanity] recovered regime accuracy={acc*100:.0f}% (>~70% = EM/Viterbi works)")


def main():
    print("REGIME STATE-DETECTORS as deploy gates -- bar = beat KAMA-rising on BOTH instruments + plateau")
    hmm_sanity()
    run_instrument("data/vantage_xauusd_h1.csv", "1h", 3.0, 500)
    run_instrument("data/vantage_btcusd_h1.csv", "4h", 2.0, 300)
    print(f"\n{'='*84}")
    print("PASS rule: CAGR/DD > KAMA-rising on gold AND BTC, plateau over window, IS~=OOS, low KAMA-")
    print("overlap (supplies context KAMA lacks). HMM = peak PBO risk -> overfit_audit before believing.")
    print("In-sample only; live-forward arbitrates regime change.")


if __name__ == "__main__":
    main()
