"""Stages 6-7 for the BTC 15m pullback gauntlet: beta null (same-frequency random-long,
same exit geometry, KAMA-on bars only) + random-drop null (CAGR/DD) + per-year tables +
timeout-share honesty check. Reuses the gauntlet's build/evaluate via exec (canonical parity)."""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np

src = open("/home/angelbell/dev/auto-trade/experiments/btc15m_pullback_gauntlet.py").read()
exec(src.split('print("\\n================ STAGE 2')[0])  # funcs + cells dict, no stage prints

from research.edge_harness import random_drop_null

FINALISTS = [("15m", 4.0, 0.3), ("15m", 4.0, 0.25), ("30m", 2.0, 0.25), ("30m", 2.0, 0.3)]
SP = 15.0

def walk_np(h, l, c, i, e, stop, tgt, risk):
    j0, j1 = i + 1, min(i + 1 + FWD, len(c))
    if j0 >= j1: return None, i
    hs, ls = h[j0:j1], l[j0:j1]
    hit_s = ls <= stop; hit_t = hs >= tgt
    s_i = np.argmax(hit_s) if hit_s.any() else 10**9
    t_i = np.argmax(hit_t) if hit_t.any() else 10**9
    if s_i == 10**9 and t_i == 10**9:
        exit_j = min(i + FWD, len(c) - 1)
        return (c[exit_j] - e) / risk, exit_j
    if s_i <= t_i: return -1.0, j0 + s_i
    return (tgt - e) / risk, j0 + t_i

for (tf, RR, frac) in FINALISTS:
    df, E, h, l, c = cells[(tf, RR, True)]
    tr, miss = evaluate(df, E, h, l, c, frac)
    trm, _ = evaluate(df, E, h, l, c, None)
    trn = net(tr, SP); real_mean = np.mean([r for _, r in trn])
    print(f"\n########## {tf} RR{RR:.0f} frac{frac} (net$15, N={len(trn)}, meanR={real_mean:+.3f}) ##########")

    # timeout share (trades that hit neither stop nor tgt within FWD)
    n_to = 0
    for (t, R, risk) in tr:
        if R != -1.0 and abs(R * risk - (RR * risk + frac * risk) * (1 if R > 0 else 1)) > 1e-9:
            pass
    # simpler: recompute effRR and count exact-target hits
    effRR = (RR + frac) / (1 - frac)
    Rg = np.array([R for _, R, _ in tr])
    n_tgt = np.isclose(Rg, effRR, rtol=1e-6).sum(); n_stp = np.isclose(Rg, -1.0).sum()
    n_to = len(Rg) - n_tgt - n_stp
    print(f"  exits: target={n_tgt} stop={n_stp} timeout={n_to} ({n_to/len(Rg)*100:.0f}%)  effRR={effRR:.2f}  "
          f"breakeven win={1/(1+effRR)*100:.1f}%  actual win={(Rg>0).mean()*100:.1f}%")

    # ---- (a) beta null: same-frequency random-long on KAMA-on bars, same (risk%, RR) geometry ----
    dck = df["close"].resample("1D").last().dropna()
    kmg = kama_adaptive(dck, 14)
    kreg = ((kmg > kmg.shift(1)).shift(1)).reindex(df.index, method="ffill").fillna(False).values
    ok_bars = np.where(kreg[:len(c) - FWD - 1])[0]
    ok_bars = ok_bars[ok_bars > 500]
    riskpct = np.array([risk / (c[0] if False else 1) for _, _, risk in tr])  # placeholder
    # real geometry pairs: (risk fraction of entry price, reward/risk)
    geo = []
    for (i_, e_, stop_, tgt_, H1_) in E:
        pass
    # derive from trades directly: need entry px; recompute from tr is lossy -> rebuild pairs from E+frac
    geo = []
    for (i_, e_, stop_, tgt_, H1_) in E:
        lim = e_ - frac * (e_ - stop_)
        if lim <= stop_ or lim >= e_: continue
        geo.append((lim, (lim - stop_) / lim, (tgt_ - lim) / (lim - stop_)))
    riskf = np.array([g[1] for g in geo]); rrs = np.array([g[2] for g in geo])
    rng = np.random.default_rng(0); means = []
    for d_ in range(200):
        picks = np.sort(rng.choice(ok_bars, size=len(trn), replace=False))
        k = rng.integers(0, len(riskf), size=len(picks))
        busy = -1; Rs = []
        for p, ki in zip(picks, k):
            if p <= busy: continue
            e_ = c[p]; risk_ = riskf[ki] * e_; stop_ = e_ - risk_; tgt_ = e_ + rrs[ki] * risk_
            R, exit_j = walk_np(h, l, c, p, e_, stop_, tgt_, risk_)
            if R is None: continue
            Rs.append(R - SP / risk_); busy = exit_j
        if len(Rs) > 20: means.append(np.mean(Rs))
    means = np.array(means)
    pct = (means < real_mean).mean() * 100
    print(f"  BETA NULL (random-long, KAMA-on, same geometry, {len(means)} draws): "
          f"null meanR med={np.median(means):+.3f} sd={means.std():.3f} p90={np.percentile(means,90):+.3f} -> real %ile={pct:.1f}")

    # ---- (b) random-drop null vs market base (CAGR/DD) ----
    print("  RANDOM-DROP NULL vs market base (CAGR/DD):", end="")
    random_drop_null(net(trm, SP), trn, years=SPAN_YRS)

    # ---- per-year table (net$15) ----
    s = stats(trn)
    print("  per-year netR($15): " + "  ".join(f"{y}:{v:+.1f}" for y, v in s["ann"].items()))
