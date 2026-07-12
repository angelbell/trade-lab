"""edge_harness.py — the STANDARD evaluation harness for this lab (v3, numba-JIT).

Drop in ONE causal signal function; get a fixed VERDICT CARD across the whole TF
ladder, every time, with nothing forgotten. Rules the user kept restating live in
CODE here, not prose: always report PF+N+risk, sweep every TF, check the beta null,
no lookahead, model cost AND stop-slippage, skip the dead session window.

SIGNAL CONTRACT (causal):
    signal(df) -> np.ndarray length len(df), values {-1,0,+1}; sig[i] uses data
    through bar i's CLOSE only. Harness ENTERS at bar i+1 OPEN.

Per TF, ALWAYS prints: N, N/yr, win%, PF, meanR, IS/OOS, green-years, maxDD(R),
ret/DD, beta-null %ile.   [= PF + count + risk, per TF, with the beta null]

Tools:
  evaluate(name, signal, ...)         -> per-TF verdict card (+ optional dict via _return)
  audit(configs, flagship=)           -> DSR(trial-haircut)/PBO/CSCV/bootCI+null (overfit_audit)
  random_drop_null(base, kept, years) -> does a filter beat dropping the same N at random?
  combine(legs, weights=)             -> multi-instrument basket: corr + combined ret/DD  (N from instruments)
  check_causal(signal, df)            -> recompute on prefixes; FAILS if the signal peeks at the future
  sweep(name, build_signal, params)   -> PF/meanR per param at one TF: PLATEAU(real) vs SPIKE(overfit)

Knobs encoding lab lessons:  stop_slip (overshoot beyond the stop), skip_hours (dead window),
  resolve_fills (M5/M1 fill+exit order), exit_mode "rr"|"mean", slip (entry),
  entry="pullback"+pullback_frac (VALIDATED lever: pin stop+tgt at market levels, lower the
    entry to a frac-of-risk pullback -> effRR balloons; miss=runaway winner=adverse-selection
    modelled. Beats market on low-cost trend legs, gold15m/BTC; use shallow frac~0.25-0.3.
    NB breakouts pull back shallowly by nature, so deep frac selects the WEAKER breaks -> lumpy).
"""
import sys; sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from numba import njit
from src.data_loader import load_mt5_csv
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
HORIZON = 300

LADDERS = {
    "GOLD":   ("data/vantage_xauusd_m5.csv", 0.40,
               [("5m", "5min"), ("15m", "15min"), ("1h", "60min"), ("2h", "120min"),
                ("4h", "240min"), ("8h", "480min"), ("1d", "1440min")]),
    "BTC":    ("data/vantage_btcusd_h1.csv", 15.0,
               [("1h", None), ("2h", "120min"), ("4h", "240min"), ("8h", "480min"), ("1d", "1440min")]),
    "USDJPY": ("data/vantage_usdjpy_h1.csv", 0.015,
               [("1h", None), ("2h", "120min"), ("4h", "240min"), ("8h", "480min"), ("1d", "1440min")]),
}


@njit(cache=True)
def _walk(starts, sides, e, stop, tgt, sd, eh, el, ec, cost, stop_slip, horizon):
    """JIT barrier walk on EXEC bars. no-overlap via busy. stop_slip = fraction of the
    overshoot beyond the stop added to the loss (realistic stop fill)."""
    n = starts.shape[0]; R = np.full(n, np.nan); busy = -1; N = ec.shape[0]
    for k in range(n):
        p0 = starts[k]
        if p0 <= busy or p0 >= N or np.isnan(e[k]) or sd[k] <= 0.0:
            continue
        s = sides[k]; st = stop[k]; tg = tgt[k]; d = sd[k]
        r = np.nan; xj = p0 + horizon if p0 + horizon < N else N - 1
        end = p0 + horizon if p0 + horizon < N else N
        for j in range(p0, end):
            if s > 0:
                if el[j] <= st:
                    over = st - el[j]; r = -1.0 - stop_slip * over / d; xj = j; break
                if eh[j] >= tg: r = (tg - e[k]) / d; xj = j; break
            else:
                if eh[j] >= st:
                    over = eh[j] - st; r = -1.0 - stop_slip * over / d; xj = j; break
                if el[j] <= tg: r = (e[k] - tg) / d; xj = j; break
        if np.isnan(r):
            r = (ec[xj] - e[k]) / d if s > 0 else (e[k] - ec[xj]) / d
        R[k] = r - cost / d; busy = xj
    return R


@njit(cache=True)
def _walk_pullback(starts, sides, e_mkt, stop, tgt, sd, eh, el, ec, cost, stop_slip, horizon, frac):
    """Pullback-limit to a FIXED target: stop & tgt are pinned at the MARKET-entry levels;
    only the entry is lowered to e_mkt - frac*sd (long). Fill on the pullback touch BEFORE the
    target is reached (else MISSED = the runaway winner, adverse selection modelled). Realized
    risk shrinks to (1-frac)*sd -> effective RR balloons. R is in units of that realized risk.
    (Validated lever: cheaper entry + far fixed target beats market on low-cost trend legs.)"""
    n = starts.shape[0]; R = np.full(n, np.nan); busy = -1; N = ec.shape[0]
    for k in range(n):
        p0 = starts[k]
        if p0 <= busy or p0 >= N or np.isnan(e_mkt[k]) or sd[k] <= 0.0:
            continue
        s = sides[k]; st = stop[k]; tg = tgt[k]; d = sd[k]
        lim = e_mkt[k] - frac * d if s > 0 else e_mkt[k] + frac * d
        end = p0 + horizon if p0 + horizon < N else N
        fill = -1
        for j in range(p0, end):                        # locate the pullback fill (miss if tgt first)
            if s > 0:
                if eh[j] >= tg: break
                if el[j] <= lim: fill = j; break
            else:
                if el[j] <= tg: break
                if eh[j] >= lim: fill = j; break
        if fill < 0:
            continue                                    # never pulled back = missed (runaway)
        risk = (lim - st) if s > 0 else (st - lim)
        if risk <= 0.0:
            continue
        r = np.nan; xj = fill + horizon if fill + horizon < N else N - 1
        end2 = fill + horizon if fill + horizon < N else N
        for j in range(fill, end2):                     # barrier walk from the fill bar
            if s > 0:
                if el[j] <= st:
                    over = st - el[j]; r = -1.0 - stop_slip * over / risk; xj = j; break
                if eh[j] >= tg: r = (tg - lim) / risk; xj = j; break
            else:
                if eh[j] >= st:
                    over = eh[j] - st; r = -1.0 - stop_slip * over / risk; xj = j; break
                if el[j] <= tg: r = (lim - tg) / risk; xj = j; break
        if np.isnan(r):
            r = (ec[xj] - lim) / risk if s > 0 else (lim - ec[xj]) / risk
        R[k] = r - cost / risk; busy = xj
    return R


def _prep_and_walk(idx, sides, atr, mean, rr, katr, exit_mode, slip, cost, stop_slip,
                   exec_o, exec_h, exec_l, exec_c, starts, horizon, entry="market", limit_atr=0.0,
                   pullback_frac=0.0):
    sgn = np.where(sides > 0, 1.0, -1.0)
    sd = katr * atr[idx]
    if entry == "pullback":                                   # fixed stop+tgt at market levels, entry lowered
        e_mkt = exec_o[starts] + slip * sgn
        stop = np.where(sides > 0, e_mkt - sd, e_mkt + sd).astype(np.float64)
        tgt = np.where(sides > 0, e_mkt + rr * sd, e_mkt - rr * sd).astype(np.float64)
        return _walk_pullback(starts.astype(np.int64), sides.astype(np.int64), e_mkt.astype(np.float64),
                              stop, tgt, sd.astype(np.float64), exec_h, exec_l, exec_c, cost, stop_slip,
                              horizon, pullback_frac)
    if entry == "limit":
        lim = exec_o[starts] - limit_atr * sd * sgn            # long: below open; short: above
        fill = np.where(sides > 0, exec_l[starts] <= lim, exec_h[starts] >= lim)  # else MISSED (adverse selection)
        e = np.where(fill, lim, np.nan)
    else:
        e = exec_o[starts] + slip * sgn                         # market: pay slip at next open
    stop = np.where(sides > 0, e - sd, e + sd)
    if exit_mode == "mean":
        m = mean[np.minimum(idx + 1, len(mean) - 1)]
        tgt = m.astype(np.float64)
        bad = np.where(sides > 0, ~(m > e), ~(m < e))
        e = e.astype(np.float64).copy(); e[bad] = np.nan
    else:
        tgt = np.where(sides > 0, e + rr * sd, e - rr * sd).astype(np.float64)
    return _walk(starts.astype(np.int64), sides.astype(np.int64), e.astype(np.float64),
                 stop.astype(np.float64), tgt.astype(np.float64), sd.astype(np.float64),
                 exec_h, exec_l, exec_c, cost, stop_slip, horizon)


def _card_stats(times, R, span_yrs):
    yr = np.array([t.year for t in times])
    pf = R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99
    yrs = np.unique(yr); half = yrs[len(yrs) // 2]
    green = sum(1 for y in yrs if R[yr == y].sum() > 0)
    cum = R.cumsum(); dd = float((np.maximum.accumulate(cum) - cum).max())
    retdd = R.sum() / dd if dd > 1e-9 else float("inf")
    return dict(N=len(R), npy=len(R) / span_yrs, win=(R > 0).mean() * 100, pf=pf,
                meanR=R.mean(), IS=R[yr < half].mean() if (yr < half).any() else float("nan"),
                OOS=R[yr >= half].mean(), green=green, ny=len(yrs), maxDD=dd, retdd=retdd)


def evaluate(name, signal, rr=2.0, katr=1.0, exit_mode="rr", resolve_fills=False,
             slip=0.0, stop_slip=0.0, skip_hours=None, beta_trials=300, only=None,
             quiet=False, cost=None, entry="market", limit_atr=0.0, pullback_frac=0.0, _return=False):
    csv, dfl_cost, tfs = LADDERS[name]
    cost = dfl_cost if cost is None else cost                  # cost=0 -> GROSS edge (find edge first)
    base = load_mt5_csv(csv)
    bt_ns = base.index.values.astype("datetime64[ns]").astype("int64")
    bo, bh, bl, bc = (base[k].values for k in ("open", "high", "low", "close"))
    skip = set(skip_hours or [])
    if not quiet:
        ex = f" exit={exit_mode}" + ("/M5fill" if resolve_fills else "") + (f" slip={slip}" if slip else "") \
             + (f" sslip={stop_slip}" if stop_slip else "") + (f" skip{sorted(skip)}UTC" if skip else "") \
             + (f" LIMIT-{limit_atr}ATR" if entry == "limit" else "") \
             + (f" PULLBACK-{pullback_frac}risk->fixedTgt" if entry == "pullback" else "") + (" GROSS" if cost == 0 else "")
        print(f"\n========== {name}  ({signal.__name__}, RR{rr}/{katr}ATR{ex}, cost={cost}) ==========")
        print(f"  {'TF':<4}{'N':>6}{'N/yr':>6}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>14}{'green':>7}{'maxDD-R':>9}{'ret/DD':>8}{'beta%':>7}")
    out = {}
    for lbl, fr in tfs:
        if only and lbl not in only: continue
        df = base if fr is None else base.resample(fr).agg(AGG).dropna()
        if len(df) < 300: continue
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        atr = ta.atr(df["high"], df["low"], df["close"], 14).values
        mean = df["close"].rolling(20).mean().values
        t_ns = df.index.values.astype("datetime64[ns]").astype("int64")
        sig = np.asarray(signal(df))
        idx = np.where(sig != 0)[0]; idx = idx[idx + 1 < len(c)]
        if skip and len(idx):
            hrs = df.index[idx + 1].hour.values
            idx = idx[~np.isin(hrs, list(skip))]
        if len(idx) < 12:
            if not quiet: print(f"  {lbl:<4}{len(idx):>6}  (too few)")
            continue
        sides = sig[idx].astype(np.int64)
        if resolve_fills and fr is not None:
            starts = np.searchsorted(bt_ns, t_ns[idx + 1])
            ratio = max(1, int(round((t_ns[1] - t_ns[0]) / (bt_ns[1] - bt_ns[0]))))
            R = _prep_and_walk(idx, sides, atr, mean, rr, katr, exit_mode, slip, cost, stop_slip, bo, bh, bl, bc, starts, HORIZON * ratio, entry, limit_atr, pullback_frac)
        else:
            R = _prep_and_walk(idx, sides, atr, mean, rr, katr, exit_mode, slip, cost, stop_slip, o, h, l, c, idx + 1, HORIZON, entry, limit_atr, pullback_frac)
        ok = ~np.isnan(R); R = R[ok]; et = df.index[idx + 1][ok]
        if len(R) < 12:
            if not quiet: print(f"  {lbl:<4}{len(R):>6}  (too few fills)")
            continue
        span = (df.index[-1] - df.index[0]).days / 365.25
        s = _card_stats(list(et), R, span)
        if not quiet:
            bpct = _beta_pct(idx, sides, o, h, l, c, atr, mean, rr, katr, exit_mode, slip, cost, stop_slip, s["pf"], beta_trials, entry, limit_atr, pullback_frac) if beta_trials else float("nan")
            isoos = f"{s['IS']:+.2f}/{s['OOS']:+.2f}"; grn = f"{s['green']}/{s['ny']}"
            bs = f"{bpct:>5.0f}%" if beta_trials else "    -"
            print(f"  {lbl:<4}{s['N']:>6}{s['npy']:>6.0f}{s['win']:>5.0f}%{s['pf']:>7.2f}{s['meanR']:>+8.3f}"
                  f"{isoos:>14}{grn:>7}{s['maxDD']:>9.1f}{s['retdd']:>8.2f}{bs:>7}")
        out[lbl] = (list(zip(et, R)), s)
    if not quiet:
        print("  (beta% = real PF's %ile vs random SAME-SIDE entries; <70 drift/beta, >90 real selection)")
    return out if _return else None


def _beta_pct(idx, sides, o, h, l, c, atr, mean, rr, katr, exit_mode, slip, cost, stop_slip, real_pf, trials, entry="market", limit_atr=0.0, pullback_frac=0.0, seed=0):
    rng = np.random.default_rng(seed)
    valid = np.where(~np.isnan(atr) & (atr > 0))[0]; valid = valid[valid + 1 < len(c)]
    nlong = int((sides > 0).sum()); ntot = len(sides); pfs = []
    for _ in range(trials):
        pick = np.sort(rng.choice(valid, size=min(ntot, len(valid)), replace=False))
        sd = np.where(np.arange(len(pick)) < nlong, 1, -1); rng.shuffle(sd)
        R = _prep_and_walk(pick, sd.astype(np.int64), atr, mean, rr, katr, exit_mode, slip, cost, stop_slip, o, h, l, c, pick + 1, HORIZON, entry, limit_atr, pullback_frac)
        R = R[~np.isnan(R)]
        if len(R) >= 8:
            pfs.append(R[R > 0].sum() / abs(R[R <= 0].sum()) if (R <= 0).any() else 9.99)
    return (np.array(pfs) < real_pf).mean() * 100 if pfs else float("nan")


def audit(configs, flagship=None, trials=2000, seed=0):
    configs = {k: (v[0] if isinstance(v, tuple) else v) for k, v in configs.items()}  # accept evaluate() output
    cols, srs = {}, []
    for nm, tr in configs.items():
        s = pd.Series([r for _, r in tr], index=pd.to_datetime([t for t, _ in tr]))
        cols[nm] = s.groupby(pd.Grouper(freq="M")).sum(); srs.append(s.mean() / s.std(ddof=1))
    M = pd.concat(cols, axis=1).fillna(0.0).values; V = float(np.nanvar(srs))
    flagship = flagship or max(configs, key=lambda k: len(configs[k]))
    R = np.array([r for _, r in configs[flagship]])
    t0, t1 = configs[flagship][0][0], configs[flagship][-1][0]
    yrs = (pd.Timestamp(t1) - pd.Timestamp(t0)).days / 365.25
    print(f"\n--- AUDIT (flagship={flagship}, configs={M.shape[1]}, V_SR={V:.4f}) ---")
    print("  DSR: " + "  ".join(f"@{N}={psr(R, sr0(N, V))[0]:.2f}" for N in (1, 10, 50, 100, 200, 400)))
    pbo, oosm, pl = cscv(M)
    print(f"  PBO={pbo:.2f} (IS-best OOS-SR={oosm:+.2f}, P(OOS loss)={pl:.2f})  [<0.2 robust, ~0.5 noise]")
    rng = np.random.default_rng(seed); obs = cdd_R(R, yrs)[2]
    boot = np.array([cdd_R(block_resample(R, 20, rng), yrs)[2] for _ in range(trials)])
    nul = np.array([cdd_R(block_resample(R - R.mean(), 20, rng), yrs)[2] for _ in range(trials)])
    print(f"  CAGR/DD obs={obs:+.2f} bootCI[5/50/95]={np.percentile(boot,5):+.2f}/{np.percentile(boot,50):+.2f}/"
          f"{np.percentile(boot,95):+.2f} null p={(nul>=obs).mean():.3f}")


def random_drop_null(base, kept, years, trials=2000, seed=0):
    bR = np.array([r for _, r in base]); kR = np.array([r for _, r in kept])
    obs = cdd_R(kR, years)[2]; rng = np.random.default_rng(seed)
    nul = np.array([cdd_R(rng.choice(bR, len(kR), replace=False), years)[2] for _ in range(trials)])
    print(f"  random-drop null: kept CAGR/DD={obs:+.2f} vs null med={np.median(nul):+.2f} "
          f"pctile={(nul < obs).mean()*100:.0f}%  (>90 = filter adds real edge, not n-trimming)")
    return (nul < obs).mean() * 100


def combine(legs, weights=None, freq="Q"):
    """legs = {name: [(time,R)]} (one chosen leg per instrument). Reports cross-correlation
    (period-R) + combined ret/DD at given weights. Serves 'N from multiple instruments'."""
    series = {}
    for nm, tr in legs.items():
        s = pd.Series([r for _, r in tr], index=pd.to_datetime([t for t, _ in tr]))
        series[nm] = s.groupby(pd.Grouper(freq=freq)).sum()
    df = pd.concat(series, axis=1).fillna(0.0)
    w = np.array([(weights or {}).get(c, 1.0) for c in df.columns]); w = w / w.sum()
    comb = (df * w).sum(axis=1)
    cum = comb.cumsum(); dd = float((cum.cummax() - cum).max()); retdd = comb.sum() / dd if dd > 1e-9 else float("inf")
    print(f"\n--- COMBINE ({', '.join(df.columns)}; weights={dict(zip(df.columns, w.round(2)))}; corr on {freq}-R) ---")
    print(df.corr().round(2).to_string().replace("\n", "\n   "))
    indiv = {c: (df[c].sum() / (df[c].cumsum().cummax() - df[c].cumsum()).max()) for c in df.columns}
    print(f"   per-leg ret/DD: " + "  ".join(f"{c}={v:+.2f}" for c, v in indiv.items()))
    print(f"   COMBINED: totR={comb.sum():+.1f}  maxDD={dd:.1f}  ret/DD={retdd:+.2f}")
    return retdd


def check_causal(signal, df, ntest=12, tail=40, seed=0):
    """recompute the signal on truncated prefixes; if values for ALREADY-CLOSED bars change
    when future bars are added, the signal PEEKS at the future = lookahead. FAILS loudly."""
    rng = np.random.default_rng(seed); full = np.asarray(signal(df), float)
    ks = rng.integers(max(300, len(df) // 2), len(df), size=ntest); bad = 0
    for k in ks:
        cut = np.asarray(signal(df.iloc[:int(k)]), float)
        a = np.nan_to_num(full[:int(k)][-tail:]); b = np.nan_to_num(cut[-tail:])
        if a.shape != b.shape or not np.array_equal(a, b): bad += 1
    ok = bad == 0
    print(f"  causal check: {ntest-bad}/{ntest} prefixes match -> {'PASS (no lookahead)' if ok else 'FAIL: signal uses FUTURE data'}")
    return ok


def sweep(name, build_signal, params, tf="4h", pname="p", **kw):
    """PF/meanR/N per param at ONE TF -> read PLATEAU (neighbors agree = real) vs SPIKE (overfit)."""
    print(f"\n-- sweep {name} @ {tf} ({pname}: plateau=real, spike=overfit) --")
    for p in params:
        r = evaluate(name, build_signal(p), only=[tf], beta_trials=0, quiet=True, _return=True, **kw)
        if tf in r:
            _, s = r[tf]
            print(f"  {pname}={p}:  N={s['N']:>4}  PF={s['pf']:.2f}  meanR={s['meanR']:+.3f}  ret/DD={s['retdd']:+.2f}")
        else:
            print(f"  {pname}={p}:  (too few)")


# ---------------- demo signals (causal) ----------------
def demo_breakout(df):
    c = df["close"].values; sma = df["close"].rolling(100).mean().values
    dch = pd.Series(df["high"].values).rolling(20).max().shift(1).values
    dcl = pd.Series(df["low"].values).rolling(20).min().shift(1).values
    sig = np.zeros(len(c)); sig[(c > dch) & (c > sma)] = 1; sig[(c < dcl) & (c < sma)] = -1
    sig[:101] = 0; return sig


def demo_rsi_fade(df):
    rsi = ta.rsi(df["close"], 14).values
    sig = np.zeros(len(df)); sig[rsi <= 20] = 1; sig[rsi >= 80] = -1; return sig


def _don(df, n):
    c = df["close"].values; sma = df["close"].rolling(100).mean().values
    dch = pd.Series(df["high"].values).rolling(n).max().shift(1).values
    dcl = pd.Series(df["low"].values).rolling(n).min().shift(1).values
    sig = np.zeros(len(c)); sig[(c > dch) & (c > sma)] = 1; sig[(c < dcl) & (c < sma)] = -1
    sig[:max(n, 100) + 1] = 0; return sig


if __name__ == "__main__":
    only = ["1h", "4h", "1d"]
    g = evaluate("GOLD", demo_breakout, rr=3.0, only=only, stop_slip=0.5, skip_hours=(12, 13, 14), _return=True)
    check_causal(demo_breakout, load_mt5_csv(LADDERS["GOLD"][0]).resample("240min").agg(AGG).dropna())
    b = evaluate("BTC", demo_breakout, rr=3.0, only=only, stop_slip=0.5, _return=True)
    u = evaluate("USDJPY", demo_breakout, rr=3.0, only=only, stop_slip=0.5, _return=True)
    if g and b and u:
        combine({"GOLD": g["4h"][0], "BTC": b["4h"][0], "USDJPY": u["4h"][0]})
    sweep("GOLD", lambda n: (lambda df: _don(df, n)), [10, 20, 30, 40], tf="4h", pname="donch", rr=3.0)
