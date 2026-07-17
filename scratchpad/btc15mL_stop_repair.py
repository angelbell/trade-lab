"""btc15mL_stop_repair.py -- test the user's two live-chart observations on btc15m_L
(BTC 15m ZigZag(2.0xATR) Pattern-B, trend_ema80, 4h-KAMA(14) gate, RR4.5, pullback_frac=0.30,
fill_win=200, cost $15/risk, start 2018-10-01):

  obs1: "stop is too shallow, gets wicked -- should sit near L0 (the wave origin), 2 swings back"
  obs2: "price ran to near target then gave it back to a stop-out" (runner give-back / ratchet)

Frozen spec card, executed faithfully. Sections:
  E1  sweep --sl-b x --pullback-frac, --tgt-ref stop (widening the stop ALSO pushes the target
      out -- R-multiple per win is unchanged by construction; this is the "just move the stop"
      idea, mechanised literally)
  E2  same grid, --tgt-ref l2 (target price held at the TIGHT swinglow-implied level; a wider
      stop buys win-rate at a lower R per win) -- the DIRECT test of "the target is fine, the
      stop is too tight"
  E3  TP2-ratchet re-test AT RR4.5 (previous kill was measured at RR4.0 on the 4h gate:
      totR/DD 9.47->5.07, docs/findings/s02_exits.md). Ratchet mechanism (u = lim-stop, the
      REALIZED risk at the pullback-limit fill): once high >= e_px + TP2*u, stop -> e_px + TP1*u.
  E4  small-account impact: 2025+ median stop-width($) per sl-b mode -> %-risk a 100,000-JPY
      account is forced into at the minimum 0.01 BTC lot (USDJPY=155).

Machinery REUSED, not reinvented:
  - breakout_wave.run/resample (the actual trade engine: --sl-b, --sl-b-k, --tgt-ref, all
    already implemented in the engine -- E1/E2 call run() directly, no re-implementation)
  - breakout_wave.swings_zigzag / kama_adaptive (imported for E3's entry list, since the
    engine's run() has no ratchet-exit option; the entry/gate/fill ASSEMBLY in build_entries_L
    below is a parameter-only adaptation of pine_replica_btc15m.build_entries -- gate_tf and rr
    made configurable instead of pine_replica's hardcoded daily-gate/RR4.0 -- so it can be
    tie-backed against the CURRENT adopted leg (4h gate, RR4.5) rather than the old one)
  - book_spec_fix.build/book/w_trade (the frozen 6-leg book + trade-resolution-DD arbiter)
  - btc15mS_symmetry.make_S (btc15m_S rebuilt at the CURRENT adopted RR4.5, per spec card)
  - btc15mS_symmetry.boot_pair / book_series, book_leave_one_out.cdd (block bootstrap machinery)

TIE-BACKS (both checked at runtime, script aborts loudly if either fails):
  T1  leg:  the exact CLI in the spec card must give n=759, win=23%, meanR=+0.59
  T2  book: build('2018-01-01', False) with btc15m_S swapped to make_S(d15,'1D',4.5,None)
      must give 6-leg CAGR/DD = 8.48 (maxDD 7.49%)
  T3  (E3 only) the custom ratchet-off walk must reproduce T1's 759 trades with ZERO
      per-trade R mismatches vs breakout_wave.run()'s own trade table (verified in dev,
      re-checked at runtime below)

Run:
  .venv/bin/python scratchpad/btc15mL_stop_repair.py --smoke
  .venv/bin/python scratchpad/btc15mL_stop_repair.py
"""
import argparse
import contextlib
import io
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from src.data_loader import load_mt5_csv
    from breakout_wave import run, resample, swings_zigzag, kama_adaptive
    from radar_gate_race import BASE
    from book_spec_fix import build, book, w_trade
    from btc15mS_symmetry import make_S, boot_pair, book_series
    from book_leave_one_out import cdd

ROOT = "/home/angelbell/dev/auto-trade"
NEW = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
COST_L = 15.0     # $/risk, matches CLAUDE.md live-account BTC cost canon
USDJPY = 155.0
ACCOUNT_JPY = 100_000.0
LOT_BTC = 0.01

SL_MODES = [
    ("swinglow(現行)",  dict(sl_b="swinglow", sl_b_k=1.5)),
    ("band k=0.5",      dict(sl_b="band",     sl_b_k=0.5)),
    ("band k=1.0",      dict(sl_b="band",     sl_b_k=1.0)),
    ("band k=1.5",      dict(sl_b="band",     sl_b_k=1.5)),
    ("band k=2.0",      dict(sl_b="band",     sl_b_k=2.0)),
    ("origin",          dict(sl_b="origin",   sl_b_k=1.5)),
    ("atr k=1.5",       dict(sl_b="atr",      sl_b_k=1.5)),
    ("atr k=2.0",       dict(sl_b="atr",      sl_b_k=2.0)),
    ("atr k=3.0",       dict(sl_b="atr",      sl_b_k=3.0)),
]
PFS_FULL = [0.30, 0.25, 0.20, 0.15]


# --------------------------------------------------------------------- E1/E2 leg builder ---
def make_L(d15, sl_b="swinglow", sl_b_k=1.5, tgt_ref="stop", pf=0.30, rr=4.5, cost=COST_L):
    """Rebuild btc15m_L via the ENGINE (breakout_wave.run, --sl-b/--sl-b-k/--tgt-ref already
    implemented there -- nothing re-derived here) with one axis (stop placement / pullback
    frac / target reference) swapped in. Same PDH soft-sizing + cost application as
    book_spec_fix.build()'s btc15m_L leg."""
    kw = dict(BASE)
    kw.update(gate_kama=14, gate_kama_tf="240min", pullback_frac=pf, rr=rr,
              sl_b=sl_b, sl_b_k=sl_b_k, tgt_ref=tgt_ref, fill_win=200, cost=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        t = run(d15, SimpleNamespace(**kw))
    if t is None or len(t) < 5:
        return None, None
    RL = t["R"].values - cost / t["risk"].values
    ei = d15.index.get_indexer(t["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
    s = pd.Series(RL * w, index=pd.DatetimeIndex(t["time"]))
    return s, t


def leg_stats(s, t, span):
    n = len(s)
    pos, neg = s[s > 0].sum(), abs(s[s <= 0].sum())
    pf_ = pos / neg if neg > 0 else np.inf
    eq = np.cumprod(1 + 0.01 * s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    days = (s.index[-1] - s.index[0]).days
    cagr = (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100
    yrs = s.index.year.values; half = np.median(yrs)
    return dict(n=n, npyr=n / span, win=(s > 0).mean() * 100, pf=pf_, meanR=s.mean(),
                medR=s.median(), stdR=s.std(), totRyr=s.sum() / span, maxdd=dd,
                cagr_dd=(cagr / dd if dd > 0 else np.nan),
                isoos=f"{s[yrs < half].mean():+.2f}/{s[yrs >= half].mean():+.2f}",
                stopmed=t["risk"].median())


# ------------------------------------------------------------------------- E3 ratchet ------
def build_entries_L(df, gate_tf="240min", rr=4.5, bo_window=20):
    """Parameter-only adaptation of pine_replica_btc15m.build_entries: gate_tf and rr made
    configurable (pine_replica hardcoded daily-gate/RR4.0) so this ties back to the CURRENT
    adopted btc15m_L (4h gate, RR4.5) instead of the old one. swings_zigzag/kama_adaptive are
    the imported engine primitives -- not re-derived. Returns (entry_bar, entry_px, stop, tgt)
    with NO pdh weight baked in (weight is computed post-walk from the FILL price, matching
    book_spec_fix's own convention -- see note in main())."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = ta.atr(df["high"], df["low"], df["close"], 14).values
    es = df["close"].ewm(span=80, adjust=False).mean().values
    dck = df["close"].resample(gate_tf).last().dropna()
    kmg = kama_adaptive(dck, 14)
    kreg = ((kmg > kmg.shift(1)).shift(1)).reindex(df.index, method="ffill").fillna(False).values
    sw = swings_zigzag(h, l, a, 2.0)
    E = []
    for t in range(2, len(sw)):
        (cL2, iL2, pL2, kL2), (cH1, iH1, pH1, kH1), (cL0, iL0, pL0, kL0) = sw[t], sw[t - 1], sw[t - 2]
        if not (kL2 == -1 and kH1 == +1 and kL0 == -1):
            continue
        if pL2 <= pL0 or pH1 - pL0 <= 0:
            continue
        if not np.isnan(es[cL2]) and pH1 < es[cL2]:
            continue
        e_i = None
        for j in range(cL2 + 1, min(cL2 + 1 + bo_window, len(c))):
            if c[j] > pH1:
                e_i = j; break
        if e_i is None or not kreg[e_i]:
            continue
        e = c[e_i]; stop = pL2
        if e - stop <= 0:
            continue
        E.append((e_i, e, stop, e + rr * (e - stop)))
    E.sort(key=lambda x: x[0])
    seen = set(); U = []
    for en in E:
        if en[0] in seen:
            continue
        seen.add(en[0]); U.append(en)
    return U


def walk_ratchet(df, E, pf=0.30, fill_win=200, fwd=500, tp2=None, tp1=None, cost=COST_L):
    """Exit-walk that mirrors breakout_wave.run()'s pullback-limit path EXACTLY (identical
    fill-loop range, identical exit-loop range, stop checked before target on a same-bar
    conflict -- the engine's own conservative convention), with an ADDITIONAL optional
    ratchet arm: once high touches e_px + tp2*risk, stop is raised to e_px + tp1*risk
    (tp2=None => ratchet off => bit-identical to the engine, verified in main()).
    Returns (fill_time, e_px, stop, R_raw) -- unweighted, uncosted-except-cost-arg;
    PDH weighting and the $/risk cost are applied by the caller (matches book_spec_fix's
    order: cost then weight)."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    busy_until = []
    out = []
    for (i, e, stop, tgt) in E:
        busy_until = [x for x in busy_until if x >= i]
        if len(busy_until) >= 1:
            continue
        lim = e - pf * (e - stop)
        fj = None
        for j in range(i + 1, min(i + 1 + fill_win, len(c))):
            if h[j] >= tgt:
                break
            if l[j] <= lim:
                fj = j; break
        if fj is None:
            continue
        e_px, e_bar = lim, fj
        risk = e_px - stop
        reward = tgt - e_px
        if risk <= 0:
            continue
        exit_j = min(e_bar + fwd, len(c) - 1)
        R = None
        cur_stop = stop
        ratched = False
        for j in range(e_bar + 1, min(e_bar + 1 + fwd, len(c))):
            if l[j] <= cur_stop:
                R = (cur_stop - e_px) / risk; exit_j = j; break
            if h[j] >= tgt:
                R = reward / risk; exit_j = j; break
            if tp2 is not None and not ratched and h[j] >= e_px + tp2 * risk:
                cur_stop = max(cur_stop, e_px + tp1 * risk); ratched = True
        if R is None:
            R = (c[exit_j] - e_px) / risk
        out.append((df.index[e_bar], e_px, stop, R))
        busy_until.append(exit_j)
    return out


def weighted_series(tr, d15, cost=COST_L):
    """apply the SAME PDH soft-sizing + $/risk cost as book_spec_fix.build()'s btc15m_L leg
    (weight decided from the FILL price vs the prior-day-high AS OF THE FILL BAR -- not the
    signal bar; this matches book_spec_fix exactly, verified via the E3 ratchet-off tie-back)."""
    if not tr:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([x[0] for x in tr])
    e_px = np.array([x[1] for x in tr]); stop = np.array([x[2] for x in tr])
    R = np.array([x[3] for x in tr]); risk = e_px - stop
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(idx)
    w = np.where(e_px > pdh[ei], 1.0, 0.5)
    RL = (R - cost / risk) * w
    return pd.Series(RL, index=idx)


# --------------------------------------------------------------------------- E4 -------------
def e4_row(t, tag):
    """2025+ trades' median stop-width($) -> %-risk a 100,000-JPY account is forced into
    at the minimum 0.01 BTC lot (USDJPY=155)."""
    sub = t[pd.DatetimeIndex(t["time"]).year >= 2025] if t is not None else None
    if sub is None or len(sub) < 3:
        return dict(tag=tag, n=0, stopmed=np.nan, riskpct=np.nan)
    stopmed = sub["risk"].median()
    acct_usd = ACCOUNT_JPY / USDJPY
    risk_usd = stopmed * LOT_BTC
    riskpct = risk_usd / acct_usd * 100
    return dict(tag=tag, n=len(sub), stopmed=stopmed, riskpct=riskpct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    if args.smoke:
        d15 = d15.loc["2023-01-01":]
    span = (d15.index[-1] - d15.index[0]).days / 365.25
    print(f"d15: {d15.index[0].date()} -> {d15.index[-1].date()}  ({span:.2f}yr)"
          + ("  [SMOKE]" if args.smoke else ""))

    # ---------------------------------------------------------------- T1 leg tie-back ------
    with contextlib.redirect_stdout(io.StringIO()):
        s_cur, t_cur = make_L(d15, pf=0.30)   # current adopted leg, all defaults
    if not args.smoke:
        ok = (len(t_cur) == 759 and abs((t_cur["R"].mean()) - 0.59) < 0.01
              and abs((t_cur["R"] > 0).mean() * 100 - 23) < 1.0)
        print(f"T1 leg tie-back: n={len(t_cur)} win={(t_cur['R']>0).mean()*100:.0f}% "
              f"meanR={t_cur['R'].mean():+.2f}  (spec: n=759 win=23% meanR=+0.59)"
              f"  {'OK' if ok else '*** MISMATCH ***'}")
        if not ok:
            print("ABORTING: leg tie-back failed."); return
    else:
        print(f"T1 leg tie-back SKIPPED under --smoke (truncated data): n={len(t_cur)}")

    # --------------------------------------------------------------- T2 book tie-back ------
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        L_fixed = build("2018-01-01", False)
        L_fixed["btc15m_S"] = make_S(d15, "1D", 4.5, None)
    if not args.smoke:
        c0 = book(L_fixed, NEW)
        ok2 = abs(c0[2] - 8.48) < 0.01 and abs(c0[1] - 7.49) < 0.01
        print(f"T2 book tie-back: CAGR/DD={c0[2]:.2f} maxDD={c0[1]:.2f}%  "
              f"(spec: 8.48 / 7.49%)  {'OK' if ok2 else '*** MISMATCH ***'}")
        if not ok2:
            print("ABORTING: book tie-back failed."); return
    else:
        print("T2 book tie-back SKIPPED under --smoke (build() always uses full data)")

    def book_with_L(s):
        LL = dict(L_fixed); LL["btc15m_L"] = s
        return book(LL, NEW)

    sl_modes = SL_MODES if not args.smoke else SL_MODES[:3]
    pfs = PFS_FULL if not args.smoke else [0.30, 0.20]

    # ------------------------------------------------------------------- n0 baselines ------
    n0 = {}
    for tgt_ref in ("stop", "l2"):
        for label, kw in sl_modes:
            s0, t0 = make_L(d15, tgt_ref=tgt_ref, pf=0.0, **kw)
            n0[(tgt_ref, label)] = len(s0) if s0 is not None else 0

    # ---------------------------------------------------------------------- E1 / E2 --------
    all_rows = {}
    e4_all = {}
    for tgt_ref, tag in (("stop", "E1"), ("l2", "E2")):
        print("\n" + "=" * 118)
        print(f"{tag}  --tgt-ref {tgt_ref}" + ("  (目標も損切りと一緒に遠のく)" if tgt_ref == "stop"
              else "  (目標は現行の価格のまま=スイングロー基準に固定)"))
        print("=" * 118)
        hdr = (f"{'sl-b':<16}{'pf':>5}{'n':>5}{'N/yr':>6}{'fill%':>7}{'win%':>6}{'PF':>6}"
               f"{'meanR':>8}{'medR':>7}{'stdR':>7}{'stop$med':>10}{'legCAGR/DD':>11}"
               f"{'bookCAGR/DD':>12}{'IS/OOS':>16}")
        print(hdr)
        for label, kw in sl_modes:
            for pf in pfs:
                s, t = make_L(d15, tgt_ref=tgt_ref, pf=pf, **kw)
                key = (tag, label, pf)
                if s is None:
                    print(f"{label:<16}{pf:>5.2f}   no trades")
                    all_rows[key] = None
                    continue
                m = leg_stats(s, t, span)
                m["fill_pct"] = 100 * m["n"] / n0[(tgt_ref, label)] if n0[(tgt_ref, label)] else np.nan
                bc = book_with_L(s)
                m["book_cagr_dd"] = bc[2]
                all_rows[key] = m
                e4_all[key] = e4_row(t, f"{tag} {label} pf={pf}")
                print(f"{label:<16}{pf:>5.2f}{m['n']:>5}{m['npyr']:>6.1f}{m['fill_pct']:>6.0f}%"
                      f"{m['win']:>5.0f}%{m['pf']:>6.2f}{m['meanR']:>+8.3f}{m['medR']:>+7.2f}"
                      f"{m['stdR']:>7.2f}{m['stopmed']:>10.1f}{m['cagr_dd']:>11.2f}"
                      f"{m['book_cagr_dd']:>12.2f}{m['isoos']:>16}")

    # ------------------------------------------------------------------------- E3 ----------
    print("\n" + "=" * 118)
    print("E3  TP2-ratchet re-test AT RR4.5 (current adopted stop/target: swinglow / --tgt-ref stop)")
    print("=" * 118)
    E = build_entries_L(d15, gate_tf="240min", rr=4.5, bo_window=20)
    print(f"candidate armed breaks: {len(E)}")

    # T3: ratchet-off must reproduce the T1 leg tie-back exactly (per-trade, not just aggregate)
    tr_off = walk_ratchet(d15, E, pf=0.30, fill_win=200, fwd=500, tp2=None, tp1=None, cost=0.0)
    off_df = pd.DataFrame(tr_off, columns=["time", "e_px", "stop", "R"]).set_index("time")
    off_official = t_cur.set_index("time")[["e_px", "risk", "R"]].copy()
    off_official["stop"] = off_official["e_px"] - off_official["risk"]
    joined = off_df.join(off_official, lsuffix="_mine", rsuffix="_off", how="outer")
    mism = joined.dropna()
    mism["Rdiff"] = (mism["R_mine"] - mism["R_off"]).abs()
    n_bad = (mism["Rdiff"] > 1e-6).sum()
    only_mine = off_df.index.difference(off_official.index)
    only_off = off_official.index.difference(off_df.index)
    ok3 = (n_bad == 0 and len(only_mine) == 0 and len(only_off) == 0 and len(mism) == len(off_official))
    print(f"T3 ratchet-off tie-back: n(mine)={len(off_df)} n(official)={len(off_official)} "
          f"mismatched-R={n_bad}  {'OK' if ok3 else '*** MISMATCH ***'}")
    if not ok3 and not args.smoke:
        print("ABORTING: E3 walk does not reproduce the engine's own trade table."); return

    tp2s = [2.5, 3.0, 3.5, 4.0] if not args.smoke else [3.0]
    tp1s = [1.0, 1.5, 2.0, 2.5] if not args.smoke else [1.0, 1.5]
    combos = [(None, None, "ラチェット無し(現行)")]
    for tp2 in tp2s:
        for tp1 in tp1s:
            if tp1 < tp2:
                combos.append((tp2, tp1, f"TP2={tp2} TP1={tp1}"))

    print(f"\n{'combo':<24}{'n':>5}{'win%':>6}{'PF':>6}{'meanR':>8}{'totR/yr':>9}"
          f"{'legCAGR/DD':>11}{'bookCAGR/DD':>12}")
    e3_series = {}
    for tp2, tp1, label in combos:
        tr = walk_ratchet(d15, E, pf=0.30, fill_win=200, fwd=500, tp2=tp2, tp1=tp1, cost=COST_L)
        s = weighted_series(tr, d15, cost=COST_L)
        e3_series[label] = s
        if len(s) < 5:
            print(f"{label:<24}  no trades"); continue
        pos, neg = s[s > 0].sum(), abs(s[s <= 0].sum())
        pfv = pos / neg if neg > 0 else np.inf
        eq = np.cumprod(1 + 0.01 * s.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        days = (s.index[-1] - s.index[0]).days
        cagr = (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100
        cagr_dd = cagr / dd if dd > 0 else np.nan
        bc = book_with_L(s)
        print(f"{label:<24}{len(s):>5}{(s>0).mean()*100:>5.0f}%{pfv:>6.2f}{s.mean():>+8.3f}"
              f"{s.sum()/span:>9.2f}{cagr_dd:>11.2f}{bc[2]:>12.2f}")

    # block bootstrap: current (ratchet off) vs the best-by-book-CAGR/DD ratchet combo
    valid = [(lbl, s) for lbl, s in e3_series.items() if len(s) >= 5 and lbl != "ラチェット無し(現行)"]
    if valid:
        best_label, best_s = max(valid, key=lambda kv: book_with_L(kv[1])[2])
        print(f"\nE3 book CAGR/DD 最良: {best_label}")
        sA = book_series(dict(L_fixed, btc15m_L=e3_series["ラチェット無し(現行)"]), NEW)
        sB = book_series(dict(L_fixed, btc15m_L=best_s), NEW)
        ptA = cdd(sA.values, (sA.index[-1] - sA.index[0]).days)
        print(f"  series点推定 tie-back: 現行(series経由)={ptA[2]:.2f}  "
              f"(book()経由={book_with_L(e3_series['ラチェット無し(現行)'])[2]:.2f})")
        res = boot_pair(sA, sB, ndraw=(300 if args.smoke else 2000))
        print(f"  ブロック・ブートストラップ（6レッグ・ブック, 現行 vs {best_label}）")
        print(f"  {'block':<8}{'現行 median':>16}{'最良 median':>16}{'P(最良>現行)':>16}")
        for blk, (da, db, p) in res.items():
            print(f"  {str(blk)+'mo':<8}{da:>16.2f}{db:>16.2f}{p:>15.1f}%")

    # ------------------------------------------------------------------------- E4 ----------
    print("\n" + "=" * 118)
    print("E4  小口座インパクト（2025年以降のトレードの損切り幅$中央値 -> 0.01 BTCロットが強いる"
          f"リスク%、口座={ACCOUNT_JPY:,.0f}円、USDJPY={USDJPY}）")
    print("=" * 118)
    print(f"{'sl-b / pf':<28}{'tgt-ref':>9}{'n(2025+)':>10}{'stop$med':>10}{'risk%(口座)':>12}")
    for (tag, label, pf), row in e4_all.items():
        tgt_ref = "stop" if tag == "E1" else "l2"
        if row["n"] == 0:
            continue
        flag = "  <-- 3%超" if row["riskpct"] > 3.0 else ""
        print(f"{label + f' pf={pf:.2f}':<28}{tgt_ref:>9}{row['n']:>10}"
              f"{row['stopmed']:>10.1f}{row['riskpct']:>11.2f}%{flag}")


if __name__ == "__main__":
    main()
