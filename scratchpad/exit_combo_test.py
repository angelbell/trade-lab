"""COMBINATION test (user GO 2026-07-12): taker-exit (new, PASS) x TP2-ratchet (existing,
BTC-specific DD overlay) on btc15m_L canon. Do they add, interfere, or cancel?

WALKER   one bar loop, conservative order per bar: (1) stop (incl. ratcheted stop) intrabar,
         (2) final target intrabar, (3) taker decision-point check on the bar that first dips
         to <= lim+1R after a >= lim+2R touch -> exit NEXT bar open if flow is weak,
         (4) ratchet arm on the bar's high (effective from the next bar).
         Ratchet: once high >= lim + trig*u (u = lim - stop), stop := lim + floor*u.
         Canon adopted example from the exec sweep: trig 3.5 -> floor 2.0 (also 3.0->2.0 shown).
         Taker rule: frozen threshold, official taker_4h < 1.10 (proxy r_up variant also run).
CELLS    A base | B taker only | D ratchet only | B+D combined (both ratchet settings)
JUDGE    PF / totR/yr / maxDD(R) / totR/DD / meanR + IS/OOS, plus DD in equity-% terms
         (the ratchet's claim was a DD cut: 31.8% -> 22-26% at 1% risk).
Fixed trade set, no re-arm; cost $15 net. Run: .venv/bin/python scratchpad/exit_combo_test.py
"""
import sys, os
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from src.data_loader import load_mt5_csv
from swing_pivot_ab import build, FWD, START, ROOT

FRAC, RR, COST = 0.3, 4.0, 15.0
CACHE = os.path.join(ROOT, "data/ext_btc_oi_metrics.csv")
CACHE_WIN = os.path.join(ROOT, "scratchpad/cache_taker_proxy_windows.csv")
T0 = pd.Timestamp("2020-09-01", tz="UTC")


def walk(df, E, h, l, c, o, flow, th, ratchet):
    """flow: dict[timestamp]->value (None = taker exit off). ratchet: (trig, floor) or None."""
    busy = -1; out = []
    for (i, e, stop0, tgt, H1, minleg) in E:
        if i <= busy: continue
        lim = e - FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fill_j = None
        for j in range(i + 1, min(i + 1 + FWD, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill_j = j; break
        if fill_j is None: continue
        u = lim - stop0; reward = tgt - lim
        lv2, lv1 = lim + 2 * u, lim + 1 * u
        if l[fill_j] <= stop0:
            out.append((df.index[fill_j], -1.0 - COST / u)); busy = fill_j; continue
        cur_stop = stop0
        trig_px = lim + ratchet[0] * u if ratchet else None
        floor_px = lim + ratchet[1] * u if ratchet else None
        t2_hit = False; pending_exit = False
        R = None; exit_j = min(fill_j + FWD, len(c) - 1)
        for j in range(fill_j + 1, min(fill_j + 1 + FWD, len(c))):
            if pending_exit:                                   # taker exit executes here
                R = (o[j] - lim) / u; exit_j = j; break
            if l[j] <= cur_stop:                               # 1. stop (possibly ratcheted)
                R = (cur_stop - lim) / u; exit_j = j; break
            if h[j] >= tgt:                                    # 2. target
                R = reward / u; exit_j = j; break
            if flow is not None and t2_hit and l[j] <= lv1:    # 3. taker decision point
                v = flow.get(df.index[j], np.nan)
                if np.isfinite(v) and v < th and j + 1 < len(c):
                    pending_exit = True
                flow_done = True
                t2_hit = False                                 # decision fires once (first dip)
            if h[j] >= lv2: t2_hit = True
            if trig_px is not None and h[j] >= trig_px:        # 4. arm ratchet (next-bar effect)
                cur_stop = max(cur_stop, floor_px)
        if R is None: R = (c[exit_j] - lim) / u
        out.append((df.index[fill_j], R - COST / u)); busy = exit_j
    return out


def stats(tr, span, risk=0.01):
    t = [x[0] for x in tr]; R = np.array([x[1] for x in tr])
    cum = np.cumsum(R); ddR = (np.maximum.accumulate(cum) - cum).max()
    eq = np.cumprod(1 + risk * R); ddpct = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    yrs = np.array([x.year for x in t]); uy = np.unique(yrs); half = uy[len(uy) // 2]
    return dict(n=len(R), pf=R[R > 0].sum() / abs(R[R <= 0].sum()), totyr=R.sum() / span,
                ddR=ddR, ddpct=ddpct, retdd=R.sum() / ddR, meanR=R.mean(),
                IS=R[yrs < half].sum(), OOS=R[yrs >= half].sum())


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    span = (df.index[-1] - df.index[0]).days / 365.25
    o = df["open"].values
    E, h, l, c, _ = build(df, RR, "zigzag", 2.0)

    m = pd.read_csv(CACHE, index_col=0); m.index = pd.to_datetime(m.index, utc=True)
    m = m.sort_index(); m = m[~m.index.duplicated(keep="last")]; m = m[m["sum_open_interest"] > 0]
    tk = m["sum_taker_long_short_vol_ratio"]
    tk4 = tk.rolling("4h").mean()                       # official, causal
    flow_off = {t: (tk4.asof(t) if t >= T0 else np.nan) for t in df.index}

    cw = pd.read_csv(CACHE_WIN, index_col=0, parse_dates=[0])
    flow_prx = {pd.Timestamp(i): r for i, r in cw["r_up"].items()}   # only at decision points

    print(f"{'cell':<30}{'n':>5}{'PF':>6}{'totR/yr':>9}{'maxDD_R':>9}{'maxDD%':>8}{'tot/DD':>8}{'meanR':>8}{'IS/OOS':>14}")
    def row(name, tr):
        s = stats(tr, span)
        print(f"{name:<30}{s['n']:>5}{s['pf']:>6.2f}{s['totyr']:>9.2f}{s['ddR']:>9.1f}{s['ddpct']:>7.1f}%"
              f"{s['retdd']:>8.2f}{s['meanR']:>+8.3f}{f'{s[chr(73)+chr(83)]:+.0f}/{s[chr(79)+chr(79)+chr(83)]:+.0f}R':>14}")
        return s

    A = row("A base (RR4 only)", walk(df, E, h, l, c, o, None, None, None))
    B = row("B taker exit (official<1.10)", walk(df, E, h, l, c, o, flow_off, 1.10, None))
    Bp = row("B' taker exit (proxy r_up)", walk(df, E, h, l, c, o, flow_prx, 0.970, None))
    for trig, fl in ((3.5, 2.0), (3.0, 2.0)):
        row(f"D ratchet {trig}->{fl}", walk(df, E, h, l, c, o, None, None, (trig, fl)))
    for trig, fl in ((3.5, 2.0), (3.0, 2.0)):
        row(f"B+D taker + ratchet {trig}->{fl}", walk(df, E, h, l, c, o, flow_off, 1.10, (trig, fl)))
    for trig, fl in ((3.5, 2.0),):
        row(f"B'+D proxy + ratchet {trig}->{fl}", walk(df, E, h, l, c, o, flow_prx, 0.970, (trig, fl)))

if __name__ == "__main__":
    main()
