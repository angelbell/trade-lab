"""jpy_sweep_reclaim.py -- B1 STEP1-2: USDJPY 15m failed-break (sweep & reclaim) FADE screen.

Spec (docs/proposals.md B1, pre-registered): level pierced intrabar, then within k<=4 bars a
CLOSE back inside = reclaim event -> fade toward the range. Levels: (a) prior-day high/low,
(b) last confirmed ZigZag pivot (zz-k2). One event per outside-episode; a sustained break
(no inside-close within k bars) = breakout succeeded, no event, episode consumed.
STEP1: barrier race +-1 ATR from the reclaim close (favorable side first = win; tie = loss)
vs SAME-HOUR random baseline of the same directional race. STEP2 strata: k, sweep depth
(max pierce distance / ATR), side, level type; per-year of the headline. Cost enters STEP3+.
PASS bar: delta >= +5pt vs beta. Predicted kill: reject-bar carries no info (exhaustion grave).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag

K_RECLAIM = 4
K_RACE = 96          # 1 day of 15m bars
BAR = 1.0            # ATR barrier


def main():
    d = load_mt5_csv("data/vantage_usdjpy_m15.csv")
    cnt = d.groupby(d.index.date).size()
    ok = cnt[cnt.rolling(30).median() >= 80]
    d = d[d.index.date >= ok.index[0]]
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = ta.atr(d["high"], d["low"], d["close"], 14).shift(1).values
    n = len(c)
    span = (d.index[-1] - d.index[0]).days / 365.25
    print(f"span {d.index[0].date()} -> {d.index[-1].date()} ({span:.1f}yr, {n:,} bars)")

    # ---------- barrier race from every bar, both directions ----------
    up_l, dn_l = c + BAR * atr, c - BAR * atr
    t_up = np.full(n, K_RACE + 1, np.int32)
    t_dn = np.full(n, K_RACE + 1, np.int32)
    for k in range(1, K_RACE + 1):
        hs, ls = np.empty(n), np.empty(n)
        hs[:n - k], ls[:n - k] = h[k:], l[k:]
        hs[n - k:], ls[n - k:] = -np.inf, np.inf
        t_up = np.where((t_up > K_RACE) & (hs >= up_l), k, t_up)
        t_dn = np.where((t_dn > K_RACE) & (ls <= dn_l), k, t_dn)
    win_long = (np.minimum(t_up, t_dn) <= K_RACE) & (t_up < t_dn)   # tie -> loss
    win_short = (np.minimum(t_up, t_dn) <= K_RACE) & (t_dn < t_up)
    valid = ~np.isnan(atr) & (np.arange(n) < n - K_RACE)
    hours = d.index.hour.values
    beta_L = {hh: win_long[valid & (hours == hh)].mean() for hh in range(24)}
    beta_S = {hh: win_short[valid & (hours == hh)].mean() for hh in range(24)}

    # ---------- level series ----------
    dh = d["high"].resample("1D").max().dropna()
    dl = d["low"].resample("1D").min().dropna()
    pdh = dh.shift(1).reindex(d.index, method="ffill").values
    pdl = dl.shift(1).reindex(d.index, method="ffill").values
    sw = swings_zigzag(h, l, atr, 2.0)
    zzh = np.full(n, np.nan)
    zzl = np.full(n, np.nan)
    cur_h = cur_l = np.nan
    si = 0
    for i in range(n):
        while si < len(sw) and sw[si][0] <= i:
            if sw[si][3] == 1: cur_h = sw[si][2]
            else: cur_l = sw[si][2]
            si += 1
        zzh[i] = cur_h
        zzl[i] = cur_l

    # ---------- sweep & reclaim events ----------
    def events(level, side):
        """side='short': pierce above level then close back below. Returns
        list of (event_bar, k_used, depth_atr)."""
        ev = []
        i = 1
        while i < n - K_RACE:
            L = level[i]
            if np.isnan(L) or np.isnan(atr[i]) or atr[i] <= 0:
                i += 1; continue
            pierced = h[i] > L if side == "short" else l[i] < L
            prev_in = c[i - 1] <= L if side == "short" else c[i - 1] >= L
            if pierced and prev_in:
                depth = 0.0
                done = False
                for k in range(0, K_RECLAIM + 1):
                    j = i + k
                    if j >= n - 1: break
                    if side == "short":
                        depth = max(depth, (h[j] - L) / atr[i])
                        if c[j] < L:
                            ev.append((j, k, depth)); done = True; break
                        if c[j] < L and k == 0: pass
                    else:
                        depth = max(depth, (L - l[j]) / atr[i])
                        if c[j] > L:
                            ev.append((j, k, depth)); done = True; break
                # consume the episode either way: skip past the window
                i = i + (k if done else K_RECLAIM) + 1
            else:
                i += 1
        return ev

    yr = d.index.year.values

    def report(tag, ev, side):
        if len(ev) < 30:
            print(f"  {tag:<30} n={len(ev)} too few"); return
        idx = np.array([e[0] for e in ev])
        ks = np.array([e[1] for e in ev])
        dep = np.array([e[2] for e in ev])
        w = win_short[idx] if side == "short" else win_long[idx]
        beta = np.mean([(beta_S if side == "short" else beta_L)[hh] for hh in hours[idx]])
        print(f"  {tag:<30} n={len(idx):5d} N/yr={len(idx)/span:5.0f}  win={w.mean()*100:4.1f}%"
              f"  beta={beta*100:4.1f}%  diff={w.mean()*100-beta*100:+4.1f}pt")
        return idx, ks, dep, w

    print(f"\n===== STEP1: reclaim event, {BAR}ATR race {K_RACE} bars, k<={K_RECLAIM} =====")
    ev_pdh = events(pdh, "short")
    ev_pdl = events(pdl, "long")
    ev_zzh = events(zzh, "short")
    ev_zzl = events(zzl, "long")
    r1 = report("PDH sweep -> short", ev_pdh, "short")
    r2 = report("PDL sweep -> long", ev_pdl, "long")
    r3 = report("ZZ swing-high -> short", ev_zzh, "short")
    r4 = report("ZZ swing-low -> long", ev_zzl, "long")

    print("\n===== STEP2 strata (PDH+PDL pooled) =====")
    for tag, evs, side in [("PDH short", ev_pdh, "short"), ("PDL long", ev_pdl, "long")]:
        idx = np.array([e[0] for e in evs]); ks = np.array([e[1] for e in evs])
        dep = np.array([e[2] for e in evs])
        w = win_short[idx] if side == "short" else win_long[idx]
        bmap = beta_S if side == "short" else beta_L
        for lab, m in [("k=0 (同足ヒゲ)", ks == 0), ("k=1-2", (ks >= 1) & (ks <= 2)),
                       ("k=3-4", ks >= 3), ("depth>=0.3ATR", dep >= 0.3), ("depth<0.3", dep < 0.3)]:
            if m.sum() < 30: continue
            b = np.mean([bmap[hh] for hh in hours[idx[m]]])
            print(f"  {tag} {lab:<16} n={m.sum():5d}  win={w[m].mean()*100:4.1f}%  "
                  f"beta={b*100:4.1f}%  diff={w[m].mean()*100-b*100:+4.1f}pt")
    print("\n  per-year diff (PDH short / PDL long):")
    for tag, evs, side in [("PDHs", ev_pdh, "short"), ("PDLl", ev_pdl, "long")]:
        idx = np.array([e[0] for e in evs])
        w = win_short[idx] if side == "short" else win_long[idx]
        bmap = beta_S if side == "short" else beta_L
        line = []
        for y in np.unique(yr[idx]):
            m = yr[idx] == y
            if m.sum() < 20: continue
            b = np.mean([bmap[hh] for hh in hours[idx[m]]])
            line.append(f"{y}:{(w[m].mean()-b)*100:+.0f}(n={m.sum()})")
        print(f"   {tag}: " + "  ".join(line))


if __name__ == "__main__":
    main()
