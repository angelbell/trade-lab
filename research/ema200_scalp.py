"""ema200_scalp.py -- faithfully mechanize & FALSIFY the video's "200-EMA pullback + structure-break"
gold 5m scalp. Long: uptrend(EMA20>EMA200) -> price pulls back and TOUCHES EMA200 -> a descending
trendline OR double-bottom neckline breaks on a confirmed close -> enter next-bar open. SL = recent
swing low (structural). TP = N-calc (AB=CD measured move). Skip filters tested SEPARATELY as luck-sorters.

Prior = LOW EV: gold 5m intraday is on the dead list; the trendline break is +0.89 redundant w/ gold_bo;
the skip filters are luck-sorters (RR-band was backwards on TLB); 92%/13-trades/4-months = cherry-pick.
So: ALL-SIGNALS BASE first (no skip filters); if it has no edge net of 5m cost, the filters can't make one.

  .venv/bin/python research/ema200_scalp.py
"""
import os, sys, warnings
from bisect import bisect_right
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag
from research.real_trendline import active_resistance, detect

RNG = np.random.default_rng(7)
CSV = "data/vantage_xauusd_m5.csv"
ZZ, TOL, MT, M, W, BUF = 2.0, 0.5, 2, 6, 24, 0.05    # trendline params (from real_trendline) ; W=touch->trigger window
TOUCH_W = 36                                          # bars a 200EMA touch stays "armed" (~3h on M5)
DB_TOL = 0.5                                           # double-bottom equal-low tolerance (ATR)


def structure(seq_c, seq_p, seq_k, si):
    """last L0(low)->H1(high)->L2(low) confirmed by bar si -> (L0p,H1p,L2p) for measured move/stop."""
    k = bisect_right(seq_c, si)
    lows = [(seq_p[j], j) for j in range(k) if seq_k[j] == -1]
    highs = [(seq_p[j], j) for j in range(k) if seq_k[j] == +1]
    if len(lows) < 2 or not highs:
        return None
    L2p, L2j = lows[-1]
    hb = [(p, j) for (p, j) in highs if j < L2j]
    if not hb:
        return None
    H1p, H1j = hb[-1]
    l0 = [(p, j) for (p, j) in lows if j < H1j]
    if not l0:
        return None
    return l0[-1][0], H1p, L2p


def double_bottom_signals(seq_c, seq_p, seq_k, close, atr, n):
    """two near-equal confirmed lows + neckline (high between) broken on a confirmed close."""
    out = []
    lows = [(seq_c[j], seq_p[j]) for j in range(len(seq_c)) if seq_k[j] == -1]
    for t in range(1, len(seq_c)):
        if seq_k[t] != -1:
            continue
        # find prior low and the high between
        pj = next((u for u in range(t - 1, -1, -1) if seq_k[u] == -1), None)
        hj = next((u for u in range(t - 1, -1, -1) if seq_k[u] == +1), None)
        if pj is None or hj is None or not (pj < hj < t):
            continue
        if abs(seq_p[t] - seq_p[pj]) > DB_TOL * (atr[seq_c[t]] if seq_c[t] < n else atr[-1]):
            continue
        neck = seq_p[hj]; c0 = seq_c[t]
        for j in range(c0, min(c0 + W, n)):
            if close[j] > neck:
                out.append(j); break
    return out


def make_trades(d, entries, seq_c, seq_p, seq_k, cost, fwd=300):
    op = d["open"].values; H = d["high"].values; Lo = d["low"].values; C = d["close"].values
    tm = d.index; n = len(C); rows = []; busy = -1
    for si in entries:
        ei = si + 1
        if ei <= busy or ei >= n:
            continue
        st = structure(seq_c, seq_p, seq_k, si)
        if st is None:
            continue
        L0, H1, L2 = st
        e = op[ei]; stop = L2; tgt = L2 + (H1 - L0)          # structural stop + measured (AB=CD) TP
        risk = e - stop
        if risk <= 0 or tgt <= e:
            continue
        rr = (tgt - e) / risk; R = None; xj = None
        for j in range(ei, min(ei + fwd, n)):
            if Lo[j] <= stop: R, xj = -1.0, j; break
            if H[j] >= tgt:  R, xj = rr, j; break
        if R is None:
            xj = min(ei + fwd, n - 1); R = (C[xj] - e) / risk
        R -= cost / risk
        rows.append((tm[ei], R, rr)); busy = xj
    return pd.DataFrame(rows, columns=["time", "R", "rr"])


def stat(tag, t):
    if len(t) < 10:
        print(f"  {tag:<26} n={len(t)} (too few)"); return
    be = (1.0 / (1.0 + t.rr.median())) * 100
    w = t.R[t.R > 0].sum(); l = t.R[t.R < 0].sum(); pf = w / abs(l) if l else 9.9
    isr = t[t.time.dt.year < 2024].R; oos = t[t.time.dt.year >= 2024].R
    print(f"  {tag:<26} n={len(t):>4} win={(t.R>0).mean()*100:>3.0f}%(be~{be:.0f}) PF={pf:4.2f} "
          f"meanR={t.R.mean():+.3f} medRR={t.rr.median():.1f} | IS={isr.mean():+.3f} OOS={oos.mean():+.3f}")


def main():
    d = load_mt5_csv(CSV)
    c = d["close"].values; lo = d["low"].values; n = len(d)
    print(f"ema200_scalp -- gold M5 {d.index[0].date()}->{d.index[-1].date()} (n={n})  ALL-SIGNALS BASE first")
    ema20 = d["close"].ewm(span=20, adjust=False).mean().values
    ema200 = d["close"].ewm(span=200, adjust=False).mean().values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    up = ema20 > ema200
    print(f"  uptrend(20>200): {up.mean()*100:.0f}% of bars")

    # 200EMA touch arming
    touch = up & (lo <= ema200)
    last_touch = np.where(touch, np.arange(n), -1)
    last_touch = pd.Series(last_touch).replace(-1, np.nan).ffill().values
    armed = (np.arange(n) - np.nan_to_num(last_touch, nan=-1e9)) <= TOUCH_W

    sw = swings_zigzag(d["high"].values, lo, atr, ZZ)
    seq = sorted([(cc, pr, k) for (cc, pp, pr, k) in sw], key=lambda x: x[0])
    seq_c = [x[0] for x in seq]; seq_p = [x[1] for x in seq]; seq_k = [x[2] for x in seq]

    # trigger A: descending trendline break (reuse real_trendline), gated by uptrend + recent touch
    L, wid = active_resistance(d, atr, sw, TOL, MT, M)
    tl_all = detect(d, atr, L, wid, "break", W, BUF, TOL)
    tl = [i for i in tl_all if i < n and up[i] and armed[i]]
    # trigger B: double-bottom neckline break, gated likewise
    db_all = double_bottom_signals(seq_c, seq_p, seq_k, c, atr, n)
    db = [i for i in db_all if i < n and up[i] and armed[i]]
    either = sorted(set(tl) | set(db))
    print(f"  entries: trendline={len(tl)}  double-bottom={len(db)}  either={len(either)}")

    print("\n  == 1. ALL-SIGNALS BASE (no skip filters), full history, cost=0.30usd ==")
    for tag, ents in [("trendline-break", tl), ("double-bottom", db), ("either", either)]:
        stat(tag, make_trades(d, ents, seq_c, seq_p, seq_k, 0.30))

    print("\n  == 2. COST REALISM (either), gold RT x1/x2/x3 (5m spread is the killer) ==")
    for cm in (0.30, 0.60, 0.90):
        stat(f"either cost={cm}", make_trades(d, either, seq_c, seq_p, seq_k, cm))

    print("\n  == 3. THE 92% WINDOW: 2025-04..08 vs full (cherry-pick check) ==")
    t = make_trades(d, either, seq_c, seq_p, seq_k, 0.30)
    win = t[(t.time >= "2025-04-01") & (t.time < "2025-09-01")]
    stat("2025-04..08 window", win)
    stat("full history", t)

    print("\n  == 4. SKIP FILTERS vs LUCK-SORTER (either) ==")
    base = t
    # video filters: skip RR<1, RR>3, (target<=recent high & range-y proxied by rr extremes)
    filt = base[(base.rr >= 1.0) & (base.rr <= 3.0)]
    stat("base (all RR)", base)
    stat("filtered 1<=RR<=3", filt)
    if len(base) >= 20:
        keepn = len(filt)
        vals = []
        for _ in range(300):
            idx = np.sort(RNG.choice(len(base), min(keepn, len(base)), replace=False))
            vals.append(base.iloc[idx].R.mean())
        pct = (np.array(vals) < filt.R.mean()).mean() * 100 if len(filt) else np.nan
        print(f"    -> filtered meanR={filt.R.mean():+.3f} vs random same-drop pctile={pct:.0f} "
              f"(>=90 = real, else luck-sorter)")
    print("\n  verdict inputs: base edge? survives cost? window not lucky? filter beats random-drop?")


if __name__ == "__main__":
    main()
