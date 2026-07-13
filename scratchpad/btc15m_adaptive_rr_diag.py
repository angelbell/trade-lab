"""Can the TARGET be conditioned on entry-time trend strength?  (user: "strong trend -> let it run
with a bigger R; weak trend -> take profit earlier. I do this discretionarily.")

Why this is NOT one of today's corpses:
  * a GATE vetoes trades (weekly ER / cycle position -> dead)
  * an in-hold EXIT cuts a running trade (ratchet / floor / scale-out -> dead, 4 ways)
  * THIS keeps every trade and only moves the TARGET. Untested.
Prior that says it might live: PDH (break close above the prior-day high) already separates this
leg's trades violently (meanR +0.954 vs +0.181, random-drop null 100th %ile) and is currently used
only for SIZE. If "strength" shifts the MFE distribution, a conditional RR is the natural lever.

BASE RATE FIRST (the lab's rule): do NOT sweep targets yet. Run the leg with NO TARGET (stop +
time cap only) and record each trade's MFE (peak unrealised R). Then split by 5 candidate
definitions of "strong", all computed on CLOSED bars at the break:
  S1 PDH        break close > prior-day high            (the leg's own known separator)
  S2 D20        break close > 20-day high               (new-high territory, slower)
  S3 HH4H       break close > the last completed 4H swing high (the user's "4H structure updating")
  S4 ER72       Kaufman efficiency ratio over the last 72h, above its median
  S5 KAMAslope  |daily KAMA slope| / ATR, above its median
If the MFE distributions of strong vs weak are the SAME, an adaptive RR cannot work and we stop
here. If strong reaches far targets materially more often, the optimal RR differs by group.
Run: .venv/bin/python scratchpad/btc15m_adaptive_rr_diag.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data_loader import load_mt5_csv
import pine_replica_btc15m as P, btc15m_gate_ab as G
from btc15m_gate_ab import kama_rising
from breakout_wave import swings_zigzag, kama_adaptive
from trend_leg_aging import atr as atr_fn

ROOT, START = "/home/angelbell/dev/auto-trade", "2018-10-01"


def er(s, n):
    mom = (s - s.shift(n)).abs()
    vol = s.diff().abs().rolling(n).sum()
    return (mom / vol).replace([np.inf, -np.inf], np.nan)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    h, l, c = (df[k].values for k in ("high", "low", "close"))

    # ---- entry-time strength features (all past-only) ------------------------
    pdh = df["high"].resample("1D").max().shift(1).reindex(df.index, method="ffill").values
    d20 = df["high"].resample("1D").max().rolling(20).max().shift(1)\
        .reindex(df.index, method="ffill").values
    h4 = df.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)
    sw4 = swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0)
    # last CONFIRMED 4H swing high, known only at its confirm bar -> no lookahead
    hh = pd.Series(np.nan, index=h4.index)
    for (ci, pi, px, kind) in sw4:
        if kind == +1:
            hh.iloc[ci] = px
    hh4 = hh.ffill().reindex(df.index, method="ffill").values
    er72 = er(df["close"], 288).values                       # 72h = 288 x 15m bars
    dcl = df["close"].resample("1D").last()
    km = kama_adaptive(dcl, 14)
    slope = ((km - km.shift(1)) / dcl.rolling(14).std()).shift(1)\
        .reindex(df.index, method="ffill").values

    # ---- run the leg with NO TARGET: pure MFE distribution -------------------
    G.RR = 99.0; P.RR = 99.0                                  # target effectively unreachable
    E = G.build_entries(df, kama_rising(df, "4h"))
    busy = -1
    rows = []
    for (i, e, stop0, tgt, w) in E:
        if i <= busy: continue
        lim = e - G.FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fill = None
        for j in range(i + 1, min(i + 1 + P.FILLWIN, len(c))):
            if l[j] <= lim: fill = j; break
        if fill is None: continue
        u = lim - stop0
        mfe = 0.0; ej = min(fill + P.FWD, len(c) - 1)
        if l[fill] <= stop0:
            rows.append((i, fill, 0.0, w)); busy = fill; continue
        for j in range(fill + 1, min(fill + 1 + P.FWD, len(c))):
            mfe = max(mfe, (h[j] - lim) / u)
            if l[j] <= stop0: ej = j; break
        rows.append((i, fill, mfe, w)); busy = ej
    d = pd.DataFrame(rows, columns=["ebar", "fill", "mfe", "w"])
    print(f"trade set (no target, stop + {P.FWD}-bar cap): n={len(d)}   "
          f"MFE median {d.mfe.median():.2f}R  mean {d.mfe.mean():.2f}R\n")

    feats = {
        "S1 PDH (above prior-day high)": c[d.ebar.values] > pdh[d.ebar.values],
        "S2 D20 (above 20-day high)": c[d.ebar.values] > d20[d.ebar.values],
        "S3 HH4H (above last 4H swing high)": c[d.ebar.values] > hh4[d.ebar.values],
        "S4 ER72 (efficiency > median)": er72[d.ebar.values] > np.nanmedian(er72[d.ebar.values]),
        "S5 KAMA slope > median": slope[d.ebar.values] > np.nanmedian(slope[d.ebar.values]),
    }
    ladder = [3, 4, 5, 6, 7, 8]
    print("MFE distribution by 'strength'.  P(MFE >= xR) = the chance a target at xR would be hit.")
    print(f"{'feature':<36}{'grp':<8}{'n':>5}{'med':>7}{'mean':>7}"
          + "".join(f"{'P>=' + str(x) + 'R':>8}" for x in ladder))
    for nm, m in feats.items():
        m = np.asarray(m, dtype=bool)
        for tag, sub in (("STRONG", d[m]), ("weak", d[~m])):
            if len(sub) < 20: continue
            print(f"{nm if tag=='STRONG' else '':<36}{tag:<8}{len(sub):>5}"
                  f"{sub.mfe.median():>7.2f}{sub.mfe.mean():>7.2f}"
                  + "".join(f"{100*(sub.mfe >= x).mean():>7.0f}%" for x in ladder))
        print()

    # ---- what fixed RR would each group choose on its own? -------------------
    print("optimal target PER GROUP (expected R of a fixed target at xR, cost ignored here;")
    print("R at target = (0.3 + RR)/0.7 in u units -- so 'RR' here is the leg's RR input):")
    print(f"{'feature':<36}{'grp':<8}" + "".join(f"{'RR' + str(x):>8}" for x in [3, 3.5, 4, 4.5, 5, 5.5, 6]))
    for nm, m in feats.items():
        m = np.asarray(m, dtype=bool)
        for tag, sub in (("STRONG", d[m]), ("weak", d[~m])):
            if len(sub) < 20: continue
            row = []
            for rr in [3, 3.5, 4, 4.5, 5, 5.5, 6]:
                tgtR = (0.3 + rr) / 0.7                       # target in u units
                hit = sub.mfe >= tgtR
                ev = (hit * tgtR + (~hit) * (-1.0)).mean()    # crude: miss -> assume stop
                row.append(ev)
            best = int(np.argmax(row))
            print(f"{nm if tag=='STRONG' else '':<36}{tag:<8}"
                  + "".join(f"{v:>8.2f}" for v in row)
                  + f"   best RR{[3,3.5,4,4.5,5,5.5,6][best]}")
        print()


if __name__ == "__main__":
    main()
