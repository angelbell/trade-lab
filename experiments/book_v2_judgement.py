"""BOOK JUDGEMENT: does btc15m_L v2 improve the 6-leg book, or only the leg?

A leg is judged at the book, not on its own: v2 halves the leg's drawdown by down-sizing weak
setups, but if those are exactly the trades that were decorrelated from the other legs, the book
can get WORSE while the leg gets better. That is the whole point of this test.

  book = gold_bo + btc_bo_kama + btc_pull + gold15m + btc15m_L + btc15m_S   (as in book_integration)
  v1 leg = the book's current btc15m_L: 4h-KAMA gate, RR4 (BASE), PDH soft 0.5
  v2 leg = 4h-KAMA gate, RR4.5, no ratchet, size ladder (4H-swing-high AND PDH = 1.0 / one = 0.5 /
           neither = 0.25)
Everything else (the other 5 legs, inv-vol weighting at constant total risk, the monthly joint
bootstrap that preserves cross-leg correlation) is reused from book_integration.py unchanged.
Run: .venv/bin/python experiments/book_v2_judgement.py
"""
import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from research.portfolio_kama import get_legs
from radar_gate_race import BASE
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, swings_zigzag
from short_mirror_15m import invert
import pine_replica_btc15m as P, btc15m_gate_ab as G
from btc15m_gate_ab import kama_rising
from trend_leg_aging import atr as atr_fn

ROOT = "/home/angelbell/dev/auto-trade"


def btc15m_v2(d15):
    """the proposed leg, built with the replica machinery (4h gate, RR4.5, HH4H+PDH size ladder)."""
    G.RR = 4.5; P.RR = 4.5
    h, l, c = (d15[k].values for k in ("high", "low", "close"))
    pdh = d15["high"].resample("1D").max().shift(1).reindex(d15.index, method="ffill").values
    h4 = d15.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = atr_fn(h4["high"].values, h4["low"].values, h4["close"].values)
    sw = swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0)
    s = pd.Series(np.nan, index=h4.index)
    for (ci, pi, px, kind) in sw:
        if kind == +1:
            s.iloc[ci] = px
    hh = s.ffill().shift(1).reindex(d15.index, method="ffill").values      # no lookahead
    E = G.build_entries(d15, kama_rising(d15, "4h"))
    busy = -1; out = []
    for (i, e, stop0, tgt, _w) in E:
        if i <= busy: continue
        s_hh = np.isfinite(hh[i]) and c[i] > hh[i]
        s_pd = np.isfinite(pdh[i]) and c[i] > pdh[i]
        w = 1.0 if (s_hh and s_pd) else (0.5 if (s_hh or s_pd) else 0.25)
        lim = e - G.FRAC * (e - stop0)
        if lim <= stop0 or lim >= e: continue
        fill = None
        for j in range(i + 1, min(i + 1 + P.FILLWIN, len(c))):
            if h[j] >= tgt: break
            if l[j] <= lim: fill = j; break
        if fill is None: continue
        u = lim - stop0; rew = (tgt - lim) / u
        if l[fill] <= stop0:
            out.append((d15.index[fill], w * (-1.0 - G.COST / u))); busy = fill; continue
        R = None; ej = min(fill + P.FWD, len(c) - 1)
        for j in range(fill + 1, min(fill + 1 + P.FWD, len(c))):
            if l[j] <= stop0: R = -1.0; ej = j; break
            if h[j] >= tgt: R = rew; ej = j; break
        if R is None: R = (c[ej] - lim) / u
        out.append((d15.index[ej], w * (R - G.COST / u))); busy = ej
    return pd.Series([x[1] for x in out], index=pd.DatetimeIndex([x[0] for x in out]))


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = {k: pd.Series(t.R.values, index=pd.DatetimeIndex(t.time))
                for k, t in get_legs().items()}
        g = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                      "ext_cap": 8.0, "pullback_frac": 0.25}))
        legs["gold15m"] = pd.Series(t["R"].values - 0.3 / t["risk"].values,
                                    index=pd.DatetimeIndex(t["time"]))
        b = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":]
        d15 = resample(b, "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3}))
        Rn = t["R"].values - 15.0 / t["risk"].values
        pdh = d15["high"].resample("1D").max().dropna().shift(1)\
            .reindex(d15.index, method="ffill").values
        ab = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
        v1 = pd.Series(Rn * np.where(ab, 1.0, 0.5), index=pd.DatetimeIndex(t["time"]))
        inv = invert(d15); C = 2 * d15["high"].max()
        ts_ = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3}))
        Rs = ts_["R"].values - 15.0 / ts_["risk"].values
        pdl = d15["low"].resample("1D").min().dropna().shift(1)\
            .reindex(d15.index, method="ffill").values
        mS = (C - ts_["e_px"].values) < pdl[d15.index.get_indexer(ts_["time"])]
        legs["btc15m_S"] = pd.Series(Rs[mS], index=pd.DatetimeIndex(ts_["time"])[mS])
        v2 = btc15m_v2(d15)

    print(f"btc15m_L  v1: n={len(v1)} totR={v1.sum():+.0f}   v2: n={len(v2)} totR={v2.sum():+.0f}")
    OLD = ["gold_bo", "btc_bo_kama", "btc_pull"]
    NEW = OLD + ["gold15m", "btc15m_L", "btc15m_S"]
    rng = np.random.default_rng(7)

    for tag, leg in (("v1 (current book leg)", v1), ("v2 (proposed)", v2)):
        L = dict(legs); L["btc15m_L"] = leg
        mon = {k: s.groupby(s.index.to_period("M")).sum() for k, s in L.items()}
        start = max(s.index.min() for s in mon.values())
        end = min(s.index.max() for s in mon.values())
        midx = pd.period_range(start, end, freq="M")
        M = pd.DataFrame({k: v.reindex(midx, fill_value=0.0) for k, v in mon.items()})
        sig = M.std()

        def weights(names, total, parity=False):
            w = (1.0 / sig[names]); w = w / w.sum() * total
            if parity:
                btc = [n for n in names if "btc" in n]; gld = [n for n in names if "gold" in n]
                sb, sg = w[btc].sum(), w[gld].sum()
                if sb > sg:
                    w[btc] *= sg / sb; w = w / w.sum() * total
            return w

        def ladder(w, t2):
            port = (M[w.index] * w).sum(axis=1).values
            mult = np.array([np.prod(1 + port[rng.integers(0, len(port), 12)]) for _ in range(4000)])
            eq = np.cumprod(1 + port)
            dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
            yrs = len(port) / 12
            cagr = (eq[-1] ** (1 / yrs) - 1) * 100
            print(f"    {t2:<30} CAGR={cagr:5.1f}% maxDD={dd:4.1f}% CAGR/DD={cagr/dd:5.2f} | "
                  f"1yr mult med={np.median(mult):.2f} p10={np.percentile(mult,10):.2f} "
                  f"p90={np.percentile(mult,90):.2f} P(halve)={(mult<=0.5).mean()*100:.1f}%")

        print(f"\n=== BOOK with btc15m_L = {tag}   ({len(midx)} common months) ===")
        print(f"  monthly corr of btc15m_L vs the others: "
              + "  ".join(f"{k} {M['btc15m_L'].corr(M[k]):+.2f}" for k in NEW if k != "btc15m_L"))
        for total in (0.02, 0.03):
            print(f"  -- total risk {total*100:.0f}% --")
            ladder(weights(OLD, total), "old book (3 legs)")
            ladder(weights(NEW, total), "new book (6 legs)")
            ladder(weights(NEW, total, parity=True), "new 6 legs + BTC<=gold parity")
        print("  inv-vol weights (3%): "
              + "  ".join(f"{k} {v*100:.2f}" for k, v in weights(NEW, 0.03).items()))


if __name__ == "__main__":
    main()
