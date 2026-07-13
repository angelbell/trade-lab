"""Does Binance taker flow predict ANYTHING, at the horizon it actually lives on?

Why: the frozen in-hold screen (inhold_flow_screen.py) found no cell that beat its max-statistic
null. Two explanations survive: (a) flow carries no information, or (b) it does, but we asked it
the wrong question -- we used a 4h window to predict whether a trade that resolves over DAYS
finishes positive. This test separates them, unconditionally (no trades involved):

    for each window W: does the trailing taker imbalance over W predict the NEXT W of return?

  imbalance = (taker_buy - taker_sell) / (taker_buy + taker_sell) over (t-W, t]   [past-only]
  target    = close[t+W] / close[t] - 1                                            [next W]
  sampling  = NON-OVERLAPPING (step W bars) so n is honest.

Also computed, and load-bearing:
  * LAG SCAN (the tz guard the retracted work skipped): corr between the imbalance over W and the
    CONTEMPORANEOUS return over the same W, at lags -8..+8 bars. Aggressive buying moves price up,
    so this must peak sharply at lag 0. If the peak sits off zero, the clocks are misaligned and
    every other number here is garbage.
  * ECONOMICS: quintile of imbalance -> forward return in basis points, vs the $15 round-trip cost
    expressed in bp on the same bars. A predictive IC that cannot clear the spread is not a lever.
  * CONTROLS: up/dn volume ratio (a price feature in disguise) and dOI, same treatment.

Run: .venv/bin/python scratchpad/flow_horizon_test.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.data_loader import load_mt5_csv
from inhold_flow_screen import load_flow

ROOT = "/home/angelbell/dev/auto-trade"
START = "2018-10-01"
WINDOWS = {"30m": 2, "1h": 4, "2h": 8, "4h": 16, "12h": 48, "24h": 96}
COST = 15.0


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[START:]
    # FEED GLITCH: 12 bars on 2020-08-10 01:00-03:45 are priced ~296-301 while BTC was ~11,700
    # (the loader's guard drops only the bridging bar). Harmless to btc15m_L (verified: identical
    # trades) but it destroys any Pearson correlation or bp-level statistic on this file.
    med = df["close"].rolling(96, center=True, min_periods=20).median()
    bad = ((df["close"] / med - 1).abs() > 0.4) & (df.index >= "2020-08-01") & (df.index < "2020-09-01")
    if bad.any():
        print(f"[data guard] dropping {int(bad.sum())} feed-glitch bars on {df.index[bad][0].date()}")
        df = df[~bad]
    idx, close = df.index, df["close"]

    spot = load_flow(os.path.join(ROOT, "data/ext_btc_5m_flow_spot.csv"))
    perp = load_flow(os.path.join(ROOT, "data/ext_btc_5m_flow.csv"))
    S = spot[["taker_buy", "taker_sell", "up_vol", "dn_vol"]].resample("15min").sum().reindex(idx)
    F = perp[["taker_buy", "taker_sell"]].resample("15min").sum().reindex(idx)
    oi = pd.read_csv(os.path.join(ROOT, "data/ext_btc_oi_metrics.csv"), index_col=0)
    oi.index = pd.to_datetime(oi.index, utc=True, format="mixed")
    oi = oi[~oi.index.duplicated(keep="first")].sort_index()
    oi = oi[oi["sum_open_interest"] > 0]
    oi.index = oi.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    O = oi["sum_open_interest"].resample("15min").last().reindex(idx)

    cost_bp = (COST / close * 1e4)
    print(f"span {idx[0].date()} -> {idx[-1].date()}  ({len(idx)} 15m bars)")
    print(f"round-trip cost ${COST:.0f} = {cost_bp.median():.1f} bp at the median price "
          f"(range {cost_bp.quantile(0.9):.1f} bp early / {cost_bp.quantile(0.1):.1f} bp late)\n")

    # ---------- LAG SCAN: the tz guard --------------------------------------
    w = 16                                                     # 4h
    imb4 = ((S["taker_buy"] - S["taker_sell"]).rolling(w).sum()
            / (S["taker_buy"] + S["taker_sell"]).rolling(w).sum())
    ret4 = close / close.shift(w) - 1.0
    print("LAG SCAN (4h taker imbalance vs CONTEMPORANEOUS 4h return; must peak at lag 0):")
    line = []
    for lag in range(-8, 9, 2):
        c = imb4.shift(lag).corr(ret4)
        line.append((lag, c))
    peak = max(line, key=lambda x: abs(x[1]))
    print("  " + "  ".join(f"{l:+d}:{c:+.2f}" for l, c in line))
    print(f"  peak at lag {peak[0]:+d} (corr {peak[1]:+.2f}) -> "
          f"{'clocks ALIGNED' if peak[0] == 0 else '*** MISALIGNED -- STOP ***'}\n")

    # ---------- the horizon test --------------------------------------------
    feats = {}
    for wn, wb in WINDOWS.items():
        num_s = (S["taker_buy"] - S["taker_sell"]).rolling(wb).sum()
        den_s = (S["taker_buy"] + S["taker_sell"]).rolling(wb).sum()
        num_f = (F["taker_buy"] - F["taker_sell"]).rolling(wb).sum()
        den_f = (F["taker_buy"] + F["taker_sell"]).rolling(wb).sum()
        feats[("taker imb SPOT", wn)] = (num_s / den_s)
        feats[("taker imb PERP", wn)] = (num_f / den_f)
        uv = S["up_vol"].rolling(wb).sum(); dv = S["dn_vol"].rolling(wb).sum()
        feats[("up/dn vol (control)", wn)] = ((uv - dv) / (uv + dv))
        feats[("dOI (control)", wn)] = (O / O.shift(wb) - 1.0).replace([np.inf, -np.inf], np.nan)

    names = ["taker imb SPOT", "taker imb PERP", "up/dn vol (control)", "dOI (control)"]
    print("FORWARD IC (Spearman, non-overlapping samples): does the trailing window predict the NEXT window?")
    print(f"{'variable':<22}" + "".join(f"{w:>12}" for w in WINDOWS))
    ics = {}
    for v in names:
        row = []
        for wn, wb in WINDOWS.items():
            x = feats[(v, wn)]
            y = close.shift(-wb) / close - 1.0
            k = np.arange(0, len(idx), wb)                     # non-overlapping
            xx, yy = x.values[k], y.values[k]
            m = np.isfinite(xx) & np.isfinite(yy)
            ic, p = spearmanr(xx[m], yy[m])
            ics[(v, wn)] = (ic, p, m.sum())
            row.append(f"{ic:+.3f}{'*' if p < 0.01 else ' '}({m.sum()})")
        print(f"{v:<22}" + "".join(f"{s:>12}" for s in row))
    print("  * = p<0.01.  IC is the rank correlation with the NEXT window's return.")

    # ---------- economics of the best forward cell ---------------------------
    best = max(ics, key=lambda k: abs(ics[k][0]) if "control" not in k[0] else -1)
    v, wn = best
    wb = WINDOWS[wn]
    x = feats[best]
    y = close.shift(-wb) / close - 1.0
    k = np.arange(0, len(idx), wb)
    d = pd.DataFrame({"x": x.values[k], "y": y.values[k] * 1e4,
                      "t": idx[k], "cost": cost_bp.values[k]}).dropna()
    d["q"] = pd.qcut(d["x"], 5, labels=False)
    print(f"\nECONOMICS of the strongest TRUE-flow cell: {v} / {wn}  "
          f"(IC {ics[best][0]:+.3f}, p={ics[best][1]:.1e}, n={ics[best][2]})")
    print(f"  {'quintile':<10}{'mean bp':>10}{'median bp':>11}{'sd bp':>9}{'n':>7}")
    for q in range(5):
        g = d[d["q"] == q]["y"]
        print(f"  Q{q+1:<9}{g.mean():>+10.1f}{g.median():>+11.1f}{g.std():>9.0f}{len(g):>7}")
    spread = d[d["q"] == 4]["y"].mean() - d[d["q"] == 0]["y"].mean()
    print(f"  Q5-Q1 spread {spread:+.1f} bp   vs round-trip cost {d['cost'].median():.1f} bp"
          f"  -> {'clears cost' if spread > d['cost'].median() else 'DOES NOT clear cost'}")

    # per-year stability of the IC
    d["yr"] = d["t"].dt.year
    print(f"\n  per-year IC ({v} / {wn}):")
    for yr, g in d.groupby("yr"):
        ic, p = spearmanr(g["x"], g["y"])
        print(f"    {yr}  IC {ic:+.3f}  (n={len(g)})", end="")
        print("   <- negative" if ic < 0 else "")


if __name__ == "__main__":
    main()
