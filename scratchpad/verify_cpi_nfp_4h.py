"""Does the 4-hour post-release hold transfer from FOMC to CPI and NFP?

Pre-registered before running, because the prior is that it does NOT: the FOMC effect looks
like a policy-direction dollar drift that unfolds over hours (statement -> presser -> repricing
of the path). CPI and NFP are statistics, not policy decisions -- no direction is announced and
the number is arguably priced in minutes. The ledger already records CPI as 2021-22 inflation-
era beta and NFP as below null, though both were tested as a 1-minute momentum follow, not as
this rule.

PASS requires all four:
  1. median > 0
  2. >= 95th percentile against the same-clock non-event control
  3. per-year median positive in a majority of years
  4. survives dropping 2021-2022 (the inflation era)

Same machine as the FOMC test: always long at release +1 min, exit 4h later, cost $0.30/oz.
Release is 08:30 ET = JST 21:30/22:30, so unlike FOMC this would fall in waking hours.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.fomc_event_study import price_before

COST = 0.30
B = 10000
rng = np.random.default_rng(42)
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END, PX = df.index.max(), float(df["close"].iloc[-1])


def hold(ts, h, stop=None):
    t_in = ts + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=h)
    if t_out > T_END:
        return None
    pe = price_before(df, t_in)
    if pe is None or not np.isfinite(pe) or pe <= 0:
        return None
    if stop is not None:
        seg = df.loc[t_in:t_out]
        if len(seg) < 2:
            return None
        lvl = pe * (1 - stop)
        if (seg["low"] <= lvl).any():
            return (lvl - COST - pe) / pe
    px = price_before(df, t_out)
    if px is None or not np.isfinite(px):
        return None
    return (px - COST - pe) / pe


def controls(evts):
    """Non-event weekdays at the same clock, +/-2 days around any event excluded."""
    ex = {t.date() for t in evts}
    clock = evts[0].time()
    lo, hi = min(evts).normalize(), max(evts).normalize()
    out = []
    for d in pd.bdate_range(lo, hi):
        if any(abs((d.date() - x).days) <= 2 for x in ex):
            continue
        out.append(pd.Timestamp.combine(d.date(), clock).tz_localize("UTC"))
    return out


for name, path in [("CPI", "data/ext_cpi_dates.csv"), ("NFP", "data/ext_nfp_dates.csv"),
                   ("FOMC(基準)", None)]:
    if path:
        e = pd.read_csv(path, parse_dates=["dt_broker"])
        evts = list(e["dt_broker"].dt.tz_localize("UTC").sort_values())
    else:
        e = pd.read_csv("scratchpad/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
        evts = list(e["dt_broker"].dt.tz_localize("UTC").sort_values())
    ctl = controls(evts)
    print(f"\n########## {name}  n={len(evts)}  発表 broker {evts[0].time()} ##########")
    for h in [60, 240]:
        r = np.array([x for x in (hold(t, h) for t in evts) if x is not None])
        c = np.array([x for x in (hold(t, h) for t in ctl) if x is not None])
        if len(r) < 5 or len(c) < 30:
            continue
        d = rng.integers(0, len(c), size=(B, len(r)))
        pct = np.mean(np.median(c[d], axis=1) < np.median(r)) * 100
        print(f"  H={h:>3}分 n={len(r):>3} | 中央値 {np.median(r)*100:+7.3f}%  平均 {r.mean()*100:+7.3f}%  "
              f"勝率 {np.mean(r > 0)*100:5.1f}%  対照 {np.median(c)*100:+7.3f}%  %ile {pct:5.1f}  "
              f"| 現在価格換算 中央値 {np.median(r)*PX:+6.2f}$ 平均 {r.mean()*PX:+6.2f}$")
    # per-year and the 2021-22 drop, at H=240
    rows = [(t.year, hold(t, 240)) for t in evts]
    rows = [(y, v) for y, v in rows if v is not None]
    t = pd.DataFrame(rows, columns=["yr", "r"])
    g = t.groupby("yr")["r"].agg(n="count", med=lambda s: s.median() * 100)
    pos = (g["med"] > 0).sum()
    print(f"  年別(H=240) 中央値プラス {pos}/{len(g)}年:  " +
          " ".join(f"{y}:{v:+.2f}" for y, v in g["med"].items()))
    ex = t[~t["yr"].isin([2021, 2022])]["r"].values
    print(f"  2021-22を除く: n={len(ex)} 中央値 {np.median(ex)*100:+.3f}%  平均 {ex.mean()*100:+.3f}%")
