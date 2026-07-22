"""Enter the FOMC scalp on a limit inside the impulse leg instead of at market.

The stop test showed 62.1% of these trades retrace half the leg and 27.6% retrace all of it
within five minutes -- so a limit placed inside the leg would fill often and at a much better
price. Against that, the repo's standing result is that deep pullback entries select the WEAK
breakouts ("strong breakouts don't come back"), which is why the book uses 0.25-0.30.

Compared honestly per EVENT, not per fill: an event where the limit never fills earns zero, so
a better fill price bought with a lower fill rate has to clear that bar.

Execution realism carried over from the ledger:
  - buy limits fill on the ASK, so on this (bid) data require low <= limit - one-side spread
  - the stop scan includes the FILL bar and the stop wins ties on that bar (falsifier 11) --
    skipping the fill bar is exactly the bug that once inflated the book
  - exit at the same wall-clock moment as the market version, so only the ENTRY differs
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, COST_ROUNDTRIP
from experiments.event_scalp_cond import threshold_subset
from experiments.fomc_event_study import price_before

COST = COST_ROUNDTRIP["GOLD"]["base"]   # round trip, $/oz
SPREAD1 = 0.15                          # one-side, for ASK-based limit fills
H = 5
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END, PX = df.index.max(), float(df["close"].iloc[-1])

ev = pd.read_csv("experiments/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
real = build_scalp_table(df, events, 1, [5, 10], "pb")
sub, _ = threshold_subset(real, "confirm_move_atr", 0.50).pipe(
    lambda x: (x[0].sort_values("t0").reset_index(drop=True), x[1])) if False else (
    threshold_subset(real, "confirm_move_atr", 0.50)[0].sort_values("t0").reset_index(drop=True), None)


def market(t0, d):
    """Baseline: market at the confirm close, leg-full-retrace stop, time exit."""
    P0 = price_before(df, t0)
    t_in = t0 + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=H)
    if t_out > T_END:
        return None
    pe = price_before(df, t_in)
    if P0 is None or pe is None or pe <= 0:
        return None
    bars = df.loc[t_in:t_out]
    if len(bars) < 2:
        return None
    L = P0
    hit = (bars["low"] <= L).any() if d > 0 else (bars["high"] >= L).any()
    if hit:
        return (d * (L - pe) - COST) / pe
    px = price_before(df, t_out)
    return None if px is None else (d * (px - pe) - COST) / pe


def limit(t0, d, f):
    """Limit at f of the way back into the leg. Returns (filled?, return per EVENT)."""
    P0 = price_before(df, t0)
    t_in = t0 + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=H)
    if t_out > T_END:
        return None
    pe = price_before(df, t_in)
    if P0 is None or pe is None or pe <= 0:
        return None
    leg = abs(pe - P0)
    lim = pe - d * f * leg
    bars = df.loc[t_in:t_out]
    if len(bars) < 2:
        return None
    # ASK-based fill: for a long the bid must trade through the limit by the spread
    if d > 0:
        touched = bars.index[bars["low"] <= lim - SPREAD1]
    else:
        touched = bars.index[bars["high"] >= lim + SPREAD1]
    if len(touched) == 0:
        return (False, 0.0)
    t_fill = touched[0]
    seg = df.loc[t_fill:t_out]           # includes the fill bar
    L = P0
    hit = (seg["low"] <= L).any() if d > 0 else (seg["high"] >= L).any()
    if hit:
        return (True, (d * (L - lim) - COST) / lim)
    px = price_before(df, t_out)
    return None if px is None else (True, (d * (px - lim) - COST) / lim)


base = np.array([x for x in (market(r["t0"], r["d"]) for _, r in sub.iterrows()) if x is not None])
print(f"n={len(base)}  H={H}分  損切り=脚の全戻し(P0)  片側スプレッド ${SPREAD1}")
print(f"\n{'入口':>18} {'約定率%':>7} {'約定時 中央値%':>13} | {'1イベント当たり':>26}")
print(f"{'':>18} {'':>7} {'':>13} | {'中央値%':>9} {'平均%':>8} {'現在換算 平均':>12}")
print(f"{'成行(基準)':>18} {100.0:>7.1f} {np.median(base)*100:>13.3f} | "
      f"{np.median(base)*100:>9.3f} {base.mean()*100:>8.3f} {base.mean()*PX:>11.2f}$")
for f in [0.25, 0.50, 0.70, 1.00]:
    out = [x for x in (limit(r["t0"], r["d"], f) for _, r in sub.iterrows()) if x is not None]
    filled = np.array([o[1] for o in out if o[0]])
    per_event = np.array([o[1] for o in out])
    if len(filled) == 0:
        continue
    print(f"{('指値 ' + str(f) + '戻し'):>18} {len(filled)/len(out)*100:>7.1f} "
          f"{np.median(filled)*100:>13.3f} | {np.median(per_event)*100:>9.3f} "
          f"{per_event.mean()*100:>8.3f} {per_event.mean()*PX:>11.2f}$")
print("\n※ 1イベント当たり = 約定しなかった回を 0 として全イベントで平均（見送りを損得ゼロで数える）")
