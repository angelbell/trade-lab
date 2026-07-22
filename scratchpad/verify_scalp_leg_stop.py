"""Stop the 1-minute scalp at the impulse leg, not at an ATR multiple.

The earlier stop test only swept k x ATR14 (0.5-3.0) and fixed dollars ($1-$5) and concluded
"no stop works". Both were sized off the PRE-release 1-minute ATR, which is far too small once
the statement hits -- 0.5-1.0x ATR stopped out 100% of trades. The stop the rule actually
implies was never tested: the trade is taken because the first minute moved from P0 to P_entry,
so a full retrace back through P0 means the premise is gone.

Levels tested (d = +1 long / -1 short, L = the leg |P_entry - P0|):
    f=0.5  half the leg given back
    f=1.0  back to P0 -- the pre-statement price, the leg fully retraced
    f=1.5  through P0 by half a leg again
plus the confirmation bar's own low/high.

Judged on the median (n=29, the mean is owned by one event), with the stop scan including the
entry bar itself and the stop winning ties on the same bar.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from scratchpad.event_scalp import build_scalp_table, null_scalp_table, COST_ROUNDTRIP
from scratchpad.event_scalp_cond import threshold_subset
from scratchpad.fomc_event_study import price_before

COST = COST_ROUNDTRIP["GOLD"]["base"]
B = 10000
rng = np.random.default_rng(42)
df = load_mt5_csv("data/vantage_xauusd_m1.csv")
T_END, PX = df.index.max(), float(df["close"].iloc[-1])

ev = pd.read_csv("scratchpad/fomc_stmt_2019.csv", parse_dates=["dt_broker"])
ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
events = list(ev["dt_broker"].sort_values())
real = build_scalp_table(df, events, 1, [5, 10], "leg")
sub, _ = threshold_subset(real, "confirm_move_atr", 0.50)
sub = sub.sort_values("t0").reset_index(drop=True)


def run(t0, d, H, mode, f=None):
    P0 = price_before(df, t0)
    t_in = t0 + pd.Timedelta(minutes=1)
    t_out = t_in + pd.Timedelta(minutes=H)
    if t_out > T_END:
        return None
    pe = price_before(df, t_in)
    if P0 is None or pe is None or not np.isfinite(pe) or pe <= 0:
        return None
    bars = df.loc[t_in:t_out]          # includes the entry bar (falsifier 11)
    if len(bars) < 2:
        return None
    if mode == "leg":
        L = pe - d * f * abs(pe - P0)
    elif mode == "bar":
        b0 = df.loc[df.index <= t_in].iloc[-1]
        L = b0["low"] if d > 0 else b0["high"]
    else:
        L = None
    if L is not None:
        hit = (bars["low"] <= L).any() if d > 0 else (bars["high"] >= L).any()
        if hit:
            return (d * (L - pe) - COST) / pe, True, abs(pe - L) / pe
    px = price_before(df, t_out)
    if px is None or not np.isfinite(px):
        return None
    return (d * (px - pe) - COST) / pe, False, (abs(pe - L) / pe if L is not None else np.nan)


for H in [5, 10]:
    print(f"\n########## H={H}分  (n候補 {len(sub)}) ##########")
    print(f"{'損切り':>16} {'n':>3} | {'中央値%':>8} {'平均%':>7} {'勝率%':>6} {'発動%':>6} "
          f"{'損切り幅%':>9} | {'最悪%':>7} {'現在換算 中央値':>13} {'最悪':>7}")
    base = None
    for mode, f, lab in [(None, None, "無し"), ("bar", None, "確認足の安値/高値"),
                         ("leg", 0.5, "脚の0.5戻し"), ("leg", 1.0, "脚の全戻し(=P0)"),
                         ("leg", 1.5, "脚の1.5倍")]:
        out = [run(r["t0"], r["d"], H, mode, f) for _, r in sub.iterrows()]
        out = [o for o in out if o is not None]
        if not out:
            continue
        v = np.array([o[0] for o in out])
        hit = np.mean([o[1] for o in out]) * 100
        wid = np.nanmedian([o[2] for o in out]) * 100
        if base is None:
            base = v
        print(f"{lab:>16} {len(v):>3} | {np.median(v)*100:+8.3f} {v.mean()*100:+7.3f} "
              f"{np.mean(v > 0)*100:6.1f} {hit:6.1f} {wid:9.3f} | {v.min()*100:+7.3f} "
              f"{np.median(v)*PX:+13.2f}$ {v.min()*PX:+7.2f}$")
