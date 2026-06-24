"""equity_gate.py -- the EQUITY-CURVE meta-gate: deploy a leg only when the leg's OWN recent
realized equity is healthy. An UNTESTED regime lever whose mechanism is ORTHOGONAL to every gate
tried so far (SMA/slope/ER/stack/persist/KAMA/ADX/dist/vol/macro all read the MARKET regime, which
regime_headroom.py declared unpredictable-from-price). This does NOT predict the market; it reacts
to whether the strategy is in sync -- side-stepping the headroom null.

CRISP FALSIFIER (known result): equity-curve trading on an IID trade sequence is mathematically
worthless and only burns cost. So PERSISTENCE in trade-R must exist FIRST (Part 1, the gate-keeper)
or the idea is dead on arrival -- we do NOT fish for a lucky parameter on a non-persistent base.

Strictly causal: each trade's R is realized at exit_time = entry_time + hold(days); a candidate
entry at time T is gated using ONLY trades that CLOSED before T (exit < T). No lookahead.

  .venv/bin/python research/equity_gate.py
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG, SPLIT, metrics
from research.portfolio_kama import PB, kama_gate_btc

RNG = np.random.default_rng(7)


# ============================== legs WITH hold (for causal exit times) ==============================
def legs_with_hold():
    """gold_bo, btc_bo_kama, btc_pull as (time,R,hold) -- re-run configs since get_legs strips hold."""
    gold = run_bo(resample(load_mt5_csv("data/vantage_xauusd_h1.csv"), "1h"),
                  SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                     "daily_sma": 150, "daily_slope_k": 10}))[["time", "R", "hold"]]
    btc = run_bo(resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h"),
                 SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300}))[["time", "R", "hold"]]
    btc_k = kama_gate_btc(btc)
    pb = run_pb(resample(load_mt5_csv("data/vantage_btcusd_h1.csv"), "4h"), "long",
                SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0)[["time", "R", "hold"]]
    out = {}
    for k, v in [("gold_bo", gold), ("btc_bo_kama", btc_k), ("btc_pull", pb)]:
        out[k] = v.sort_values("time").reset_index(drop=True)
    return out


# ============================== Part 1: PERSISTENCE PRE-TEST ==============================
def runs_z(signs):
    """Wald-Wolfowitz runs test z-stat: <<0 = streaky (fewer runs than random)."""
    s = signs[signs != 0]
    n1 = int((s > 0).sum()); n2 = int((s < 0).sum()); n = n1 + n2
    if n1 == 0 or n2 == 0:
        return np.nan, 0
    runs = 1 + int((s[1:] != s[:-1]).sum())
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    return (runs - mu) / np.sqrt(var) if var > 0 else np.nan, runs


def cond_spread(R, K):
    """meanR after a winning last trade minus after a losing one, AND after last-K cum>0 vs <0."""
    prev_win = np.sign(R[:-1]) > 0
    after = R[1:]
    s1 = after[prev_win].mean() - after[~prev_win].mean() if prev_win.any() and (~prev_win).any() else np.nan
    cum = pd.Series(R).rolling(K).sum().shift(1).values
    up = cum > 0
    nxt = R[~np.isnan(cum)]
    upm = up[~np.isnan(cum)]
    s2 = nxt[upm].mean() - nxt[~upm].mean() if upm.any() and (~upm).any() else np.nan
    return s1, s2


def part1(legs, K=8, B=2000):
    print("\n" + "=" * 80)
    print("1. PERSISTENCE PRE-TEST (gate-keeper). Equity-curve trading needs streaky trade-R;")
    print("   iid R => the gate is worthless. acf>0 / runs-z<0 / after-win>after-loss = persistent.")
    print(f"\n  {'leg':<14} {'n':>4} {'acf1':>6} {'acf1(sgn)':>9} {'runsZ':>6}  "
          f"{'dR|W-L':>7} {'p(shuf)':>7}  {'dR|cumK':>7} {'p':>6}")
    alive = {}
    for name, leg in legs.items():
        R = leg.R.values; n = len(R)
        acf1 = np.corrcoef(R[:-1], R[1:])[0, 1]
        sg = np.sign(R)
        acf1s = np.corrcoef(sg[:-1], sg[1:])[0, 1]
        z, _ = runs_z(sg)
        s1, s2 = cond_spread(R, K)
        # reshuffle null (one-sided: persistence => observed spread > shuffled)
        sh1 = np.empty(B); sh2 = np.empty(B)
        for b in range(B):
            Rs = RNG.permutation(R)
            sh1[b], sh2[b] = cond_spread(Rs, K)
        p1 = np.nanmean(sh1 >= s1) if not np.isnan(s1) else np.nan
        p2 = np.nanmean(sh2 >= s2) if not np.isnan(s2) else np.nan
        print(f"  {name:<14} {n:>4} {acf1:>6.2f} {acf1s:>9.2f} {z:>6.2f}  "
              f"{s1:>7.3f} {p1:>7.3f}  {s2:>7.3f} {p2:>6.3f}")
        alive[name] = (p1 < 0.10) or (p2 < 0.10)
    # sanity: synthetic iid stream must give p ~ 0.5
    iid = RNG.standard_normal(400)
    s1, s2 = cond_spread(iid, K)
    sh = np.array([cond_spread(RNG.permutation(iid), K)[0] for _ in range(B)])
    print(f"  {'[iid sanity]':<14}  (synthetic N(0,1)) dR|W-L={s1:+.3f}  p={np.nanmean(sh>=s1):.3f} (must be ~0.5)")
    return alive


# ============================== Part 2-4: causal equity gate ==============================
def causal_gate(leg, K, mode):
    """keep[i] = leg's recent realized equity (closed trades only, exit<entry) is healthy."""
    leg = leg.sort_values("time").reset_index(drop=True)
    entry = leg.time.values
    exit_ = (leg.time + pd.to_timedelta(leg.hold, unit="D")).values
    R = leg.R.values
    order = np.argsort(exit_, kind="stable")
    exit_sorted = exit_[order]; R_by_exit = R[order]; eq = np.cumsum(R_by_exit)
    keep = np.zeros(len(leg), bool)
    for i in range(len(leg)):
        m = int(np.searchsorted(exit_sorted, entry[i], side="left"))   # #trades closed before entry
        if m == 0:
            keep[i] = True; continue                                   # warmup: allow
        if mode == "streak":
            keep[i] = R_by_exit[max(0, m - K):m].sum() >= 0
        else:                                                          # equity vs its own MA
            keep[i] = eq[m - 1] >= eq[max(0, m - K):m].mean()
    return leg[keep]


def cdd(t):
    m = metrics(t, risk=0.01)
    return m


def show(name, t):
    m = cdd(t)
    if m is None:
        print(f"  {name:<28} (too few)"); return
    print(f"  {name:<28} n={m['n']:>4} CAGR={m['cagr']:+5.1f}% DD={m['dd']:4.1f}% "
          f"CAGR/DD={m['cdd']:5.2f} | IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | yr+ {m['py']}")


def random_pctile(leg, gated_cdd, keep_frac, draws=300):
    """percentile of the gate's CAGR/DD among random gates dropping the SAME fraction."""
    n = len(leg); k = max(3, int(round(keep_frac * n)))
    vals = []
    for _ in range(draws):
        idx = np.sort(RNG.choice(n, k, replace=False))
        m = cdd(leg.iloc[idx])
        if m is not None:
            vals.append(m["cdd"])
    vals = np.array(vals)
    return (vals < gated_cdd).mean() * 100, np.median(vals)


def part234(legs):
    print("\n" + "=" * 80)
    print("2-4. CAUSAL EQUITY GATE (skip entry when own recent equity unhealthy) vs ungated;")
    print("     SCREEN K for a PLATEAU (not a spike); RANDOM-gate percentile (luck-sorter check).")
    for mode in ("streak", "ma"):
        print(f"\n  --- gate mode = {mode}  (a:last-K cumR>=0 / b:equity>=MA(K)) ---")
        for name, leg in legs.items():
            base = cdd(leg)
            print(f"\n  {name}:  ungated CAGR/DD={base['cdd']:.2f} (n={base['n']}, "
                  f"IS={base['isr']:+.2f} OOS={base['oos']:+.2f})")
            for K in (3, 5, 8, 12, 20):
                g = causal_gate(leg, K, mode)
                m = cdd(g)
                if m is None:
                    print(f"    K={K:<3} (too few)"); continue
                kf = len(g) / len(leg)
                pct, rmed = random_pctile(leg, m["cdd"], kf)
                tag = "PLATEAU?" if m["cdd"] > base["cdd"] else ""
                print(f"    K={K:<3} CAGR/DD={m['cdd']:5.2f} (keep {kf*100:3.0f}%) "
                      f"IS={m['isr']:+.2f} OOS={m['oos']:+.2f} | rand p{pct:3.0f} (med {rmed:.2f}) {tag}")


def main():
    legs = legs_with_hold()
    alive = part1(legs)
    print("\n  persistence verdict:", {k: ("persistent" if v else "iid-like") for k, v in alive.items()})
    if not any(alive.values()):
        print("\n  >>> NO leg shows exploitable persistence -> equity-curve gate is DEAD on arrival.")
        print("      (the honest kill: no fishing for a lucky parameter on an iid base.)")
        return
    part234(legs)
    print("\n" + "=" * 80)
    print("NOTE: in-sample lift only. A real gate = PLATEAU across K + beats RANDOM (>=~90 pctile) +")
    print("IS~=OOS + per-year better, on >=2 legs. Then run overfit_audit. Live-forward arbitrates regime.")


if __name__ == "__main__":
    main()
