"""Position sizing for the structure_sma base strategy (B in the roadmap).

Goal: turn the trade distribution into a live risk plan — how much to risk per
trade, the resulting expected drawdown, terminal wealth, ruin probability, and
the leverage/margin that implies on Vantage.

Method:
  1. Run the LOCKED base config (1h/4h/SMA150/breakout, fractal_n=3, atr_sl=2.0)
     over the full 2022→today sample → 104 trades.
  2. R-multiple per trade = trade_return / planned_risk, where planned_risk =
     ATR(14)*atr_sl/entry_price (the stop distance set at entry). R is unit-free
     and sizing-independent, so we can rescale risk freely.
  3. Kelly: f* maximizes E[log(1 + f*R)]. Report full / half / quarter Kelly.
  4. For each fixed-fractional risk%, Monte-Carlo by resampling the R sequence
     (block bootstrap to keep streaks) → median/P5 terminal multiple, expected
     & worst MaxDD, P(equity halves). Effective leverage = risk% / sl_frac.
"""

import argparse

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import fetch_ohlcv
from src.strategy import compute_signals_structure, run_portfolio

LOCK = dict(fractal_n=3, entry_mode="breakout", sma_regime=150, htf_timeframe="4h")
ATR_SL = 2.0


def r_multiples(start: str, end: str, tf: str = "1h") -> tuple[np.ndarray, float]:
    d = fetch_ohlcv("BTC/USDT", tf, start, end)
    le, lx, se, sx = compute_signals_structure(d["close"], d["high"], d["low"], **LOCK)
    pf = run_portfolio(d["close"], d["high"], d["low"], le, lx, se, sx,
                       atr_mult_sl=ATR_SL, freq="1h")
    r = pf.trades.records_readable
    atr = ta.atr(d["high"], d["low"], d["close"], length=14)
    sl_frac = (atr * ATR_SL / d["close"]).clip(upper=0.5)
    entry_sl = sl_frac.reindex(r["Entry Timestamp"]).to_numpy()
    R = (r["Return"].to_numpy() / entry_sl)
    R = R[np.isfinite(R)]
    return R, float(np.nanmean(sl_frac))


def kelly_f(R: np.ndarray) -> float:
    """f* maximizing E[log(1+f*R)] on the empirical R distribution."""
    fs = np.linspace(0.001, 1.0, 1000)
    best_f, best_g = 0.0, -np.inf
    for f in fs:
        x = 1 + f * R
        if (x <= 0).any():
            break  # f too large: a single -R would bankrupt -> stop
        g = np.mean(np.log(x))
        if g > best_g:
            best_g, best_f = g, f
    return best_f


def block_bootstrap(R: np.ndarray, n_paths: int, horizon: int, block: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Return array (n_paths, horizon) of resampled R, preserving short streaks."""
    out = np.empty((n_paths, horizon))
    n = len(R)
    for p in range(n_paths):
        seq = []
        while len(seq) < horizon:
            i = rng.integers(0, n)
            seq.extend(R[i:i + block])
        out[p] = seq[:horizon]
    return out


def simulate(R: np.ndarray, risk: float, paths: np.ndarray) -> dict:
    """Compound equity along each bootstrapped path at fixed fractional risk."""
    n_paths, horizon = paths.shape
    eq = np.ones(n_paths)
    peak = np.ones(n_paths)
    maxdd = np.zeros(n_paths)
    ever_half = np.zeros(n_paths, dtype=bool)
    for t in range(horizon):
        eq = eq * (1 + risk * paths[:, t])
        eq = np.maximum(eq, 1e-9)
        peak = np.maximum(peak, eq)
        dd = eq / peak - 1
        maxdd = np.minimum(maxdd, dd)
        ever_half |= eq <= 0.5
    return dict(
        median=np.median(eq), p5=np.percentile(eq, 5), p95=np.percentile(eq, 95),
        exp_maxdd=np.mean(maxdd), worst_maxdd=np.min(maxdd),
        p_half=np.mean(ever_half),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end",   default="2026-06-09")
    args = p.parse_args()

    R, avg_sl = r_multiples(args.start, args.end)
    wins = R[R > 0]; losses = R[R < 0]
    win_rate = len(wins) / len(R)
    payoff = wins.mean() / abs(losses.mean()) if len(losses) else float("inf")
    exp_R = R.mean()

    print(f"\n=== Trade distribution ({args.start}→{args.end}, base config) ===")
    print(f"  trades={len(R)}  win rate={win_rate:.1%}  payoff(win/loss)={payoff:.2f}")
    print(f"  E[R]={exp_R:+.3f}  median R={np.median(R):+.3f}  "
          f"best={R.max():+.1f}  worst={R.min():+.1f}")
    print(f"  avg planned stop (sl_frac)={avg_sl:.3%} of notional")

    f_star = kelly_f(R)
    print(f"\n=== Kelly ===")
    print(f"  full Kelly f*  = {f_star:.2%} of equity risked per trade")
    print(f"  half  Kelly    = {f_star/2:.2%}   quarter Kelly = {f_star/4:.2%}")
    print(f"  (full Kelly is too aggressive in practice — half/quarter is standard)")

    # Monte Carlo over a realistic 1-year horizon (~ trades/yr from the sample)
    yrs = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days / 365.25
    trades_per_yr = max(1, round(len(R) / yrs))
    rng = np.random.default_rng(7)
    paths = block_bootstrap(R, n_paths=5000, horizon=trades_per_yr, block=4, rng=rng)

    print(f"\n=== Fixed-fractional risk — 1yr Monte Carlo "
          f"({trades_per_yr} trades/yr, 5000 paths) ===")
    print(f"{'risk/trade':>10} {'medianX':>8} {'P5 X':>7} {'P95 X':>8} "
          f"{'E[MaxDD]':>9} {'worstDD':>8} {'P(halve)':>9} {'eff.lev':>8}")
    for risk in (0.005, 0.01, 0.02, 0.03, 0.05):
        s = simulate(R, risk, paths)
        eff_lev = risk / avg_sl  # notional / equity to risk `risk` over an sl_frac stop
        print(f"{risk:>9.1%} {s['median']:>8.2f} {s['p5']:>7.2f} {s['p95']:>8.2f} "
              f"{s['exp_maxdd']:>9.1%} {s['worst_maxdd']:>8.1%} "
              f"{s['p_half']:>9.2%} {eff_lev:>7.2f}x")

    print(f"\n  eff.lev = notional/equity per position. Vantage offers up to 1000x,"
          f"\n  so margin is NEVER the binding constraint — risk%/drawdown is.")


if __name__ == "__main__":
    main()
