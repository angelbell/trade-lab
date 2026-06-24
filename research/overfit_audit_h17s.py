"""overfit_audit_h17s.py -- QUANTIFY statistical-overfit risk on H17-S
(gold ORB short-only + 1H-EMA80 gate + daily-SMA-falling gate), reusing the
SAME math as overfit_audit.py (DSR / PBO-CSCV / block-bootstrap+null) but fed by
the scalp_lab ORB trade stream instead of the breakout get_legs() family.

Flagship config: --dir short --no-tp --htf-tf 1h --htf-ema 80 --daily-sma 80
                 --daily-slope-k 10  (1H exec = --resample 60min).
PBO grid = the H17-S configs we'd actually consider:
  resample {15min,30min,60min} x daily_sma {30,50,80,100,120} x htf_ema {50,80,120}.

Discipline note: the sealed TEST (2025+) was already consumed by H17, so this audit
uses the full 2018-06+ history (a statistical-overfit measurement, not a fresh peek).
Like overfit_audit.py it CANNOT measure regime-change risk -> live-forward = arbiter.

  .venv/bin/python research/overfit_audit_h17s.py
"""
import os, sys, re, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv
import research.scalp_lab as sl
# reuse the published audit math verbatim
from research.overfit_audit import psr, sr0, cscv, cdd_R, block_resample

CSV = "data/vantage_xauusd_m15.csv"
START = "2018-06-01"          # H17 development window start (matches scalp_lab SPLITS)


def scalp_lab_defaults():
    """parse scalp_lab.py argparse defaults so the param object is faithful to the tool."""
    src = open(os.path.join(os.path.dirname(__file__), "scalp_lab.py")).read()
    d = {}
    for m in re.finditer(r'add_argument\(\s*"--([\w-]+)"(.*?)\)', src, re.S):
        dest = m.group(1).replace("-", "_"); body = m.group(2)
        if "store_true" in body:
            d[dest] = False; continue
        dm = re.search(r'default\s*=\s*([^,)\n]+)', body)
        if not dm:
            d[dest] = None; continue
        raw = dm.group(1).strip()
        try:
            d[dest] = eval(raw, {"__builtins__": {}}, {})
        except Exception:
            d[dest] = raw.strip('"\'')
    return d

BASE = scalp_lab_defaults()


def make_p(**over):
    p = dict(BASE); p.update(strat="orb", no_tp=True, dir="short",
                             htf_tf="1h", htf_ema=80, daily_sma=80, daily_slope_k=10)
    p.update(over)
    return SimpleNamespace(**p)


def load_resampled(tf):
    d = load_mt5_csv(CSV).loc[START:]
    if tf and tf not in ("native",):
        d = d.resample(tf, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return d


def run_cfg(d, p):
    """full scalp_lab ORB pipeline -> trades df with t_in/pips. Returns None if too thin."""
    dir_, sl_px, tp_px = sl.orb_signals(d, p)
    dir_, sl_px, tp_px = sl.vol_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = sl.daily_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = sl.htf_trend_gate(d, dir_, sl_px, tp_px, p)
    dir_, sl_px, tp_px = sl.kama_gate(d, dir_, sl_px, tp_px, p)
    t = sl.backtest(d, dir_, sl_px, tp_px, p)
    return t if len(t) >= 20 else None


def to_R(t):
    """pips -> R-multiples using avg-loss as 1R (matches scalp_lab.metrics)."""
    loss = t.pips[t.pips < 0]
    one_r = abs(loss.mean()) if len(loss) else 1.0
    return (t.pips.values / one_r)


# ----- build the PBO grid once: monthly matrix + per-trial SR variance -----
def build_grid():
    cache = {tf: load_resampled(tf) for tf in ("15min", "30min", "60min")}
    cols, srs = {}, []; cid = 0
    for tf, ds, te in itertools.product(("15min", "30min", "60min"),
                                        (30, 50, 80, 100, 120), (50, 80, 120)):
        t = run_cfg(cache[tf], make_p(daily_sma=ds, htf_ema=te))
        if t is None:
            continue
        R = to_R(t)
        s = pd.DataFrame({"R": R, "m": t.t_in.dt.tz_localize(None).dt.to_period("M")})
        cols[f"c{cid}"] = s.groupby("m").R.sum()
        srs.append(R.mean() / R.std(ddof=1))
        cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0)
    return M.values, float(np.var(srs)), cache


def part_a(R, V):
    print("\n" + "=" * 74)
    print("A. DEFLATED SHARPE (per-trade) -- DSR=P(true SR>deflation benchmark for N trials)")
    Ns = [1, 10, 25, 45, 100, 200]
    _, sr, g1, g4 = psr(R, 0.0)
    t = sr * np.sqrt(len(R))
    dsrs = [psr(R, sr0(N, V))[0] for N in Ns]
    print(f"  {'H17-S(1H,sma80)':<16} n={len(R):>3} SR/tr={sr:>5.3f} t={t:>4.2f} "
          f"skew={g1:>5.2f} kurt={g4:>4.1f}")
    print("   " + "  ".join(f"DSR@{N}={d:.2f}" for N, d in zip(Ns, dsrs)))
    print("   (DSR>0.95 = per-trade edge survives the N-trial haircut)")


def part_b(M, S=10, seeds=24):
    print("\n" + "=" * 74)
    print("B. PBO via CSCV -- P(IS-best config is OOS-below-median). <0.5 better; <~0.2 robust.")
    # only ~55 active months -> a single CSCV run is unstable; average real vs a same-shape
    # noise baseline over many seeds (noise must center on ~0.50 for the estimate to be valid).
    real, noise, oosm, plos = [], [], [], []
    for sd in range(seeds):
        pbo, om, pl = cscv(M, S=S, seed=sd)
        real.append(pbo); oosm.append(om); plos.append(pl)
        noise.append(cscv(np.random.default_rng(sd).standard_normal(M.shape), S=S, seed=sd)[0])
    real, noise = np.array(real), np.array(noise)
    print(f"  H17-S grid = {M.shape[1]} cfg x {M.shape[0]} active months  (S={S} blocks, {seeds} seeds)")
    print(f"  REAL  PBO={real.mean():.2f}   IS-best mean OOS-Sharpe={np.mean(oosm):+.2f}   "
          f"P(OOS loss)={np.mean(plos):.2f}")
    print(f"  NOISE PBO={noise.mean():.2f} (sd{noise.std():.2f}, must center ~0.50 = CSCV valid)   "
          f"gap REAL-NOISE={real.mean()-noise.mean():+.2f}")


def part_c(t):
    print("\n" + "=" * 74)
    print("C. BLOCK-BOOTSTRAP CI on CAGR/DD (1% risk) + mean-removed NULL (empirical p).")
    R = to_R(t)
    yrs = max((t.t_in.max() - t.t_in.min()).days / 365.25, 0.5)
    obs = cdd_R(R, yrs)[2]
    B, L = 4000, 5
    rng = np.random.default_rng(7)
    boot = np.array([cdd_R(block_resample(R, L, rng), yrs)[2] for _ in range(B)])
    rn = R - R.mean()
    nul = np.array([cdd_R(block_resample(rn, L, rng), yrs)[2] for _ in range(B)])
    p = (nul >= obs).mean()
    c5, c50, c95 = np.percentile(boot, [5, 50, 95])
    print(f"  H17-S(1H,sma80)  obs CAGR/DD={obs:+.2f}  CI[5/50/95]={c5:+.2f}/{c50:+.2f}/{c95:+.2f}"
          f"  null p={p:.3f}")
    print("   (small p => edge unlikely to be luck; CI shows how uncertain the SIZE is)")


def main():
    print("H17-S overfit audit  (gold ORB short + 1H-EMA80 + daily-SMA-falling)")
    M, V, cache = build_grid()
    t_flag = run_cfg(cache["60min"], make_p(daily_sma=80, htf_ema=80))
    part_b(M)
    part_a(to_R(t_flag), V)
    part_c(t_flag)
    print("\n" + "=" * 74)
    print("NOTE: A/B/C quantify STATISTICAL overfit only. Regime-concentration (profit in")
    print("gold's 2021-22 downtrend, dormant in bull years) is a REGIME risk these cannot")
    print("measure -- live-forward stays the arbiter for H17-S's SIZE.")


if __name__ == "__main__":
    main()
