"""Validation 1 & 2 for the standalone daily-regime size overlay on btc15m_L (found: size DOWN on
daily-down entries beats the random-subset null; SMA150 x0.5 ~ +15pt same-DD CAGR). Before it can be
trusted it must clear two things the 1-day cut never did honestly:

  1  NOT ONE ERA.   per-year raw R, plus LEAVE-ONE-YEAR-OUT: drop each calendar year, re-equalise the
     drawdown on the remaining base, and recompute the gain. If dropping 2022 (BTC bear) collapses it,
     the +15pt was that one drawdown. A real regime effect survives every single-year deletion.
  2  WF STABLE.     each Jan 1, pick the best (daily-definition, multiplier) from PRIOR TRADES ONLY by
     CAGR/DD, apply it forward. Compare the walk-forward-selected overlay's same-DD CAGR to the fixed
     x0.5-SMA150 and to base. If the forward choice wanders or fails to beat base, the fixed number was
     hindsight. Also print what each year would have chosen.

Arbiter throughout: standalone, de-lever to equal bootstrapped-median maxDD, compare CAGR (leverage-
free). Daily state from the CONFIRMED daily bar, shift(1), ffill -- no lookahead.
Run: .venv/bin/python scratchpad/L_daily_validate.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from arb_common import Boot, cd
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
BTC_PCT_YR = 30.0
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "rr": 4.5,
       "fill_win": 200, "fwd": 500}
DEFS = ["KAMA14", "SMA50", "SMA150"]
MS = [0.25, 0.35, 0.50, 0.75, 1.0]


def eqdd_cagr(s, D0, nb=800):
    """s（f 込み）を中央値maxDD=D0 に揃えた時の CAGR。専用 Boot で。"""
    bt = Boot(sorted(set(s.index.to_period("M"))), nb=nb, k=3)
    days = max((s.index[-1] - s.index[0]).days, 1)
    return cd((s * bt.equal_dd_cagr(s, D0)[1]).values, days)[0], bt


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    dly = d15.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last"}).dropna()
    kd = kama_adaptive(dly["close"], 14)
    raw = {"KAMA14": (kd > kd.shift(1)).shift(1),
           "SMA50": (dly["close"] > dly["close"].rolling(50).mean()).shift(1),
           "SMA150": (dly["close"] > dly["close"].rolling(150).mean()).shift(1)}
    raw = {k: v.reindex(d15.index, method="ffill") for k, v in raw.items()}

    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d15, SimpleNamespace(**CFG))
    ii = d15.index.get_indexer(t["time"])
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
    risk = t["risk"].values / w
    R = (t["R"].values * w - 15.0 / risk
         - (BTC_PCT_YR / 365.0 / 100.0) * (t["e_px"].values / risk) * t["hold"].values)
    ti = pd.DatetimeIndex(t["time"]); yr = ti.year.values
    UP = {d: raw[d].values[ii].astype(bool) for d in DEFS}       # True=日足↑

    def series(defn, m, mask=None):
        lab = np.where(UP[defn], 1.0, m)
        idx = np.ones(len(R), bool) if mask is None else mask
        return pd.Series((R * lab)[idx] * 0.01, index=ti[idx])

    base = pd.Series(R * 0.01, index=ti)
    D0 = Boot(sorted(set(ti.to_period("M"))), nb=1000).dd_median(base)
    cbase, _ = eqdd_cagr(base, D0, nb=1000)
    print(f"基準 maxDD = {D0:.2f}%（btc15m_L 単独・1%）、base の同DD CAGR = {cbase:+.1f}%\n")

    # ---- 1. 年別 + leave-one-year-out（SMA150・↓×0.5）----------------------------------------
    ov = series("SMA150", 0.5)
    print("1. 年別R（素・f=1名目）と、その年を抜いた時の同DD利得（leave-one-year-out）\n")
    print(f"  {'年':<7}{'現行R':>9}{'SMA150↓×0.5':>13}{'日足↓本数':>11}"
          f"{'   ｜抜いた年の同DD利得':>22}")
    yrs = sorted(set(yr))
    for y in yrs:
        my = yr == y
        rr_cur = R[my].sum(); rr_ov = (R * np.where(UP["SMA150"], 1.0, 0.5))[my].sum()
        ndn = int((~UP["SMA150"][my]).sum())
        keep = ~my
        bK = base[keep[np.arange(len(R))]] if False else pd.Series(R[keep] * 0.01, index=ti[keep])
        oK = pd.Series((R * np.where(UP["SMA150"], 1.0, 0.5))[keep] * 0.01, index=ti[keep])
        DK = Boot(sorted(set(ti[keep].to_period("M"))), nb=600).dd_median(bK)
        g = eqdd_cagr(oK, DK, nb=600)[0] - eqdd_cagr(bK, DK, nb=600)[0]
        print(f"  {y:<7}{rr_cur:>+8.0f}R{rr_ov:>+12.0f}R{ndn:>11}"
              f"{f'{y}を除外 → {g:+.1f}pt':>22}")
    gfull = eqdd_cagr(ov, D0, nb=1000)[0] - cbase
    print(f"\n  全期間（除外なし）の同DD利得 = {gfull:+.1f}pt。"
          f"どの1年を抜いても符号が変わらなければ一時代依存ではない。")

    # ---- 2. ウォークフォワード選択 ----------------------------------------------------------
    print("\n2. ウォークフォワード: 毎年1月1日、その年より前のトレードだけで (定義,m) を CAGR/DD で選択\n")
    first = yrs[0] + 2
    wf_full = np.ones(len(R)); wf_sma150 = np.ones(len(R)); picks = {}
    for y in yrs:
        my = yr == y
        if y < first:
            picks[y] = "(warmup: overlay無)"
            continue
        past = yr < y
        # フルグリッド選択
        best, bd, bm = -9, "SMA150", 1.0
        for d in DEFS:
            for m in MS:
                sp = (R[past] * np.where(UP[d][past], 1.0, m)) * 0.01
                c, dd = cd(sp, max((ti[past][-1] - ti[past][0]).days, 1))
                if c / max(dd, 1e-9) > best:
                    best, bd, bm = c / max(dd, 1e-9), d, m
        wf_full[my] = np.where(UP[bd][my], 1.0, bm)
        # 定義を SMA150 に固定して m だけ選択
        best2, bm2 = -9, 1.0
        for m in MS:
            sp = (R[past] * np.where(UP["SMA150"][past], 1.0, m)) * 0.01
            c, dd = cd(sp, max((ti[past][-1] - ti[past][0]).days, 1))
            if c / max(dd, 1e-9) > best2:
                best2, bm2 = c / max(dd, 1e-9), m
        wf_sma150[my] = np.where(UP["SMA150"][my], 1.0, bm2)
        picks[y] = f"フル→ {bd}×{bm}   SMA150固定→ ×{bm2}"

    print("  各年の選択:")
    for y in yrs:
        print(f"    {y}: {picks[y]}")
    arms = {
        "base（overlay無し）":          base,
        "固定 SMA150 ↓×0.5":           ov,
        "固定 SMA150 ↓×0.35":          series("SMA150", 0.35),
        "WF: SMA150固定・m だけ選択":    pd.Series(R * wf_sma150 * 0.01, index=ti),
        "WF: (定義,m) フル選択":         pd.Series(R * wf_full * 0.01, index=ti),
    }
    print(f"\n  {'アーム':<24}{'同DD CAGR':>11}{'base差':>9}")
    for nm, s in arms.items():
        c = eqdd_cagr(s, D0, nb=1000)[0]
        print(f"  {nm:<24}{c:>+10.1f}%{c-cbase:>+8.1f}pt"
              + ("  ← 基準" if nm.startswith("base") else ""))


if __name__ == "__main__":
    main()
