"""Q5 (the widest-stop quintile) has meanR -0.013 and PF 0.97 -- it is a dead 20 trades/yr.
Is "skip the wide-stop breakouts" a real filter, or a luck-sorter?

Two traps to clear before believing it:
  1. A DOLLAR threshold is a disguised TIME filter. A $1,000 stop is 12.5% of BTC in 2019 and 1% in
     2025, so cutting on dollars quietly deletes the early years. Cut on stop/PRICE (%) instead --
     scale-free, and the same rule a trader could actually apply live at any BTC price.
  2. Any selection rule is a luck-sorter (checklist 7). It must beat
       (a) a RANDOM-DROP null that removes the same NUMBER of trades at random, judged on totR/yr and
           CAGR/DD -- not on PF, which any culling of losers can inflate; and
       (b) a CIRCULAR BLOCK BOOTSTRAP over months (1/3/6/12), where a real improvement's P RISES with
           block length and a path-fit's P falls.
Report N at every threshold. A filter that buys PF by deleting the book's frequency has bought nothing.
Run: .venv/bin/python scratchpad/btc15mL_wide_stop_filter.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
NDRAW = 2000


def pf(x):
    return x[x > 0].sum() / abs(x[x <= 0].sum()) if (x <= 0).any() else np.inf


def cdd(s):
    """CAGR%, maxDD%, ratio on the trade-resolution equity curve at 1% risk (the arbiter)."""
    eq = np.cumprod(1 + 0.01 * s.values)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    days = max((s.index[-1] - s.index[0]).days, 1)
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def block_boot(base, arm, blocks=(1, 3, 6, 12)):
    """Paired circular block bootstrap over months. P = share of resamples where arm beats base."""
    out = {}
    months = sorted(set(base.index.to_period("M")))
    bm = {m: base[base.index.to_period("M") == m].values for m in months}
    am = {m: arm[arm.index.to_period("M") == m].values for m in months}
    M = len(months)
    for L in blocks:
        nb = int(np.ceil(M / L))
        wins = 0
        for _ in range(NDRAW):
            st = RNG.integers(0, M, nb)
            order = np.concatenate([(np.arange(s, s + L) % M) for s in st])[:M]
            b = np.concatenate([bm[months[i]] for i in order if len(bm[months[i]])])
            a = np.concatenate([am[months[i]] for i in order if len(am[months[i]])])
            if len(b) < 10 or len(a) < 10:
                continue
            eb = np.cumprod(1 + 0.01 * b); ea = np.cumprod(1 + 0.01 * a)
            db = ((np.maximum.accumulate(eb) - eb) / np.maximum.accumulate(eb)).max()
            da = ((np.maximum.accumulate(ea) - ea) / np.maximum.accumulate(ea)).max()
            wins += (ea[-1] ** (1 / max(da, 1e-9))) > (eb[-1] ** (1 / max(db, 1e-9)))
        out[L] = 100 * wins / NDRAW
    return out


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])
    w = np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
    risk = t["risk"].values / w
    R = pd.Series(t["R"].values * w - 15.0 / risk, index=pd.DatetimeIndex(t["time"]))
    px = t["e_px"].values
    stop_pct = 100.0 * risk / px                      # scale-free: stop as % of entry price
    yrs = (R.index[-1] - R.index[0]).days / 365.25

    print("損切り幅を「価格に対する%」で測り直す（絶対額のしきい値は時代フィルタに化けるため）\n")
    print(f"  損切り/価格(%):  中央値 {np.median(stop_pct):.2f}%  "
          f"10/90%点 {np.percentile(stop_pct,10):.2f}% / {np.percentile(stop_pct,90):.2f}%")
    print("\n  年ごとの中央値（絶対額だと時代で意味が変わるが、%なら安定しているはず）")
    y = pd.Series(stop_pct, index=R.index)
    print("    " + "  ".join(f"{a}:{b:.2f}%" for a, b in y.groupby(y.index.year).median().items()))
    dol = pd.Series(risk, index=R.index)
    print("    （参考・絶対額の中央値: "
          + "  ".join(f"{a}:${b:,.0f}" for a, b in dol.groupby(dol.index.year).median().items()) + "）")

    print("\n\n  損切り/価格 の五分位")
    print(f"    {'帯':<6}{'損切り/価格':>12}{'n':>6}{'本/年':>7}{'PF':>7}{'meanR':>9}{'totR/年':>10}")
    q = pd.qcut(stop_pct, 5, labels=False)
    for i in range(5):
        m = q == i
        print(f"    Q{i+1}{'':<4}{np.median(stop_pct[m]):>11.2f}%{m.sum():>6}{m.sum()/yrs:>7.0f}"
              f"{pf(R.values[m]):>7.2f}{R.values[m].mean():>+9.3f}{R.values[m].sum()/yrs:>+10.1f}")

    print("\n\n  『損切りが広すぎるブレイクを見送る』しきい値スイープ")
    print("  ★ PF ではなく **N と totR/年 と CAGR/DD** で判定する（選別は運のソーター）")
    b_c, b_d, b_r = cdd(R)
    print(f"    {'条件':<24}{'n':>6}{'本/年':>7}{'PF':>7}{'meanR':>9}{'totR/年':>10}"
          f"{'CAGR/DD':>10}{'ランダム除去null':>16}")
    print(f"    {'全部（現行）':<24}{len(R):>6}{len(R)/yrs:>7.0f}{pf(R.values):>7.2f}"
          f"{R.mean():>+9.3f}{R.sum()/yrs:>+10.1f}{b_r:>10.2f}{'—':>16}")
    keep = {}
    for thr in (6.0, 5.0, 4.0, 3.0, 2.5, 2.0):
        m = stop_pct <= thr
        if m.sum() < 100:
            continue
        s = R[m]
        c, d, r = cdd(s)
        # random-drop null: remove the SAME number of trades at random, 2000 times
        ndrop = len(R) - m.sum()
        null = []
        for _ in range(NDRAW):
            k = RNG.choice(len(R), len(R) - ndrop, replace=False)
            null.append(cdd(R.iloc[np.sort(k)])[2])
        pctile = 100 * np.mean(r > np.array(null))
        print(f"    {'損切り <= 価格の '+str(thr)+'%':<24}{m.sum():>6}{m.sum()/yrs:>7.0f}"
              f"{pf(s.values):>7.2f}{s.mean():>+9.3f}{s.sum()/yrs:>+10.1f}{r:>10.2f}"
              f"{pctile:>14.0f}%ile")
        keep[thr] = s

    print("\n\n  巡回ブロック・ブートストラップ（本物なら、ブロックを伸ばすほど P が上がる）")
    print(f"    {'条件':<24}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    for thr, s in keep.items():
        p = block_boot(R, s)
        print(f"    {'損切り <= 価格の '+str(thr)+'%':<24}"
              + "".join(f"{p[L]:>7.0f}%" for L in (1, 3, 6, 12)))


if __name__ == "__main__":
    main()
