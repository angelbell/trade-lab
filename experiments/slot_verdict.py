"""btc15m_L holds ONE position at a time (max_pos=1). While it is holding, every new signal is thrown
away -- 1,238 of them over 7.7 years. Two ways to stop throwing them away:

  A  cut the open trade at 1 day, so the slot refills   -> costs -72R of amputated winners (law 4/9)
  B  open a SECOND position alongside                   -> costs nothing in R, but doubles exposure

B was measured raw and looked awful (book 4.84 -> 3.27), but that comparison was rigged: two full-size
positions risk 2x the account, so the drawdown doubles by construction. The only fair arbiter is EQUAL
maxDD -- de-lever both arms to the same bootstrapped-median drawdown and compare CAGR. Under that
arbiter, "more slots" is free to prove itself, because taking two half-size positions has exactly the
same peak exposure as one full-size one.

Judged under two weightings, because sigma-inv-vol is known to reward tail amputation (skew +1.84):
  sigma  the book's current rule
  cvar   mean of the worst 10% -- blind to the right tail, so it cannot be gamed by cutting winners
Run: .venv/bin/python experiments/slot_verdict.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from rr_with_swap import leg, SIX

RNG = np.random.default_rng(20260714)
ROOT = "/home/angelbell/dev/auto-trade"
BTC_PCT_YR = 30.0


def risk_of(x, how):
    v = x.values
    if how == "sigma":
        return v.std()
    return abs(np.mean(np.sort(v)[:max(1, int(0.10 * len(v)))]))       # cvar 10%


def w_of(L, how, budget=0.03):
    r = pd.Series({k: risk_of(L[k], how) for k in SIX})
    return (1 / r) / (1 / r).sum() * budget


def stream(L, w, scale=1.0):
    st = max(L[k].index.min() for k in SIX); en = min(L[k].index.max() for k in SIX)
    return pd.concat([pd.Series(L[k][(L[k].index >= st) & (L[k].index <= en)].values * w[k] * scale,
                                index=L[k][(L[k].index >= st) & (L[k].index <= en)].index)
                      for k in SIX]).sort_index()


def cd(v, days):
    eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
    return (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100, ((pk - eq) / pk).max() * 100


def boot_dd(s, nb=250, k=3):
    mk = s.index.to_period("M"); months = sorted(mk.unique())
    by = {m: s.values[mk == m] for m in months}
    nm = len(months); nblk = int(np.ceil(nm / k)); days = (s.index[-1] - s.index[0]).days
    return float(np.median([cd(np.concatenate([by[months[(b + j) % nm]]
                                               for b in RNG.integers(0, nm, nblk)
                                               for j in range(k)])[:len(s)], days)[1]
                            for _ in range(nb)]))


def eq_cagr(L, how, D0):
    w = w_of(L, how); lo, hi = 0.15, 4.0
    for _ in range(20):
        m = (lo + hi) / 2
        if boot_dd(stream(L, w, m)) > D0:
            hi = m
        else:
            lo = m
    s = stream(L, w, lo)
    return cd(s.values, (s.index[-1] - s.index[0]).days)[0], lo, w['btc15m_L']


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    cfg = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "rr": 4.5,
           "fill_win": 200, "fwd": 500}

    def L15(fwd=500, mp=1, per=1.0):
        """btc15m_L の変種。per = 1本あたりのサイズ（枠を増やすときは 1/mp にして総エクスポージャを保つ）。"""
        with contextlib.redirect_stderr(io.StringIO()):
            t = run(d15, SimpleNamespace(**{**cfg, "fwd": fwd, "max_pos": mp}))
        ii = d15.index.get_indexer(t["time"])
        w = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
        rk = t["risk"].values / w
        R = (t["R"].values * w - 15.0 / rk
             - (BTC_PCT_YR / 365.0 / 100.0) * (t["e_px"].values / rk) * t["hold"].values)
        return pd.Series(R * per, index=pd.DatetimeIndex(t["time"]))

    B0 = {k: leg(k)[0] for k in SIX}
    D0 = boot_dd(stream(B0, w_of(B0, "sigma")))
    print(f"基準 maxDD = {D0:.2f}%（現行ブック・巡回ブロック3か月の中央値）")
    print("全ての行をこの maxDD にそろえて CAGR で比べる（レバレッジを完全に排除）\n")

    V = {"現行 (1枠・打切り無し)":          L15(),
         "A  1日で強制決済 (1枠)":          L15(fwd=96),
         "B  2枠（各1/2サイズ）":           L15(mp=2, per=0.5),
         "B  3枠（各1/3サイズ）":           L15(mp=3, per=1 / 3),
         "B  無制限（各1/2サイズ）":         L15(mp=99, per=0.5),
         "A+B 1日で切る＋2枠(各1/2)":       L15(fwd=96, mp=2, per=0.5)}

    print(f"  {'btc15m_L の枠と出口':<26}{'n':>6}{'totR':>8}{'σ':>7}{'歪度':>7}"
          f"{'σ重み: CAGR':>14}{'差':>8}{'cvar重み: CAGR':>17}{'差':>8}")
    b_s = b_c = None
    for nm, s in V.items():
        L = dict(B0); L["btc15m_L"] = s
        cs, _, ws = eq_cagr(L, "sigma", D0)
        cc, _, wc = eq_cagr(L, "cvar", D0)
        if b_s is None:
            b_s, b_c = cs, cc
        v = s.values
        sk = ((v - v.mean()) ** 3).mean() / max(v.std() ** 3, 1e-9)
        print(f"  {nm:<26}{len(s):>6}{s.sum():>+8.0f}{v.std():>7.2f}{sk:>+7.2f}"
              f"{cs:>+13.1f}%{cs-b_s:>+7.1f}pt{cc:>+16.1f}%{cc-b_c:>+7.1f}pt"
              + ("  ← 現行" if b_s == cs else ""))

    print("\n※ 「各1/2サイズ」= 2枠同時に持っても口座のエクスポージャは1枠と同じ（比較を公平にするため）。")
    print("   totR はそのサイズでの口座R。σ重み＝現行、cvar重み＝下方だけを見る（右の裾を切っても得しない）。")


if __name__ == "__main__":
    main()
