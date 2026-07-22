"""The 1-day cut on btc15m_L survives the PAIRED arbiter under all five weightings (+2.4 to +3.7 CAGR
pt at equal bootstrapped-median maxDD, P 81-92% at 12-month blocks). So it is not the sigma artifact I
called it. But the decomposition that said "the time stop itself is destructive" (book 4.84 -> 4.04)
was measured on the SAME broken single-path arbiter, so it has to be redone before anything is
believed. The two claims cannot both stand:

  claim 1  cutting the SAME 763 trades at 1 day costs -72.4R of amputated winners (law 4/9), and
           trades still alive at 1 day are worth +1.59R if simply held -- both re-verified, both
           bit-exact against the canonical leg.
  claim 2  the arm with the cut is worth +3.5 CAGR pt at equal drawdown.

If both survive the paired arbiter, then the cut destroys R and still compounds better -- which can
only mean the gain is a POSITION-SIZING effect (more, smaller, smoother bets at a fixed drawdown),
not an exit edge. That is a real thing, but it is a different claim, and it would mean the same money
is available by simply betting more per trade WITHOUT cutting anything. So the decisive control is
the leverage dial: take the base leg and just scale its risk up until its drawdown matches. If the
cut's advantage survives THAT, it is genuinely adding something.

Arms (btc15m_L only; the other five legs are untouched):
  base        763 trades, hold to target/stop
  A  cut      the SAME 763 entries, force-closed at bar+96, slot occupancy unchanged (re-walk)
  A+B run     fwd=96 -- cut AND the freed slot refills, n = 1002 (this is the +3.5pt arm)
  B  slots    max_pos=2 at half size each (same peak exposure), no cut, n = 1307
  dial        base leg, risk scaled 1.25x / 1.5x -- the "just bet more" null
Run: .venv/bin/python experiments/cut1d_paired.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from arb_common import Boot, months_union, cd, BUDGET
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from rr_with_swap import leg, SIX

ROOT = "/home/angelbell/dev/auto-trade"
BTC_PCT_YR = 30.0
PF, RR = 0.3, 4.5
RR_REAL = (RR + PF) / (1.0 - PF)


def cvar(v, q=0.10):
    return abs(np.mean(np.sort(v)[:max(1, int(q * len(v)))]))


def raw(L, how):
    o = {}
    for k in SIX:
        v = L[k].values
        o[k] = 1.0 if len(v) < 5 else (1.0 / max(v.std(), 1e-9) if how == "sigma"
                                       else 1.0 / max(cvar(v), 1e-9))
    r = pd.Series(o)
    return r / r.sum() * BUDGET


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    cfg = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": PF, "rr": RR,
           "fill_win": 200, "fwd": 500}
    hi, lo_, cl = d15["high"].values, d15["low"].values, d15["close"].values
    day = d15.index.values.astype("datetime64[s]").astype(np.int64) / 86400.0

    def leg15(fwd=500, mp=1, per=1.0):
        with contextlib.redirect_stderr(io.StringIO()):
            t = run(d15, SimpleNamespace(**{**cfg, "fwd": fwd, "max_pos": mp}))
        ii = d15.index.get_indexer(t["time"])
        w = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
        rk = t["risk"].values / w
        R = (t["R"].values * w - 15.0 / rk
             - (BTC_PCT_YR / 365.0 / 100.0) * (t["e_px"].values / rk) * t["hold"].values)
        return pd.Series(R * per, index=pd.DatetimeIndex(t["time"]))

    def cut_same_entries(cap):
        """同じ 763 本の入口。cap 本で強制決済。枠の占有は変えない（＝入口は1本も増えない）。"""
        with contextlib.redirect_stderr(io.StringIO()):
            t = run(d15, SimpleNamespace(**cfg))
        ii = d15.index.get_indexer(t["time"])
        w = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
        e, rk = t["e_px"].values, t["risk"].values
        stop, tgt = e - rk, e + RR_REAL * rk
        cost = 15.0 * w / rk
        swd = (BTC_PCT_YR / 365.0 / 100.0) * (e / rk) * w
        out = np.empty(len(e))
        for i in range(len(e)):
            j0 = ii[i]; lim = min(j0 + min(cap, 500), len(cl) - 1)
            if lo_[j0] <= stop[i]:
                r, jj = -1.0, j0
            else:
                r = None
                for j in range(j0 + 1, lim + 1):
                    if lo_[j] <= stop[i]: r, jj = -1.0, j; break
                    if hi[j] >= tgt[i]:   r, jj = RR_REAL, j; break
                if r is None:
                    jj = lim; r = (cl[jj] - e[i]) / rk[i]
            out[i] = r * w[i] - cost[i] - swd[i] * (day[jj] - day[j0])
        return pd.Series(out, index=pd.DatetimeIndex(t["time"]))

    B0 = {k: leg(k)[0] for k in SIX}
    ARMS = {
        "現行（打切り無し）":            leg15(),
        "A  同じ763本を1日で切るだけ":   cut_same_entries(96),
        "A+B 1日で切る（枠が空く）":      leg15(fwd=96),
        "B  2枠（各1/2・切らない）":      leg15(mp=2, per=0.5),
        "ダイヤル 現行×1.25倍":          leg15() * 1.25,
        "ダイヤル 現行×1.50倍":          leg15() * 1.50,
    }
    st = max(B0[k].index.min() for k in SIX); en = min(B0[k].index.max() for k in SIX)
    B0 = {k: B0[k][(B0[k].index >= st) & (B0[k].index <= en)] for k in SIX}
    ARMS = {k: v[(v.index >= st) & (v.index <= en)] for k, v in ARMS.items()}
    yrs = sorted({y for k in SIX for y in B0[k].index.year}); first = yrs[0] + 2

    def mix(s15, how):
        L = dict(B0); L["btc15m_L"] = s15
        by = {}
        for y in yrs:
            past = {k: L[k][L[k].index.year < y] for k in SIX}
            by[y] = raw(past, how) if (y >= first and min(len(past[k]) for k in SIX) >= 5) \
                else pd.Series(BUDGET / len(SIX), index=SIX)      # 履歴が足りない最初の2年は等分
        return pd.concat([pd.Series(L[k].values * np.array([by[y][k] for y in L[k].index.year]),
                                    index=L[k].index) for k in SIX]).sort_index()

    S = {(nm, h): mix(s, h) for nm, s in ARMS.items() for h in ("sigma", "cvar")}
    bt = Boot(months_union(*S.values()), nb=1000, k=3)
    b12 = Boot(bt.months, nb=800, k=12)
    D0 = bt.dd_median(S[("現行（打切り無し）", "sigma")])
    print(f"基準 maxDD = {D0:.2f}%（現行・σ重み・WF・巡回ブロック3か月・中央値）")
    print("全アームを同じ1000経路で同DDに揃えて CAGR。重みはウォークフォワード。\n")
    print(f"  {'btc15m_L の腕':<26}{'n':>6}{'totR':>8}{'σ重み CAGR':>13}{'差':>8}"
          f"{'P(12か月)':>11}{'cvar重み CAGR':>16}{'差':>8}{'P(12か月)':>11}")
    b = {}
    for nm in ARMS:
        row = f"  {nm:<26}{len(ARMS[nm]):>6}{ARMS[nm].sum():>+8.0f}"
        for h in ("sigma", "cvar"):
            c = bt.equal_dd_cagr(S[(nm, h)], D0)[0]
            if nm == "現行（打切り無し）":
                b[h] = c
            p = 100 * np.mean(b12.ratios(S[(nm, h)]) > b12.ratios(S[("現行（打切り無し）", h)]))
            row += f"{c:>+12.1f}%{c-b[h]:>+7.1f}pt{p:>10.0f}%"
        print(row + ("  ← 現行" if nm == "現行（打切り無し）" else ""))
    print("\n  ※ totR は口座R（そのアームのサイズでの合計）。ダイヤル行は『同じ763本を、ただ大きく張る』null。")


if __name__ == "__main__":
    main()
