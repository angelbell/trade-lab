"""RETRACTION: I called btc15m_L's 1-day max hold "a leverage dial" and I was wrong. The dial I tested
scaled the leg's R series, which makes sigma bigger, which makes inv-vol shrink the weight -- while
the leg still TRADES at the scaled size. The sum of the weights stayed 3% but the risk actually taken
did not, so the "+2.8pt dial" was a leverage leak. Done honestly (move budget between legs, total
pinned at 3%), the dial is a HILL peaking at x1.25 and worth at most +0.6pt. The 1-day cut delivers
+3.8pt at essentially that same leg weight. A dial cannot buy it.

So the gain is real and it is not an exit edge and it is not a bet size. It is THROUGHPUT. btc15m_L
holds one position at a time; a trade that sits for five days blocks about a third of the leg's
opportunities. Cutting at one day gives up ~63R of amputated tail and recovers ~60R from the signals
it can now take -- same money, 30% more trades, lower variance, and at a fixed drawdown that
compounds better.

If that story is right, the shape must be a PLATEAU in the max-hold parameter, not a spike at 96 --
and it must not be reproducible by cutting at a RANDOM time with the same average hold (which would
free the slot just as often but would carry no "one day" information). Two falsifiers:

  sweep    fwd = 48 / 96 / 144 / 192 / 300 / 500 bars
  random   release the slot after a random number of bars drawn from the SAME holding distribution
           as the 96-bar arm (matched mean hold, matched trade count -- only the WHICH is random)
  peryear  a throughput gain must show up broadly, not in one era

Paired arbiter throughout: walk-forward weights, leverage fixed per arm at equal bootstrapped-median
maxDD, CAGR compared on 1000 identical resampled histories.
Run: .venv/bin/python experiments/throughput_sweep.py
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
RNG = np.random.default_rng(20260714)
BTC_PCT_YR = 30.0


class Pair(Boot):
    def cagrs(self, s):
        mk = s.index.to_period("M")
        by = {m: s.values[mk == m] for m in self.months}
        days = max((s.index[-1] - s.index[0]).days, 1)
        n = len(s)
        return np.array([cd(np.concatenate([by[self.months[j]] for j in seq])[:n], days)[0]
                         for seq in self.layout])


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    cfg = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "rr": 4.5,
           "fill_win": 200, "fwd": 500}

    def leg15(fwd=500, cap_series=None):
        a = SimpleNamespace(**{**cfg, "fwd": fwd})
        if cap_series is not None:
            a.fwd_random = cap_series
        with contextlib.redirect_stderr(io.StringIO()):
            t = run(d15, a)
        ii = d15.index.get_indexer(t["time"])
        w = np.where(t["e_px"].values > pdh[ii], 1.0, 0.5)
        rk = t["risk"].values / w
        R = (t["R"].values * w - 15.0 / rk
             - (BTC_PCT_YR / 365.0 / 100.0) * (t["e_px"].values / rk) * t["hold"].values)
        return pd.Series(R, index=pd.DatetimeIndex(t["time"]))

    B0 = {k: leg(k)[0] for k in SIX}
    ARMS = {"500本 = 5.2日（現行）": leg15(500)}
    for f in (48, 96, 144, 192, 300):
        ARMS[f"{f}本 = {f*0.25/24:.1f}日"] = leg15(f)

    st = max(B0[k].index.min() for k in SIX); en = min(B0[k].index.max() for k in SIX)
    B0 = {k: B0[k][(B0[k].index >= st) & (B0[k].index <= en)] for k in SIX}
    ARMS = {k: v[(v.index >= st) & (v.index <= en)] for k, v in ARMS.items()}
    yrs = sorted({y for k in SIX for y in B0[k].index.year}); first = yrs[0] + 2
    EQ = pd.Series(BUDGET / len(SIX), index=SIX)

    def mix(s15):
        L = dict(B0); L["btc15m_L"] = s15
        by = {}
        for y in yrs:
            past = {k: L[k][L[k].index.year < y] for k in SIX}
            if y >= first and min(len(past[k]) for k in SIX) >= 5:
                r = pd.Series({k: 1.0 / max(past[k].values.std(), 1e-9) for k in SIX})
                by[y] = r / r.sum() * BUDGET
            else:
                by[y] = EQ
        return pd.concat([pd.Series(L[k].values * np.array([by[y][k] for y in L[k].index.year]),
                                    index=L[k].index) for k in SIX]).sort_index()

    S = {k: mix(v) for k, v in ARMS.items()}
    bt = Pair(months_union(*S.values()), nb=1000, k=3)
    D0 = bt.dd_median(S["500本 = 5.2日（現行）"])
    base = bt.cagrs(S["500本 = 5.2日（現行）"] * bt.equal_dd_cagr(S["500本 = 5.2日（現行）"], D0)[1])
    print(f"基準 maxDD = {D0:.2f}%（現行・σ重み・WF）。全アームを同DDに揃え、同じ1000経路で CAGR を対比較。\n")
    print("1. 最大保有を振る（本物ならプラトー、当てはめなら 96 だけ尖る）")
    print(f"  {'最大保有':<20}{'n':>6}{'totR':>8}{'σ(R)':>8}{'CAGR中央値':>12}{'差':>9}{'P(現行に勝つ)':>15}")
    for k in ["500本 = 5.2日（現行）", "48本 = 0.5日", "96本 = 1.0日", "144本 = 1.5日",
              "192本 = 2.0日", "300本 = 3.1日"]:
        c = bt.cagrs(S[k] * bt.equal_dd_cagr(S[k], D0)[1])
        print(f"  {k:<20}{len(ARMS[k]):>6}{ARMS[k].sum():>+8.0f}{ARMS[k].std():>8.2f}"
              f"{np.median(c):>+11.1f}%{np.median(c-base):>+8.1f}pt{100*np.mean(c>base):>13.0f}%"
              + ("  ← 現行" if k.startswith("500") else ""))

    print("\n2. 年別（スループットの利得なら、一つの時代に偏らないはず）")
    s0, s1 = S["500本 = 5.2日（現行）"], S["96本 = 1.0日"]
    sc0 = bt.equal_dd_cagr(s0, D0)[1]; sc1 = bt.equal_dd_cagr(s1, D0)[1]
    print(f"  {'年':<7}{'現行':>9}{'1日で切る':>11}{'差':>9}{'本数 現行→1日':>16}")
    for y in yrs:
        a = (s0[s0.index.year == y] * sc0).sum() * 100
        b = (s1[s1.index.year == y] * sc1).sum() * 100
        na = (ARMS["500本 = 5.2日（現行）"].index.year == y).sum()
        nb = (ARMS["96本 = 1.0日"].index.year == y).sum()
        print(f"  {y:<7}{a:>+8.1f}%{b:>+10.1f}%{b-a:>+8.1f}pt{f'{na} → {nb}':>16}")


if __name__ == "__main__":
    main()
