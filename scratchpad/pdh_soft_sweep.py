"""The PDH soft-size factor (0.5) has never been swept.

Today's autopsy of btc15m_L: the breaks that happen ABOVE the previous day's high (full size) carry
the leg -- n=208, PF 2.67, meanR +1.228. The breaks INSIDE the previous day's range (half size) are
70% of the leg and earn almost nothing -- n=534, PF ~1.17, meanR +0.078 at half size.

The 0.5 was assumed, not derived. Structural law 11 says the with-drift leg should "size down, not
skip" -- but it never said 0.5. The autopsy now gives a reason to ask: is 0.5 too big (the group
barely earns) or too small (the group is 70% of the frequency and carries the book's decorrelation)?

One lever. Sweep w in {0 = skip, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0 = no rule at all} and let the BOOK
decide. Report N and totR/yr next to PF, because a factor that shrinks the bet also shrinks the leg.
Run: .venv/bin/python scratchpad/pdh_soft_sweep.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import w_trade

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260714)
NDRAW = 2000


def book_of(legs):
    w = w_trade(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    s = pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                             index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                   for k in SIX]).sort_index()
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
    return cagr, dd, cagr / max(dd, 1e-9)


def cdd_v(v, days):
    eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
    return (eq[-1] ** (365.25 / max(days, 1)) - 1) / max(((pk - eq) / pk).max(), 1e-9)


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    above = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    Rraw = t["R"].values - 15.0 / t["risk"].values           # コスト込み・サイズ倍率なし
    idx = pd.DatetimeIndex(t["time"])
    r0 = book_of(legs0)[2]
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    yrs = (idx[-1] - idx[0]).days / 365.25
    print(f"現行ブック = {r0:.2f}   （前日高値の上でブレイク = {above.sum()}本 / 中 = {(~above).sum()}本）\n")
    print(f"  {'半分サイズの倍率':>16}{'n':>6}{'年本数':>8}{'PF':>7}{'meanR':>9}{'totR/年':>9}"
          f"{'σ(R)':>8}{'ブックCAGR':>11}{'ブックDD':>9}{'CAGR/DD':>9}{'差':>8}")
    arms = {}
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        f = np.where(above, 1.0, w)
        keep = f > 0
        s = pd.Series(Rraw[keep] * f[keep], index=idx[keep])
        legs = dict(legs0); legs["btc15m_L"] = s
        c, d, rb = book_of(legs)
        tag = ("0.0（見送る）" if w == 0 else "1.0（ルール無効）" if w == 1 else f"{w}") + \
              ("  ← 現行" if w == 0.5 else "")
        print(f"  {tag:>16}{len(s):>6}{len(s)/yrs:>8.0f}{pf(s.values):>7.2f}{s.mean():>+9.3f}"
              f"{s.sum()/yrs:>+9.1f}{s.std():>8.3f}{c:>10.1f}%{d:>8.2f}%{rb:>9.2f}{rb-r0:>+8.2f}"
              + ("  ★" if rb > r0 + 0.05 else ""))
        arms[w] = s

    print("\n  ブックの巡回ブロック・ブートストラップ（本物なら P はブロックとともに上がる）")
    base = arms[0.5]
    def series(s):
        legs = dict(legs0); legs["btc15m_L"] = s
        w = w_trade(legs, SIX)
        st = max(legs[k].index.min() for k in SIX); en = min(legs[k].index.max() for k in SIX)
        return pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                                    index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                          for k in SIX]).sort_index()
    b = series(base); bm = b.index.to_period("M"); months = sorted(set(bm)); M = len(months)
    b_by = {m: b[bm == m].values for m in months}
    print(f"  {'倍率':>8}{'ブック':>9}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    for w in (0.25, 0.4, 0.6, 0.75):
        a = series(arms[w]); am = a.index.to_period("M")
        a_by = {m: a[am == m].values for m in months}
        row = f"  {w:>8}{book_of({**legs0, 'btc15m_L': arms[w]})[2]:>9.2f}"
        for Lb in (1, 3, 6, 12):
            nb = int(np.ceil(M / Lb)); wins = 0
            for _ in range(NDRAW):
                st = RNG.integers(0, M, nb)
                order = np.concatenate([(np.arange(s2, s2 + Lb) % M) for s2 in st])[:M]
                bb = np.concatenate([b_by[months[i]] for i in order if len(b_by[months[i]])])
                aa = np.concatenate([a_by[months[i]] for i in order if len(a_by[months[i]])])
                if len(bb) < 20 or len(aa) < 20:
                    continue
                wins += cdd_v(aa, 365 * M / 12) > cdd_v(bb, 365 * M / 12)
            row += f"{100*wins/NDRAW:>7.0f}%"
        print(row)


if __name__ == "__main__":
    main()


def leverage_check():
    """Shrinking the soft trades lowers sigma(R) -> inv-vol hands btc15m_L a BIGGER weight.
    Pure leverage on this leg already reaches 8.13-8.22. So the whole 'gain' could be the dial.
    Pin every weight at its current value and re-measure: what survives is the sizing rule itself."""
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    w0 = w_trade(legs0, SIX)
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    above = t["e_px"].values > pdh[d15.index.get_indexer(t["time"])]
    Rraw = t["R"].values - 15.0 / t["risk"].values
    idx = pd.DatetimeIndex(t["time"])

    def book_pinned(legs, w):
        st = max(legs[k].index.min() for k in SIX); en = min(legs[k].index.max() for k in SIX)
        s = pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                                 index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                       for k in SIX]).sort_index()
        eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
        return cagr / max(dd, 1e-9)

    r0 = book_of(legs0)[2]
    print("\n\n【レバレッジ検定】重みを現行に固定して、サイズ倍率だけを動かす")
    print("   差が消えたら、これはただのレバレッジ・ダイヤル。\n")
    print(f"  {'倍率':>8}{'σ(R)':>8}{'重み(再計算)':>13}{'ブック(重み再計算)':>18}{'ブック(重み固定)':>17}")
    for w in (0.25, 0.4, 0.5, 0.6, 0.75):
        f = np.where(above, 1.0, w)
        s = pd.Series(Rraw * f, index=idx)
        legs = dict(legs0); legs["btc15m_L"] = s
        wn = w_trade(legs, SIX)
        print(f"  {w:>8}{s.std():>8.3f}{100*wn['btc15m_L']:>12.3f}%"
              f"{book_of(legs)[2]:>18.2f}{book_pinned(legs, w0):>17.2f}"
              + ("  ← 現行" if w == 0.5 else ""))
    print(f"\n  比較: 何も変えずに btc15m_L の重みだけ上げたら（純粋なレバレッジ）")
    for m in (1.05, 1.10, 1.12, 1.15, 1.20):
        w = w0.copy(); w["btc15m_L"] *= m
        print(f"    重み ×{m:.2f}  →  ブック {book_pinned(legs0, w):.2f}")
    print(f"\n  現行（倍率0.5・重み再計算）= {r0:.2f}")


leverage_check()
