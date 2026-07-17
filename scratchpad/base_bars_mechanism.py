"""Is the real variable "how many bars did the base take", not "how big is the wave in %"?

A 2%-of-price wave built over 50 bars is an orderly consolidation. The same 2% built over 5 bars is
a spike. That story would explain everything the stop/price filter does AND why it does not
decompose into ATR: base LENGTH is orthogonal to volatility.

It makes a hard, falsifiable PREDICTION -- the one the liquidation-cascade story failed:
    if base_bars is the true variable, filtering on it should work on the legs where stop/price
    FAILED (btc15m_S, gold15m, gold_bo), because those legs also have spiky bases.
If base_bars only works on btc15m_L, it is the same single-path artifact wearing a new costume,
and the honest conclusion is that the filter has no mechanism.

Also: is stop/price just a PROXY for base_bars? Correlate them, and control one for the other.
Run: .venv/bin/python scratchpad/base_bars_mechanism.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from radar_gate_race import BASE
from short_mirror_15m import invert
from book_spec_fix import w_trade

ROOT = "/home/angelbell/dev/auto-trade"
SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]


def legs_with_bars():
    """Every leg, plus (stop/price %, base_bars) per trade."""
    out, extra = {}, {}
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        t = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                       "daily_sma": 150, "daily_slope_k": 10}))
        out["gold_bo"] = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))
        extra["gold_bo"] = (100 * t["risk"].values / t["e_px"].values, t["base_bars"].values)

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        t = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0,
                                                     "fwd": 300})))
        out["btc_bo_kama"] = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))
        extra["btc_bo_kama"] = (100 * t["risk"].values / t["e_px"].values, t["base_bars"].values)

        t = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0))
        out["btc_pull"] = pd.Series(t["R"].values, index=pd.DatetimeIndex(t["time"]))

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                        "pullback_frac": 0.25, "fill_win": 200}))
        out["gold15m"] = pd.Series(t["R"].values - 0.30 / t["risk"].values,
                                   index=pd.DatetimeIndex(t["time"]))
        extra["gold15m"] = (100 * t["risk"].values / t["e_px"].values, t["base_bars"].values)

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        w = np.where(t["e_px"].values > pdh[d15.index.get_indexer(t["time"])], 1.0, 0.5)
        risk = t["risk"].values / w
        out["btc15m_L"] = pd.Series(t["R"].values * w - 15.0 / risk,
                                    index=pd.DatetimeIndex(t["time"]))
        extra["btc15m_L"] = (100 * risk / t["e_px"].values, t["base_bars"].values)

        inv = invert(d15); C = 2 * d15["high"].max()
        t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": 4.5,
                                        "fill_win": 200}))
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        m = (C - t["e_px"].values) < pdl[d15.index.get_indexer(t["time"])]
        px_s = C - t["e_px"].values
        out["btc15m_S"] = pd.Series(t["R"].values[m] - 15.0 / t["risk"].values[m],
                                    index=pd.DatetimeIndex(t["time"])[m])
        extra["btc15m_S"] = (100 * t["risk"].values[m] / px_s[m], t["base_bars"].values[m])
    return out, extra


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
    return cagr / max(dd, 1e-9)


def main():
    legs0, X = legs_with_bars()
    r0 = book_of(legs0)
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    print(f"現行ブック = {r0:.2f}\n")
    print("1. 「波の形成本数」と「損切り/価格」は、そもそも同じものか？（相関）\n")
    print(f"  {'leg':<14}{'形成本数 中央値':>16}{'10/90%点':>14}{'相関(本数 vs 損切り/価格)':>26}")
    for k in ["btc15m_L", "btc15m_S", "gold15m", "gold_bo", "btc_bo_kama"]:
        sp, bb = X[k]
        c = np.corrcoef(bb, sp)[0, 1]
        print(f"  {k:<14}{np.median(bb):>16.0f}{f'{np.percentile(bb,10):.0f} / {np.percentile(bb,90):.0f}':>14}"
              f"{c:>26.2f}")
    print("  → 相関が弱ければ別物。強ければ『損切り/価格』は形成本数の代理変数。\n")

    print("\n2. 事前登録の予言: 形成本数が真の変数なら、**損切り/価格が効かなかった脚にも効く**")
    print("   （btc15m_S / gold15m / gold_bo。btc15m_L だけなら、また同じ単一経路の当てはめ）\n")
    for k in ["btc15m_L", "btc15m_S", "gold15m", "gold_bo"]:
        s = legs0[k]
        sp, bb = X[k]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        print(f"  === {k}   （速い＝スパイクを捨てる。本数が少ないものを切る）")
        print(f"      {'閾値':>12}{'n':>6}{'残す率':>8}{'PF':>7}{'totR/年':>9}{'ブック':>9}{'差':>8}")
        print(f"      {'全部':>12}{len(s):>6}{'100%':>8}{pf(s.values):>7.2f}"
              f"{s.sum()/yrs:>+9.1f}{r0:>9.2f}{0.0:>+8.2f}")
        for cutpct in (10, 20, 30, 40):
            thr = np.percentile(bb, cutpct)          # 下位 cutpct%（＝速い波）を捨てる
            m = bb >= thr
            lg = dict(legs0); lg[k] = s[m]
            rb = book_of(lg)
            print(f"      {f'>= {thr:.0f}本':>12}{m.sum():>6}{100-cutpct:>7}%{pf(s.values[m]):>7.2f}"
                  f"{s.values[m].sum()/yrs:>+9.1f}{rb:>9.2f}{rb-r0:>+8.2f}"
                  + ("  ★" if rb > r0 + 0.05 else ""))
        print()

    print("\n3. 形成本数で層別したときの、素の成績（機構が実在するなら、遅い波ほど強いはず）\n")
    for k in ["btc15m_L", "btc15m_S", "gold15m"]:
        s = legs0[k]; sp, bb = X[k]
        q = pd.qcut(bb, 4, labels=False, duplicates="drop")
        print(f"  {k}")
        print(f"    {'帯':<6}{'形成本数(中央値)':>16}{'n':>6}{'勝率':>8}{'PF':>7}{'meanR':>9}")
        for i in sorted(set(q)):
            m = q == i
            print(f"    Q{i+1:<5}{np.median(bb[m]):>16.0f}{m.sum():>6}"
                  f"{100*(s.values[m]>0).mean():>7.1f}%{pf(s.values[m]):>7.2f}{s.values[m].mean():>+9.3f}")
        print()


if __name__ == "__main__":
    main()
