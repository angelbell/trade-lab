"""Did the "~90% of totR comes from 2020/23/24" caveat ever apply -- and did today's changes fix it?

The adoption caveat on btc15m_L (docs/findings/book.md) says the leg's regime concentration is
~90% of totR from 2020/2023/2024. Measuring the CURRENT canonical leg (4h-KAMA gate, RR4.5) gives
only ~60%. Two possible reasons, and they mean opposite things:

  (a) the old caveat was simply wrong  -> nothing learned, and the doc needs a correction
  (b) the caveat was TRUE for the config it described (DAILY KAMA gate, RR4.0) and today's two
      changes (gate D->4h, RR 4.0->4.5) genuinely DE-concentrated the leg
      -> then the gate/RR changes bought regime-robustness, not just return, which is a much
         bigger deal than the CAGR/DD number said

So: rebuild btc15m_L in BOTH configs on the same data and measure the same concentration stats.
Run: .venv/bin/python scratchpad/btc15m_old_vs_new_concentration.py
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


def build(d15, gate_tf, rr, pdh_soft):
    t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": gate_tf,
                                    "pullback_frac": 0.3, "rr": rr}))
    R = t["R"].values - 15.0 / t["risk"].values
    idx = pd.DatetimeIndex(t["time"])
    if pdh_soft:
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(
            d15.index, method="ffill").values
        ei = d15.index.get_indexer(t["time"])
        R = R * np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
    return pd.Series(R, index=idx)


def stats(s, tag):
    yr = s.groupby(s.index.year).sum()
    yr_full = yr[(yr.index >= 2019) & (yr.index <= 2025)]      # complete years only
    tot = yr_full.sum()
    top3 = yr_full.sort_values(ascending=False).head(3)
    trend = [y for y in (2020, 2023, 2024) if y in yr_full.index]
    trend_share = yr_full[trend].sum() / tot * 100
    rest = s[~s.index.year.isin(trend)]
    rest = rest[(rest.index.year >= 2019) & (rest.index.year <= 2025)]
    pf = rest[rest > 0].sum() / abs(rest[rest <= 0].sum())
    print(f"\n--- {tag} ---   n={len(s)}  totR(2019-25)={tot:+.1f}")
    print(f"    2020+2023+2024 のシェア : {trend_share:.0f}%")
    print(f"    上位3年({', '.join(str(y) for y in top3.index)}) のシェア : "
          f"{top3.sum()/tot*100:.0f}%")
    print(f"    その3年を抜いた残り     : n={len(rest)}  PF={pf:.2f}  "
          f"meanR={rest.mean():+.3f}  totR={rest.sum():+.1f}")
    print("    年別 totR: " + "  ".join(f"{y}:{v:+.0f}" for y, v in yr_full.items()))


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    print("btc15m_L: 旧構成（日足KAMAゲート・RR4.0）vs 現構成（4hKAMAゲート・RR4.5）")
    print("※ どちらも押し目指値0.3・PDHソフト0.5・コスト$15。完全な年（2019-2025）のみで集計。")
    stats(build(d15, "1D", 4.0, True), "旧: 日足ゲート・RR4.0（採用前の留保が書かれた構成）")
    stats(build(d15, "240min", 4.5, True), "現: 4hゲート・RR4.5（今日のPine v2）")
    stats(build(d15, "240min", 4.0, True), "中間: 4hゲート・RR4.0（ゲートだけ変えた）")
    stats(build(d15, "1D", 4.5, True), "中間: 日足ゲート・RR4.5（RRだけ変えた）")


if __name__ == "__main__":
    main()
