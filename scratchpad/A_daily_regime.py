"""The user's read of 2026: BTC is in a DAILY downtrend, so a 15m long is structurally counter-trend,
and the up-legs are short -- so they take profit early instead of letting it run.

btc15m_A's only regime gate is the 4h KAMA. It never looks at the daily. In a daily downtrend that
fast gate switches on during BOUNCES, and the leg buys counter-trend rallies. 2026 is exactly that:
14 trades, PF 0.32, -8.1%.

Structural law 9 says trend legs don't age, so a FAR fixed target is always right -- but that law was
derived by stratifying on five measures of ENTRY STRENGTH (PDH, 4h new high, ER). The DAILY TREND
DIRECTION was never one of them, and it is not a strength measure -- it is a REGIME, and law 3 says
regime is the biggest lever. So this is genuinely untested.

Follow the ledger's mandated order: measure the RAW EXCURSION first (stop only, no target), stratified
by the daily state. Only then talk about RR. And check the daily-down trades are spread across the
years -- if they are all 2026, this is curve-fitting to the current drawdown.
Run: .venv/bin/python scratchpad/A_daily_regime.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
import pandas_ta as ta
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample, kama_adaptive
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "fill_win": 200}


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    dly = d15.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last"}).dropna()
    # 日足の状態はすべて【確定足】から。shift(1) して15分足へ ffill。
    st = {}
    k = kama_adaptive(dly["close"], 14)
    st["D1 日足KAMA(14)↑"] = (k > k.shift(1)).shift(1)
    st["D2 終値>日足SMA50"] = (dly["close"] > dly["close"].rolling(50).mean()).shift(1)
    st["D3 終値>日足SMA150"] = (dly["close"] > dly["close"].rolling(150).mean()).shift(1)
    st = {kk: v.reindex(d15.index, method="ffill") for kk, v in st.items()}

    # 目標なし（損切りのみ）で走らせた素の巡行幅 -- 台帳の規律: RR の話はこの後
    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d15, SimpleNamespace(**{**CFG, "rr": 100.0, "fwd": 500}))   # rr=100 ≒ 目標なし
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])
    ab = t["e_px"].values > pdh[ei]
    e = t["e_px"].values[ab]; risk = t["risk"].values[ab]
    ti = pd.DatetimeIndex(t["time"])[ab]
    idx = d15.index.get_indexer(ti)
    stop = e - risk
    h, l = d15["high"].values, d15["low"].values
    MFE = np.full(len(e), np.nan)
    for i in range(len(e)):
        j0 = idx[i]
        mx = -np.inf
        for j in range(j0 + 1, min(j0 + 501, len(h))):
            mx = max(mx, h[j])
            if l[j] <= stop[i]:
                break
        MFE[i] = (mx - e[i]) / risk[i]                 # 損切りに当たるまでの最大順行幅（R単位）

    print(f"素の巡行幅（目標なし・損切りのみ）  n={len(MFE)}   btc15m_A の全トレード\n")
    print("1. 日足の状態で層別。**まず巡行幅。RR の話はその後**（台帳の規律）\n")
    for name, s in st.items():
        up = s.values[idx]
        print(f"  {name}")
        print(f"    {'':<10}{'n':>5}{'年別の散らばり':>16}{'MFE 中央値':>12}{'MFE 平均':>10}"
              f"{'P(MFE>=4.5R)':>14}{'P(>=3R)':>10}{'P(>=2R)':>10}{'P(>=1R)':>10}")
        for lab, m in (("日足↑", up == True), ("日足↓", up == False)):
            if m.sum() < 5:
                continue
            x = MFE[m]
            yrs_ = pd.Series(ti[m]).dt.year.value_counts().sort_index()
            spread = "/".join(f"{v}" for v in yrs_.values)
            print(f"    {lab:<10}{m.sum():>5}{spread:>16}{np.median(x):>12.2f}{x.mean():>10.2f}"
                  f"{100*np.mean(x>=4.5):>13.0f}%{100*np.mean(x>=3):>9.0f}%"
                  f"{100*np.mean(x>=2):>9.0f}%{100*np.mean(x>=1):>9.0f}%")
        print(f"      年の内訳（日足↓の n を年別に）: "
              + "  ".join(f"{y}:{v}" for y, v in
                          pd.Series(ti[up == False]).dt.year.value_counts().sort_index().items()))
        print()

    print("\n2. 各層で、RR ごとの期待値を直接出す（巡行幅から逆算。コスト込み）")
    print("   ★ 目標 RR で勝ち = MFE >= RR。負け = −1R。\n")
    cost_R = 15.0 / risk
    for name, s in st.items():
        up = s.values[idx]
        print(f"  {name}")
        print(f"    {'RR':>5}", end="")
        for rr in (1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0):
            print(f"{rr:>8.1f}", end="")
        print()
        for lab, m in (("日足↑ meanR", up == True), ("日足↓ meanR", up == False)):
            if m.sum() < 5:
                continue
            print(f"    {lab:<12}", end="")
            best, bestrr = -9, 0
            for rr in (1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0):
                win = MFE[m] >= rr
                R = np.where(win, rr, -1.0) - cost_R[m]
                if R.mean() > best:
                    best, bestrr = R.mean(), rr
                print(f"{R.mean():>+8.2f}", end="")
            print(f"    ← 最適 RR = **{bestrr}**")
        print()


if __name__ == "__main__":
    main()
