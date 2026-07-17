"""How big an account does btc15m_A need, once Vantage's 0.01-lot minimum is enforced?

The book's weights assume free position sizing. Live, the smallest trade you can place already risks
a fixed number of yen, and if that exceeds your intended risk you are FORCED to over-bet -- or to skip
the trade, which changes the strategy.

    risk_per_min_lot = stop_distance($/BTC) x 0.01 BTC x USDJPY

The daily x0.75 rule makes it worse: to size 0.75 of a 0.01-lot position you would need 0.0075 lots,
which does not exist. On a small account that rule quietly rounds back to full size.

⚠️ CONTRACT SIZE: BTCUSD 1 lot = 1 BTC is the standard assumption. CLAUDE.md flags it as UNVERIFIED.
Check it in the terminal -- every number below scales linearly with it.
Run: .venv/bin/python scratchpad/A_account_size.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE

ROOT = "/home/angelbell/dev/auto-trade"
USDJPY = 155.0
BTC_PER_LOT = 1.0          # ⚠️ 未確認。ターミナルで要確認
MIN_LOT, STEP = 0.01, 0.01
CFG = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "fill_win": 200,
       "rr": 4.5, "fwd": 500}


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**CFG))
    dly = d15["close"].resample("1D").last().dropna()
    upD = (dly > dly.rolling(150).mean()).shift(1).reindex(d15.index, method="ffill")
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])
    ab = t["e_px"].values > pdh[ei]
    risk = t["risk"].values[ab]                     # 約定値から損切りまでの $/BTC
    e = t["e_px"].values[ab]
    W = np.where(upD.values[ei][ab] == True, 1.0, 0.75)
    R = (t["R"].values - 15.0 / t["risk"].values)[ab] * W
    ti = pd.DatetimeIndex(t["time"])[ab]

    print("⚠️ 前提: BTCUSD 1ロット = 1 BTC（＝最小0.01ロット = 0.01 BTC）。**ターミナルで要確認。**")
    print(f"   USDJPY = {USDJPY}。すべての数字はこの2つに比例します。\n")
    for lab, m in (("全期間", np.ones(len(risk), bool)), ("2025年以降（＝今の価格帯）", ti >= "2025")):
        r = risk[m]; px = e[m]
        print(f"  {lab}:  n={m.sum()}   BTC価格 中央値 ${np.median(px):,.0f}")
        print(f"    損切り幅 ($/BTC):  中央値 ${np.median(r):,.0f}   "
              f"25/75%点 ${np.percentile(r,25):,.0f} / ${np.percentile(r,75):,.0f}   "
              f"90%点 ${np.percentile(r,90):,.0f}")
        jpy = r * MIN_LOT * BTC_PER_LOT * USDJPY
        print(f"    最小ロット(0.01)のリスク:  中央値 {np.median(jpy):,.0f}円   "
              f"25/75%点 {np.percentile(jpy,25):,.0f} / {np.percentile(jpy,75):,.0f}円   "
              f"90%点 {np.percentile(jpy,90):,.0f}円\n")

    m25 = ti >= "2025"
    jpy = risk[m25] * MIN_LOT * BTC_PER_LOT * USDJPY
    print("  口座サイズ別: 最小ロットが「口座の何%」のリスクになるか（2025年以降の損切り幅で測定）\n")
    print(f"  {'口座':<12}{'最小ロットのリスク(中央値)':>24}{'狙い0.5%に対して':>18}"
          f"{'狙い1%に対して':>16}{'1%を超える率':>14}{'判定':>22}")
    for acct in (100_000, 300_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000):
        pct = 100 * jpy / acct
        med = np.median(pct)
        over1 = 100 * np.mean(pct > 1.0)
        if med <= 0.5:
            v = "0.5%運用も可"
        elif med <= 1.0:
            v = "1%運用のみ可"
        elif med <= 2.0:
            v = "過大リスク"
        else:
            v = "建てられない"
        print(f"  {acct:>10,}円{np.median(jpy):>19,.0f}円{med/0.5:>16.1f}倍{med/1.0:>14.1f}倍"
              f"{over1:>13.0f}%{v:>22}")

    print("\n  ★ 日足×0.75 のルールは、最小ロットの口座では**丸め込まれて消える**")
    print("    （0.01ロットの0.75倍 = 0.0075ロット は存在しない → フルサイズに戻る）\n")
    print(f"  {'口座':<12}{'狙い1%で建てられるロット(中央値)':>30}{'×0.75 が表現できるか':>22}")
    for acct in (300_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000):
        want = acct * 0.01
        lots = want / (np.median(risk[m25]) * BTC_PER_LOT * USDJPY)
        ok = "できる" if lots >= 0.04 else ("粗い" if lots >= 0.02 else "**できない（フルサイズに戻る）**")
        print(f"  {acct:>10,}円{lots:>26.3f} ロット{ok:>22}")

    print("\n\n  実際に回したシミュレーション（MT5 忠実: 0.01ロット刻みで切り上げ、最小0.01）")
    print("  2019年から複利。※当時は BTC $8千なので損切りも小さく、今より小さい口座で建てられた点に注意\n")
    print(f"  {'開始資金':<12}{'狙いリスク':>10}{'建てた':>7}{'見送り':>7}{'実リスク中央値':>15}"
          f"{'最終資金':>16}{'maxDD':>9}")
    for start in (300_000, 1_000_000, 3_000_000):
        for tgt in (0.005, 0.01):
            eq = float(start); curve = []; taken = skipped = 0; ar = []
            for i in range(len(risk)):
                per_lot = risk[i] * BTC_PER_LOT * USDJPY          # 1ロットあたりの損失額(円)
                want = eq * tgt * W[i]                            # 日足サイズ込みの狙い
                lots = max(np.ceil(want / per_lot / STEP) * STEP, MIN_LOT)
                risk_jpy = lots * per_lot
                if risk_jpy / eq > 0.03:                          # 最小ロットでも3%超 → 見送る
                    skipped += 1; continue
                eq += (R[i] / W[i]) * risk_jpy                    # R は W 込みなので割り戻す
                if eq <= 0:
                    eq = 1.0; curve.append(eq); break
                ar.append(100 * risk_jpy / eq); curve.append(eq); taken += 1
            c = np.array(curve); pk = np.maximum.accumulate(c)
            dd = ((pk - c) / pk).max() * 100 if len(c) else 0
            print(f"  {start:>10,}円{100*tgt:>9.1f}%{taken:>7}{skipped:>7}"
                  f"{np.median(ar) if ar else 0:>14.2f}%{c[-1]:>15,.0f}円{dd:>8.1f}%")


if __name__ == "__main__":
    main()
