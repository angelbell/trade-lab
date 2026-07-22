"""What can a JPY 100,000 account ACTUALLY trade, once the 0.01-lot minimum is enforced?

The book's weights assume free position sizing. On a small account they are fiction: the minimum
0.01 lot already risks a fixed number of dollars, and for the 1H/4H legs that is 4-14x the intended
weight. So instead of asking "what does the book return", ask the only question that matters here:

    simulate the account bar by bar, size every trade as MT5 would
        lot = ceil( equity * risk% / (stop_distance * contract_units) , 0.01 lot )
    with a hard floor of 0.01 lot, and let the realized R scale with the risk actually taken.

If the forced lot is bigger than the target, the trade is over-risked -- that is real, and it is
what the account would actually experience. A trade is SKIPPED only if even 0.01 lot would exceed
`max_risk` of equity (a discretionary trader would not take it).

Contract sizes: XAUUSD+ 1 lot = 100 oz (0.01 lot = 1 oz).  BTCUSD 1 lot = 1 BTC (0.01 lot = 0.01 BTC).
⚠️ The BTC contract size is the standard assumption -- CHECK IT IN THE TERMINAL before acting.
Run: .venv/bin/python experiments/small_account_sim.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import kama_gate_btc, cycle_gate_pull, PB
from radar_gate_race import BASE
from short_mirror_15m import invert

ROOT = "/home/angelbell/dev/auto-trade"
USDJPY = 155.0
UNITS = {"gold_bo": 100.0, "gold15m": 100.0,                       # oz per 1.0 lot
         "btc_bo_kama": 1.0, "btc_pull": 1.0, "btc15m_L": 1.0, "btc15m_S": 1.0}   # BTC per 1.0 lot
MIN_LOT, LOT_STEP = 0.01, 0.01


def legs_with_risk():
    """(time, R, risk-in-price-units) per leg -- R already net of cost, PDH weight applied."""
    out = {}
    with contextlib.redirect_stderr(io.StringIO()):
        g1 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:], "1h")
        t = run(g1, SimpleNamespace(**{**CFG, "csv": "x", "tf": "1h", "rr": 3.0, "fwd": 500,
                                       "daily_sma": 150, "daily_slope_k": 10}))
        out["gold_bo"] = (pd.DatetimeIndex(t["time"]), t["R"].values, t["risk"].values)

        b4 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv"), "4h")
        t = kama_gate_btc(run(b4, SimpleNamespace(**{**CFG, "csv": "x", "tf": "4h", "rr": 2.0, "fwd": 300})))
        out["btc_bo_kama"] = (pd.DatetimeIndex(t["time"]), t["R"].values, t["risk"].values)

        t = cycle_gate_pull(run_pb(b4, "long", SimpleNamespace(**{**PB, "csv": "x", "tf": "4h"}), 0.0))
        out["btc_pull"] = (pd.DatetimeIndex(t["time"]), t["R"].values, t["risk"].values)

        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                        "ext_cap": 8.0, "pullback_frac": 0.25}))
        out["gold15m"] = (pd.DatetimeIndex(t["time"]), t["R"].values - 0.3 / t["risk"].values,
                          t["risk"].values)

        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5}))
        pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
        ei = d15.index.get_indexer(t["time"])
        R = (t["R"].values - 15.0 / t["risk"].values) * np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
        out["btc15m_L"] = (pd.DatetimeIndex(t["time"]), R, t["risk"].values)

        inv = invert(d15); C = 2 * d15["high"].max()
        t = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": 4.5}))
        pdl = d15["low"].resample("1D").min().dropna().shift(1).reindex(d15.index, method="ffill").values
        m = (C - t["e_px"].values) < pdl[d15.index.get_indexer(t["time"])]
        out["btc15m_S"] = (pd.DatetimeIndex(t["time"])[m],
                           (t["R"].values - 15.0 / t["risk"].values)[m], t["risk"].values[m])
    return out


def simulate(L, basket, start_jpy, risk_pct, max_risk):
    """MT5-faithful sizing: lot = ceil(equity*risk / (stop*units), 0.01), floor 0.01. Skip if even
    the minimum lot risks more than max_risk of equity."""
    rows = []
    for k in basket:
        idx, R, risk = L[k]
        for i in range(len(idx)):
            rows.append((idx[i], k, R[i], risk[i]))
    rows.sort(key=lambda r: r[0])

    eq = start_jpy
    curve, taken, skipped, actual_risks = [], 0, 0, []
    for ts, k, R, risk_px in rows:
        risk_per_min_lot_jpy = risk_px * MIN_LOT * UNITS[k] * USDJPY
        want_jpy = eq * risk_pct
        lots = np.ceil(want_jpy / (risk_px * UNITS[k] * USDJPY) / LOT_STEP) * LOT_STEP
        lots = max(lots, MIN_LOT)
        risk_jpy = risk_px * lots * UNITS[k] * USDJPY
        if risk_jpy / eq > max_risk:                 # even the minimum lot is too big -> skip
            skipped += 1
            continue
        eq += R * risk_jpy
        if eq <= 0:
            eq = 1.0
            curve.append(eq); break
        actual_risks.append(risk_jpy / eq)
        curve.append(eq)
        taken += 1
    if not curve:
        return None
    c = np.array(curve)
    dd = ((np.maximum.accumulate(c) - c) / np.maximum.accumulate(c)).max() * 100
    yrs = (rows[-1][0] - rows[0][0]).days / 365.25
    cagr = ((c[-1] / start_jpy) ** (1 / yrs) - 1) * 100
    return dict(final=c[-1], cagr=cagr, dd=dd, cdd=cagr / max(dd, 1e-9), taken=taken,
                skipped=skipped, med_risk=100 * np.median(actual_risks), yrs=yrs)


def feasible_today(L, start_jpy, risk_pct=0.01, max_risk=0.03):
    """今日の価格で、最小ロットが何%のリスクを強いるか。2025年以降のトレードだけで測る。"""
    print(f"=== 開始資金 {start_jpy:,}円 · 今日の価格（2025年以降のトレードで測定） ===")
    print(f"  {'leg':<13}{'損切り(中央値)':>16}{'最小ロットのリスク':>18}"
          f"{'狙い1%に対して':>15}{'3%超で見送りになる率':>21}")
    ok = []
    for k in ["btc15m_L", "btc15m_S", "gold15m", "btc_pull", "btc_bo_kama", "gold_bo"]:
        idx, R, risk = L[k]
        m = idx >= "2025"
        r = risk[m]
        jpy = r * MIN_LOT * UNITS[k] * USDJPY
        pct = jpy / start_jpy * 100
        skip = 100 * np.mean(pct > max_risk * 100)
        med = np.median(pct)
        mark = "  ← 建てられる" if med <= 1.5 else ("  ← 過大リスク" if med <= 3 else "  ← 建てられない")
        unit = "$/oz" if k.startswith("gold") else "$/BTC"
        print(f"  {k:<13}{np.median(r):>11,.0f} {unit:<4}{np.median(jpy):>15,.0f}円"
              f"{med:>13.2f}%{skip:>19.0f}%{mark}")
        if med <= 1.5:
            ok.append(k)
    print(f"  → 建てられるレッグ: {', '.join(ok) if ok else '（なし）'}\n")
    return ok


def main():
    L = legs_with_risk()
    SIX = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
    print("Vantage の最小ロットは 0.01。gold は 1ロット=100oz（0.01ロット=1oz）、BTC は 1ロット=1BTC。")
    print("損切り幅は市場構造が決めるので、口座が小さいと『最小ロットでも risk% を守れない』。\n")
    for start in (100_000, 300_000, 1_000_000, 3_000_000):
        feasible_today(L, start)

    print("=" * 92)
    print("小口座で実際に回せる構成（1トレード=資金の1%、0.01ロット切り上げ、2019-05 から複利）")
    print("※ 2019年は BTC $8千・gold $1,300 なので、当時は今より小さい口座でも建てられた。")
    print("   下の数字は『その頃から回していたら』であって、『今 10万円で始めたら』ではない。\n")
    print(f"  {'構成':<24}{'建てた':>7}{'見送り':>7}{'実リスク中央値':>14}"
          f"{'CAGR':>9}{'maxDD':>8}{'CAGR/DD':>9}")
    for tag, b in (("btc15m_L だけ", ["btc15m_L"]),
                   ("btc15m_L + btc15m_S", ["btc15m_L", "btc15m_S"])):
        r = simulate(L, b, 100_000, 0.01, 0.03)
        print(f"  {tag:<24}{r['taken']:>7}{r['skipped']:>7}{r['med_risk']:>13.2f}%"
              f"{r['cagr']:>8.1f}%{r['dd']:>7.1f}%{r['cdd']:>9.2f}")


if __name__ == "__main__":
    main()
