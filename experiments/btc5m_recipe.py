"""Port the ONE recipe that works (ZigZag Pattern-B + pullback-limit + KAMA gate + far fixed target)
from BTC 15m down to BTC 5m. The user wants PF ~2 on a short timeframe WITH frequency; btc15m_L
already gives PF 1.76 at 98/yr, so the cheapest way to more of it is the same machine on faster bars.

The thing that kills low-TF work is not the timeframe -- it is cost/stop_distance (today's identity,
confirmed three times). btc15m_L's ZigZag stop has a median of $470, so the $15 round trip is 3.2%.
On 5m the stop will be smaller; the question is whether it stays far enough above the cost floor.
Print the cost fraction explicitly so the verdict is "no edge" or "edge, killed by cost" and never a
mush of the two.

Everything else is held at btc15m_L's adopted spec: pattern B, ZigZag k=2, pullback-limit 0.30,
fill window 200 bars, 4h-KAMA(14) rising gate, PDH soft-size 0.5, RR 4.5. Only --tf moves.
Run: .venv/bin/python experiments/btc5m_recipe.py
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

ROOT = "/home/angelbell/dev/auto-trade"
COST = 15.0


def leg(tf, rr, gate_tf="240min", frac=0.3):
    with contextlib.redirect_stderr(io.StringIO()):
        d = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m5.csv").loc["2018-10-01":], tf)
        t = run(d, SimpleNamespace(**{**BASE, "tf": tf, "gate_kama": 14, "gate_kama_tf": gate_tf,
                                      "pullback_frac": frac, "rr": rr, "fill_win": 200, "fwd": 500}))
    pdh = d["high"].resample("1D").max().dropna().shift(1).reindex(d.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[d.index.get_indexer(t["time"])], 1.0, 0.5)
    risk = t["risk"].values / w
    R = pd.Series(t["R"].values * w - COST / risk, index=pd.DatetimeIndex(t["time"]))
    G = pd.Series(t["R"].values * w, index=pd.DatetimeIndex(t["time"]))
    return R, G, risk


def line(tag, R, G, risk):
    if len(R) < 20:
        print(f"  {tag:<26}n={len(R)} 少なすぎ"); return
    yrs = (R.index[-1] - R.index[0]).days / 365.25
    half = R.index[len(R) // 2]
    pfg = G[G > 0].sum() / abs(G[G <= 0].sum())
    pfn = R[R > 0].sum() / abs(R[R <= 0].sum())
    eq = np.cumprod(1 + 0.01 * R.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    print(f"  {tag:<26}{len(R):>5}{len(R)/yrs:>7.0f}{np.median(risk):>9,.0f}$"
          f"{100*np.median(COST/risk):>8.1f}%{100*(R>0).mean():>7.1f}%{pfg:>8.2f}{pfn:>8.2f}"
          f"{R.mean():>+9.3f}{R[R.index < half].mean():>+9.3f}{R[R.index >= half].mean():>+9.3f}"
          f"{R.sum()/yrs:>+9.1f}{cagr/max(dd,1e-9):>9.2f}")


def main():
    print("同じレシピ（ZigZag Pattern-B ＋ 押し目指値0.30 ＋ 4hKAMAゲート ＋ PDHソフト ＋ 遠い固定目標）")
    print("を、15分足 → 5分足へ。動かすのは足だけ。\n")
    print("コストの読み方: 今日3度確認した恒等式 → **コストのR換算 = コスト ÷ 損切り幅**。")
    print("  15分足の損切り中央値 $470 → コスト 3.2%（生きる）／ 今日死んだ構造ストップ $38 → 39%（死ぬ）。")
    print("  5分足がどちら側に落ちるかが、この実験の全て。\n")
    print(f"  {'':<26}{'n':>5}{'本/年':>7}{'損切り中央':>10}{'コスト':>8}{'勝率':>8}"
          f"{'PF素':>8}{'PF実':>8}{'meanR':>9}{'IS':>9}{'OOS':>9}{'totR/年':>9}{'CAGR/DD':>9}")
    R, G, k = leg("15min", 4.5)
    line("15分（現行 btc15m_L）", R, G, k)
    print()
    for rr in (3.0, 4.5, 6.0):
        R, G, k = leg("5min", rr)
        line(f"5分 · RR{rr}", R, G, k)
    print()
    for gt in ("60min", "240min", "1D"):
        R, G, k = leg("5min", 4.5, gate_tf=gt)
        line(f"5分 · RR4.5 · ゲート{gt}", R, G, k)
    print()
    for fr in (0.0, 0.2, 0.3, 0.4):
        R, G, k = leg("5min", 4.5, frac=fr)
        line(f"5分 · RR4.5 · 押し目{fr}", R, G, k)


if __name__ == "__main__":
    main()
