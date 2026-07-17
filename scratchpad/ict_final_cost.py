"""ICT キルゾーンの最終判定: 執行モデルを物理的に正しく分解する。
  約定判定 = ASK基準（買い指値は BID <= lim - spread のときだけ約定）
  コスト   = 手数料のみ（スプレッドは「ASKで買いBIDで決済する」形で既に損切り幅に埋まっている）
Vantage RAW/ECN 実測（CLAUDE.md）: FX 往復0.9pip = 生スプレッド~0.3 + 手数料~0.6
                                   gold 往復$0.15-0.35 = スプレッド~$0.15 + 手数料$0.06
                                   BTC 手数料0・スプレッド$10-25"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ict_killzone import (load_ny, price_and_scan, find_entries, SYMS,
                          LONDON_HOURS, KZ_HOURS, F_DEFAULT, RR_DEFAULT, STOPBUF_DEFAULT)
from ict_diag import ask_fill_entries, trades_from, PIP

# (spread_for_fill, commission_only_cost) in price units
MODEL = {
    "gbpusd": (0.3 * 1e-4, 0.6 * 1e-4), "eurusd": (0.3 * 1e-4, 0.6 * 1e-4),
    "audusd": (0.3 * 1e-4, 0.6 * 1e-4), "usdjpy": (0.3 * 1e-2, 0.6 * 1e-2),
    "gold":   (0.15, 0.06),             "btcusd": (15.0, 0.0),
}

def stats(tr, cost):
    if tr is None or not len(tr):
        return None
    g = tr["R"].values; net = g - cost / tr["risk"].values
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    yrs = pd.to_datetime(tr["date"]).dt.year.values
    half = len(g) // 2
    by = pd.Series(net).groupby(yrs).sum()
    return dict(n=len(g), win=100*(g > 0).mean(), gross=g.mean(), net=net.mean(),
                pf=pos/neg if neg > 0 else np.inf, tot=net.sum(),
                IS=net[:half].sum(), OOS=net[half:].sum(),
                gy=100*(by > 0).mean(), ny=len(by))

print("ICT NYキルゾーン: ASK約定 + 手数料のみ（スプレッド二重計上を除去）")
print(f"{'sym':7s} {'side':5s} | {'現行(BID約定/往復コスト)':^24s} | {'正しい執行モデル':^40s}")
print(f"{'':7s} {'':5s} | {'n':>5} {'gross':>7} {'net':>7} | {'n':>5} {'win%':>5} {'gross':>7} {'net':>7} {'PF':>5} {'IS':>7} {'OOS':>7} {'黒字年%':>7}")
CANON = {"gold": 0.6, "btcusd": 15.0, "eurusd": 9e-5, "gbpusd": 9e-5, "audusd": 9e-5, "usdjpy": 9e-3}
for name in ("gbpusd", "eurusd", "usdjpy", "audusd", "gold", "btcusd"):
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
    sp, comm = MODEL[name]
    r0 = price_and_scan(df, find_entries(df), STOPBUF_DEFAULT, RR_DEFAULT)
    r1 = price_and_scan(df, ask_fill_entries(df, LONDON_HOURS, KZ_HOURS, F_DEFAULT, sp),
                        STOPBUF_DEFAULT, RR_DEFAULT)
    for side in ("long", "short"):
        a = stats(trades_from(df, r0, side), CANON[name])
        b = stats(trades_from(df, r1, side), comm)
        if not (a and b):
            continue
        print(f"{name:7s} {side:5s} | {a['n']:5d} {a['gross']:+7.3f} {a['net']:+7.3f} | "
              f"{b['n']:5d} {b['win']:5.1f} {b['gross']:+7.3f} {b['net']:+7.3f} {b['pf']:5.2f} "
              f"{b['IS']:+7.1f} {b['OOS']:+7.1f} {b['gy']:6.0f}%")
