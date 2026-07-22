"""本物のキルゾーン(NY07-10h)を ASK基準の約定で測り直す。
買い指値は「BID が lim - spread 以下」でしか約定しない（Vantage CSV は BID）。
唯一コスト後プラスだった gbpusd long が生き残るかを見る。"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ict_killzone import (load_ny, price_and_scan, find_entries, COST_RT, SYMS,
                          LONDON_HOURS, KZ_HOURS, F_DEFAULT, RR_DEFAULT, STOPBUF_DEFAULT)
from ict_diag import ask_fill_entries, trades_from, stat, PIP

SPREADS = {"gbpusd": [0.0, 0.5, 0.9, 1.5], "eurusd": [0.0, 0.5, 0.9, 1.5],
           "audusd": [0.0, 0.5, 0.9, 1.5], "usdjpy": [0.0, 0.5, 0.9, 1.5],
           "gold":   [0.0, 1.5, 3.0],      # gold: 1 pip = 0.1 usd -> 1.5pip = $0.15
           "btcusd": [0.0, 10.0, 25.0]}    # btc: 1 "pip" = $1
print(f"{'sym':7s} {'side':5s} {'spread':>10s} {'n':>5} {'win%':>6} {'gross':>8} {'net':>8} {'PF':>6}")
for name in ("gbpusd", "eurusd", "usdjpy", "audusd", "gold", "btcusd"):
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
    for sp_p in SPREADS[name]:
        sp = sp_p * PIP[name]
        recs = price_and_scan(df, ask_fill_entries(df, LONDON_HOURS, KZ_HOURS, F_DEFAULT, sp),
                              STOPBUF_DEFAULT, RR_DEFAULT)
        for side in ("long", "short"):
            s = stat(trades_from(df, recs, side), COST_RT[name])
            if s:
                print(f"{name:7s} {side:5s} {sp_p:9.1f}p {s['n']:5d} {s['win']:6.1f} "
                      f"{s['gross']:+8.3f} {s['net']:+8.3f} {s['pf']:6.2f}")
