"""Does the ADOPTED book get MORE EFFICIENT if the two BREAKOUT legs (gold_bo, btc_bo_kama)
use the pullback-limit execution instead of market entry? Same legs, same gates, ONLY the
breakout entry changes. btc_pull unchanged. Faithful: reuses portfolio_kama leg generation
(now with breakout_wave --pullback-frac) + the canonical trade-level & monthly-invvol metrics."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from types import SimpleNamespace
from src.data_loader import load_mt5_csv
from breakout_wave import run as run_bo, resample
from ema_pullback import run as run_pb
from research.regime_gate_lab import CFG
from research.portfolio_kama import PB, kama_gate_btc, cycle_gate_pull, cagr_dd
from research.portfolio_alloc import monthly_matrix, w_inv_vol, w_equal, port_ret, cagr_dd_monthly
SPLIT=2022
def legs_for(pf):
    gold = run_bo(resample(load_mt5_csv("data/vantage_xauusd_h1.csv"),"1h"),
                  SimpleNamespace(**{**CFG,"csv":"x","tf":"1h","rr":3.0,"fwd":500,
                                     "daily_sma":150,"daily_slope_k":10,"pullback_frac":pf}))[["time","R"]]
    btc = run_bo(resample(load_mt5_csv("data/vantage_btcusd_h1.csv"),"4h"),
                 SimpleNamespace(**{**CFG,"csv":"x","tf":"4h","rr":2.0,"fwd":300,"pullback_frac":pf}))[["time","R"]]
    btc_k = kama_gate_btc(btc)
    dbtc = resample(load_mt5_csv("data/vantage_btcusd_h1.csv"),"4h")
    pb = run_pb(dbtc,"long",SimpleNamespace(**{**PB,"csv":"x","tf":"4h"}),0.0)[["time","R"]]
    pb = cycle_gate_pull(pb)                          # btc_pull UNCHANGED (already a pullback)
    return {"gold_bo":gold,"btc_bo_kama":btc_k,"btc_pull":pb}
def book_invvol(legs):
    M=monthly_matrix(legs); Mis=M[M.index.year<SPLIT]; budget=0.01*len(legs)
    w=w_inv_vol(Mis,budget); full=port_ret(M,w)
    cf,df_,rf=cagr_dd_monthly(full)
    co,do,ro=cagr_dd_monthly(full[full.index.year>=SPLIT])
    return rf,ro,dict(zip(M.columns,(w/0.01).round(2)))
def tradelevel(legs):
    t=pd.concat(list(legs.values()),ignore_index=True); return cagr_dd(t)
print(f"{'pf':>5} | leg CAGR/DD (trade-level, 1%)          | 3-leg trade-lvl | 3-leg monthly inv-vol (FULL/OOS)")
for pf in (0.0,0.25,0.30):
    L=legs_for(pf)
    lc={k:cagr_dd(v) for k,v in L.items()}
    tl=tradelevel(L); rf,ro,w=book_invvol(L)
    print(f"{pf:>5} | gold {lc['gold_bo'][2]:.2f} btc_k {lc['btc_bo_kama'][2]:.2f} pull {lc['btc_pull'][2]:.2f} "
          f"(n {len(L['gold_bo'])}/{len(L['btc_bo_kama'])}/{len(L['btc_pull'])}) | {tl[2]:>13.2f} | {rf:.2f}/{ro:.2f}  w={w}")
print("\nbaseline pf=0 must match canonical: 3-leg trade-lvl 2.63, monthly inv-vol 3.79/4.08")
