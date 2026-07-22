import sys, pandas as pd, numpy as np
sys.path.insert(0,"/home/angelbell/dev/auto-trade")
from src.data_loader import load_mt5_csv
from experiments.event_scalp import build_scalp_table, null_scalp_table
from experiments.event_scalp_cond import threshold_subset, pctile_of_real_in_pool
PIP=0.01; COST=0.009; HSET=[5,10,15]
ev=pd.read_csv("experiments/fomc_stmt_2019.csv",parse_dates=["dt_utc","dt_broker"])
ev["dt_broker"]=ev["dt_broker"].dt.tz_localize("UTC"); events=list(ev["dt_broker"].sort_values())
df=load_mt5_csv("data/vantage_usdjpy_m1.csv")
# fill the w_c ladder: 4 and 7
for wc in [4,7]:
    real=build_scalp_table(df,events,wc,HSET,f"wc{wc}"); null=null_scalp_table(df,events,wc,HSET,f"wc{wc}",draws_target=3000)
    for frac in [1.00,0.50]:
        sub,thr=threshold_subset(real,"confirm_move_atr",frac)
        nsub,_=threshold_subset(null,"confirm_move_atr",frac) if frac<1 else (null,-np.inf)
        cells=[]
        for h in HSET:
            g=sub[f"g_{h}"].dropna(); gn=nsub[f"g_{h}"].dropna()
            pct,_=pctile_of_real_in_pool(g,gn,COST)
            cells.append(f"H{h}:{(g-COST).mean()/PIP:+.2f}p/{(g>COST).mean()*100:.0f}%/{pct:.0f}")
        print(f"w_c={wc} frac={frac:.2f} n={len(sub):3d} | "+" | ".join(cells))
# annual for the standout cell w_c=2
real=build_scalp_table(df,events,2,HSET,"wc2")
for frac in [1.00,0.50]:
    sub,_=threshold_subset(real,"confirm_move_atr",frac)
    t=sub.dropna(subset=["g_5"]).copy(); t["yr"]=t["t0"].dt.year
    g=t.groupby("yr").apply(lambda s: pd.Series({"n":len(s),"netP":(s["g_5"]-COST).sum()/PIP,"mean":(s["g_5"]-COST).mean()/PIP}))
    print(f"\n-- w_c=2 H=5 frac={frac:.2f} annual --"); print(g.round(2).to_string())
