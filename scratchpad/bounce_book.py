"""Can the BTC 4h deep-bounce IMPROVE the book? It's a 3rd BTC-4h-long leg, so the bar is:
low annual-R correlation with the existing BTC legs (btc_bo, btc_pull) AND per-year spread
(not era-concentrated), else it's just more BTC beta. Compare 3-leg vs +bounce vs bounce-
replaces-btc_pull on CAGR/DD."""
import sys, os; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive
from research.portfolio import get_trades, get_trades_pullback, report, metrics
AGG={"open":"first","high":"max","low":"min","close":"last"}
W,FWD=40,300
def bounce_trades(frac=0.786, spread=15.0, sslip=0.5):
    df=load_mt5_csv("data/vantage_btcusd_h1.csv").resample("240min").agg(AGG).dropna()
    h,l,c=df["high"].values,df["low"].values,df["close"].values
    a=ta.atr(df["high"],df["low"],df["close"],14).values
    es=df["close"].ewm(span=80,adjust=False).mean().values
    dck=df["close"].resample("1D").last().dropna(); kmg=kama_adaptive(dck,14)
    g=((kmg>kmg.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    sw=swings_zigzag(h,l,a,2.0); imps=[]
    for t in range(1,len(sw)):
        cH,iH,pH,kH=sw[t]; cL,iL,pL,kL=sw[t-1]
        if kH!=+1 or kL!=-1 or pH-pL<=0: continue
        if es is not None and not np.isnan(es[cH]) and pH<es[cH]: continue
        imps.append((cH,pL,pH))
    busy=-1; rows=[]
    for (cH,L0,H1) in imps:
        if cH<=busy: continue
        lim=H1-frac*(H1-L0)
        if lim<=L0 or not g[min(cH,len(g)-1)]: continue
        fj=None
        for j in range(cH+1,min(cH+1+W,len(c))):
            if l[j]<=L0: break
            if l[j]<=lim: fj=j; break
        if fj is None: continue
        entry=lim; risk=entry-L0; reward=H1-entry
        if risk<=0 or reward<=0: continue
        xj=min(fj+FWD,len(c)-1); r=None
        for j in range(fj+1,min(fj+1+FWD,len(c))):
            if l[j]<=L0: r=-1.0-sslip*(L0-l[j])/risk; xj=j; break
            if h[j]>=H1: r=reward/risk; xj=j; break
        if r is None: r=(c[xj]-entry)/risk
        r-=spread/risk; rows.append((df.index[fj],r,(df.index[xj]-df.index[fj]).total_seconds()/86400)); busy=xj
    return pd.DataFrame(rows, columns=["time","R","hold"])
risk=0.01
EX=dict(exit_kama=0,gate_tf="1D",gate_kama=0,wave="all",sl_b="swinglow",sl_b_k=1.5,ext_cap=0,
        retest=0,retest_tol=0.10,tp1_frac=0.0,tp1_rr=1.0,tp1_be=1,tp=0.0)
gold=get_trades("data/vantage_xauusd_h1.csv","1h",rr=3.0,fwd=500,daily_sma=150,daily_slope_k=10,risk=risk,**EX)
btc=get_trades("data/vantage_btcusd_h1.csv","4h",rr=2.0,fwd=300,risk=risk,**EX)
btcpb=get_trades_pullback("data/vantage_btcusd_h1.csv","4h",risk=risk)
bnc=bounce_trades()
print(f"\n=== standalone (risk 1%/trade) ===")
report("GOLD breakout",gold,risk); report("BTC breakout",btc,risk)
report("BTC pullback",btcpb,risk); report("BTC deep-bounce",bnc,risk)
print(f"\n=== book combinations ===")
report("3-leg (book)",pd.concat([gold,btc,btcpb],ignore_index=True),risk)
report("4-leg (+bounce)",pd.concat([gold,btc,btcpb,bnc],ignore_index=True),risk)
report("3-leg (bounce replaces pull)",pd.concat([gold,btc,bnc],ignore_index=True),risk)
# annual-R correlations
def ann(t): return t.groupby(t.time.dt.year)["R"].sum()
al=pd.concat([ann(gold),ann(btc),ann(btcpb),ann(bnc)],axis=1).fillna(0); al.columns=["G.bo","B.bo","B.pull","B.bounce"]
print("\n=== annual-R correlation (is the bounce REDUNDANT with BTC legs?) ===")
print(al.corr().round(2).to_string())
print("\n=== per-year (era-concentration of the bounce?) ===")
print(f"  {'year':<6}{'B.bo':>8}{'B.pull':>8}{'B.bounce':>10}")
for y in sorted(al.index):
    print(f"  {y:<6}{al['B.bo'].get(y,0):>+8.0f}{al['B.pull'].get(y,0):>+8.0f}{al['B.bounce'].get(y,0):>+10.0f}")

print("\n\n==== OVERFIT AUDIT (bounce frac family, cost+slip) ====")
from research.edge_harness import audit
cfgs={f"f{fr}":[(r.time,r.R) for r in bounce_trades(frac=fr).itertuples()] for fr in (0.618,0.7,0.786,0.85)}
for k,v in cfgs.items():
    R=np.array([r for _,r in v]); print(f"  {k}: n={len(R)} meanR={R.mean():+.3f} CAGR/DD-ish")
audit(cfgs, flagship="f0.786")
print("\n==== book CAGR/DD (NOT ret/DD) at equal 1%/leg ====")
def cagrdd(t):
    t=t.sort_values("time"); eq=(1+risk*t["R"]).cumprod()
    _,cagr,dd,_=metrics(eq,t); return cagr,dd,cagr/dd
for nm,t in [("3-leg book",pd.concat([gold,btc,btcpb],ignore_index=True)),
             ("4-leg +bounce",pd.concat([gold,btc,btcpb,bnc],ignore_index=True)),
             ("3-leg bounce-replaces-pull",pd.concat([gold,btc,bnc],ignore_index=True))]:
    cg,dd,r=cagrdd(t); print(f"  {nm:<28} CAGR={cg:+.1f}% maxDD={dd:.1f}% CAGR/DD={r:.2f}")
