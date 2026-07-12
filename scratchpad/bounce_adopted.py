"""FAITHFUL test: does the ADOPTED book (portfolio_kama.get_legs = gold_bo + btc_bo_kama +
btc_pull[cycle-gated]) get MORE EFFICIENT (CAGR/DD) when the BTC 4h deep-bounce is added as a
4th leg, at CONSTANT total risk (inverse-vol, budget held), vs pile-on (budget grows)?"""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive, resample
from research.portfolio_kama import get_legs
from research.portfolio_alloc import monthly_matrix, cagr_dd_monthly, port_ret, w_inv_vol, w_equal
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
        r-=spread/risk; rows.append((df.index[fj],r)); busy=xj
    return pd.DataFrame(rows, columns=["time","R"])

legs=get_legs()
print("ADOPTED book legs (full-sample CAGR/DD, 1% each):")
for nm,t in legs.items():
    M=monthly_matrix({nm:t}); _,cg,dd=cagr_dd_monthly(M[nm]*0.01); 
    print(f"  {nm:<14} n={len(t):>4}  CAGR/DD={cg/dd if dd>0 else 0:>5.2f}")
bnc=bounce_trades(); bnc.columns=["time","R"]
def bookcagrdd(legdict, budget):
    M=monthly_matrix(legdict); w=w_inv_vol(M, budget)
    ret=port_ret(M, w); _,cg,dd=cagr_dd_monthly(ret); return cg,dd,cg/dd if dd>0 else 0, dict(zip(M.columns, (w/budget).round(2)))
print("\n=== ADOPTED 3-leg book (inverse-vol) ===")
cg,dd,r,wts=bookcagrdd(legs, 0.03); print(f"  budget3%  CAGR={cg:+.1f}% maxDD={dd:.1f}% CAGR/DD={r:.2f}  w={wts}")
cge,dde,re_=cagr_dd_monthly(port_ret(monthly_matrix(legs), w_equal(monthly_matrix(legs),0.03)))[1:]+ (0,)
me=monthly_matrix(legs); ce,de,rr_e=cagr_dd_monthly(port_ret(me,w_equal(me,0.03))); print(f"  equal     CAGR={ce:+.1f}% maxDD={de:.1f}% CAGR/DD={rr_e:.2f}")
legs4={**legs, "btc_bounce":bnc}
print("\n=== 4-leg (+bounce) ===")
cg,dd,r,wts=bookcagrdd(legs4, 0.03); print(f"  CONSTANT total 3%  CAGR={cg:+.1f}% maxDD={dd:.1f}% CAGR/DD={r:.2f}  w={wts}")
cg,dd,r,wts=bookcagrdd(legs4, 0.04); print(f"  pile-on     4%     CAGR={cg:+.1f}% maxDD={dd:.1f}% CAGR/DD={r:.2f}  w={wts}")
