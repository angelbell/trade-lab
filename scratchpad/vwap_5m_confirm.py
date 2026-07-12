"""User: VWAP ~same on 5m/15m; what if the ENTRY CONFIRMATION is on 5m? Keep the 15m SETUP
(down-cross+RSI<40+ADX<25+daily-up); confirm the VWAP-reclaim on 15m vs 5m (finer/earlier entry).
R normalized by 15m-ATR (comparable), fills/exits walked on the 5m path. GROSS, no-lookahead."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
d=load_mt5_csv("data/vantage_xauusd_m5.csv")
# --- M5 (fine) ---
o5,h5,l5,c5=(d[k].values for k in ("open","high","low","close")); v5=d["volume"].values
hlc3_5=(h5+l5+c5)/3; day5=d.index.normalize()
vwap5=(pd.Series(hlc3_5*v5,index=d.index).groupby(day5).cumsum()/pd.Series(v5,index=d.index).groupby(day5).cumsum().replace(0,np.nan)).values
t5=d.index.values.astype("datetime64[ns]").astype("int64")
# --- 15m (setup) ---
df=d.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
o15,h15,l15,c15=(df[k].values for k in ("open","high","low","close")); v15=df["volume"].values
hlc3=(h15+l15+c15)/3; day=df.index.normalize()
vwap15=(pd.Series(hlc3*v15,index=df.index).groupby(day).cumsum()/pd.Series(v15,index=df.index).groupby(day).cumsum().replace(0,np.nan)).values
rsi15=ta.rsi(df["close"],14).values; atr15=ta.atr(df["high"],df["low"],df["close"],14).values
adx15=ta.adx(df["high"],df["low"],df["close"],14)["ADX_14"].values
dc=d["close"].resample("1D").last().dropna(); dsma=dc.rolling(150).mean()
dup=((dc>dsma).shift(1)).reindex(df.index,method="ffill").fillna(False).values
t15=df.index.values.astype("datetime64[ns]").astype("int64")
N5=len(c5); N15=len(c15); STEP=np.int64(15*60*1_000_000_000)
cu15=np.zeros(N15,bool); cu15[1:]=(c15[:-1]>=vwap15[:-1])&(c15[1:]<vwap15[1:])
setup=cu15&(rsi15<40)&(adx15<25)&dup
span=(df.index[-1]-df.index[0]).days/365.25
def walk5(e_m5, a):
    """walk M5 from entry index e_m5 (enter at o5[e_m5]); stop1.5/tgt2.5 in 15m-ATR a. returns R(/1.5)."""
    e=o5[e_m5]; stop=e-1.5*a; tgt=e+2.5*a
    for j in range(e_m5,min(e_m5+900,N5)):
        if l5[j]<=stop: return -1.5/1.5, j
        if h5[j]>=tgt: return 2.5/1.5, j
    xj=min(e_m5+900,N5-1); return ((c5[xj]-e)/a)/1.5, xj
def run(mode, name, cwin15=3, cwin5=9):
    R=[];ts=[];busy=-1
    for k in np.where(setup)[0]:
        a=atr15[k]
        if np.isnan(a) or a<=0 or k+1>=N15: continue
        close_ns=t15[k]+STEP                      # 15m setup bar closes here
        if mode=="15m":
            ent5=None
            for j in range(1,cwin15+1):
                if k+j>=N15: break
                if c15[k+j]>vwap15[k+j]:
                    if k+j+1>=N15: break
                    ent5=int(np.searchsorted(t5,t15[k+j+1])); break   # o15[k+j+1] = first M5 bar of next 15m
        else:  # 5m confirm
            m0=int(np.searchsorted(t5,close_ns)); ent5=None
            for m in range(m0,min(m0+cwin5,N5)):
                if c5[m]>vwap5[m]: ent5=m+1; break
        if ent5 is None or ent5>=N5 or ent5<=busy: continue
        r,xj=walk5(ent5,a); R.append(r); ts.append(d.index[ent5]); busy=xj
    R=np.array(R); y=np.array([t.year for t in ts])
    if len(R)<15: print(f"  {name:<24} n={len(R)} too few"); return
    pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.9; half=np.median(y); ys=sorted(set(y))
    green=sum(1 for yy in ys if R[y==yy].sum()>0)
    print(f"  {name:<24} n={len(R):>4}({len(R)/span:>3.0f}/yr) PF={pf:.2f} win={(R>0).mean()*100:>3.0f}% "
          f"meanR={R.mean():+.3f} IS/OOS={R[y<half].mean():+.2f}/{R[y>=half].mean():+.2f} green={green}/{len(ys)}")
print("15m setup fixed (down-cross+RSI<40+ADX<25+daily-up); confirmation TF varies. stop1.5/tgt2.5, GROSS\n")
run("15m","15m-confirm (baseline)")
run("5m","5m-confirm (finer entry)")
run("5m","5m-confirm win=6 (tighter)",cwin5=6)
run("5m","5m-confirm win=12",cwin5=12)
