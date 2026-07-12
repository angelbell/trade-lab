"""Measure the NATURAL retrace-to-H1 depth (as frac of risk) per tested breakout config,
and the frac where meanR/ret-DD is best. Fills the breakout-vs-bounce comparison table."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive
AGG={"open":"first","high":"max","low":"min","close":"last"}
BO=20
def build_entries(df, gate):
    h,l,c=df["high"].values,df["low"].values,df["close"].values
    a=ta.atr(df["high"],df["low"],df["close"],14).values
    es=df["close"].ewm(span=80,adjust=False).mean().values
    if gate=="sma":
        dc=df["close"].resample("1D").last().dropna(); sma=dc.rolling(150).mean()
        g=((dc>sma)&(sma>sma.shift(10))).shift(1).reindex(df.index,method="ffill").fillna(False).values
    else:
        dck=df["close"].resample("1D").last().dropna(); kmg=kama_adaptive(dck,14)
        g=((kmg>kmg.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    sw=swings_zigzag(h,l,a,2.0)
    def fb(level,after):
        for j in range(after,min(after+BO,len(c))):
            if c[j]>level: return j
        return None
    depths=[]
    for t in range(2,len(sw)):
        (cL2,iL2,pL2,kL2),(cH1,iH1,pH1,kH1),(cL0,iL0,pL0,kL0)=sw[t],sw[t-1],sw[t-2]
        if not(kL2==-1 and kH1==+1 and kL0==-1): continue
        if pL2<=pL0 or pH1-pL0<=0: continue
        if es is not None and not np.isnan(es[cL2]) and pH1<es[cL2]: continue
        e_i=fb(pH1,cL2+1)
        if e_i is None: continue
        if not g[e_i]: continue
        e=c[e_i]; risk=e-pL2
        if risk<=0: continue
        depths.append((e-pH1)/risk)   # retrace-to-broken-level H1, as frac of risk
    return np.array(depths)
gold=load_mt5_csv("data/vantage_xauusd_m5.csv")
btc=load_mt5_csv("data/vantage_btcusd_h1.csv")
rows=[("GOLD 15m",gold.resample("15min").agg(AGG).dropna(),"sma"),
      ("BTC 1h",btc,"kama"),
      ("BTC 4h",btc.resample("240min").agg(AGG).dropna(),"kama")]
print(f"{'config':<10}{'n':>5}{'H1-depth med':>14}{'IQR(25-75)':>14}{'>0.3':>7}{'>0.5':>7}")
for nm,df,gate in rows:
    d=build_entries(df,gate)
    print(f"{nm:<10}{len(d):>5}{np.median(d):>13.2f}{np.percentile(d,25):>7.2f}-{np.percentile(d,75):<6.2f}"
          f"{(d>0.3).mean()*100:>5.0f}%{(d>0.5).mean()*100:>6.0f}%")
print("\n(H1-depth = how far a breakout naturally pulls back to its broken level, as frac of risk.")
print(" shallow = strong break. this is the 'effective Fib' the market actually offers on a breakout.)")
