"""Mirror of the breakout depth test, for a BOUNCE entry. Same ZigZag impulse L0->H1 (uptrend,
H1>EMA80). Place a LONG limit at H1 - frac*(H1-L0) = classic Fib retracement of the impulse.
Fill if the pullback reaches it (else missed=ran away as a breakout / didn't pull back deep).
stop=L0 (impulse origin), target=H1 (bounce back to prior high). Sweep frac; where does the
bounce edge PEAK? Hypothesis: DEEP (0.382-0.618), the opposite of breakout's shallow 0.236."""
import sys; sys.path.insert(0,"/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from breakout_wave import swings_zigzag, kama_adaptive
AGG={"open":"first","high":"max","low":"min","close":"last"}
W,FWD=40,300
def run(name, df, gate):
    h,l,c=df["high"].values,df["low"].values,df["close"].values
    a=ta.atr(df["high"],df["low"],df["close"],14).values
    es=df["close"].ewm(span=80,adjust=False).mean().values
    if gate=="kama":
        dck=df["close"].resample("1D").last().dropna(); kmg=kama_adaptive(dck,14)
        g=((kmg>kmg.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    else:
        dc=df["close"].resample("1D").last().dropna(); sma=dc.rolling(150).mean()
        g=((dc>sma)&(sma>sma.shift(10))).shift(1).reindex(df.index,method="ffill").fillna(False).values
    sw=swings_zigzag(h,l,a,2.0)
    # collect (confirm_bar_of_H1, L0_price, H1_price) uptrend impulses
    imps=[]
    for t in range(1,len(sw)):
        cH,iH,pH,kH=sw[t]; cL,iL,pL,kL=sw[t-1]
        if kH!=+1 or kL!=-1: continue
        if pH-pL<=0: continue
        if es is not None and not np.isnan(es[cH]) and pH<es[cH]: continue
        imps.append((cH,pL,pH))   # cH = bar H1 confirmed (causal), L0=pL, H1=pH
    def stats(R,ts):
        R=np.array(R); yr=np.array([t.year for t in ts]); yrs=np.unique(yr); half=yrs[len(yrs)//2] if len(yrs) else 0
        pf=R[R>0].sum()/abs(R[R<=0].sum()) if (R<=0).any() else 9.99
        return (len(R),(R>0).mean()*100,pf,R.mean(),
                R[yr<half].mean() if (yr<half).any() else float("nan"),
                R[yr>=half].mean() if (yr>=half).any() else float("nan"))
    def sweep_frac(frac):
        busy=-1; R=[]; ts=[]; fills=0; tot=0
        for (cH,L0,H1) in imps:
            if cH<=busy: continue
            tot+=1; lim=H1-frac*(H1-L0)
            if lim<=L0: continue
            if not g[min(cH,len(g)-1)]: continue
            fill_j=None
            for j in range(cH+1,min(cH+1+W,len(c))):
                if l[j]<=L0: break                 # broke impulse low before filling = dead
                if l[j]<=lim: fill_j=j; break
            if fill_j is None: continue
            fills+=1; entry=lim; risk=entry-L0; reward=H1-entry
            if risk<=0 or reward<=0: continue
            exit_j=min(fill_j+FWD,len(c)-1); r=None
            for j in range(fill_j+1,min(fill_j+1+FWD,len(c))):
                if l[j]<=L0: r=-1.0; exit_j=j; break
                if h[j]>=H1: r=reward/risk; exit_j=j; break
            if r is None: r=(c[exit_j]-entry)/risk
            R.append(r); ts.append(df.index[fill_j]); busy=exit_j
        fillpct=fills/tot*100 if tot else 0
        return R,ts,fillpct
    FIB={0.236,0.382,0.5,0.618,0.786}
    print(f"\n### {name}  (n_impulses={len(imps)}, stop=L0, target=H1) ###")
    print(f"  {'frac':>6}{'':2}{'fill%':>7}{'N':>5}{'win':>6}{'PF':>7}{'meanR':>8}{'IS/OOS':>13}")
    grid=sorted(set([round(x,3) for x in np.arange(0.2,0.86,0.0433)])|FIB)
    for frac in grid:
        R,ts,fp=sweep_frac(frac)
        if len(R)<12: continue
        n,win,pf,mr,is_,oos=stats(R,ts); star=" *" if round(frac,3) in {round(f,3) for f in FIB} else "  "
        print(f"  {frac:>6.3f}{star}{fp:>6.0f}%{n:>5}{win:>5.0f}%{pf:>7.2f}{mr:>+8.3f}{is_:>+6.2f}/{oos:>+.2f}")
btc=load_mt5_csv("data/vantage_btcusd_h1.csv")
gold=load_mt5_csv("data/vantage_xauusd_m5.csv")
run("BTC 4h", btc.resample("240min").agg(AGG).dropna(), "kama")
run("BTC 1h", btc, "kama")
run("GOLD 15m", gold.resample("15min").agg(AGG).dropna(), "sma")
print("\n(where does bounce meanR PEAK? deep 0.382-0.618 => confirms breakout=shallow/bounce=deep law)")

print("\n\n==== COST+STOP-SLIP STRESS on the DEEP bounce (tight stop = the risk) ====")
def run_stress(name, df, gate, spread, sslip):
    h,l,c=df["high"].values,df["low"].values,df["close"].values
    a=ta.atr(df["high"],df["low"],df["close"],14).values
    es=df["close"].ewm(span=80,adjust=False).mean().values
    if gate=="kama":
        dck=df["close"].resample("1D").last().dropna(); kmg=kama_adaptive(dck,14)
        g=((kmg>kmg.shift(1)).shift(1)).reindex(df.index,method="ffill").fillna(False).values
    else:
        dc=df["close"].resample("1D").last().dropna(); sma=dc.rolling(150).mean()
        g=((dc>sma)&(sma>sma.shift(10))).shift(1).reindex(df.index,method="ffill").fillna(False).values
    sw=swings_zigzag(h,l,a,2.0); imps=[]
    for t in range(1,len(sw)):
        cH,iH,pH,kH=sw[t]; cL,iL,pL,kL=sw[t-1]
        if kH!=+1 or kL!=-1 or pH-pL<=0: continue
        if es is not None and not np.isnan(es[cH]) and pH<es[cH]: continue
        imps.append((cH,pL,pH))
    def one(frac,sp,ss):
        busy=-1; R=[]; ts=[]
        for (cH,L0,H1) in imps:
            if cH<=busy: continue
            lim=H1-frac*(H1-L0)
            if lim<=L0 or not g[min(cH,len(g)-1)]: continue
            fill_j=None
            for j in range(cH+1,min(cH+1+W,len(c))):
                if l[j]<=L0: break
                if l[j]<=lim: fill_j=j; break
            if fill_j is None: continue
            entry=lim; risk=entry-L0; reward=H1-entry
            if risk<=0 or reward<=0: continue
            exit_j=min(fill_j+FWD,len(c)-1); r=None
            for j in range(fill_j+1,min(fill_j+1+FWD,len(c))):
                if l[j]<=L0: over=L0-l[j]; r=-1.0-ss*over/risk; exit_j=j; break
                if h[j]>=H1: r=reward/risk; exit_j=j; break
            if r is None: r=(c[exit_j]-entry)/risk
            r-=sp/risk; R.append(r); ts.append(df.index[fill_j]); busy=exit_j
        R=np.array(R); yr=np.array([t.year for t in ts]); yrs=np.unique(yr); half=yrs[len(yrs)//2]
        cum=np.cumsum(R); dd=(np.maximum.accumulate(cum)-cum).max()
        return len(R),(R>0).mean()*100,R.mean(),R[yr<half].mean(),R[yr>=half].mean(),R.sum()/dd if dd>0 else 0
    print(f"\n  {name} (spread=${spread}, stop_slip={sslip}):")
    print(f"  {'frac':>6}{'N':>5}{'win':>6}{'meanR gross':>13}{'meanR NET':>11}{'IS/OOS net':>13}{'ret/DD':>8}")
    for frac in (0.5,0.618,0.7,0.786,0.85):
        n0,w0,g0,_,_,_=one(frac,0.0,0.0)
        n,w,mr,is_,oos,rd=one(frac,spread,sslip)
        print(f"  {frac:>6.3f}{n:>5}{w:>5.0f}%{g0:>+13.3f}{mr:>+11.3f}{is_:>+6.2f}/{oos:>+.2f}{rd:>8.2f}")
btc=load_mt5_csv("data/vantage_btcusd_h1.csv"); gold=load_mt5_csv("data/vantage_xauusd_m5.csv")
run_stress("BTC 4h", btc.resample("240min").agg(AGG).dropna(),"kama",15.0,0.5)
run_stress("BTC 1h", btc,"kama",15.0,0.5)
run_stress("GOLD 15m", gold.resample("15min").agg(AGG).dropna(),"sma",0.3,0.5)
run_stress("GOLD 15m", gold.resample("15min").agg(AGG).dropna(),"sma",0.35,0.5)
run_stress("BTC 1h", btc, "kama", 10, 0.5)
print("\n(does the deep-bounce edge SURVIVE the tight-stop cost/slip tax? gross vs net)")
