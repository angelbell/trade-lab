"""交絡除去: 深乖離LONGフェードの第一歩を stop無し・前方MFE で測り直す。
gold 1h・上昇ゲートON。oversold(close<=MA20-2ATR)後、次20本の前方MFE/MAE(ATR単位, stop無し)を、
同レジーム内ランダム建てのnullと比較。tightストップの交絡を除いても素性が無いかを見る。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv, GOLD_H1_START
import pandas_ta as ta

H = 20  # forward horizon (bars)
d = load_mt5_csv(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:]
c, h, l = d["close"].values, d["high"].values, d["low"].values
ma20 = pd.Series(c).rolling(20).mean().values
atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
sma150 = pd.Series(c).rolling(150).mean().values
up = pd.Series(sma150).diff(10).values > 0   # daily-ish slope proxy on 1h (rough gate)

def fwd_mfe_mae(idx):
    mfe, mae = [], []
    for s in idx:
        if s + 1 + H >= len(c): continue
        e = c[s]  # approx next-open ~ close; fine for excursion shape
        fh = h[s+1:s+1+H]; fl = l[s+1:s+1+H]
        a = atr[s]
        if not np.isfinite(a) or a <= 0: continue
        mfe.append((fh.max() - e) / a)
        mae.append((e - fl.min()) / a)
    return np.array(mfe), np.array(mae)

valid = np.isfinite(ma20) & np.isfinite(atr) & np.isfinite(sma150)
sig = valid & (c <= ma20 - 2*atr) & up
sig_idx = np.where(sig)[0]
pool_idx = np.where(valid & up)[0]   # anyDip null: any bar in same uptrend regime

mfe_s, mae_s = fwd_mfe_mae(sig_idx)
print(f"gold 1h  oversold-in-uptrend  n={len(mfe_s)}  (H={H}bars, stop無し前方excursion)")
print(f"  MFE: median={np.median(mfe_s):.3f} mean={mfe_s.mean():.3f} std={mfe_s.std():.3f} "
      f"p25={np.percentile(mfe_s,25):.3f} p75={np.percentile(mfe_s,75):.3f}")
print(f"  MAE: median={np.median(mae_s):.3f} mean={mae_s.mean():.3f}")
print(f"  MFE/MAE(mean) = {mfe_s.mean()/mae_s.mean():.3f}  (>1.2で深掘り価値)")

# null: 同数ランダムを uptrend pool から2000回、中央値MFEの分布
rng = np.random.default_rng(20260719)
nboot = 2000; n = len(mfe_s)
mfe_all_pool, _ = fwd_mfe_mae(pool_idx)
meds = np.array([np.median(rng.choice(mfe_all_pool, n, replace=True)) for _ in range(nboot)])
pct = (meds < np.median(mfe_s)).mean() * 100
print(f"  anyDip null: signal MFE中央値 {np.median(mfe_s):.3f} は pool(n={len(mfe_all_pool)})ランダムの "
      f"{pct:.1f}%ile (mean媒 {meds.mean():.3f})")
# mean MFE でも
meds_mean = np.array([rng.choice(mfe_all_pool, n, replace=True).mean() for _ in range(nboot)])
pct_mean = (meds_mean < mfe_s.mean()).mean() * 100
print(f"  anyDip null(mean): signal MFE平均 {mfe_s.mean():.3f} は {pct_mean:.1f}%ile")
