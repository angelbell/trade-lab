"""ユーザーの指摘への検算（2026-07-14）:
「ICT の損切りもロンドン"安値"＝構造的なスイング安値では？ ブックのスイング安値と何が違うのか」
→ 損切りの置き場所は同じ「構造」。違うのは *エントリーを損切りにどれだけ近づけるか*。
   同じ銘柄(BTC)・同じ時間足(15m)・同じコスト($15)で、損切り幅(R の分母)を直接比べる。"""
import sys, os, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pandas_ta as ta
from src.data_loader import load_mt5_csv
from ict_killzone import load_ny, find_entries, price_and_scan, STOPBUF_DEFAULT, RR_DEFAULT, SYMS
from pine_replica_btc15m import build_entries as book_entries, FRAC as BOOK_FRAC, START as BOOK_START

ROOT = "/home/angelbell/dev/auto-trade"

def show(tag, risks, atrs, cost):
    r = np.asarray(risks, float); a = np.asarray(atrs, float)
    ra = r / a
    cr = cost / r
    print(f"  {tag:34s} n={len(r):5d}  損切り幅/ATR15m: 中央値 {np.median(ra):5.2f} 平均 {ra.mean():5.2f} "
          f"(std {ra.std():4.2f})   コスト/R: 中央値 {np.median(cr):6.3f} 平均 {cr.mean():6.3f}")

print("=" * 108)
print("BTC 15分・コスト $15 往復で、ICT(戻り0.705) と ブックの btc15m_L(戻り0.30) の『損切り幅』を直接比較")
print("=" * 108)

# --- ICT (BTC 15m) ---
with contextlib.redirect_stderr(io.StringIO()):
    dfi, _ = load_ny(SYMS["btcusd"])
recs = price_and_scan(dfi, find_entries(dfi), STOPBUF_DEFAULT, RR_DEFAULT)
atr_i = dfi["atr14"].values
for side in ("long",):
    R, A = [], []
    for r in recs:
        rec = r[side]
        if rec.get("valid") == "ok" and rec.get("filled"):
            R.append(rec["risk"]); A.append(atr_i[rec["entry_pos"]])
    show("ICT NYキルゾーン (戻り0.705)", R, A, 15.0)

# --- book btc15m_L ---
with contextlib.redirect_stderr(io.StringIO()):
    dfb = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_m15.csv")).loc[BOOK_START:]
atr_b = ta.atr(dfb["high"], dfb["low"], dfb["close"], 14).values
E = book_entries(dfb)
R, A = [], []
for (i, e, stop, tgt, w) in E:
    lim = e - BOOK_FRAC * (e - stop)          # 押し目指値
    R.append(lim - stop); A.append(atr_b[i])  # R の分母 = 指値 - 損切り
show(f"btc15m_L 正典 (戻り{BOOK_FRAC})", R, A, 15.0)

# --- ICT の入口はそのままで、戻りだけ「うちの正典の深さ」にしたら？ ---
print("\n  参考: ICT の機構のまま、戻り率だけを振ったときの損切り幅（BTC 15m）")
for f in (0.25, 0.30, 0.50, 0.705, 0.886):
    recs = price_and_scan(dfi, find_entries(dfi, f=f), STOPBUF_DEFAULT, RR_DEFAULT)
    R, A = [], []
    for r in recs:
        rec = r["long"]
        if rec.get("valid") == "ok" and rec.get("filled"):
            R.append(rec["risk"]); A.append(atr_i[rec["entry_pos"]])
    show(f"  ICT 戻り {f}", R, A, 15.0)
