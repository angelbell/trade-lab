"""ドル円掃引の最終候補2本を独立に組み直して照合する。

計測係の報告値:
  候補1 = k2.0 / B系(stop=2.0ATR) / ATR×2トレール / fwd20 / PDH>0 / 週足30MA上
          N=326 年12.3本 勝率46.9% PF1.79 平均+0.0868% maxDD(価格%)2.3% 帰無%ile 99/98 黒字年17/26
  候補2 = k2.5 / A系(拡大足の端) / ATR×3トレール / fwd20 / PDH>0 / 週足30MA上
          N=194 年7.3本 勝率57.7% PF2.27 平均+0.1400% maxDD(価格%)1.8% 帰無%ile 100/100 黒字年19/25
  ショート成行は全域で PF1.11-1.19、前日安値フィルタは帰無%ile 2-24 で不合格。
  円換算(10万円/リスク3%): 候補1 ゲート版 0.09ロット 年+10,716円 最悪年-11,496円 maxDD 26,144円(26.1%)

🚨 価格%のDD(2.3%)と円建てDD(26.1%)が10倍違う点を必ず確かめる。固定ロットでは損切り幅が
   広いトレードほど円の損失が大きくなるので、価格%で正規化したDDは実口座の痛みを表さない。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402

COST = 0.009
ACCOUNT, MAXRISK = 100000.0, 0.03


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


d = load_mt5_csv("data/vantage_usdjpy_h1.csv").loc["2000-01-01":]
o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
ap = wilder_atr(d).shift(1).to_numpy()
day = d.index.floor("D")
_pdh = d["high"].groupby(day).max().shift(1)
pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
wc = d["close"].resample("W").last().dropna()
w30 = wc.rolling(30).mean()
reg_w30 = (wc > w30).shift(1).reindex(d.index, method="ffill").fillna(False).to_numpy()
span = (d.index[-1] - d.index[0]).days / 365.25


def cell(k, stop_mode, trail, use_pdh=True, gate=True, fwd=20):
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    if use_pdh:
        s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    if gate:
        s = s[reg_w30[s]]
    ent = []
    for i in s:
        e = o[i + 1]
        st = l[i] if stop_mode == "A" else e - 2.0 * ap[i]
        if e - st > 0:
            ent.append((i, e, st, e + 1000.0 * (e - st), i))
    if len(ent) < 10:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=trail, trail_n=14)
    t, _ = walk(d, ent, None, a)
    pnl_px = (t["R"] * t["risk"] - COST).to_numpy()
    pct = pnl_px / t["e_px"].to_numpy()
    eq = np.cumsum(pct)
    dd_pct = float((np.maximum.accumulate(eq) - eq).max()) * 100
    # 円建て（固定ロット。損切り幅の中央値でリスク3%に収まる最大ロット）
    med_risk = np.median(t["risk"].to_numpy())
    lots = max(0.01, np.floor(ACCOUNT * MAXRISK / (med_risk * 1000)) * 0.01)
    yen = pnl_px * 1000 * (lots / 0.01)
    eqy = np.cumsum(yen)
    dd_yen = float((np.maximum.accumulate(eqy) - eqy).max())
    yr = pd.Series(yen).groupby(t["time"].dt.year.values).sum()
    w, ls = pct[pct > 0].sum(), -pct[pct < 0].sum()
    return dict(N=len(pct), per_yr=len(pct) / span, win=np.mean(pct > 0) * 100,
                pf=w / ls, mean=pct.mean() * 100, dd_pct=dd_pct, pct=pct,
                lots=lots, yen_yr=yen.sum() / span, dd_yen=dd_yen,
                worst=yr.min(), pos=int((yr > 0).sum()), ny=len(yr))


def drop_null(p0, obs, reps=400, seed=53):
    rng = np.random.default_rng(seed)
    n = min(len(obs), len(p0))
    f, m = [], []
    for _ in range(reps):
        x = rng.choice(p0, size=n, replace=False)
        w, ls = x[x > 0].sum(), -x[x < 0].sum()
        f.append(w / ls); m.append(x.mean())
    f, m = np.array(f), np.array(m)
    ow, ol = obs[obs > 0].sum(), -obs[obs < 0].sum()
    return (f < ow / ol).mean() * 100, (m < obs.mean()).mean() * 100


CANDS = [("候補1 k2.0/B系/TR2", 2.0, "B", 2.0), ("候補2 k2.5/A系/TR3", 2.5, "A", 3.0),
         ("参考 k2.5/B系/TR3", 2.5, "B", 3.0)]
print(f"{'':<22} {'N':>5} {'N/年':>6} {'勝率':>6} {'PF':>6} {'平均%':>8} {'DD価格%':>8} "
      f"{'ロット':>6} {'年間円':>9} {'DD円':>9} {'最悪年円':>9} {'黒字年':>7}")
for lab, k, sm, tr in CANDS:
    r = cell(k, sm, tr)
    print(f"{lab:<22} {r['N']:5d} {r['per_yr']:6.1f} {r['win']:5.1f}% {r['pf']:6.2f} "
          f"{r['mean']:+8.4f} {r['dd_pct']:8.2f} {r['lots']:6.2f} {r['yen_yr']:9,.0f} "
          f"{r['dd_yen']:9,.0f} {r['worst']:9,.0f} {r['pos']:3d}/{r['ny']:<3d}")

# PDHフィルタの帰無（ゲートON母集団の中で）
print("\n--- 前日高値フィルタの間引き帰無（週足30MA上の母集団内）")
for lab, k, sm, tr in CANDS[:2]:
    base = cell(k, sm, tr, use_pdh=False)
    obs = cell(k, sm, tr, use_pdh=True)
    npf, nm = drop_null(base["pct"], obs["pct"])
    print(f"  {lab:<22} フィルタ無し PF={base['pf']:.2f}(N={base['N']}) → "
          f"あり PF={obs['pf']:.2f}(N={obs['N']}) 帰無%ile(PF{npf:.1f} 平均{nm:.1f})")

# ゲート自体の帰無（無ゲート母集団の中で）
print("\n--- 週足30MAゲートの間引き帰無（無ゲート母集団内）")
for lab, k, sm, tr in CANDS[:2]:
    base = cell(k, sm, tr, gate=False)
    obs = cell(k, sm, tr, gate=True)
    npf, nm = drop_null(base["pct"], obs["pct"])
    print(f"  {lab:<22} 無ゲート PF={base['pf']:.2f}(N={base['N']}) → "
          f"ゲート PF={obs['pf']:.2f}(N={obs['N']}) 帰無%ile(PF{npf:.1f} 平均{nm:.1f})")

r1, r2 = cell(2.0, "B", 2.0), cell(2.5, "A", 3.0)
assert 300 <= r1["N"] <= 345, r1["N"]
assert 1.70 < r1["pf"] < 1.90, r1["pf"]
assert 180 <= r2["N"] <= 205, r2["N"]
assert 2.15 < r2["pf"] < 2.40, r2["pf"]
print(f"\nOK: 候補1 N={r1['N']} PF={r1['pf']:.2f} / 候補2 N={r2['N']} PF={r2['pf']:.2f} を再現")
